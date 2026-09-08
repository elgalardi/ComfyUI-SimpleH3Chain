"""Focused MiniMax H3 scene chaining nodes for ComfyUI.

This first release deliberately reuses the pinned, locally tested Context Loop
runtime while presenting a smaller and stable public interface.  It does not
modify the native MiniMax H3 sampling chain.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import threading
import uuid
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths
import comfy.sd
import comfy.utils
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

from .stable_engine import chain_nodes as _chain
from .stable_engine import nodes as _context
from .image_nodes import (
    NODE_CLASS_MAPPINGS as _IMAGE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .masked_context import (
    apply_masked_av_continuation,
    apply_masked_video_continuation,
)
from .long_edit_nodes import (
    NODE_CLASS_MAPPINGS as _LONG_EDIT_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LONG_EDIT_NODE_DISPLAY_NAME_MAPPINGS,
)


class _SplitTrim(int):
    """INT-compatible internal contract for head context plus H3 tail padding."""

    def __new__(cls, head: int, tail: int):
        obj = int.__new__(cls, max(0, int(head)) + max(0, int(tail)))
        obj.head = max(0, int(head))
        obj.tail = max(0, int(tail))
        return obj


def _safe_run_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "h3_chain"))
    value = value.strip("._-")
    return value or "h3_chain"


_DATE_TOKEN = re.compile(r"%date:([^%]+)%", re.IGNORECASE)


def _expand_date_tokens(value: str, now=None) -> str:
    now = now or datetime.now()

    def replace(match):
        pattern = match.group(1)
        # Comfy-style date fields use the common yyyy/MM/dd vocabulary.
        replacements = (
            ("yyyy", "%Y"), ("YYYY", "%Y"),
            ("yy", "%y"), ("YY", "%y"),
            ("MM", "%m"), ("dd", "%d"), ("DD", "%d"),
            ("HH", "%H"), ("hh", "%H"),
            ("mm", "%M"), ("ss", "%S"),
        )
        for source, target in replacements:
            pattern = pattern.replace(source, target)
        try:
            return now.strftime(pattern)
        except ValueError:
            return now.strftime("%Y-%m-%d")

    return _DATE_TOKEN.sub(replace, str(value or "final"))


def _versioned_final_name(manifest, requested: str) -> str:
    expanded = _expand_date_tokens(requested)
    base = _chain._safe_name(expanded, "final")
    run_name = _chain._safe_name(manifest.get("run_name"), "h3_chain")
    final_dir = os.path.join(
        _chain._output_root(), run_name, "final"
    )
    candidate = base
    version = 2
    while os.path.exists(os.path.join(final_dir, candidate + ".mp4")):
        candidate = f"{base}_v{version}"
        version += 1
    return candidate


def _square_panel(image, size=512):
    """Fit the first BHWC image into a square panel without cropping."""
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError("Continuity references must be ComfyUI IMAGE tensors.")
    frame = image[:1].detach().to(device="cpu", dtype=torch.float32)
    height, width = int(frame.shape[1]), int(frame.shape[2])
    if height < 1 or width < 1:
        raise ValueError("Continuity reference has an invalid resolution.")
    scale = min(float(size) / width, float(size) / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    frame = frame.permute(0, 3, 1, 2)
    frame = F.interpolate(
        frame, size=(new_height, new_width), mode="bilinear",
        align_corners=False,
    )
    pad_left = (size - new_width) // 2
    pad_right = size - new_width - pad_left
    pad_top = (size - new_height) // 2
    pad_bottom = size - new_height - pad_top
    frame = F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return frame.permute(0, 2, 3, 1).contiguous()


class SimpleH3ChainPlan(_chain.MiniMaxH3ChainPlan):
    """Small front end for the stable frame-exact chain planner."""

    RETURN_TYPES = (
        _chain.PLAN_TYPE, "STRING", "INT", "INT", "INT", "STRING", "BOOLEAN"
    )
    RETURN_NAMES = (
        "plan", "summary", "clip_count", "width", "height", "plan_preview",
        "base_preview"
    )
    OUTPUT_TOOLTIPS = _chain.MiniMaxH3ChainPlan.OUTPUT_TOOLTIPS + (
        "Readable production plan with every scene, duration, steps, seed, and prompt.",
        "Master low-resolution base-preview switch for the recursive chain.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect the JSON plan produced by H3 Story Director.",
                }),
                "width": ("INT", {
                    "default": 960, "min": 32, "max": 4096, "step": 32,
                }),
                "height": ("INT", {
                    "default": 544, "min": 32, "max": 4096, "step": 32,
                }),
                "audio_mode": (list(_chain.AUDIO_MODES), {
                    "default": "generated_audio",
                }),
                "output_name": ("STRING", {
                    "default": "h3_chain",
                    "tooltip": "Folder and final-chain name. Use a new name for a new production.",
                }),
                "base_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Decode and display complete low-resolution base scenes. "
                        "Disable to decode only the 39-frame Masked AV context "
                        "tail and skip every base MP4."
                    ),
                }),
            },
            "optional": {
                "block_config": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Optional configuration from H3 Story Director — Multi-Storyboard "
                        "Clips. It selects the join contract without duplicated widgets."
                    ),
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Turn a Story Director JSON plan into independent H3 shots joined by "
        "clean cuts and visual reference continuity."
    )

    @staticmethod
    def _context_configuration(prompt):
        """Read the recursive Context widget once so Plan and Trim agree."""
        found = []
        if isinstance(prompt, dict):
            for node in prompt.values():
                if not isinstance(node, dict):
                    continue
                if node.get("class_type") != "SimpleH3ChainContext":
                    continue
                inputs = node.get("inputs", {})
                if not isinstance(inputs, dict):
                    continue
                found.append((
                    str(inputs.get("context_frames", "cut")),
                    str(inputs.get("context_type", "masked_av")),
                    str(inputs.get("audio_context_frames", "match_video")),
                    int(inputs.get("audio_feather_ticks", 8)),
                ))
        unique = list(dict.fromkeys(found))
        if len(unique) > 1:
            raise ValueError(
                "Simple H3 found Context nodes with different settings. Keep "
                "one Context mode per chain."
            )
        return unique[0] if unique else ("39", "masked_av", "match_video", 8)

    @classmethod
    def IS_CHANGED(cls, prompt=None, **kwargs):
        """Invalidate Plan when the separately wired Context widget changes.

        Context is read from the complete prompt so the recursive graph needs no
        extra configuration cable. ComfyUI cannot otherwise see that hidden
        dependency in the normal input links and may reuse a masked_av plan
        after the user switches the Context node to video (or vice versa).
        """
        return hashlib.sha256(json.dumps(
            cls._context_configuration(prompt),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def _audio_context_value(context_frames, context_type, setting):
        if context_type == "masked_cut":
            # A masked cut is visual-only.  It must not carry a protected audio
            # prefix (or any other temporal bridge) into the next shot.
            return 0
        if context_type == "storyboard_fl2va_continuous_audio":
            # FL2VA keeps an exact independent visual first frame, while the
            # previous generated soundtrack is supplied as a latent reference
            # before the new delivered timeline. Unlike masked_av this adds no
            # protected video prefix and therefore needs no scene-head trim.
            return 39
        if setting == "off":
            return 0
        if setting == "match_video":
            if context_type == "cut_reference":
                return 22
            return 0 if context_frames == "cut" else int(context_frames)
        return int(setting)

    @staticmethod
    def _director_context(context_frames, context_type, director_mode):
        """Resolve the context contract carried inside Story Director's plan."""
        mode = str(director_mode or "Continuous Story")
        requested = str(context_type)
        if requested in (
            "storyboard_ref2va", "storyboard_fl2va_continuous_audio"
        ):
            return "cut", requested
        if requested == "source_control_cut":
            return "cut", requested
        if requested == "source_control_overlap_5":
            return "5", requested
        if requested == "source_edit_contiguous":
            return "39", requested
        if requested in ("masked_av", "masked_cut"):
            return "39", requested
        # Migrate workflows saved with the retired context modes. Cinematic
        # Cuts keeps only a weak visual state reference and begins a fresh
        # camera setup; every other mode defaults to same-take masked continuity.
        return "39", ("masked_cut" if mode == "Cinematic Cuts" else "masked_av")

    def build(
        self, plan_json_input, width, height, audio_mode, output_name,
        base_preview,
        block_config=None,
        prompt=None,
    ):
        context_frames, context_type, audio_context_frames, audio_feather_ticks = (
            self._context_configuration(prompt)
        )
        if str(block_config or "").strip():
            try:
                block_settings = json.loads(str(block_config))
            except json.JSONDecodeError as error:
                raise ValueError("Multi-storyboard block_config is not valid JSON.") from error
            join_mode = str(block_settings.get("join_mode") or "cut_audio_bridge")
            if join_mode == "masked_av":
                context_frames, context_type = "39", "masked_av"
                audio_context_frames = "match_video"
            elif join_mode == "cut_audio_bridge":
                context_frames, context_type = "cut", "storyboard_fl2va_continuous_audio"
                audio_context_frames = "match_video"
            else:
                raise ValueError(f"Unknown multi-storyboard join mode: {join_mode}")
        try:
            raw_plan = json.loads(str(plan_json_input or ""))
            director_mode = (
                raw_plan.get("director_mode", "Continuous Story")
                if isinstance(raw_plan, dict) else "Continuous Story"
            )
        except json.JSONDecodeError:
            director_mode = "Continuous Story"
        context_frames, context_type = self._director_context(
            context_frames, context_type, director_mode
        )
        if context_type == "masked_av":
            context_frames = "39"
        audio_context = self._audio_context_value(
            context_frames, context_type, audio_context_frames
        )
        cut_reference = context_type in ("cut_reference", "masked_cut")
        masked_context = context_type == "masked_av"
        if cut_reference:
            # Visual delivery is a clean cut, but the generated soundtrack keeps
            # a tested 22-frame tail. Five is the smallest native storage window
            # that safely retains the final three visual state frames.
            planned_context = 5
            anchor_mode = "before"
        elif masked_context:
            planned_context = 39
            anchor_mode = "head"
        elif context_frames == "cut":
            planned_context = 1
            anchor_mode = "before"
        else:
            planned_context = int(context_frames)
            anchor_mode = (
                "head"
                if context_type in (
                    "video", "source_edit_contiguous", "source_control_overlap_5"
                )
                else "before"
            )

        run_name = _safe_run_name(output_name)
        cut_contract = (
            ":cut_contract=visual_reference_v2"
            if context_type == "masked_cut" else ""
        )
        fingerprint = (
            "simple-h3-chain-v6-context-contract:"
            f"{width}x{height}:audio={audio_mode}:"
            f"frames={context_frames}:type={context_type}:"
            f"audio_context={audio_context_frames}:feather={audio_feather_ticks}:"
            f"base_preview={int(bool(base_preview))}{cut_contract}"
        )
        result = super().build(
            plan_json=plan_json_input,
            run_name=run_name,
            generation_fingerprint=fingerprint,
            width=width,
            height=height,
            context_length=planned_context,
            # Eleven is intentionally represented as consecutive frame guides:
            # H3's native temporal VAE grid jumps directly from 5 to 22.
            encode_mode=(
                "frames" if context_type in ("images", "cut_reference", "masked_cut")
                or (context_type == "video" and context_frames == "11")
                else "video"
            ),
            anchor_mode=anchor_mode,
            crop="disabled",
            audio_mode=audio_mode,
            audio_context_length=audio_context,
            default_duration_seconds=15.0,
            default_steps=5,
            base_seed=0,
            segment_crf=18,
            # masked_av keeps its original lossless contract: the selected
            # duration is the raw H3 length and later scenes remove only the
            # exact protected overlap. Compensating to a second 17k+5 length
            # would create extra bridge frames and then discard them at head.
            _preserve_delivered_duration=not masked_context,
            # masked_av recovers every earlier protected overlap by extending
            # only the final scene with its existing prompt.
            _compensate_final_overlap_loss=masked_context,
        )
        result[0]["base_preview"] = bool(base_preview)
        result[0]["compatibility"]["base_preview"] = bool(base_preview)
        return result + (
            self._format_plan_preview(result[0], context_type), bool(base_preview)
        )

    @staticmethod
    def _format_plan_preview(plan, context_type="unknown"):
        compatibility = plan.get("compatibility", {})
        width = compatibility.get("width", "?")
        height = compatibility.get("height", "?")
        compensated = any(
            bool(shot.get("duration_compensated", False))
            for shot in plan.get("shots", [])[1:]
        )
        lines = [
            "SIMPLE H3 CHAIN PLAN",
            "=" * 58,
            str(plan.get("summary", "")),
            f"Resolution: {width} × {height}",
            f"Output: {plan.get('run_name', 'h3_chain')}",
            f"Context mode: {context_type} · duration compensation: "
            f"{'on' if compensated else 'off'}",
        ]

        prefix = str(plan.get("prompt_prefix") or "").strip()
        if prefix:
            lines.extend(["", "GLOBAL INSTRUCTIONS", "-" * 58, prefix])

        for shot in plan.get("shots", []):
            index = int(shot.get("index", 0))
            shot_id = str(shot.get("id") or f"scene_{index:02d}")
            duration = float(shot.get("audio_duration_seconds", 0.0))
            steps = int(shot.get("steps", 0))
            seed = int(shot.get("seed", 0))
            delivered = int(shot.get("delivered_frames", 0))
            prompt = str(shot.get("scene_prompt") or shot.get("prompt") or "").strip()
            lines.extend([
                "",
                f"SCENE {index:02d} — {shot_id}",
                "-" * 58,
                f"Duration: {duration:.2f} s | Delivered: {delivered} frames | Steps: {steps}",
                f"Seed: {seed}",
                "",
                prompt,
            ])

        return "\n".join(lines).strip()


class SimpleH3ChainLoopStart(_chain.MiniMaxH3ChainLoopStart):
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (_chain.PLAN_TYPE, {
                    "tooltip": "Plan from Simple H3 Chain Plan.",
                }),
                "start_clip": ("INT", {
                    "default": 1, "min": 1, "max": _chain.MAX_SHOTS,
                    "tooltip": "Use 1 for a new chain. Choose a later scene to resume from its saved predecessor.",
                }),
            },
            "optional": {
                "source_audio": ("AUDIO", {
                    "tooltip": "Source soundtrack or short audio reference, depending on audio_mode.",
                }),
            },
            "hidden": {
                "initial_state": (_chain.STATE_TYPE,),
            },
        }

    DESCRIPTION = "Start a new chain or resume it from a saved scene checkpoint."


class SimpleH3ChainCurrent(_chain.MiniMaxH3ChainCurrent):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainContext(_chain.MiniMaxH3ChainContext):
    CATEGORY = "MiniMax H3/Simple Chain"

    DESCRIPTION = (
        "Choose a clean cut or carry 1, 5, 11, 22, or 39 frames from the previous "
        "scene as a trimmed video overlap, independent images, or a three-frame "
        "cut reference with continuous audio."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE, {
                    "tooltip": "Current state from Simple H3 Current Scene.",
                }),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning from MiniMax H3 Ref2VA, I2V, or FL2VA.",
                }),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used for visual context.",
                }),
                "latent": ("LATENT", {
                    "tooltip": "Current scene's empty H3 latent.",
                }),
                "context_frames": (["cut", "1", "5", "11", "22", "39"], {
                    "default": "39",
                    "tooltip": (
                        "cut disables previous-scene visual context; audio may still "
                        "continue through audio_context_frames. Other values carry "
                        "that many tail frames into continuation scenes. In video "
                        "mode, 11 is a hybrid midpoint encoded as consecutive frame "
                        "guides because H3's native video grid jumps from 5 to 22."
                    ),
                }),
                "audio_context_frames": ([
                    "match_video", "off", "1", "5", "11", "22", "39"
                ], {
                    "default": "match_video",
                    "tooltip": (
                        "Audio continuity independent from the visual window. "
                        "match_video follows context_frames; for cut_reference it "
                        "keeps the established 22-frame audio tail. off disables "
                        "generated-audio carry. Source-track audio is already continuous."
                    ),
                }),
                "context_type": ([
                    "masked_av",
                    "masked_cut",
                    "storyboard_ref2va",
                    "storyboard_fl2va_continuous_audio",
                    "source_edit_contiguous",
                    "source_control_cut",
                    "source_control_overlap_5",
                ], {
                    "default": "masked_av",
                    "tooltip": (
                        "masked_av preserves an exact 39-frame latent AV prefix for "
                        "the strongest same-shot continuation. masked_cut uses only "
                        "the last accepted visual state as a weak identity/look guide "
                        "before the new timeline; it creates a real camera cut with no "
                        "protected AV prefix or overlap compensation. "
                        "storyboard_ref2va starts each scene independently from its "
                        "matching generated storyboard panel. "
                        "storyboard_fl2va_continuous_audio keeps that exact visual "
                        "cut while supplying the previous 39-frame generated-audio "
                        "tail as a nondelivered rhythm/timbre reference. "
                        "source_edit_contiguous is exclusive to long source-video "
                        "editing: it carries 39 decoded visual frames, trims them "
                        "from delivery, and keeps source audio on its exact timeline."
                    ),
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "Used only by masked_av. Smoothly releases the final audio "
                        "context edge. 8 ticks equals 0.2 seconds at H3's 40 Hz "
                        "audio latent rate; 0 uses a hard boundary."
                    ),
                }),
            },
            "optional": {
                "audio_vae": ("VAE", {
                    "tooltip": "Optional H3 audio VAE for imported first-scene context.",
                }),
                "block_config": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Join contract from the Multi-Storyboard Block Plan node.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "BOOLEAN", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "is_continuation", "latent")
    OUTPUT_TOOLTIPS = _chain.MiniMaxH3ChainContext.OUTPUT_TOOLTIPS + (
        "Masked latent ready for sampling. Connect this output to the sampler.",
    )

    def apply(
        self, state, conditioning, vae, latent, context_frames,
        audio_context_frames, context_type, audio_feather_ticks=8, audio_vae=None,
        block_config=None,
    ):
        if str(block_config or "").strip():
            try:
                block_settings = json.loads(str(block_config))
            except json.JSONDecodeError as error:
                raise ValueError("Multi-storyboard block_config is not valid JSON.") from error
            join_mode = str(block_settings.get("join_mode") or "cut_audio_bridge")
            if join_mode == "masked_av":
                context_frames, context_type = "39", "masked_av"
                audio_context_frames = "match_video"
            elif join_mode == "cut_audio_bridge":
                context_frames, context_type = "cut", "storyboard_fl2va_continuous_audio"
                audio_context_frames = "match_video"
            else:
                raise ValueError(f"Unknown multi-storyboard join mode: {join_mode}")
        director_mode = str(
            state.get("plan", {}).get("director_mode") or "Continuous Story"
        )
        context_frames, context_type = SimpleH3ChainPlan._director_context(
            context_frames, context_type, director_mode
        )
        if context_type == "masked_av":
            context_frames = "39"
        cut_reference = context_type in ("cut_reference", "masked_cut")
        requested_audio = SimpleH3ChainPlan._audio_context_value(
            str(context_frames), context_type, str(audio_context_frames)
        )

        index = int(state["index"])
        plan = state["plan"]
        current_shot = plan["shots"][index - 1]
        planned_trim = max(
            0,
            int(current_shot["raw_frames"])
            - int(current_shot["delivered_frames"]),
        )
        external_first = index == 1 and bool(state.get("external_context"))
        if context_type == "masked_av" and external_first:
            # Imported media has no original sampler latent to copy. Seed the
            # first generated scene through the established VAE-guide path;
            # subsequent generated scenes switch to lossless masked AV.
            context_type = "video"
            context_frames = "39"
            cut_reference = False
        if index == 1 and not external_first:
            first_tail_trim = max(
                0,
                int(current_shot["raw_frames"])
                - int(current_shot["delivered_frames"]),
            )
            if state["plan"]["compatibility"]["audio_mode"] == "source_intro_generated":
                intro = state.get("source_audio_intro")
                if intro is None:
                    raise ValueError("Simple H3 is missing the source audio intro.")
                if audio_vae is None:
                    raise ValueError(
                        "source_intro_generated requires the MiniMax H3 audio VAE "
                        "connected to Simple H3 Context."
                    )
                guided = MiniMaxH3AddGuide.execute(
                    positive=_chain._prepare_native_guide_conditioning(conditioning),
                    latent=latent,
                    frame_idx=0,
                    audio_vae=audio_vae,
                    audio=intro,
                )
                return (
                    guided[0], _SplitTrim(0, first_tail_trim), False, latent
                )
            return (
                _chain._prepare_native_guide_conditioning(conditioning),
                _SplitTrim(0, first_tail_trim),
                False,
                latent,
            )

        previous_frames = state.get("previous_frames")
        if previous_frames is None:
            raise ValueError("Simple H3 context has no previous scene frames.")

        cfg = plan["compatibility"]
        if context_type == "masked_av":
            previous_latent = state.get("previous_latent")
            if previous_latent is None:
                raise ValueError("Masked AV Continuation has no previous sampled H3 latent.")
            masked_latent, trim = apply_masked_av_continuation(
                latent, previous_latent, 39, audio_feather_ticks
            )
            head_trim = int(trim)
            tail_trim = max(0, int(planned_trim) - head_trim)
            return (
                _chain._prepare_native_guide_conditioning(conditioning),
                _SplitTrim(head_trim, tail_trim),
                True,
                masked_latent,
            )
        use_latent_audio = cfg["audio_mode"] in (
            "generated_audio", "source_plus_timeline", "source_intro_generated"
        )
        effective_audio = requested_audio if use_latent_audio else 0
        visual_cut = str(context_frames) == "cut" and not cut_reference
        if visual_cut and effective_audio <= 0:
            return (
                _chain._prepare_native_guide_conditioning(conditioning),
                _SplitTrim(0, planned_trim), False, latent,
            )

        previous_latent = (
            state.get("previous_latent")
            if use_latent_audio and effective_audio > 0 else None
        )
        previous_audio = (
            state.get("previous_audio")
            if use_latent_audio and effective_audio > 0 and external_first else None
        )
        if (effective_audio > 0 and previous_latent is None
                and previous_audio is None and not external_first):
            raise ValueError("Simple H3 context has no previous AV latent.")

        # A cut reference deliberately observes only the final accepted state.
        # Three consecutive tail frames are enough to disambiguate the subject's
        # latest wardrobe/props while remaining weak enough to permit a new shot.
        selected = (
            min(3, int(previous_frames.shape[0])) if cut_reference
            else (0 if visual_cut else int(context_frames))
        )
        out, trim = _context.MiniMaxH3MotionContext().apply(
            conditioning=conditioning,
            vae=vae,
            latent=latent,
            context_frames=previous_frames,
            context_length=selected,
            encode_mode=(
                "video"
                if context_type in (
                    "video", "source_edit_contiguous", "source_control_overlap_5"
                )
                and selected != 11
                else "frames"
            ),
            # Video follows Context Loop's tested head-overlap contract. Image
            # guides and cut references live before the delivered timeline.
            anchor_mode=(
                "head"
                if context_type in (
                    "video", "source_edit_contiguous", "source_control_overlap_5"
                )
                else "before"
            ),
            crop=cfg["crop"],
            # A regular Cut Reference can keep generated sound continuous. A
            # masked_cut deliberately requests zero audio context as well as a
            # fresh camera timeline.
            audio_context_length=(effective_audio if effective_audio > 0 else -1),
            audio_mode="timeline",
            context_latent=previous_latent,
            audio_vae=audio_vae,
            context_audio=previous_audio,
        )
        if context_type in ("video", "source_control_overlap_5"):
            # Duration compensation can add H3-grid padding beyond the actual
            # repeated guide. Remove the guide from the head, but move that
            # extra padding to the tail so the first newly generated motion is
            # not accidentally discarded at every seam.
            head_trim = max(0, int(trim))
            tail_trim = max(0, int(planned_trim) - head_trim)
            trim_contract = _SplitTrim(head_trim, tail_trim)
        else:
            trim_contract = planned_trim
        return (out, trim_contract, True, latent)


class SimpleH3CutReferenceSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE, {
                    "tooltip": "Current state from Simple H3 Current Scene.",
                }),
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Prompt from Simple H3 Current Scene.",
                }),
                "identity_image": ("IMAGE", {
                    "tooltip": (
                        "Clean fallback image used only on scene 1, before any "
                        "previous-scene frames exist."
                    ),
                }),
                "continuity_tag": ([
                    "<Picture 2>", "<Picture 3>", "<Picture 4>",
                    "<Picture 5>", "<Picture 6>", "<Picture 7>",
                    "<Picture 8>", "<Picture 9>",
                ], {"default": "<Picture 4>"}),
                "panel_size": ([384, 512, 640, 768], {"default": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("continuity_sheet", "augmented_prompt", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Build a Ref2VA cut-reference picture from the final three consecutive "
        "frames of the previous accepted scene and append its exact tag to the prompt."
    )

    def build(self, state, prompt, identity_image, continuity_tag, panel_size):
        size = int(panel_size)
        previous = state.get("previous_frames")
        has_previous = bool(
            torch.is_tensor(previous)
            and previous.ndim == 4
            and previous.shape[0]
        )
        panels = []
        if has_previous:
            # The final consecutive frames describe the latest accepted state.
            # This matters when wardrobe, hair, props or the environment changed
            # during the scene: sampling an earlier frame could reintroduce the
            # obsolete state in the following cut.
            tail = previous[-min(3, int(previous.shape[0])):]
            panels.extend(
                _square_panel(tail[index:index + 1], size)
                for index in range(int(tail.shape[0]))
            )
        else:
            panels.append(_square_panel(identity_image, size))
        sheet = torch.cat(panels, dim=2)
        scene = int(state.get("index", 1))
        if not has_previous:
            instruction = (
                f"{continuity_tag} is the clean identity fallback for the first "
                "scene; use the primary reference pictures for exact identity."
            )
            status = f"scene {scene}: clean fallback; no predecessor"
        else:
            instruction = (
                f"{continuity_tag} contains the final {len(panels)} consecutive "
                "frames from the previously accepted scene. Treat them as the "
                "primary and authoritative visual state at the cut. Preserve the current wardrobe, "
                "hairstyle, accessories, physical changes, and persistent props "
                "shown there. Original connected Picture references are secondary "
                "identity anchors only: use them for immutable facial and body identity, "
                "but never use them to restore an obsolete outfit, pose, expression, "
                "prop, background, or lighting state. Treat this scene as a clean cinematic cut "
                "with freedom to choose a new camera angle, framing, and shot size. "
                "Do not continue the previous camera motion and do not recreate "
                "the three-panel layout in the generated video. Copy the background "
                "only when the scene prompt explicitly keeps the same location."
            )
            status = (
                f"scene {scene}: final {len(panels)} consecutive predecessor frames as "
                f"{continuity_tag}"
            )
        augmented_prompt = f"{instruction}\n\n{str(prompt).strip()}"
        return (sheet, augmented_prompt, status)


class SimpleH3SelectContinuityFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The complete, trimmed and accepted scene frames.",
                }),
                "first_position": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "second_position": ("FLOAT", {
                    "default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("frames_for_loop_end", "frame_1", "frame_2", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Pass the complete scene to Loop End so selectable motion context remains "
        "available, while also exposing two isolated continuity frames."
    )

    def select(self, images, first_position, second_position):
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("Continuity Frame Selector requires a non-empty IMAGE batch.")
        count = int(images.shape[0])
        first = round(float(first_position) * (count - 1))
        second = round(float(second_position) * (count - 1))
        if count > 1 and second == first:
            second = min(count - 1, first + 1) if first < count - 1 else first - 1
        frame_1 = images[first:first + 1].detach().to("cpu").clone()
        frame_2 = images[second:second + 1].detach().to("cpu").clone()
        return (
            images, frame_1, frame_2,
            f"carrying full {count}-frame scene; isolated frames "
            f"{first + 1} and {second + 1} selected",
        )


class SimpleH3BaseContextDecode:
    """Decode a full base scene or only the Masked AV visual context tail."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE,),
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status")
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Decode the complete base scene when Plan base_preview is enabled; "
        "otherwise decode only the final Masked AV context tail."
    )

    def decode(self, state, samples, vae):
        plan = state["plan"]
        full_preview = bool(plan.get("base_preview", True))
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        if not torch.is_tensor(latent) or latent.ndim != 5:
            raise ValueError(
                "Simple H3 Base Context Decode requires a 5D H3 video latent."
            )
        if full_preview:
            selected = latent
            mode = "complete base scene"
        else:
            from .masked_context import _pixel_frames
            context_frames = int(plan["compatibility"]["context_length"])
            context_tokens = next(
                (value for value in range(1, int(latent.shape[2]) + 1)
                 if _pixel_frames(value) == context_frames),
                None,
            )
            if context_tokens is None:
                raise ValueError(
                    f"The {context_frames}-frame context has no exact H3 latent boundary."
                )
            if int(latent.shape[2]) < context_tokens:
                raise ValueError("The current H3 latent is shorter than its context tail.")
            selected = latent[:, :, -context_tokens:].contiguous()
            mode = f"context tail only ({context_frames} frames)"
        images = vae.decode(selected)
        if images.ndim == 5:
            images = images.reshape(
                -1, images.shape[-3], images.shape[-2], images.shape[-1]
            )
        status = f"Decoded {mode}: {int(images.shape[0])} frames"
        _chain._LOG.info("Simple H3 %s", status)
        return (images, status)


class SimpleH3LoopTrim(_context.MiniMaxH3LoopTrim):
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Decoded frames from the current H3 scene.",
                }),
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Connect trim_frames from Simple H3 Auto Context.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded scene audio. It is trimmed and frame-locked automatically at 24 fps.",
                }),
                "state": (_chain.STATE_TYPE, {
                    "tooltip": (
                        "Connect Current Scene state to enable Plan-controlled "
                        "tail-only base decoding. Older workflows may leave this empty."
                    ),
                }),
            },
        }

    def trim(self, images, trim_frames, audio=None, state=None):
        head_trim = int(getattr(trim_frames, "head", int(trim_frames)))
        tail_trim = int(getattr(trim_frames, "tail", 0))
        plan = state["plan"] if isinstance(state, dict) else None
        base_preview = bool(plan.get("base_preview", True)) if plan else True
        shot = plan["shots"][int(state["index"]) - 1] if plan else None
        if not base_preview:
            context_length = int(plan["compatibility"]["context_length"])
            if int(images.shape[0]) != context_length:
                raise ValueError(
                    "Latent-only base decode returned %d frames; expected the "
                    "%d-frame context tail." % (int(images.shape[0]), context_length)
                )
            out_images = images
            out_audio = audio
            if audio is not None:
                waveform = audio["waveform"]
                sample_rate = int(audio["sample_rate"])
                head_samples = (
                    0 if audio.get("_simple_h3_source_delivered") else
                    int(round(head_trim / 24.0 * sample_rate))
                )
                wanted = int(round(
                    int(shot["delivered_frames"]) / 24.0 * sample_rate
                ))
                available = waveform[..., head_samples:]
                if int(available.shape[-1]) >= wanted:
                    delivered_waveform = available[..., :wanted]
                else:
                    delivered_waveform = F.pad(
                        available, (0, wanted - int(available.shape[-1]))
                    )
                out_audio = {
                    "waveform": delivered_waveform,
                    "sample_rate": sample_rate,
                }
        elif tail_trim:
            total = int(images.shape[0])
            if head_trim + tail_trim >= total:
                raise ValueError(
                    "Simple H3 split trim would remove the complete clip: "
                    f"head={head_trim}, tail={tail_trim}, total={total}."
                )
            out_images = images[head_trim:total - tail_trim]
            out_audio = audio
            if audio is not None:
                waveform = audio["waveform"]
                sample_rate = int(audio["sample_rate"])
                head_samples = (
                    0 if audio.get("_simple_h3_source_delivered") else
                    int(round(head_trim / 24.0 * sample_rate))
                )
                wanted = int(round(
                    int(out_images.shape[0]) / 24.0 * sample_rate
                ))
                available = waveform[..., head_samples:]
                if int(available.shape[-1]) >= wanted:
                    delivered_waveform = available[..., :wanted]
                else:
                    delivered_waveform = F.pad(
                        available,
                        (0, wanted - int(available.shape[-1])),
                    )
                out_audio = {
                    "waveform": delivered_waveform,
                    "sample_rate": sample_rate,
                }
            print(
                "[Simple H3 Trim] split seam: removed "
                f"{head_trim} context frames from head and {tail_trim} H3-grid "
                "padding frames from tail."
            )
        else:
            out_images, out_audio = super().trim(
                images=images,
                trim_frames=head_trim,
                audio=audio,
                fps=24.0,
                match_tail=True,
            )
        # Segment Save decides whether the active plan is masked_av. Preserve
        # the frame-locked raw decode as private AUDIO payload so masked-only
        # final assembly can let the new extension own its protected overlap.
        # Other context modes continue saving/assembling ``waveform`` exactly
        # as before.
        if audio is not None and out_audio is not None:
            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"])
            raw_frames = (
                int(shot["raw_frames"]) if shot is not None else int(images.shape[0])
            )
            raw_samples = int(round(raw_frames / 24.0 * sample_rate))
            if int(waveform.shape[-1]) > raw_samples:
                raw_waveform = waveform[..., :raw_samples]
            elif int(waveform.shape[-1]) < raw_samples:
                raw_waveform = F.pad(
                    waveform, (0, raw_samples - int(waveform.shape[-1]))
                )
            else:
                raw_waveform = waveform
            out_audio = dict(out_audio)
            out_audio["_simple_h3_raw_waveform"] = raw_waveform
            out_audio["_simple_h3_raw_frames"] = raw_frames
        return (out_images, out_audio)


class SimpleH3ChainSegmentSave(_chain.MiniMaxH3ChainSegmentSave):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainReview(_chain.MiniMaxH3ChainReview):
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Review the current saved Simple H3 scene with synchronized audio. "
        "Approve, edit and retry, reroll the seed, stop, or enable Continue to "
        "display every segment while running automatically to the final video. Review audio is "
        "automatically fitted to the delivered video clock without changing "
        "the saved scene or checkpoint."
    )


class SimpleH3ChainLoopEnd(_chain.MiniMaxH3ChainLoopEnd):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainAssemble(_chain.MiniMaxH3ChainAssemble):
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (_chain.MANIFEST_TYPE, {
                    "tooltip": "Completed chain manifest from Simple H3 Chain End.",
                }),
                "filename": ("STRING", {
                    "default": "%date:yyyy-MM-dd%",
                    "tooltip": "Final MP4 name. Date tokens are supported.",
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep the final video and all recovery files. Disable to "
                        "show one temporary final preview and remove this run's "
                        "segments, checkpoints, prompts, reviews and manifest."
                    ),
                }),
            },
            "optional": {
                "source_audio": ("AUDIO", {
                    "tooltip": "Connect the full source song only when the plan uses source audio.",
                }),
                "preserve_recovery": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Keep checkpoints available for a downstream refined pass even "
                        "when the base final itself is only a temporary preview."
                    ),
                }),
            },
        }

    @staticmethod
    def _temporary_preview_path(manifest):
        output_root = os.path.abspath(folder_paths.get_output_directory())
        run_name = _safe_run_name(manifest.get("run_name", "h3_chain"))
        preview_dir = os.path.abspath(os.path.join(
            output_root, run_name, "previews", "base"
        ))
        if os.path.commonpath([output_root, preview_dir]) != output_root:
            raise ValueError("Simple H3 temporary preview escaped the output folder.")
        os.makedirs(preview_dir, exist_ok=True)
        for name in os.listdir(preview_dir):
            if not name.startswith("temporary_base_preview."):
                continue
            path = os.path.join(preview_dir, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except OSError as exc:
                # A browser may briefly retain an old range request on Windows.
                # A unique next filename lets this run finish; the stale preview
                # is retried on the next temporary assembly.
                _chain._LOG.warning(
                    "Simple H3 could not remove old temporary preview %s: %s",
                    path, exc,
                )
        return os.path.join(
            preview_dir, "temporary_base_preview.%s.mp4" % uuid.uuid4().hex
        )

    @staticmethod
    def _remove_completed_run(manifest, keep_path=None, keep_paths=None):
        output_root = os.path.abspath(folder_paths.get_output_directory())
        run_name = _safe_run_name(manifest.get("run_name", "h3_chain"))
        run_dir = os.path.abspath(os.path.join(output_root, run_name))
        retained = {
            os.path.abspath(value) for value in (
                list(keep_paths or []) + ([keep_path] if keep_path else [])
            ) if value
        }
        if (os.path.commonpath([output_root, run_dir]) != output_root
                or run_dir == output_root
                or any(os.path.commonpath([run_dir, value]) != run_dir
                       for value in retained)):
            raise ValueError("Simple H3 refused to clean an unsafe run path.")

        def purge(path):
            resolved = os.path.abspath(path)
            if resolved in retained:
                return
            if os.path.isdir(resolved) and not os.path.islink(resolved):
                for name in os.listdir(resolved):
                    purge(os.path.join(resolved, name))
                try:
                    os.rmdir(resolved)
                except OSError:
                    # The directory either contains the retained preview or a
                    # browser still has one file open. Both are safe to keep.
                    pass
            else:
                _chain._safe_unlink(resolved)

        def remove(attempt=0):
            if not os.path.isdir(run_dir):
                return
            try:
                for name in os.listdir(run_dir):
                    purge(os.path.join(run_dir, name))
            except OSError as exc:
                if attempt >= 3:
                    _chain._LOG.warning(
                        "Simple H3 temporary run cleanup still failed after "
                        "retries (%s): %s", run_dir, exc,
                    )
                    return
                timer = threading.Timer(5.0 * (attempt + 1), remove, [attempt + 1])
                timer.daemon = True
                timer.start()
        remove()

    def assemble(self, manifest, filename, save_output, source_audio=None,
                 preserve_recovery=False):
        final_name = _versioned_final_name(manifest, filename)
        result = super().assemble(
            manifest=manifest,
            audio_source="plan",
            filename=final_name,
            audio_bitrate=256,
            source_audio=source_audio,
            publish_review=bool(save_output),
        )
        if save_output:
            return result

        final_path = os.path.abspath(result["result"][0])
        preview_path = self._temporary_preview_path(manifest)
        os.replace(final_path, preview_path)
        if not bool(preserve_recovery):
            self._remove_completed_run(manifest, keep_path=preview_path)
        status = (
            "temporary final preview ready; " + (
                "kept recovery artifacts for downstream refinement"
                if bool(preserve_recovery) else
                "removed this run's segments, checkpoints, prompts, reviews, "
                "manifest and permanent output"
            )
        )
        _chain._LOG.info("Simple H3 %s -> %s", status, preview_path)
        _chain._publish_final_review_preview(manifest, preview_path, status)
        return {
            "ui": {
                "text": [status],
                "videos": [_chain._video_output_item(preview_path)],
            },
            "result": (preview_path,),
        }

    DESCRIPTION = (
        "Join every accepted scene and automatically follow the plan's audio "
        "mode. Save Output can retain the production or clean its intermediate "
        "files after publishing one temporary final preview."
    )


class SimpleH3BasePreviewAssemble(SimpleH3ChainAssemble):
    """Base comparison assembler that never destroys refinement checkpoints."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (_chain.MANIFEST_TYPE,),
                "filename": ("STRING", {"forceInput": True}),
                "save_output": ("BOOLEAN", {"forceInput": True}),
                "assemble_base_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Create and publish the complete low-resolution base comparison "
                        "video. Disable to pass directly to final latent refinement."
                    ),
                }),
            },
            "optional": {"source_audio": ("AUDIO",)},
        }

    DESCRIPTION = (
        "Assemble the automatic base preview while always preserving scene "
        "checkpoints for the downstream final latent upscale/refinement path."
    )

    def assemble(self, manifest, filename, save_output, assemble_base_video=True,
                 source_audio=None):
        if not bool(assemble_base_video):
            status = (
                "Base final assembly disabled; preserved latent checkpoints for "
                "downstream refinement"
            )
            _chain._LOG.info("Simple H3 %s", status)
            return {"ui": {"text": [status]}, "result": ("",)}
        return super().assemble(
            manifest=manifest,
            filename=filename,
            save_output=save_output,
            source_audio=source_audio,
            preserve_recovery=True,
        )


STORYBOARD_LAYOUT_TYPE = "H3_STORYBOARD_LAYOUT"


def _storyboard_grid(scene_count):
    """Storyboard Cuts uses one unambiguous left-to-right timeline."""
    return max(1, int(scene_count)), 1


class SimpleH3StoryboardPlanSplit:
    """Separate one Director response into still-image and Ref2VA contracts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect plan_json from H3 Story Director — Storyboard Cuts.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("storyboard_plan_json", "ref2va_plan_json")
    FUNCTION = "split"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Creates a still-only storyboard plan and a compact video-only Ref2VA plan. "
        "Every cropped panel is normalized to <Picture 1> in the video plan."
    )

    def split(self, plan_json):
        try:
            full_plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Plan Split requires valid Director JSON.") from error
        shots = list(full_plan.get("shots") or [])
        if not 1 <= len(shots) <= 6:
            raise ValueError("Storyboard Plan Split requires between 1 and 6 scenes.")

        storyboard_plan = {
            "storyboard_prompt_prefix": str(
                full_plan.get("storyboard_prompt_prefix")
                or full_plan.get("prompt_prefix")
                or ""
            ).strip(),
            "shots": [
                {
                    "id": str(shot.get("id") or f"scene_{index:02d}"),
                    "storyboard_prompt": str(
                        shot.get("storyboard_prompt") or shot.get("prompt") or ""
                    ).strip(),
                }
                for index, shot in enumerate(shots, 1)
            ],
        }
        sole_reference = (
            "<Picture 1> is the sole authoritative first-frame reference for the "
            "current scene. Preserve every visible identity, wardrobe, hairstyle, "
            "accessory, prop, environment and composition shown in it."
        )
        video_prefix = re.sub(
            r"<Picture\s+\d+>", "<Picture 1>",
            str(full_plan.get("prompt_prefix") or "").strip(),
            flags=re.IGNORECASE,
        )
        video_prefix = "\n".join(
            line for line in video_prefix.splitlines() if line.strip()
        )
        defaults = dict(full_plan.get("defaults") or {})
        ref2va_plan = {
            "prompt_prefix": "\n\n".join(
                value for value in (sole_reference, video_prefix) if value
            ),
            "director_mode": "Cinematic Cuts",
            "defaults": defaults,
            "shots": [
                {
                    "id": str(shot.get("id") or f"scene_{index:02d}"),
                    "prompt": re.sub(
                        r"<Picture\s+\d+>", "<Picture 1>",
                        str(shot.get("prompt") or "").strip(),
                        flags=re.IGNORECASE,
                    ),
                    "duration_seconds": shot.get(
                        "duration_seconds", defaults.get("duration_seconds", 5.0)
                    ),
                    "steps": shot.get("steps", defaults.get("steps", 6)),
                }
                for index, shot in enumerate(shots, 1)
            ],
        }
        return (
            json.dumps(storyboard_plan, ensure_ascii=False, indent=2),
            json.dumps(ref2va_plan, ensure_ascii=False, indent=2),
        )


class SimpleH3StoryboardSheetPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect plan_json from H3 Story Director — Clean Cuts.",
                }),
                "scene_width": ("INT", {
                    "default": 960, "min": 32, "max": 4096, "step": 32,
                }),
                "scene_height": ("INT", {
                    "default": 544, "min": 32, "max": 4096, "step": 32,
                }),
                "max_sheet_width": ("INT", {
                    "default": 8192, "min": 1024, "max": 16384, "step": 32,
                    "tooltip": (
                        "Maximum horizontal canvas. Each panel keeps the requested "
                        "video ratio; all panels scale down together only if needed."
                    ),
                }),
            }
        }

    RETURN_TYPES = (
        "STRING", STORYBOARD_LAYOUT_TYPE, "INT", "INT", "INT", "STRING"
    )
    RETURN_NAMES = (
        "storyboard_prompt", "layout", "sheet_width", "sheet_height",
        "scene_count", "layout_summary"
    )
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Turn a Clean Cuts Director plan into one strict H3 storyboard contact-sheet prompt."
    )

    def build(self, plan_json, scene_width, scene_height, max_sheet_width):
        try:
            plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Sheet Prompt requires valid Director plan JSON.") from error
        shots = plan.get("shots") or []
        count = len(shots)
        if not 1 <= count <= 6:
            raise ValueError("A horizontal H3 storyboard sheet supports between 1 and 6 scenes.")
        cols, rows = _storyboard_grid(count)
        cell_count = cols * rows
        panel_width = max(32, round(int(scene_width) / 32) * 32)
        panel_height = max(32, round(int(scene_height) / 32) * 32)
        requested_sheet_width = panel_width * count
        limit = max(1024, int(max_sheet_width))
        if requested_sheet_width > limit:
            scale = limit / float(requested_sheet_width)
            panel_width = max(32, math.floor(panel_width * scale / 32) * 32)
            panel_height = max(32, round(panel_height * scale / 32) * 32)
        sheet_width = panel_width * count
        sheet_height = panel_height

        prefix = str(
            plan.get("storyboard_prompt_prefix") or plan.get("prompt_prefix") or ""
        ).strip()
        separator_count = max(0, count - 1)
        separator_positions = ", ".join(
            f"{index}/{count}" for index in range(1, count)
        )
        lines = [
            "subject_definitions:",
            prefix or "Preserve the exact identities and requested visual design from the connected Picture references.",
            "",
            "summary:",
            f"[reference generation] One coherent cinematic narrative sequence presented as exactly {count} successive static key moments in one horizontal storyboard strip.",
            "",
            "retention_analysis:",
            "Preserve the same subject identities, facial features, body proportions, wardrobe, hairstyle, accessories, persistent props, architecture, palette and lighting logic throughout every panel, except for explicit chronological changes described below.",
            "",
            "detailed_description:",
            f"Create exactly {count} equal shots in one horizontal row, read strictly from left to right: SHOT 1 through SHOT {count}.",
            "Treat every shot as a successive key moment from one coherent chronological sequence, not as an independent variation of the same reference.",
            "Distribute the complete requested action across the entire strip as clear cause-and-effect milestones. Do not complete the whole story in the first shot or repeat the same climax in later shots.",
            "Each panel is one finished static first-frame composition with precise subject placement, pose, expression, wardrobe, environment, shot size, camera angle, lens, focus and lighting.",
            "Give every adjacent panel decisively different cinematic coverage and a different dramatic purpose. Do not repeat nearly identical framing, camera height, angle, lens impression, subject arrangement or negative-space pattern.",
            "Every adjacent panel must show a later, visibly different chronological action state. Never render the same pose and activity twice with only stronger expressions, a tighter crop or a slightly shifted viewpoint.",
            "Build a coherent visual progression using motivated wide shots, medium coverage, over-the-shoulder views, profiles, close-ups, inserts, foreground depth or high/low viewpoints as appropriate to the story, while preserving screen direction, geography and eyelines.",
            "Reserve the strongest framing and greatest visual emphasis for the most important beat instead of giving every panel equal intensity.",
            "The connected Picture references preserve subject identity and explicitly requested lasting visual traits; their original pose, framing, expression, camera angle and background are free to change unless explicitly requested.",
            "Characters remain absorbed in their objectives, environment and one another. Their gaze, hands and posture are motivated by the visible task; they do not pose or look into the camera unless explicitly requested.",
            "The canvas is divided into exact equal-width rectangular cells with no partial, extra or implied cell at either edge.",
            (
                f"Draw exactly {separator_count} thin straight vertical structural separator"
                f"{'s' if separator_count != 1 else ''} at the exact canvas fractions: {separator_positions}."
                if separator_count else
                "Use the complete canvas as the single storyboard cell with no subdivision."
            ),
            "The structural separators run continuously from the top edge to the bottom edge. They are layout guides, not decorative frames.",
            "Keep every person and object fully inside its assigned cell and leave a small safe margin beside every separator.",
            "No text, captions, numbers, labels, decorative outer borders, speech bubbles, arrows, watermarks, overlapping panels or subjects crossing cell boundaries.",
        ]
        for index, shot in enumerate(shots, 1):
            static_prompt = str(
                shot.get("storyboard_prompt") or shot.get("prompt") or ""
            ).strip()
            if not static_prompt:
                raise ValueError(f"Scene {index} has no storyboard prompt.")
            lines.append(f"SHOT {index}: {static_prompt}")
        lines.extend((
            "",
            "overall_soundscape:",
            "N/A",
            "",
            "non_diegetic_music:",
            "N/A",
        ))
        layout = {
            "version": 1,
            "scene_count": count,
            "columns": cols,
            "rows": rows,
            "scene_width": panel_width,
            "scene_height": panel_height,
            "sheet_width": sheet_width,
            "sheet_height": sheet_height,
        }
        summary = (
            f"{count} scenes · one horizontal row · {sheet_width}×{sheet_height} sheet "
            f"· {panel_width}×{panel_height} per panel"
        )
        storyboard_prompt = "\n\n".join(lines)
        return (
            storyboard_prompt, layout, sheet_width, sheet_height, count, summary
        )


class SimpleH3StoryboardGridSheetPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_json": ("STRING", {"forceInput": True}),
            "panel_size": ("INT", {
                "default": 768, "min": 384, "max": 8192, "step": 32,
                "tooltip": "Long edge of every storyboard panel before sheet safety scaling.",
            }),
            "panel_aspect": ([
                "landscape 16:9", "landscape 3:2", "landscape 4:3",
                "square 1:1",
                "portrait 3:4", "portrait 2:3", "portrait 9:16",
            ], {
                "default": "landscape 16:9",
                "tooltip": "Real aspect ratio of each generated and later regenerated panel.",
            }),
            "max_sheet_edge": ("INT", {
                "default": 4096, "min": 1024, "max": 8192, "step": 32,
                "tooltip": "Safety limit; all cells scale down together if required.",
            }),
        }}

    RETURN_TYPES = (
        "STRING", STORYBOARD_LAYOUT_TYPE, "INT", "INT", "INT", "STRING"
    )
    RETURN_NAMES = (
        "storyboard_prompt", "layout", "sheet_width", "sheet_height",
        "scene_count", "layout_summary"
    )
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Build one coherent storyboard sheet with square, landscape, or portrait "
        "cells for exactly 1, 2, 4, or 6 scenes."
    )

    def build(self, plan_json, panel_size, panel_aspect, max_sheet_edge):
        try:
            plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Grid Sheet requires valid Director plan JSON.") from error
        shots = plan.get("shots") or []
        count = len(shots)
        grids = {1: (1, 1), 2: (1, 2), 4: (2, 2), 6: (3, 2)}
        if count not in grids:
            raise ValueError(
                f"Coherent Grid Sheet requires exactly 1, 2, 4, or 6 scenes; received {count}."
            )
        columns, rows = grids[count]
        aspect_ratios = {
            "landscape 16:9": 16 / 9,
            "landscape 3:2": 3 / 2,
            "landscape 4:3": 4 / 3,
            "square 1:1": 1.0,
            "portrait 3:4": 3 / 4,
            "portrait 2:3": 2 / 3,
            "portrait 9:16": 9 / 16,
        }
        ratio = aspect_ratios[str(panel_aspect)]
        long_edge = max(384, round(int(panel_size) / 32) * 32)
        if ratio >= 1.0:
            cell_width = long_edge
            cell_height = max(256, round((long_edge / ratio) / 32) * 32)
        else:
            cell_height = long_edge
            cell_width = max(256, round((long_edge * ratio) / 32) * 32)
        edge_limit = max(1024, int(max_sheet_edge))
        sheet_width, sheet_height = columns * cell_width, rows * cell_height
        if max(sheet_width, sheet_height) > edge_limit:
            scale = edge_limit / max(sheet_width, sheet_height)
            cell_width = max(256, math.floor((cell_width * scale) / 32) * 32)
            cell_height = max(256, math.floor((cell_height * scale) / 32) * 32)
            sheet_width, sheet_height = columns * cell_width, rows * cell_height
        prefix = str(
            plan.get("storyboard_prompt_prefix") or plan.get("prompt_prefix") or ""
        ).strip()
        positions = {
            1: ("full-frame",),
            2: ("top", "bottom"),
            4: ("top-left", "top-right", "bottom-left", "bottom-right"),
            6: (
                "top-left", "top-center", "top-right",
                "bottom-left", "bottom-center", "bottom-right",
            ),
        }[count]
        coverage_designs = {
            1: (
                "DECISIVE SINGLE-SHOT COVERAGE — one strong full-frame cinematic composition that establishes the visible action, geography and dramatic intention without looking like a contact sheet",
            ),
            2: (
                "ESTABLISHING COVERAGE — wide or medium-wide environmental composition, clear geography, layered depth, subjects relatively small in frame; avoid a centered portrait two-shot",
                "PAYOFF COVERAGE — intimate close-up, profile, over-the-shoulder or low/high-angle detail selected for the decisive action; decisively different scale, axis and camera height from SHOT 1",
            ),
            4: (
                "GEOGRAPHY — wide establishing composition with foreground, midground and background separation; establish screen direction and avoid portrait-like staging",
                "RELATIONSHIP — medium over-the-shoulder, profile two-shot or lateral composition on a new axis; use asymmetry and motivated eyelines",
                "ACTION DETAIL — close-up, insert, high/low viewpoint or foreground-obstructed angle centered on the story-critical hands, object, expression or physical mechanism; do not repeat the prior two-shot",
                "PAYOFF — the strongest final composition, using an intimate close-up or dramatic wide/low-angle tableau opposite in scale and visual rhythm to SHOT 3",
            ),
            6: (
                "GEOGRAPHY — wide establishing view with layered environmental depth and clear screen direction",
                "APPROACH — medium lateral, tracking-like or over-the-shoulder composition on a distinct axis",
                "TACTICAL DETAIL — close-up or insert of the key face, hands, prop or interaction, with shallow focus",
                "REVERSAL — low-angle, high-angle, profile or foreground-obstructed view that changes camera height and spatial dominance",
                "ESCALATION — dynamic medium-wide diagonal composition with strong depth, asymmetry and a clearly advanced action state",
                "PAYOFF — visually strongest decisive close-up or dramatic wide tableau, opposite in scale and axis to SHOT 5",
            ),
        }[count]
        lines = [
            "subject_definitions:",
            prefix or "Preserve the exact identities and requested visual design from the connected Picture references.",
            "",
            "summary:",
            f"[reference generation] One single coherent storyboard sheet containing exactly {count} {str(panel_aspect)} cinematic panels in a strict {columns}-column by {rows}-row grid.",
            "",
            "retention_analysis:",
            "All panels belong to one chronological production. Preserve identity and every visual fact that the story keeps unchanged. Apply explicit changes of wardrobe, location, props, time or physical state only in the panel where the story introduces them, then retain those changes until replaced.",
            "",
            "detailed_description:",
            f"Create exactly {count} equal {str(panel_aspect)} panels in a strict {columns} by {rows} grid. Every cell is exactly {cell_width} by {cell_height} pixels within a {sheet_width} by {sheet_height} sheet. Read in chronological order across the top row from left to right, then across the bottom row from left to right.",
            "Each grid cell contains exactly one finished static cinematic composition. Do not create extra cells, partial cells, inset pictures, montages inside a cell, or imagery outside the grid.",
            "Use one shared visual world across the entire sheet so faces, hair, body proportions, wardrobe state, persistent props, production design, palette and lighting remain coherent whenever the story says they persist.",
            "Every successive panel must advance the action and use purposeful new coverage. Do not repeat the same action state with only a tighter crop or stronger expression.",
            "COVERAGE CONTRACT: adjacent SHOTs must differ in at least four of these six properties: shot scale, camera axis, camera height, lens character, subject arrangement, and foreground-depth pattern. Never use the same centered eye-level two-shot twice.",
            "Current-shot instructions are authoritative for intentional changes. Never carry an earlier location, wardrobe, pose or prop into a later panel after the story replaces it.",
            f"Keep all story-critical subjects and objects inside the central safe area of each {str(panel_aspect)} panel.",
            "Thin straight separators define the exact grid. No text, captions, numbers, labels, decorative borders, speech bubbles, arrows or watermarks.",
        ]
        for index, (shot, position, coverage) in enumerate(zip(shots, positions, coverage_designs), 1):
            static_prompt = str(
                shot.get("storyboard_prompt") or shot.get("prompt") or ""
            ).strip()
            if not static_prompt:
                raise ValueError(f"Scene {index} has no storyboard prompt.")
            lines.append(
                f"SHOT {index}, {position} cell — MANDATORY CINEMATIC ROLE: {coverage}. "
                f"Adapt the following story beat to that coverage role without changing its action or continuity state: {static_prompt}"
            )
        lines.extend(("", "overall_soundscape:", "N/A", "", "non_diegetic_music:", "N/A"))
        layout = {
            "version": 2, "scene_count": count, "columns": columns, "rows": rows,
            "scene_width": cell_width, "scene_height": cell_height,
            "sheet_width": sheet_width, "sheet_height": sheet_height,
            "panel_aspect": str(panel_aspect),
        }
        return (
            "\n\n".join(lines), layout, sheet_width, sheet_height, count,
            f"{count} scenes · {columns}×{rows} grid · {str(panel_aspect)} · "
            f"{sheet_width}×{sheet_height} sheet · {cell_width}×{cell_height} panels",
        )


class SimpleH3StoryboardGridConsistencyPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard_prompt": ("STRING", {"forceInput": True}),
            "consistency_strength": (["low", "medium", "strong"], {
                "default": "medium",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("consistency_prompt",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Use a deterministically assembled draft grid as the structural reference for "
        "a second whole-sheet consistency pass."
    )

    def build(self, storyboard_prompt, consistency_strength):
        # Picture 1 is reserved for the assembled draft. Original identity references
        # shift one slot to the right without changing their semantic order.
        shifted = re.sub(
            r"<Picture\s+(\d+)>",
            lambda match: f"<Picture {int(match.group(1)) + 1}>",
            str(storyboard_prompt),
            flags=re.IGNORECASE,
        )
        strength_rules = {
            "low": (
                "Use a conservative recast: keep the draft composition extremely close while "
                "correcting recognizable faces and unmistakable wardrobe mistakes."
            ),
            "medium": (
                "Re-render every cell, replacing the draft people with the exact recurring subjects "
                "from the identity references and correcting wardrobe and visible state from each SHOT."
            ),
            "strong": (
                "Perform a strong identity recast and state correction in every cell while retaining "
                "the approved shot design, grid, actions and intentional chronological changes."
            ),
        }
        instruction = "\n\n".join((
            "REPLICATE AND RECAST THE STORYBOARD IN <Picture 1>.",
            "<Picture 1> is the authoritative storyboard template. Copy its exact canvas ratio, grid geometry, "
            "cell boundaries, reading order, composition, camera, pose, action, placement and environment for "
            "each numbered SHOT. Do not add, remove, merge, split, reorder, resize or crop cells.",
            "Match <Picture 1>'s exposure, white balance, saturation, contrast, skin texture and material response. "
            "The result is a faithful identity-and-state correction, with natural skin and restrained color rather "
            "than a glossy, plastic or hyper-saturated restyle.",
            "<Picture 2>, <Picture 3> and <Picture 4>, when connected, are CHARACTER IDENTITY references only, "
            "in that order. Recast the corresponding recurring people inside every relevant cell with their "
            "recognizable face, hair, skin and body identity. Never copy the identity-reference background, "
            "pose, crop, framing or lighting into the storyboard.",
            "The written SHOT specification below is authoritative for wardrobe, nudity or dressed state, "
            "props, participating characters and location. Correct any contradiction visible in the draft. "
            "A persistent garment must have the same cut, material, color and details in every SHOT until the "
            "text explicitly changes or removes it. A removed garment must not reappear later.",
            strength_rules[str(consistency_strength)],
            "Treat every cell independently while preserving story continuity. Include only the characters named "
            "for that SHOT, exactly once. A character absent from a SHOT must not appear, not even in the background.",
            "SHOT-BY-SHOT RECONSTRUCTION SPECIFICATION:",
            shifted,
        ))
        return (instruction,)


class SimpleH3StoryboardPromptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect plan_json from H3 Story Director — Storyboard Cuts.",
                }),
                "base_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "continuity_mode": ([
                    "storyboard_driven",
                    "identity_and_wardrobe",
                    "strict_story_continuity",
                    "identity_only",
                ], {
                    "default": "storyboard_driven",
                    "tooltip": (
                        "Identity only gives the camera maximum freedom. Identity + wardrobe is the "
                        "recommended clean-cut mode. Strict also carries location, props and visible state."
                    ),
                }),
                "generation_scope": (["all_scenes", "selected_scene_only"], {
                    "default": "all_scenes",
                    "tooltip": "Generate the complete storyboard or test/regenerate only one scene.",
                }),
                "selected_scene": ("INT", {
                    "default": 1, "min": 1, "max": 12, "step": 1,
                }),
                "detail_priority": ([
                    "balanced",
                    "faces_and_identity",
                    "wardrobe_and_props",
                    "environment_and_composition",
                ], {"default": "faces_and_identity"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("scene_prompt", "scene_seed", "scene_number", "scene_count", "status")
    OUTPUT_IS_LIST = (True, True, True, False, False)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Create one independent full-resolution H3 image prompt and seed per scene."
    )

    def build(
        self, plan_json, base_seed, continuity_mode,
        generation_scope, selected_scene, detail_priority,
    ):
        try:
            plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Prompt List requires valid Director plan JSON.") from error
        shots = plan.get("shots") or []
        count = len(shots)
        if not 1 <= count <= 12:
            raise ValueError("Individual H3 storyboards support between 1 and 12 scenes.")

        prefix = str(
            plan.get("storyboard_prompt_prefix") or plan.get("prompt_prefix") or ""
        ).strip()
        prefix_lower = prefix.lower()
        if "onlyfans" in prefix_lower or "creator video" in prefix_lower:
            local_genre_direction = (
                "OnlyFans-style adults-only creator direction: intimate self-produced energy, "
                "confident seductive body language, strong mutual attraction, close physical tension, "
                "deliberate flirtatious eyelines and tactile interaction appropriate to the current beat. "
                "In public scenes express this through charged teasing and proximity; in private scenes "
                "allow the performance to become more openly passionate."
            )
        else:
            local_genre_direction = ""
        if generation_scope == "selected_scene_only":
            selected = int(selected_scene)
            if selected > count:
                raise ValueError(
                    f"Selected scene {selected} does not exist; the plan contains {count} scenes."
                )
            indexed_shots = [(selected, shots[selected - 1])]
        else:
            indexed_shots = list(enumerate(shots, 1))

        continuity_rules = {
            "storyboard_driven": "",
            "identity_only": (
                "Preserve the exact identities from the connected Picture references. "
                "The shot, wardrobe, environment, lighting, pose, and camera may change freely as described."
            ),
            "identity_and_wardrobe": (
                "Preserve the exact identities from the connected Picture references and keep each subject's "
                "established wardrobe, hairstyle, accessories, and persistent physical changes consistent. "
                "Treat this as a clean cinematic cut: do not continue the previous camera framing."
            ),
            "strict_story_continuity": (
                "Preserve exact identity, wardrobe, hairstyle, accessories, persistent physical changes, "
                "location state, important props, time of day, and lighting logic established by the story. "
                "The current scene composition is authoritative for every chronological change. "
                "This is a deliberate clean cut with independent framing, not a continuation frame. "
                "Show each listed subject exactly once; never carry a previous pose, action, framing, or "
                "extra copy of a subject into the current image."
            ),
        }
        detail_rules = {
            "director_prompt_only": "",
            "balanced": "Balance subject detail, environment, lighting, and composition.",
            "faces_and_identity": "Spend visual detail first on faces, eyes, skin, hair, hands, and identity-defining features.",
            "wardrobe_and_props": "Spend visual detail first on wardrobe materials, accessories, hands, and story-critical props.",
            "environment_and_composition": "Spend visual detail first on spatial construction, environment, lighting, and camera composition.",
        }

        prompts = []
        seeds = []
        scene_numbers = []
        coverage_by_count = {
            1: ("decisive composition appropriate to the story beat",),
            2: (
                "environmental medium-wide establishing composition with layered depth",
                "intimate payoff close-up or dramatic opposing angle",
            ),
            3: (
                "wide geographic setup with strong foreground and background separation",
                "asymmetric medium relationship coverage on a new camera axis",
                "tight decisive payoff emphasizing the key face, hands or interaction",
            ),
            4: (
                "medium-wide environmental setup with clear geography and screen direction",
                "low-angle or lateral medium relationship shot with asymmetric blocking",
                "high-angle, mirror, over-the-shoulder or foreground-layered close coverage",
                "extreme close-up or dramatic final tableau reserved for the payoff",
            ),
            5: (
                "wide geographic setup", "lateral medium relationship coverage",
                "close action detail", "high/low-angle reversal", "decisive payoff composition",
            ),
            6: (
                "wide geographic setup", "lateral or over-the-shoulder approach",
                "close action insert", "high/low-angle reversal",
                "dynamic diagonal escalation", "decisive payoff composition",
            ),
        }
        coverage_plan = coverage_by_count.get(count) or tuple(
            "a distinct cinematic composition on a new camera axis" for _ in range(count)
        )
        for index, shot in indexed_shots:
            static_prompt = str(
                shot.get("storyboard_prompt") or shot.get("prompt") or ""
            ).strip()
            if not static_prompt:
                raise ValueError(f"Scene {index} has no storyboard prompt.")
            if continuity_mode == "storyboard_driven":
                visible_subjects = sorted(set(re.findall(r"\bS([1-9]\d*)\b", static_prompt)))
                subject_names = ", ".join(f"S{value}" for value in visible_subjects) or "the subjects named in the description"
                coverage = coverage_plan[index - 1]
                shot_genre = str(shot.get("genre_execution") or "").strip()
                performance_direction = " ".join(
                    value for value in (local_genre_direction, shot_genre) if value
                ) or "Direct the visible performance according to the selected production genre and the emotional purpose of this scene."
                parts = [
                    "subject_definitions:",
                    f"The visible cast is {subject_names}. Preserve each referenced identity while applying the wardrobe and current physical state described for this scene.",
                    "summary:",
                    f"[reference generation] One finished cinematic opening frame for scene {index}.",
                    "retention_analysis:",
                    "Preserve recognizable faces, hair, skin and body identity. The current scene description defines wardrobe, location, props and visible state; newly requested styling is a deliberate restyle rather than clothing inherited from a reference image.",
                    "performance_and_genre:",
                    performance_direction,
                    "detailed_description:",
                    f"Mandatory cinematic coverage: {coverage}. This coverage is distinct from the adjacent scenes in scale, axis, camera height, subject arrangement and depth pattern.",
                    static_prompt,
                ]
                prompts.append("\n\n".join(parts))
                seeds.append(int((int(base_seed) + index - 1) & 0xffffffffffffffff))
                scene_numbers.append(index)
                continue
            parts = [
                f"Create one single sharp finished cinematic still image for scene {index}.",
                "One full-frame image only. Not a storyboard, contact sheet, grid, collage, split screen, or sequence.",
                "No text, captions, numbers, labels, panel borders, speech bubbles, arrows, or watermarks.",
                "Prioritize facial identity, skin detail, wardrobe detail, environmental detail, coherent anatomy, and a clean cinematic composition.",
            ]
            if continuity_rules[continuity_mode]:
                parts.append(continuity_rules[continuity_mode])
            if detail_rules[detail_priority]:
                parts.append(detail_rules[detail_priority])
            # Storyboard-driven panels are deliberately self-contained. Repeating
            # the project-wide prefix leaks future cast and locations into them.
            if prefix and continuity_mode != "storyboard_driven":
                parts.extend((
                    "Apply these global identity, appearance, wardrobe, prop, continuity, and visual-style rules:",
                    prefix,
                ))
            parts.extend((
                "CAST AND LOCATION ISOLATION: show only the people explicitly requested in the current Scene composition, exactly once each. Other connected reference people, earlier or future characters, and people from other scenes are absent, including from the background. Use only the current scene's stated location and lighting; never blend an earlier or future location into it.",
                "Scene composition:", static_prompt,
            ))
            prompts.append("\n\n".join(parts))
            seeds.append(int((int(base_seed) + index - 1) & 0xffffffffffffffff))
            scene_numbers.append(index)

        generated_count = len(prompts)
        scope_label = "all scenes" if generation_scope == "all_scenes" else f"scene {int(selected_scene)} only"
        debug_text = "\n\n".join(
            f"================ SCENE {scene_number} · SEED {seed} ================\n{prompt}"
            for scene_number, seed, prompt in zip(scene_numbers, seeds, prompts)
        )
        return (
            prompts,
            seeds,
            scene_numbers,
            generated_count,
            f"prepared {scope_label} as independent full-resolution images · {continuity_mode}\n\n{debug_text}",
        )


class SimpleH3FL2VAStoryboardPromptList(SimpleH3StoryboardPromptList):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_json": ("STRING", {
                "forceInput": True,
                "tooltip": "Connect plan_json from the FL2VA Keyframe Storyboard Director.",
            }),
            "base_seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True,
            }),
            "generation_scope": (["all_scenes", "selected_scene_only"], {
                "default": "all_scenes",
            }),
            "selected_scene": ("INT", {
                "default": 1, "min": 1, "max": 12, "step": 1,
            }),
        }}

    DESCRIPTION = (
        "Build FL2VA still prompts directly from the Director. Visual continuity is controlled "
        "by the generated storyboard images and keyframes, not extra reference-strength wording."
    )

    def build(self, plan_json, base_seed, generation_scope, selected_scene):
        return super().build(
            plan_json=plan_json,
            base_seed=base_seed,
            continuity_mode="storyboard_driven",
            generation_scope=generation_scope,
            selected_scene=selected_scene,
            detail_priority="director_prompt_only",
        )


class SimpleH3StoryboardCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scene_numbers": ("INT",),
                "plan_json": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("storyboard_frames", "scene_count", "status")
    FUNCTION = "collect"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = "Collect individually generated scene images into one ordered IMAGE batch."

    def collect(self, images, scene_numbers, plan_json):
        if not images:
            raise ValueError("Storyboard Collect received no generated images.")
        plan_text = str(plan_json[0] if isinstance(plan_json, list) else plan_json)
        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Collect requires valid Director plan JSON.") from error
        total = len(plan.get("shots") or [])
        if not 1 <= total <= 12:
            raise ValueError("Storyboard Collect requires a plan containing 1 to 12 scenes.")
        numbers = [int(value) for value in scene_numbers]
        if len(numbers) != len(images):
            raise ValueError("Storyboard images and scene numbers became misaligned.")

        cache_key = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(folder_paths.get_output_directory(), "h3_storyboards", cache_key)
        os.makedirs(cache_dir, exist_ok=True)
        for image, scene_number in zip(images, numbers):
            if not 1 <= scene_number <= total:
                raise ValueError(f"Generated scene number {scene_number} is outside the plan.")
            if not torch.is_tensor(image) or image.ndim not in (3, 4):
                raise ValueError(f"Storyboard image {scene_number} is not a valid IMAGE tensor.")
            if image.ndim == 3:
                image = image.unsqueeze(0)
            array = (
                image[0].detach().to(device="cpu", dtype=torch.float32)
                .clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
            )
            Image.fromarray(array).save(
                os.path.join(cache_dir, f"scene_{scene_number:02d}.png"),
                compress_level=2,
            )

        missing = [
            number for number in range(1, total + 1)
            if not os.path.exists(os.path.join(cache_dir, f"scene_{number:02d}.png"))
        ]
        if missing:
            missing_text = ", ".join(str(number) for number in missing)
            raise ValueError(
                "The storyboard cache is not complete. Generate all_scenes once before regenerating "
                f"an individual scene. Missing scene(s): {missing_text}."
            )

        frames = []
        target_size = None
        for scene_number in range(1, total + 1):
            array = np.asarray(
                Image.open(os.path.join(cache_dir, f"scene_{scene_number:02d}.png")).convert("RGB")
            ).copy()
            image = torch.from_numpy(array).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
            size = (int(image.shape[1]), int(image.shape[2]))
            if target_size is None:
                target_size = size
            elif size != target_size:
                image = F.interpolate(
                    image.movedim(-1, 1),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).movedim(1, -1)
            frames.append(image)
        batch = torch.cat(frames, dim=0).contiguous()
        count = len(frames)
        return batch, count, f"assembled {count} cached storyboard images · updated scenes {numbers}"


class SimpleH3StoryboardGridGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard_prompt": ("STRING", {"forceInput": True}),
            "layout": (STORYBOARD_LAYOUT_TYPE,),
            "dark_level": ("FLOAT", {
                "default": 0.08, "min": 0.0, "max": 0.45, "step": 0.01,
            }),
            "light_level": ("FLOAT", {
                "default": 0.18, "min": 0.05, "max": 0.65, "step": 0.01,
            }),
            "separator_width": ("INT", {
                "default": 12, "min": 0, "max": 64, "step": 2,
            }),
            "safe_margin_percent": ("FLOAT", {
                "default": 3.0, "min": 0.0, "max": 12.0, "step": 0.5,
            }),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("grid_guide", "guided_prompt", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Draw an exact neutral storyboard grid and append geometry-only instructions. "
        "The guide is a structural reference, never a color or content reference."
    )

    def build(
        self, storyboard_prompt, layout, dark_level, light_level,
        separator_width, safe_margin_percent,
    ):
        if not isinstance(layout, dict):
            raise ValueError("Storyboard Grid Guide requires a valid storyboard layout.")
        columns = max(1, int(layout.get("columns", layout.get("scene_count", 1))))
        rows = max(1, int(layout.get("rows", 1)))
        count = int(layout.get("scene_count", columns * rows))
        width = int(layout.get("sheet_width", 0))
        height = int(layout.get("sheet_height", 0))
        if width <= 0 or height <= 0:
            cell_w = int(layout.get("scene_width", 768))
            cell_h = int(layout.get("scene_height", 768))
            width, height = columns * cell_w, rows * cell_h
        guide = torch.empty((1, height, width, 3), dtype=torch.float32)
        dark = float(dark_level)
        light = max(dark + 0.02, float(light_level))
        for row in range(rows):
            y0 = round(row * height / rows)
            y1 = round((row + 1) * height / rows)
            for column in range(columns):
                index = row * columns + column
                x0 = round(column * width / columns)
                x1 = round((column + 1) * width / columns)
                value = dark if index % 2 == 0 else light
                guide[:, y0:y1, x0:x1, :] = value
        half = max(0, int(separator_width) // 2)
        separator = min(0.75, max(light + 0.18, 0.42))
        if half > 0:
            for column in range(1, columns):
                x = round(column * width / columns)
                guide[:, :, max(0, x - half):min(width, x + half), :] = separator
            for row in range(1, rows):
                y = round(row * height / rows)
                guide[:, max(0, y - half):min(height, y + half), :, :] = separator
        # A subtle inner frame communicates the safe composition area without
        # adding semantic content, labels, or saturated colors.
        margin_ratio = float(safe_margin_percent) / 100.0
        frame = min(0.55, max(light + 0.08, 0.28))
        frame_width = max(1, int(separator_width) // 4)
        if margin_ratio > 0:
            for row in range(rows):
                y0 = round(row * height / rows)
                y1 = round((row + 1) * height / rows)
                for column in range(columns):
                    index = row * columns + column
                    if index >= count:
                        continue
                    x0 = round(column * width / columns)
                    x1 = round((column + 1) * width / columns)
                    mx = max(frame_width, round((x1 - x0) * margin_ratio))
                    my = max(frame_width, round((y1 - y0) * margin_ratio))
                    xa, xb = x0 + mx, x1 - mx
                    ya, yb = y0 + my, y1 - my
                    guide[:, ya:ya + frame_width, xa:xb, :] = frame
                    guide[:, yb - frame_width:yb, xa:xb, :] = frame
                    guide[:, ya:yb, xa:xa + frame_width, :] = frame
                    guide[:, ya:yb, xb - frame_width:xb, :] = frame
        boundary_rule = (
            f"separator positions and {float(safe_margin_percent):g}% internal safe margins"
            if int(separator_width) > 0 else
            f"seam positions with no visible separator and {float(safe_margin_percent):g}% internal safe margins"
        )
        geometry = (
            "GEOMETRY-ONLY GRID REFERENCE: One connected Picture contains only neutral "
            f"alternating blank cells in an exact {columns} columns by {rows} rows grid. "
            f"Use that Picture exclusively to copy the exact panel boundaries, equal cell "
            f"dimensions, {boundary_rule}. Ignore its black, gray and separator tones "
            "completely. Do not "
            "copy its colors, brightness, texture or blank content. Fill every requested "
            "cell edge-to-edge with the corresponding cinematic shot while keeping all "
            "important subjects inside the safe frame. Produce exactly "
            f"{count} panels with no partial, extra or merged panel."
        )
        prompt = f"{str(storyboard_prompt).strip()}\n\n{geometry}"
        return guide, prompt, (
            f"neutral geometry guide {width}x{height}; {columns}x{rows}; "
            f"{count} panels; separator {int(separator_width)}px; "
            f"safe margin {float(safe_margin_percent):g}%"
        )


class SimpleH3StoryboardSheetRefinePrepare:
    """Prepare one complete storyboard sheet for a higher-resolution reference edit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard_sheet": ("IMAGE",),
            "layout": (STORYBOARD_LAYOUT_TYPE,),
            "base_seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True,
            }),
            "detail_goal": ([
                "balanced_microdetail", "identity_skin_hair",
                "wardrobe_props_textures", "environment_materials",
            ], {"default": "balanced_microdetail"}),
            "target_panel_long_edge": ([
                768, 1024, 1280, 1440, 1536, 1920, 2048,
            ], {
                "default": 1024,
                "tooltip": (
                    "Virtual long edge of each grid cell. The complete sheet dimensions are "
                    "calculated from the Stage 1 layout; 1024 is the recommended first test."
                ),
            }),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "source_sheet", "refine_prompt", "seed", "width", "height", "status",
    )
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Regenerate the complete approved storyboard grid once at higher resolution, "
        "before any panels are split. This node makes no director/API call."
    )

    def prepare(self, storyboard_sheet, layout, base_seed, detail_goal, target_panel_long_edge):
        if not isinstance(storyboard_sheet, torch.Tensor) or storyboard_sheet.ndim != 4:
            raise ValueError("Storyboard Sheet Refine requires one valid IMAGE sheet.")
        source = storyboard_sheet[:1].contiguous()
        columns = max(1, int(layout["columns"]))
        rows = max(1, int(layout["rows"]))
        count = max(1, int(layout["scene_count"]))
        source_cell_width = max(1, int(layout["scene_width"]))
        source_cell_height = max(1, int(layout["scene_height"]))
        target = int(target_panel_long_edge)
        if source_cell_width >= source_cell_height:
            cell_width = target
            cell_height = max(32, round(target * source_cell_height / source_cell_width / 32) * 32)
        else:
            cell_height = target
            cell_width = max(32, round(target * source_cell_width / source_cell_height / 32) * 32)
        width = int(cell_width * columns)
        height = int(cell_height * rows)
        goals = {
            "balanced_microdetail": (
                "Improve fine natural detail evenly in faces, skin, hair, hands, fabric, "
                "props, architecture and background materials."
            ),
            "identity_skin_hair": (
                "Prioritize recognizable facial microstructure, natural skin, eyes and "
                "individual hair strands without beautification drift."
            ),
            "wardrobe_props_textures": (
                "Prioritize exact garment construction, fabric weave, seams, accessories, "
                "hands and story-critical prop surfaces."
            ),
            "environment_materials": (
                "Prioritize architecture, background objects, depth cues, lighting detail "
                "and physically coherent materials."
            ),
        }
        instruction = " ".join((
            "High-resolution reconstruction of the complete storyboard grid in <Picture 1>.",
            f"Recreate exactly the same {columns} by {rows} grid with exactly {count} ordered shots.",
            "Copy every cell boundary, composition, crop, camera angle, subject count, identity,",
            "anatomy, pose, expression, action state, wardrobe, hairstyle, accessories, props,",
            "location, background geometry, lighting direction and color palette from <Picture 1>.",
            goals[str(detail_goal)],
            "Use additional Pictures only to verify the identities already present; never import",
            "their pose, clothing, background or framing into the approved storyboard.",
            "Do not merge, reorder, duplicate, omit or invent panels, people, limbs or objects.",
            "Do not advance the story, redesign the shots, increase saturation or contrast,",
            "beautify the subjects, add text, labels, numbers, borders or a new layout.",
            "Return one complete finished storyboard sheet, not separate images.",
        ))
        return source, instruction, int(base_seed), width, height, (
            f"complete {columns}x{rows} storyboard prepared for one high-resolution reference edit · "
            f"{width}x{height} sheet · {cell_width}x{cell_height} cells · no API call"
        )


class SimpleH3StoryboardRefineDispatch:
    """Expose an ordered storyboard IMAGE batch as sequential source-latent edits."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard_frames": ("IMAGE",),
            "base_seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff,
                "control_after_generate": True,
            }),
            "detail_goal": ([
                "balanced_microdetail", "identity_skin_hair",
                "wardrobe_props_textures", "environment_materials",
            ], {"default": "balanced_microdetail"}),
            "target_panel_size": ([
                768, 1024, 1280, 1440, 1536, 1920, 2048,
                2560, 2880, 3840, 4096, 5120, 5760, 7680, 8192,
            ], {
                "default": 1024,
                "tooltip": (
                    "Target long edge for every independently regenerated panel. "
                    "1440 is the lightweight 2K test tier; 2048 is true 2K square. "
                    "Values above 2048 are experimental and can require extreme VRAM."
                ),
            }),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "source_panel", "refine_prompt", "seed", "width", "height",
        "panel_number", "status",
    )
    OUTPUT_IS_LIST = (True, True, True, True, True, True, False)
    FUNCTION = "dispatch"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Send every storyboard panel through a separate source-latent reference "
        "edit. Its local instructions consume no director/API credits."
    )

    def dispatch(self, storyboard_frames, base_seed, detail_goal, target_panel_size):
        if not isinstance(storyboard_frames, torch.Tensor) or storyboard_frames.ndim != 4:
            raise ValueError("Storyboard Refine Dispatch requires an IMAGE batch.")
        count = int(storyboard_frames.shape[0])
        if count < 1:
            raise ValueError("Storyboard Refine Dispatch received no panels.")
        goals = {
            "balanced_microdetail": (
                "Improve fine natural detail in faces, skin, hair, hands, fabric, "
                "props and environment materials evenly."
            ),
            "identity_skin_hair": (
                "Prioritize recognizable facial microstructure, natural skin texture, "
                "eyes and individual hair strands without beautification drift."
            ),
            "wardrobe_props_textures": (
                "Prioritize exact garment construction, fabric weave, seams, accessories, "
                "hands and story-critical prop surfaces."
            ),
            "environment_materials": (
                "Prioritize architectural surfaces, background objects, depth cues, "
                "lighting detail and physically coherent materials."
            ),
        }
        instruction = " ".join((
            "Reference refinement of <Picture 1>.",
            "Reconstruct <Picture 1> as the same single finished cinematic frame.",
            "Keep its exact composition, crop, camera angle, lens impression, subject count,",
            "identities, anatomy, pose, facial expression, action state, wardrobe, hairstyle,",
            "accessories, props, location, background geometry, lighting direction and palette.",
            goals[str(detail_goal)],
            "Use additional Pictures only to verify identity; never replace the approved",
            "clothing, pose, framing, location or visible state in <Picture 1>.",
            "Do not add or remove people or objects. Do not redesign, restage, recolor,",
            "beautify, increase saturation or contrast, or advance the story.",
            "No grid, collage, split screen, text, labels or borders. One image only.",
        ))
        source_height = int(storyboard_frames.shape[1])
        source_width = int(storyboard_frames.shape[2])
        target = int(target_panel_size)
        # Storyboard cells are square in this workflow. Keep a defensive ratio
        # path so the node remains usable if a future layout changes that rule.
        if source_width >= source_height:
            width = target
            height = max(32, int(round(target * source_height / source_width / 32.0)) * 32)
        else:
            height = target
            width = max(32, int(round(target * source_width / source_height / 32.0)) * 32)
        return (
            [storyboard_frames[i:i + 1].contiguous() for i in range(count)],
            [instruction for _ in range(count)],
            [int((int(base_seed) + i) & 0xffffffffffffffff) for i in range(count)],
            [width for _ in range(count)], [height for _ in range(count)],
            list(range(1, count + 1)),
            f"prepared {count} panels for sequential full-generation reference refinement "
            f"at {width}x{height}; no API call",
        )


class SimpleH3StoryboardSourceLatentInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prepared_latent": ("LATENT",),
            "source_panel": ("IMAGE",),
            "vae": ("VAE",),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("source_latent", "status")
    FUNCTION = "inject"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Replace the empty H3 VIDEO latent with an encoding of the approved panel. "
        "Audio is masked from denoising, making low-denoise sampling real img2img."
    )

    def inject(self, prepared_latent, source_panel, vae):
        import comfy.nested_tensor

        samples = prepared_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("Source Latent Inject expected an H3 joint AV NestedTensor.")
        members = list(samples.unbind())
        if not members:
            raise ValueError("The prepared H3 latent has no video member.")
        template = members[0]
        frame_count = max(1, int(prepared_latent.get(
            "h3_context_frames", prepared_latent.get("h3_requested_frames", 1)
        )))
        frames = source_panel[:1, ..., :3].repeat(frame_count, 1, 1, 1)
        encoded = vae.encode(frames)
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(0).movedim(1, 2)
        if encoded.ndim != 5:
            raise RuntimeError(
                f"Source panel VAE returned an unexpected latent shape: {tuple(encoded.shape)}"
            )
        target_t, target_h, target_w = map(int, template.shape[-3:])
        got_t, got_h, got_w = map(int, encoded.shape[-3:])
        if (got_h, got_w) != (target_h, target_w):
            raise RuntimeError(
                f"Source panel latent is {got_w}x{got_h}; prepared H3 canvas expects "
                f"{target_w}x{target_h}."
            )
        if got_t > target_t:
            encoded = encoded[..., :target_t, :, :]
        elif got_t < target_t:
            tail = encoded[..., -1:, :, :].expand(
                *encoded.shape[:-3], target_t - got_t, got_h, got_w
            )
            encoded = torch.cat((encoded, tail), dim=-3)
        members[0] = encoded.to(device=template.device, dtype=template.dtype)
        result = dict(prepared_latent)
        result["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
        masks = [torch.ones_like(members[0])]
        masks.extend(torch.zeros_like(member) for member in members[1:])
        result["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(masks))
        result["simple_h3_storyboard_source_latent"] = True
        return result, (
            f"real source latent injected: video={tuple(members[0].shape)}; "
            "audio protected from denoising"
        )


class SimpleH3StoryboardRefineCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "refined_panel": ("IMAGE",),
            "panel_number": ("INT",),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("refined_storyboard_frames", "status")
    FUNCTION = "collect"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = "Reassemble sequentially refined panels into one ordered IMAGE batch."

    def collect(self, refined_panel, panel_number):
        if len(refined_panel) != len(panel_number):
            raise ValueError("Refined panels and panel numbers became misaligned.")
        ordered = sorted(
            ((int(number), image) for number, image in zip(panel_number, refined_panel)),
            key=lambda item: item[0],
        )
        received = [number for number, _ in ordered]
        if received != list(range(1, len(ordered) + 1)):
            raise ValueError(f"Refined panel order is incomplete: {received}.")
        images = []
        for number, image in ordered:
            if not isinstance(image, torch.Tensor) or image.ndim != 4:
                raise ValueError(f"Refined panel {number} is not a valid IMAGE tensor.")
            images.append(image[:1].contiguous())
        return torch.cat(images, dim=0), f"collected {len(images)} refined panels in story order"


class SimpleH3StoryboardContactSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard_frames": ("IMAGE",),
            "max_sheet_width": ("INT", {
                "default": 8192, "min": 1024, "max": 16384, "step": 32,
            }),
            "gutter_pixels": ("INT", {
                "default": 4, "min": 0, "max": 64, "step": 1,
                "tooltip": "Deterministic separator width between assembled panels.",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("storyboard_sheet", "storyboard_frames", "status")
    FUNCTION = "assemble"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Assemble independently generated storyboard frames into a compact preview "
        "grid. This is display-only; the original ordered frame batch passes through "
        "unchanged for Ref2VA."
    )

    def assemble(self, storyboard_frames, max_sheet_width, gutter_pixels):
        if not torch.is_tensor(storyboard_frames) or storyboard_frames.ndim != 4:
            raise ValueError("Storyboard Contact Sheet requires an IMAGE batch.")
        count = int(storyboard_frames.shape[0])
        if not 1 <= count <= 12:
            raise ValueError("Storyboard Contact Sheet supports 1 to 12 frames.")
        frames = storyboard_frames[..., :3].float()
        height, width = int(frames.shape[1]), int(frames.shape[2])
        gutter = max(0, int(gutter_pixels))
        columns = 1 if count == 2 else (count if count == 1 else math.ceil(count / 2))
        rows = math.ceil(count / columns)
        requested_width = width * columns + gutter * (columns - 1)
        limit = max(1024, int(max_sheet_width))
        if requested_width > limit:
            usable = max(columns * 32, limit - gutter * (columns - 1))
            panel_width = max(32, math.floor((usable / columns) / 32) * 32)
            panel_height = max(32, round((height * panel_width / width) / 32) * 32)
            frames = F.interpolate(
                frames.movedim(-1, 1), size=(panel_height, panel_width),
                mode="bilinear", align_corners=False, antialias=True,
            ).movedim(1, -1)
            height, width = panel_height, panel_width
        sheet_width = columns * width + gutter * (columns - 1)
        sheet_height = rows * height + gutter * (rows - 1)
        sheet = torch.zeros(
            (1, sheet_height, sheet_width, 3),
            dtype=frames.dtype, device=frames.device,
        )
        for index in range(count):
            row, column = divmod(index, columns)
            y = row * (height + gutter)
            x = column * (width + gutter)
            sheet[:, y:y + height, x:x + width, :] = frames[index:index + 1]
        sheet = sheet.contiguous()
        return (
            sheet, storyboard_frames,
            f"assembled preview grid {columns}×{rows} for {count} panels · "
            f"{sheet_width}×{sheet_height} · {width}×{height} panels · "
            f"{gutter}px gutters; ordered Ref2VA frames passed through unchanged",
        )


class SimpleH3StoryboardSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_sheet": ("IMAGE",),
                "layout": (STORYBOARD_LAYOUT_TYPE,),
                "inset_percent": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": (
                        "Safety crop applied equally to the top, bottom, left and right of every "
                        "panel before resizing. 0.5% removes small grid alignment residues."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("storyboard_frames", "status")
    FUNCTION = "split"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = "Split one generated H3 storyboard sheet into ordered first-frame images."

    def split(self, storyboard_sheet, layout, inset_percent):
        if not torch.is_tensor(storyboard_sheet) or storyboard_sheet.ndim != 4:
            raise ValueError("Storyboard Split requires a valid IMAGE batch.")
        sheet = storyboard_sheet[:1]
        height, width = int(sheet.shape[1]), int(sheet.shape[2])
        cols = int(layout["columns"])
        rows = int(layout["rows"])
        count = int(layout["scene_count"])
        target_width = int(layout["scene_width"])
        target_height = int(layout["scene_height"])
        inset = max(0.0, min(0.10, float(inset_percent) / 100.0))
        panels = []
        for index in range(count):
            row, col = divmod(index, cols)
            x0 = round(col * width / cols)
            x1 = round((col + 1) * width / cols)
            y0 = round(row * height / rows)
            y1 = round((row + 1) * height / rows)
            dx = round((x1 - x0) * inset)
            dy = round((y1 - y0) * inset)
            panel = sheet[:, y0 + dy:y1 - dy, x0 + dx:x1 - dx, :]
            if panel.shape[1] < 1 or panel.shape[2] < 1:
                raise ValueError(f"Storyboard panel {index + 1} became empty after inset cropping.")
            panel = F.interpolate(
                panel.movedim(-1, 1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            panels.append(panel)
        frames = torch.cat(panels, dim=0).contiguous()
        return frames, (
            f"split {count} ordered panels from {cols}×{rows} storyboard grid · "
            f"{float(inset_percent):.2f}% safety crop on every edge"
        )


class SimpleH3CurrentStoryboardFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE,),
                "storyboard_frames": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("first_frame", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = "Select the storyboard panel matching the current Simple H3 loop scene."

    def select(self, state, storyboard_frames):
        index = int(state.get("index", 1)) - 1
        count = int(storyboard_frames.shape[0])
        if index < 0 or index >= count:
            raise ValueError(
                f"Current scene {index + 1} has no storyboard panel; only {count} were supplied."
            )
        return storyboard_frames[index:index + 1], f"scene {index + 1} uses storyboard panel {index + 1}/{count}"


class SimpleH3CurrentStoryboardGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE,),
                "storyboard_frames": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "storyboard_tag": ([
                    "<Picture 2>", "<Picture 3>", "<Picture 4>",
                    "<Picture 5>", "<Picture 6>", "<Picture 7>",
                    "<Picture 8>", "<Picture 9>",
                ], {
                    "default": "<Picture 3>",
                    "tooltip": (
                        "Must match the actual REF2VA input order. With two identity "
                        "references connected first, the storyboard panel is Picture 3."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("storyboard_reference", "guided_prompt", "status")
    FUNCTION = "guide"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Select the current storyboard panel and tell REF2VA to use it as the "
        "approved shot design rather than an exact first-frame lock."
    )

    def guide(self, state, storyboard_frames, prompt, storyboard_tag):
        index = int(state.get("index", 1)) - 1
        count = int(storyboard_frames.shape[0])
        if index < 0 or index >= count:
            raise ValueError(
                f"Current scene {index + 1} has no storyboard panel; only {count} were supplied."
            )
        instruction = (
            f"{storyboard_tag} is the approved storyboard design for this scene. "
            "Use it as the authoritative reference for shot size, camera angle, "
            "composition, character placement, wardrobe state, environment, "
            "lighting, and visual focus. Animate the action described below from "
            "this planned composition. Do not reproduce it as a contact sheet, "
            "split panel, captioned storyboard, or motionless image. The preceding "
            "Picture references remain authoritative for exact subject identity."
        )
        guided = f"{instruction}\n\n{str(prompt).strip()}"
        return (
            storyboard_frames[index:index + 1],
            guided,
            f"scene {index + 1} uses panel {index + 1}/{count} as {storyboard_tag}",
        )


class SimpleH3StoryboardBlockPlanList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_json": ("STRING", {"forceInput": True}),
            "join_mode": (["cut_audio_bridge", "masked_av"], {
                "default": "cut_audio_bridge",
                "tooltip": (
                    "cut_audio_bridge keeps a hard visual cut and supplies prior audio "
                    "as a 39-frame latent rhythm/timbre reference. masked_av carries "
                    "the exact protected AV tail and trims the overlap."
                ),
            }),
        }}

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "block_storyboard_plan", "block_number", "video_chain_plan",
        "block_config", "preview",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard Blocks"
    DESCRIPTION = (
        "Divide one Director plan into 1-4 storyboard sheets and compile one "
        "Ref2VA video scene per complete sheet without another LLM call."
    )

    def build(self, plan_json, join_mode):
        try:
            plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Block Plan requires valid Director JSON.") from error
        config = plan.get("storyboard_block_config") or {}
        storyboard_count = int(config.get("storyboard_count", 0))
        panels_per = int(config.get("panels_per_storyboard", 0))
        duration = int(config.get("clip_duration_seconds", 0))
        if storyboard_count not in (1, 2, 3, 4):
            raise ValueError("Storyboard count must be between 1 and 4.")
        if panels_per not in (1, 2, 4):
            raise ValueError("Panels per storyboard must be 1, 2, or 4.")
        if duration not in (5, 10, 15):
            raise ValueError("Clip duration must be 5, 10, or 15 seconds.")
        shots = list(plan.get("shots") or [])
        expected = storyboard_count * panels_per
        available = len(shots)
        reused_subset = False
        if available < expected:
            raise ValueError(
                f"The held Director plan contains only {available} panels, but the current "
                f"block settings require {expected}. Disable Hold Plan only when increasing "
                "the requested production beyond the cached plan."
            )
        if available > expected:
            # Reducing storyboard_count while Hold Plan is active is safe: reuse
            # the earliest complete chronological blocks without another paid
            # Director request. Increasing still requires a new plan.
            shots = shots[:expected]
            reused_subset = True
        effective = {
            "version": 1,
            "storyboard_count": storyboard_count,
            "panels_per_storyboard": panels_per,
            "clip_duration_seconds": duration,
            "total_panels": expected,
            "join_mode": str(join_mode),
        }
        block_plans = []
        block_numbers = []
        video_shots = []
        preview_lines = []
        if reused_subset:
            preview_lines.append(
                f"HOLD REUSE: using the first {expected}/{available} cached panels; "
                "no Director request required"
            )
        for block_index in range(storyboard_count):
            start = block_index * panels_per
            subset = shots[start:start + panels_per]
            still_plan = dict(plan)
            still_plan["shots"] = subset
            still_plan["scene_count"] = panels_per
            still_plan["summary"] = (
                f"Storyboard block {block_index + 1}/{storyboard_count}: "
                f"panels {start + 1}-{start + panels_per}."
            )
            block_plans.append(json.dumps(still_plan, ensure_ascii=False))
            block_numbers.append(block_index + 1)

            seconds_per_panel = float(duration) / panels_per
            shot_lines = []
            for local_index, shot in enumerate(subset, 1):
                timestamp = (local_index - 1) * seconds_per_panel
                motion = str(shot.get("prompt") or "").strip()
                shot_lines.append(
                    f"[SHOT {local_index} at {timestamp:05.2f}s] {motion}"
                )
            block_prompt = "\n\n".join(filter(None, (
                "subject_definitions:\n<Picture 1> is the complete authoritative "
                f"{panels_per}-panel storyboard for this clip. Read it in normal "
                "grid order, left-to-right and then top-to-bottom. Treat every panel "
                "as a successive full-frame shot, never as a split screen.",
                "detailed_description:\nCreate one continuous finished video that "
                f"follows all {panels_per} storyboard shots during exactly {duration} "
                "seconds. Use motivated camera cuts at the stated times. Preserve the "
                "people, wardrobe, props, geography and visible state shown in each "
                "panel. Never display the grid, borders, labels or multiple panels at once.",
                "\n".join(shot_lines),
                "Maintain natural action and continuous sound across every internal cut. "
                "Do not duplicate subjects, skip panels, reverse their order, freeze on "
                "the storyboard, or introduce an unplanned final shot.",
            )))
            block_shot = dict(subset[0])
            block_shot.update({
                "scene": block_index + 1,
                "title": f"Storyboard block {block_index + 1}",
                "prompt": block_prompt,
                "duration_seconds": float(duration),
                "seed": int(subset[0].get("seed", block_index)),
                "steps": int(subset[0].get("steps", 6)),
            })
            video_shots.append(block_shot)
            preview_lines.append(
                f"BLOCK {block_index + 1}: panels {start + 1}-{start + panels_per} · "
                f"{duration}s · {join_mode}"
            )

        video_plan = dict(plan)
        video_plan["shots"] = video_shots
        video_plan["scene_count"] = storyboard_count
        video_plan["summary"] = (
            f"{storyboard_count} storyboard-driven Ref2VA clips · "
            f"{panels_per} panels × {duration}s each"
        )
        video_plan["storyboard_block_config"] = effective
        return (
            block_plans,
            block_numbers,
            json.dumps(video_plan, ensure_ascii=False),
            json.dumps(effective, ensure_ascii=False, separators=(",", ":")),
            "\n".join(preview_lines),
        )


class SimpleH3StoryboardBlockCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard_sheets": ("IMAGE",),
            "block_numbers": ("INT",),
            "block_config": ("STRING", {"forceInput": True}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("storyboard_sheets", "storyboard_count", "status")
    FUNCTION = "collect"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard Blocks"

    def collect(self, storyboard_sheets, block_numbers, block_config):
        config_text = str(block_config[0] if isinstance(block_config, list) else block_config)
        config = json.loads(config_text)
        expected = int(config["storyboard_count"])
        numbers = [int(value) for value in block_numbers]
        if len(storyboard_sheets) != expected or sorted(numbers) != list(range(1, expected + 1)):
            raise ValueError("Generated storyboard blocks are incomplete or out of order.")
        ordered = [image for _, image in sorted(zip(numbers, storyboard_sheets))]
        normalized = []
        target = None
        for image in ordered:
            if not torch.is_tensor(image) or image.ndim not in (3, 4):
                raise ValueError("Storyboard Block Collect received an invalid IMAGE.")
            image = image.unsqueeze(0) if image.ndim == 3 else image[:1]
            if target is None:
                target = (int(image.shape[1]), int(image.shape[2]))
            elif (int(image.shape[1]), int(image.shape[2])) != target:
                image = F.interpolate(
                    image.movedim(-1, 1), size=target, mode="bilinear",
                    align_corners=False,
                ).movedim(1, -1)
            normalized.append(image[..., :3])
        batch = torch.cat(normalized, dim=0).contiguous()
        return batch, expected, f"collected {expected} complete storyboard sheets"


class SimpleH3StoryboardBlockReferenceRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": (_chain.STATE_TYPE,),
            "storyboard_sheets": ("IMAGE",),
            "prompt": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("storyboard_reference", "block_prompt", "status")
    FUNCTION = "route"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard Blocks"

    def route(self, state, storyboard_sheets, prompt):
        index = int(state["index"]) - 1
        count = int(storyboard_sheets.shape[0])
        if not 0 <= index < count:
            raise ValueError(f"Storyboard block {index + 1} does not exist; only {count} exist.")
        return (
            storyboard_sheets[index:index + 1].contiguous(),
            str(prompt),
            f"video block {index + 1}/{count} uses complete storyboard sheet {index + 1}",
        )


class SimpleH3StoryboardVideoReferenceRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE,),
                "storyboard_frames": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("current_storyboard_image", "guided_prompt", "status")
    FUNCTION = "route"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Select the storyboard image matching the current loop scene. Connect its single image "
        "output only to REF2VA ref_image_0, where it is always <Picture 1>."
    )

    def route(
        self, state, storyboard_frames, prompt,
    ):
        index = int(state.get("index", 1)) - 1
        count = int(storyboard_frames.shape[0])
        if index < 0 or index >= count:
            raise ValueError(f"Current scene {index + 1} has no storyboard image; only {count} exist.")
        storyboard = storyboard_frames[index:index + 1]
        storyboard_tag = "<Picture 1>"
        instruction = (
            f"{storyboard_tag} is the approved final visual design for this scene and is the authoritative "
            "reference for composition, camera angle, shot size, character placement, wardrobe, environment, "
            "lighting, props, identity presentation, and visual focus. Animate the action from this design. "
            "Do not reproduce a contact sheet, split panel, captioned storyboard, "
            "or motionless image."
        )
        guided = f"{instruction}\n\n{str(prompt).strip()}"
        return (
            storyboard, guided,
            f"scene {index + 1}/{count} · current storyboard routed as {storyboard_tag}",
        )


class SimpleH3StoryboardVideoIdentityRouter(SimpleH3StoryboardVideoReferenceRouter):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": (_chain.STATE_TYPE,),
            "storyboard_frames": ("IMAGE",),
            "prompt": ("STRING", {"forceInput": True}),
            "keyframe_mode": (["first_frame_only", "bridge_to_next_panel"], {
                "default": "first_frame_only",
                "tooltip": (
                    "first_frame_only anchors the exact opening panel and lets the "
                    "scene evolve freely. bridge_to_next_panel also anchors the next "
                    "panel at the final frame for continuous transitions."
                ),
            }),
            "identity_reference_count": (["one", "two", "three", "four"], {
                "default": "two",
                "tooltip": (
                    "Original identity images connected after the storyboard panel in REF2VA. "
                    "Picture 1 remains the panel; Picture 2+ map to Subject 1+ in order."
                ),
            }),
        }}

    DESCRIPTION = (
        "Route the current panel as Picture 1 while declaring later REF2VA images as "
        "identity-only references for the recurring subjects."
    )

    def route(self, state, storyboard_frames, prompt, identity_reference_count):
        index = int(state.get("index", 1)) - 1
        count = int(storyboard_frames.shape[0])
        if index < 0 or index >= count:
            raise ValueError(f"Current scene {index + 1} has no storyboard image; only {count} exist.")
        identity_count = {"one": 1, "two": 2, "three": 3, "four": 4}[
            str(identity_reference_count)
        ]
        mappings = " ".join(
            f"<Picture {subject + 1}> is the identity-only reference for <Subject {subject}>."
            for subject in range(1, identity_count + 1)
        )
        instruction = (
            "<Picture 1> is the approved final visual design for this scene and controls composition, "
            "camera angle, shot size, character placement, current wardrobe state, environment, lighting, "
            "props and visual focus. Animate the requested action from this design. "
            f"{mappings} The identity-only references control face, facial structure, skin tone, hair identity, "
            "body identity and stable distinguishing features. They must not restore their original background, "
            "pose, framing or outdated clothing when those conflict with <Picture 1> or the current scene. "
            "When identity appearance conflicts, the identity-only reference wins; for every intentional story "
            "change and shot design, <Picture 1> and the current prompt win. Do not reproduce a contact sheet, "
            "split panel, captioned storyboard or motionless image."
        )
        return (
            storyboard_frames[index:index + 1],
            f"{instruction}\n\n{str(prompt).strip()}",
            f"scene {index + 1}/{count} · panel Picture 1 + {identity_count} identity references",
        )


class SimpleH3StoryboardFL2VAKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": (_chain.STATE_TYPE,),
            "storyboard_frames": ("IMAGE",),
            "prompt": ("STRING", {"forceInput": True}),
            "keyframe_mode": (["first_frame_only", "bridge_to_next_panel"], {
                "default": "first_frame_only",
                "tooltip": (
                    "first_frame_only anchors each cut to its own exact panel. "
                    "bridge_to_next_panel also uses the following panel as the FL2VA last frame."
                ),
            }),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "bridge_prompt", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Use the current storyboard image as the exact FL2VA first frame. Optionally "
        "use the next panel as last frame for a continuous bridge."
    )

    def select(self, state, storyboard_frames, prompt, keyframe_mode):
        index = int(state.get("index", 1)) - 1
        count = int(storyboard_frames.shape[0])
        if index < 0 or index >= count:
            raise ValueError(f"Scene {index + 1} has no storyboard image; only {count} exist.")
        first = storyboard_frames[index:index + 1]
        if str(keyframe_mode) == "bridge_to_next_panel" and index + 1 < count:
            last = storyboard_frames[index + 1:index + 2]
            direction = (
                "The connected first_frame is the exact approved opening composition and the connected "
                "last_frame is the exact approved destination composition. Create natural continuous action "
                "that physically bridges them without cuts, resets, morphing, duplicated subjects, or early arrival. "
                "Preserve identity, wardrobe, props, and environment state while reaching the destination only "
                "near the end of the clip."
            )
            status = f"scene {index + 1}/{count}: bridge storyboard {index + 1} -> {index + 2}"
        else:
            last = None
            if str(keyframe_mode) == "bridge_to_next_panel":
                direction = (
                    "The connected first_frame is the exact approved opening composition. This is the final scene: "
                    "develop the described action freely toward a coherent, deliberate ending without introducing "
                    "a new unresolved event. No last_frame is imposed."
                )
                status = f"scene {index + 1}/{count}: bridge mode final scene; ending free"
            else:
                direction = (
                    "The connected first_frame is the exact approved opening composition and frame zero of this "
                    "independent cinematic shot. Begin from its exact subject placement, pose, wardrobe, props, "
                    "camera, environment and lighting, then develop the requested natural motion without an "
                    "immediate pose jump, camera reset, morph or premature action. No last_frame is imposed."
                )
                status = f"scene {index + 1}/{count}: exact first frame anchored; ending free"
        return first, last, f"{direction}\n\n{str(prompt).strip()}", status


class SimpleH3ChainManifestLoad(_chain.MiniMaxH3ChainManifestLoad):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3OptionalLoraLoader:
    """Model-only LoRA loader with an explicit, API-friendly None option."""

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model to pass through or patch."},
                ),
                "lora_name": (
                    ["None", *folder_paths.get_filename_list("loras")],
                    {
                        "tooltip": (
                            "Choose None to return the input MODEL unchanged. This "
                            "keeps optional LoRA slots permanently connected."
                        )
                    },
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "LoRA strength for the diffusion model. Zero also acts "
                            "as an exact passthrough."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The unchanged or LoRA-patched diffusion model.",)
    FUNCTION = "load_optional_lora"
    CATEGORY = "MiniMax H3/Simple Chain/Loaders"
    DESCRIPTION = (
        "Model-only LoRA loader for fixed API workflows. None and strength 0 "
        "return the exact input model without reading or applying a LoRA."
    )


    SEARCH_ALIASES = ["optional lora", "h3 lora", "lora none", "load lora"]

    def load_optional_lora(self, model, lora_name, strength_model):
        if str(lora_name).strip().lower() == "none" or float(strength_model) == 0.0:
            return (model,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        lora_metadata = None
        if self.loaded_lora is not None:
            cached_path, lora, lora_metadata = self.loaded_lora
            if cached_path != lora_path:
                self.loaded_lora = None
                lora = None
                lora_metadata = None

        if lora is None:
            lora, lora_metadata = comfy.utils.load_torch_file(
                lora_path,
                safe_load=True,
                return_metadata=True,
            )
            self.loaded_lora = (lora_path, lora, lora_metadata)

        patched_model, _ = comfy.sd.load_lora_for_models(
            model,
            None,
            lora,
            float(strength_model),
            0.0,
            lora_metadata=lora_metadata,
        )
        return (patched_model,)


class SimpleH3BasePreview(_chain.MiniMaxH3ChainReview):
    """Automatic scene monitor; the downstream assembler publishes its final here."""

    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_chain.STATE_TYPE,),
                "segment": (_chain.SEGMENT_TYPE,),
                "filename": ("STRING", {"default": "%date:yyyy-MM-dd%"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "show_scene_previews": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Show the accumulated base video through the current scene. "
                        "Disable to pass the scene through without encoding previews."
                    ),
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
                "source_audio": ("AUDIO",),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (_chain.SEGMENT_TYPE, "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("segment", "status", "filename", "save_output")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Automatically displays one accumulated base preview through the current scene, "
        "continues without a review decision, and receives the complete assembled video "
        "in the same player."
    )

    async def preview(self, state, segment, filename, save_output,
                      show_scene_previews=True, audio=None,
                      source_audio=None, dynprompt=None, unique_id=None):
        if not bool(show_scene_previews):
            status = "Base scene preview disabled; continuing directly"
            return {
                "ui": {"text": [status]},
                "result": (segment, status, str(filename), bool(save_output)),
            }
        preview_segment = segment
        partial_warning = ""
        try:
            partial_path, partial_warning = _chain._assemble_review_partial(
                state, segment, "checkpointed", source_audio
            )
            preview_segment = dict(segment)
            preview_segment["segment"] = partial_path
            preview_segment["delivered_frames"] = sum(
                int(item.get("delivered_frames", 0))
                for item in list(state.get("segments", [])) + [segment]
            )
            preview_segment["segment_sha256"] = _chain._file_sha256(partial_path)
            preview_segment["embedded_audio"] = True
        except Exception as error:
            partial_warning = f"Accumulated base preview unavailable; showing current scene ({error})"
            _chain._LOG.warning("Simple H3 %s", partial_warning)
        response = await super().review(
            state=state, segment=preview_segment, enabled=True, Continue=True,
            play_notification_sound=False, auto_continue_timeout_minutes=0.0,
            unload_models_while_waiting=False, assemble_partial_on_stop=False,
            partial_audio_source="checkpointed", audio=None,
            source_audio=source_audio, dynprompt=dynprompt, unique_id=unique_id,
        )
        result = tuple(response["result"])
        status = str(result[1])
        if partial_warning:
            status += f"; {partial_warning}"
        response["result"] = (
            segment, status, str(filename), bool(save_output)
        )
        return response


class SimpleH3DirectEditPreview:
    """Encode and display one complete H3 edit without chain state or segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Complete decoded IMAGE batch from the one-pass H3 edit.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                }),
                "filename": ("STRING", {
                    "default": "%date:yyyy-MM-dd%_h3_direct_edit",
                }),
                "save_output": ("BOOLEAN", {"default": True}),
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Display the completed MP4 inside this node. Disabling it "
                        "does not disable permanent saving when save_output is enabled."
                    ),
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Connect the original source audio for exact-track preview.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "video_path", "status")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Simple Chain/Preview"
    DESCRIPTION = (
        "Simple final player for one-pass H3 video edits. It accepts decoded frames "
        "and optional source audio directly, without chain plan, state or segment inputs."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def preview(
        self, images, fps, filename, save_output, show_preview, audio=None,
        prompt=None, extra_pnginfo=None,
    ):
        if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[0]) < 1:
            raise ValueError("Direct Edit Preview requires IMAGE [frames,height,width,channels].")
        frame_rate = int(round(float(fps)))
        if frame_rate < 1:
            raise ValueError("Direct Edit Preview fps must be positive.")
        height, width = int(images.shape[1]), int(images.shape[2])
        if width % 2 or height % 2:
            raise ValueError(
                f"Direct Edit Preview requires even dimensions; received {width}x{height}."
            )
        if not bool(save_output) and not bool(show_preview):
            status = "Direct edit preview and permanent saving disabled"
            return {"ui": {"text": [status]}, "result": (images, "", status)}

        root = (
            folder_paths.get_output_directory()
            if bool(save_output) else folder_paths.get_temp_directory()
        )
        subfolder = os.path.join("video", "h3_direct_edit")
        directory = os.path.join(root, subfolder)
        os.makedirs(directory, exist_ok=True)
        base = _chain._safe_name(_expand_date_tokens(str(filename)), "h3_direct_edit")
        path = os.path.join(directory, base + ".mp4")
        version = 2
        while os.path.exists(path):
            path = os.path.join(directory, f"{base}_v{version}.mp4")
            version += 1
        transaction = uuid.uuid4().hex
        silent = os.path.join(directory, f".{transaction}.silent.mp4")
        wav_path = os.path.join(directory, f".{transaction}.wav")
        muxed = os.path.join(directory, f".{transaction}.muxed.mp4")
        metadata_path = os.path.join(directory, f".{transaction}.metadata.txt")
        try:
            _chain._write_segment_video(images, silent, frame_rate, 19)
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError(
                    "Direct Edit Preview requires ffmpeg to embed workflow metadata."
                )
            video_metadata = {}
            if prompt is not None:
                video_metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    video_metadata[key] = value
            metadata = json.dumps(video_metadata)
            for source, escaped in (
                ("\\", "\\\\"), (";", "\\;"), ("#", "\\#"),
                ("=", "\\="), ("\n", "\\\n"),
            ):
                metadata = metadata.replace(source, escaped)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                metadata_file.write(";FFMETADATA1\n")
                metadata_file.write("comment=" + metadata)

            if audio is None:
                _chain._run_ffmpeg([
                    ffmpeg, "-y", "-i", silent, "-i", metadata_path,
                    "-map", "0:v:0", "-map_metadata", "1",
                    "-c:v", "copy", "-movflags", "+faststart", muxed,
                ])
            else:
                _chain._write_wav(audio, wav_path)
                duration = float(images.shape[0]) / float(frame_rate)
                _chain._run_ffmpeg([
                    ffmpeg, "-y", "-i", silent, "-i", wav_path,
                    "-i", metadata_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-map_metadata", "2",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-t", f"{duration:.9f}", "-movflags", "+faststart", muxed,
                ])
            os.replace(muxed, path)
            status = (
                f"One-pass H3 edit preview ready · {int(images.shape[0])} frames · "
                f"{width}x{height} · {frame_rate} fps · "
                f"{'source audio' if audio is not None else 'silent'} · workflow metadata"
            )
            ui_result = {"text": [status]}
            if bool(show_preview):
                ui_result["videos"] = [_chain._video_output_item(path)]
            return {
                "ui": ui_result,
                "result": (images, os.path.abspath(path), status),
            }
        finally:
            for temporary in (silent, wav_path, muxed, metadata_path):
                if os.path.exists(temporary):
                    _chain._safe_unlink(temporary)


class SimpleH3LatentUpscaleResolution:
    """Resolve a low-resolution H3 first pass from the requested final size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "latent_upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Off keeps the requested resolution unchanged. On generates the "
                        "chain near half width and half height, then restores the requested "
                        "size with the learned H3 latent upscaler and a short refinement pass."
                    ),
                }),
                "first_pass_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Spatial scale used for the first H3 pass. 0.5 means approximately "
                        "half width and half height, or one quarter of the final pixels."
                    ),
                }),
                "align": ("INT", {
                    "default": 32, "min": 16, "max": 256, "step": 16,
                    "tooltip": "Align the generated width and height to the H3 pixel grid.",
                }),
            },
            "optional": {
                "refine_without_upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Runs the final refinement pass at the requested native resolution "
                        "without spatially upscaling the latent."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "first_pass_width", "first_pass_height", "final_width", "final_height",
        "latent_upscale", "status", "refine",
    )
    FUNCTION = "resolve"
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Calculates an aligned low-resolution first pass while preserving the user's "
        "requested final dimensions. It can also request refinement at native resolution "
        "without resizing the latent. Connect latent_upscale to the windowed upscaler and "
        "refine to the final refinement node."
    )

    @staticmethod
    def _aligned(value, scale, align):
        target = float(value) * float(scale)
        return max(int(align) * 2, int(round(target / int(align))) * int(align))

    def resolve(
        self, width, height, latent_upscale, first_pass_scale, align,
        refine_without_upscale=False,
    ):
        final_width = int(width)
        final_height = int(height)
        enabled = bool(latent_upscale) and float(first_pass_scale) < 0.999
        if not enabled:
            refine = bool(refine_without_upscale)
            return (
                final_width, final_height, final_width, final_height, False,
                (
                    f"Refine only: native {final_width}x{final_height}; no latent resize"
                    if refine else
                    f"Latent upscale and refinement off: native {final_width}x{final_height}"
                ),
                refine,
            )

        first_width = min(
            final_width,
            self._aligned(final_width, first_pass_scale, align),
        )
        first_height = min(
            final_height,
            self._aligned(final_height, first_pass_scale, align),
        )
        if first_width == final_width and first_height == final_height:
            enabled = False
        first_pixels = first_width * first_height
        final_pixels = max(1, final_width * final_height)
        return (
            first_width, first_height, final_width, final_height, enabled,
            (
                f"Latent upscale {'on' if enabled else 'off'}: "
                f"{first_width}x{first_height} -> {final_width}x{final_height} "
                f"({first_pixels / final_pixels:.1%} first-pass pixels)"
            ),
            enabled or bool(refine_without_upscale),
        )


class SimpleH3LatentUpscaleRefine:
    """Optionally upscale an H3 video latent and refine only its video stream."""

    UPSCALE_FOLDER = "latent_upscale_models"

    @classmethod
    def _models(cls):
        if cls.UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                cls.UPSCALE_FOLDER,
                os.path.join(folder_paths.models_dir, cls.UPSCALE_FOLDER),
            )
        models = folder_paths.get_filename_list(cls.UPSCALE_FOLDER)
        return models or ["(place an H3 upscaler in models/latent_upscale_models)"]

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampled_latent": ("LATENT", {
                    "tooltip": "Completed low-resolution H3 AV latent from the primary sampler.",
                }),
                "latent_upscale": ("BOOLEAN", {"default": False}),
                "upscaler_model": (cls._models(),),
                "final_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "final_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "generation_steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "refine_denoise": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.75, "step": 0.01,
                    "tooltip": (
                        "Short high-resolution detail recovery. Lower values preserve the "
                        "low-resolution motion more strictly; higher values may add detail but "
                        "can alter faces or motion."
                    ),
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta57"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("delivery_latent", "context_latent", "status")
    OUTPUT_TOOLTIPS = (
        "Final-resolution AV latent for video/audio decode.",
        "Original low-resolution sampled latent for Masked AV continuation and checkpoints.",
        "Upscale/refinement summary.",
    )
    FUNCTION = "upscale_refine"
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Optional learned 3D latent upscale followed by a short H3 refinement. The audio "
        "latent is preserved exactly. Use delivery_latent for decode and context_latent "
        "for checkpoints plus Simple H3 Loop End so every continuation remains on the "
        "low-resolution grid."
    )

    def upscale_refine(
        self, model, positive, negative, sampled_latent, latent_upscale,
        upscaler_model, final_width, final_height, seed, generation_steps,
        refine_denoise, sampler_name, scheduler,
    ):
        if not bool(latent_upscale):
            return (sampled_latent, sampled_latent, "Latent upscale off; native latent delivered.")

        try:
            from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
                MinimaxH3LatentUpscaler3D,
                UpscaleMode,
            )
            from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
            import nodes as comfy_nodes
        except Exception as error:
            raise RuntimeError(
                "Simple H3 latent upscale requires LBH-123-AI/"
                "Comfyui_Minimax_h3_latent_Upscaler to be installed."
            ) from error

        separated = LTXVSeparateAVLatent.execute(sampled_latent)
        video_latent, audio_latent = separated[0], separated[1]
        source_shape = tuple(video_latent["samples"].shape)
        mode = {
            "mode": UpscaleMode.TARGET_DIMENSIONS,
            "width": int(final_width),
            "height": int(final_height),
        }
        upscaled_video = MinimaxH3LatentUpscaler3D.execute(
            latent=video_latent,
            model_name=str(upscaler_model),
            mode=mode,
            align=32,
            enable_temporal_chunking=True,
            force_unload=True,
            device="cuda",
            precision="fp16",
        )[0]

        combined = LTXVConcatAVLatent.execute(upscaled_video, audio_latent)[0]

        # The low-resolution pass already established composition and motion. A
        # bounded partial denoise restores high-frequency detail without paying
        # for a second full generation at the requested resolution.
        refine_steps = max(2, min(6, int(math.ceil(int(generation_steps) * 0.5))))
        refined = comfy_nodes.common_ksampler(
            model,
            int(seed),
            refine_steps,
            1.0,
            str(sampler_name),
            str(scheduler),
            positive,
            negative,
            combined,
            denoise=float(refine_denoise),
        )[0]

        # H3 jointly predicts audio and video. Keep the original generated audio
        # bit-for-bit at latent level so refinement cannot degrade voices/music.
        refined_video = LTXVSeparateAVLatent.execute(refined)[0]
        delivery = LTXVConcatAVLatent.execute(refined_video, audio_latent)[0]
        delivery = dict(delivery)
        delivery["_simple_h3_context_latent"] = sampled_latent
        target_shape = tuple(refined_video["samples"].shape)
        return (
            delivery,
            sampled_latent,
            (
                f"H3 latent upscale {source_shape[-1] * 16}x{source_shape[-2] * 16} "
                f"-> {target_shape[-1] * 16}x{target_shape[-2] * 16}; "
                f"{refine_steps} refinement steps at denoise {float(refine_denoise):.2f}; "
                "original audio latent preserved."
            ),
        )


class SimpleH3LatentUpscaleRefineAdvanced(SimpleH3LatentUpscaleRefine):
    """Learned H3 latent upscale with native KSampler Advanced controls."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampled_latent": ("LATENT", {
                    "tooltip": "Completed low-resolution H3 AV latent from the primary sampler.",
                }),
                "latent_upscale": ("BOOLEAN", {"default": False}),
                "upscaler_model": (cls._models(),),
                "final_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "final_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "add_noise": (["enable", "disable"], {"default": "enable"}),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0,
                    "step": 0.1, "round": 0.01,
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "start_at_step": ("INT", {
                    "default": 3, "min": 0, "max": 10000,
                    "tooltip": "Skip earlier schedule steps exactly like KSampler Advanced.",
                }),
                "end_at_step": ("INT", {
                    "default": 10000, "min": 0, "max": 10000,
                    "tooltip": "10000 means the end of the selected step schedule.",
                }),
                "return_with_leftover_noise": (["disable", "enable"], {
                    "default": "disable",
                }),
            },
        }

    FUNCTION = "upscale_refine_advanced"
    DESCRIPTION = (
        "Separate Advanced variant of the optional H3 learned latent upscaler. "
        "It has no denoise control: refinement is defined exclusively by the full "
        "step schedule, start_at_step, end_at_step, noise and leftover-noise settings."
    )

    def upscale_refine_advanced(
        self, model, positive, negative, sampled_latent, latent_upscale,
        upscaler_model, final_width, final_height, add_noise, noise_seed,
        steps, cfg, sampler_name, scheduler, start_at_step, end_at_step,
        return_with_leftover_noise,
    ):
        if not bool(latent_upscale):
            return (sampled_latent, sampled_latent, "Latent upscale off; native latent delivered.")

        try:
            from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
                MinimaxH3LatentUpscaler3D,
                UpscaleMode,
            )
            from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
            import nodes as comfy_nodes
        except Exception as error:
            raise RuntimeError(
                "Simple H3 latent upscale requires LBH-123-AI/"
                "Comfyui_Minimax_h3_latent_Upscaler to be installed."
            ) from error

        separated = LTXVSeparateAVLatent.execute(sampled_latent)
        video_latent, audio_latent = separated[0], separated[1]
        source_shape = tuple(video_latent["samples"].shape)
        mode = {
            "mode": UpscaleMode.TARGET_DIMENSIONS,
            "width": int(final_width),
            "height": int(final_height),
        }
        upscaled_video = MinimaxH3LatentUpscaler3D.execute(
            latent=video_latent,
            model_name=str(upscaler_model),
            mode=mode,
            align=32,
            enable_temporal_chunking=True,
            force_unload=True,
            device="cuda",
            precision="fp16",
        )[0]
        combined = LTXVConcatAVLatent.execute(upscaled_video, audio_latent)[0]
        schedule_steps = max(1, int(steps))
        schedule_start = max(0, min(int(start_at_step), schedule_steps - 1))
        requested_end = int(end_at_step)
        schedule_end = (
            schedule_steps
            if requested_end >= 10000
            else max(schedule_start + 1, min(requested_end, schedule_steps))
        )
        refined = comfy_nodes.common_ksampler(
            model,
            int(noise_seed),
            schedule_steps,
            float(cfg),
            str(sampler_name),
            str(scheduler),
            positive,
            negative,
            combined,
            denoise=1.0,
            disable_noise=str(add_noise) == "disable",
            start_step=schedule_start,
            last_step=schedule_end,
            force_full_denoise=str(return_with_leftover_noise) == "disable",
        )[0]

        refined_video = LTXVSeparateAVLatent.execute(refined)[0]
        delivery = LTXVConcatAVLatent.execute(refined_video, audio_latent)[0]
        delivery = dict(delivery)
        delivery["_simple_h3_context_latent"] = sampled_latent
        target_shape = tuple(refined_video["samples"].shape)
        return (
            delivery,
            sampled_latent,
            (
                f"H3 latent upscale {source_shape[-1] * 16}x{source_shape[-2] * 16} "
                f"-> {target_shape[-1] * 16}x{target_shape[-2] * 16}; "
                f"advanced steps {schedule_start}->{schedule_end} of {schedule_steps} "
                f"({schedule_end - schedule_start} evaluated); "
                "original audio latent preserved."
            ),
        )


class SimpleH3LatentUpscaleRefineMasked(SimpleH3LatentUpscaleRefineAdvanced):
    """Experimental Advanced refine with a protected high-resolution video prefix."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["audio_output"] = (["original", "refined"], {
            "default": "original",
            "tooltip": (
                "original restores the untouched audio latent from the base generation. "
                "refined delivers the audio produced by the extra Advanced sampling pass."
            ),
        })
        inputs["optional"] = {
            "state": (_chain.STATE_TYPE, {
                "tooltip": (
                    "Connect Simple H3 Current Scene state. From scene two onward, the "
                    "previous refined 39-frame tail becomes a protected video prefix."
                ),
            }),
        }
        return inputs

    FUNCTION = "upscale_refine_masked"
    DESCRIPTION = (
        "Experimental isolated variant of Advanced. It preserves the previous refined "
        "39-frame video tail during the next high-resolution pass while leaving native "
        "low-resolution Masked AV and audio unchanged."
    )

    def upscale_refine_masked(
        self, model, positive, negative, sampled_latent, latent_upscale,
        upscaler_model, final_width, final_height, add_noise, noise_seed,
        steps, cfg, sampler_name, scheduler, start_at_step, end_at_step,
        return_with_leftover_noise, audio_output, state=None,
    ):
        if not bool(latent_upscale):
            return (sampled_latent, sampled_latent, "Latent upscale off; native latent delivered.")

        try:
            from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
                MinimaxH3LatentUpscaler3D,
                UpscaleMode,
            )
            from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
            import nodes as comfy_nodes
        except Exception as error:
            raise RuntimeError(
                "Simple H3 latent upscale requires LBH-123-AI/"
                "Comfyui_Minimax_h3_latent_Upscaler to be installed."
            ) from error

        separated = LTXVSeparateAVLatent.execute(sampled_latent)
        video_latent, audio_latent = separated[0], separated[1]
        source_shape = tuple(video_latent["samples"].shape)
        upscaled_video = MinimaxH3LatentUpscaler3D.execute(
            latent=video_latent,
            model_name=str(upscaler_model),
            mode={
                "mode": UpscaleMode.TARGET_DIMENSIONS,
                "width": int(final_width),
                "height": int(final_height),
            },
            align=32,
            enable_temporal_chunking=True,
            force_unload=True,
            device="cuda",
            precision="fp16",
        )[0]
        combined = LTXVConcatAVLatent.execute(upscaled_video, audio_latent)[0]

        previous_refined = (
            state.get("previous_refined_latent")
            if isinstance(state, dict) else None
        )
        refined_continuation = previous_refined is not None
        if refined_continuation:
            combined, _ = apply_masked_video_continuation(
                combined,
                previous_refined,
                context_frames=39,
            )

        schedule_steps = max(1, int(steps))
        schedule_start = max(0, min(int(start_at_step), schedule_steps - 1))
        requested_end = int(end_at_step)
        schedule_end = (
            schedule_steps if requested_end >= 10000
            else max(schedule_start + 1, min(requested_end, schedule_steps))
        )
        refined = comfy_nodes.common_ksampler(
            model,
            int(noise_seed),
            schedule_steps,
            float(cfg),
            str(sampler_name),
            str(scheduler),
            positive,
            negative,
            combined,
            denoise=1.0,
            disable_noise=str(add_noise) == "disable",
            start_step=schedule_start,
            last_step=schedule_end,
            force_full_denoise=str(return_with_leftover_noise) == "disable",
        )[0]

        refined_streams = LTXVSeparateAVLatent.execute(refined)
        refined_video, refined_audio = refined_streams[0], refined_streams[1]
        selected_audio = (
            refined_audio if str(audio_output) == "refined" else audio_latent
        )
        delivery = LTXVConcatAVLatent.execute(refined_video, selected_audio)[0]
        context = dict(sampled_latent)
        context["_simple_h3_refined_latent"] = delivery
        target_shape = tuple(refined_video["samples"].shape)
        return (
            delivery,
            context,
            (
                f"H3 refined masked upscale {source_shape[-1] * 16}x{source_shape[-2] * 16} "
                f"-> {target_shape[-1] * 16}x{target_shape[-2] * 16}; "
                f"advanced steps {schedule_start}->{schedule_end} of {schedule_steps}; "
                f"refined 39-frame continuity "
                f"{'active' if refined_continuation else 'initialized'}; "
                f"audio output {str(audio_output)}."
            ),
        )


def _h3_video_tokens_for_total_frames(total_frames):
    """Return the exact H3 video-token count for one delivered timeline."""
    from .masked_context import _pixel_frames

    total_frames = int(total_frames)
    if total_frames < 1:
        raise ValueError("The assembled H3 timeline must contain at least one frame.")
    # H3 frame lengths are sparse but the loop planner guarantees an exact
    # representable cumulative timeline. Keep the search explicit so a corrupt
    # or foreign manifest fails instead of silently shifting a seam.
    approximate = max(1, int(math.ceil(total_frames / 3.4)))
    for tokens in range(max(1, approximate - 16), approximate + 32):
        value = _pixel_frames(tokens)
        if value == total_frames:
            return tokens
        if value > total_frames:
            break
    raise ValueError(
        f"The delivered H3 timeline length {total_frames} does not map to an "
        "exact latent-token boundary."
    )


def _h3_video_tokens_at_least_frames(total_frames):
    """Return the first H3 token boundary at or beyond a delivered timeline."""
    from .masked_context import _pixel_frames

    requested = max(1, int(total_frames))
    tokens = max(1, int(math.floor(requested / 3.4)))
    while _pixel_frames(tokens) < requested:
        tokens += 1
    return tokens, _pixel_frames(tokens)


class SimpleH3FinalWindowedLatentUpscale(SimpleH3LatentUpscaleRefine):
    """Rebuild the completed base timeline and upscale it with temporal windows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (_chain.MANIFEST_TYPE, {
                    "tooltip": (
                        "Completed manifest from Simple H3 Loop Until Final Scene. "
                        "All base latents are read from its verified checkpoints."
                    ),
                }),
                "latent_upscale": ("BOOLEAN", {"default": True}),
                "upscaler_model": (cls._models(),),
                "final_width": ("INT", {
                    "default": 1280, "min": 64, "max": 4096, "step": 8,
                }),
                "final_height": ("INT", {
                    "default": 720, "min": 64, "max": 4096, "step": 8,
                }),
                "temporal_windowing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Process the complete latent timeline in overlapping temporal "
                        "windows inside the learned 3D upscaler to limit peak VRAM."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("final_latent", "delivered_frames", "status")
    OUTPUT_TOOLTIPS = (
        "One exact assembled AV latent, optionally upscaled after all scenes finish.",
        "Exact user-facing frame count. Trim the decoded result to this value.",
        "Timeline reconstruction, seam trimming, windowing and resolution summary.",
    )
    FUNCTION = "assemble_and_upscale"
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Experimental post-chain path. It reconstructs the exact low-resolution "
        "Masked AV timeline from scene checkpoints, removes repeated 39-frame heads "
        "and final grid padding, then applies one learned 3D upscale over overlapping "
        "temporal windows. It performs no second diffusion pass and preserves the "
        "original generated audio latent."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def assemble_and_upscale(
        self, manifest, latent_upscale, upscaler_model, final_width,
        final_height, temporal_windowing,
    ):
        if (not isinstance(manifest, dict) or
                manifest.get("format") != "h3_chain_manifest_v3"):
            raise ValueError(
                "Final Windowed Latent Upscale requires a completed Simple H3 manifest."
            )
        segments = list(manifest.get("segments") or [])
        if not segments:
            raise ValueError("The completed Simple H3 manifest contains no scenes.")

        try:
            from safetensors.torch import load_file as load_safetensors
            from comfy_extras.nodes_lt import LTXVConcatAVLatent
        except Exception as error:
            raise RuntimeError(
                "Final Windowed Latent Upscale requires safetensors and ComfyUI's "
                "native joint AV latent nodes."
            ) from error

        compatibility = manifest.get("compatibility") or {}
        context_frames = int(compatibility.get("context_length", 0))
        fingerprint = str(compatibility.get("generation_fingerprint") or "")
        masked = "type=masked_av" in fingerprint
        if masked and context_frames != 39:
            raise ValueError(
                "Final Windowed Latent Upscale currently requires the exact 39-frame "
                "Masked AV/Cut boundary."
            )

        video_parts = []
        audio_parts = []
        delivered_total = 0
        latent_total_frames = 0
        total_audio_ticks = 0
        video_shape = None
        audio_shape = None
        removed_video_tokens = 0
        removed_audio_ticks = 0

        for position, segment in enumerate(segments):
            index = int(segment.get("index", position + 1))
            _chain._verify_segment_artifacts(segment, index)
            checkpoint = _chain._absolute_output_path(segment["checkpoint"])
            tensors = load_safetensors(checkpoint, device="cpu")
            if "video" not in tensors or "audio" not in tensors:
                raise ValueError(
                    f"Scene {index} checkpoint has no base MiniMax H3 AV latent."
                )
            video = tensors["video"].detach().cpu().contiguous()
            audio = tensors["audio"].detach().cpu().contiguous()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if audio.ndim == 3:
                audio = audio.unsqueeze(0)
            if video.ndim != 5 or audio.ndim != 4:
                raise ValueError(
                    f"Scene {index} has invalid latent shapes: "
                    f"video={tuple(video.shape)}, audio={tuple(audio.shape)}."
                )
            current_video_shape = tuple(video.shape[1:2] + video.shape[3:])
            current_audio_shape = tuple(audio.shape[1:3])
            if video_shape is None:
                video_shape = current_video_shape
                audio_shape = current_audio_shape
            elif current_video_shape != video_shape or current_audio_shape != audio_shape:
                raise ValueError(
                    "All base scenes must use identical latent resolution and audio geometry."
                )

            delivered = int(segment.get("delivered_frames", 0))
            if delivered <= 0:
                raise ValueError(f"Scene {index} has no valid delivered frame count.")
            desired_cumulative = delivered_total + delivered
            is_last = position == len(segments) - 1
            previous_total_tokens = sum(int(value.shape[2]) for value in video_parts)
            if not masked:
                # Independent storyboard cuts each own a complete valid H3 token
                # grid. Their delivered-frame cumulative (for example 124+124=248)
                # need not itself be an H3 boundary. Preserve every scene token and
                # trim only the final decoded timeline to the delivered frame count.
                from .masked_context import _pixel_frames
                delivered_tokens = int(video.shape[2])
                next_total_tokens = previous_total_tokens + delivered_tokens
                next_latent_frames = _pixel_frames(next_total_tokens)
            elif is_last:
                next_total_tokens, next_latent_frames = (
                    _h3_video_tokens_at_least_frames(desired_cumulative)
                )
                delivered_tokens = next_total_tokens - previous_total_tokens
            else:
                next_total_tokens = _h3_video_tokens_for_total_frames(
                    desired_cumulative
                )
                next_latent_frames = desired_cumulative
                delivered_tokens = next_total_tokens - previous_total_tokens
            head_tokens = 0
            head_audio = 0
            if position and masked:
                head_tokens = _h3_video_tokens_for_total_frames(context_frames)
                head_audio = int(round(context_frames / 24.0 * 40.0))
            if head_tokens + delivered_tokens > int(video.shape[2]):
                raise ValueError(
                    f"Scene {index} cannot provide {delivered_tokens} delivered video "
                    f"tokens after its {head_tokens}-token context head."
                )

            # Audio follows the exact delivered timeline, not the padded H3 video
            # token boundary. The final video latent may need a few extra grid
            # frames, especially on the last scene, but those frames are removed
            # after decode and must never demand nonexistent audio from the scene.
            next_audio_ticks = int(round(desired_cumulative / 24.0 * 40.0))
            delivered_audio_ticks = next_audio_ticks - total_audio_ticks
            if head_audio + delivered_audio_ticks > int(audio.shape[-1]):
                raise ValueError(
                    f"Scene {index} cannot provide {delivered_audio_ticks} delivered "
                    f"audio ticks after its {head_audio}-tick context head."
                )
            video_parts.append(
                video[:, :, head_tokens:head_tokens + delivered_tokens].clone()
            )
            audio_parts.append(
                audio[..., head_audio:head_audio + delivered_audio_ticks].clone()
            )
            removed_video_tokens += int(video.shape[2]) - delivered_tokens
            removed_audio_ticks += int(audio.shape[-1]) - delivered_audio_ticks
            delivered_total = desired_cumulative
            latent_total_frames = next_latent_frames
            total_audio_ticks = next_audio_ticks

        assembled_video = torch.cat(video_parts, dim=2).contiguous()
        assembled_audio = torch.cat(audio_parts, dim=-1).contiguous()
        expected_tokens = _h3_video_tokens_for_total_frames(latent_total_frames)
        if int(assembled_video.shape[2]) != expected_tokens:
            raise RuntimeError(
                "Internal H3 timeline reconstruction produced an incorrect video length."
            )
        if int(assembled_audio.shape[-1]) != total_audio_ticks:
            raise RuntimeError(
                "Internal H3 timeline reconstruction produced an incorrect audio length."
            )

        base_video = {"samples": assembled_video}
        if bool(latent_upscale):
            try:
                from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
                    MinimaxH3LatentUpscaler3D,
                    UpscaleMode,
                )
            except Exception as error:
                raise RuntimeError(
                    "Final Windowed Latent Upscale requires LBH-123-AI/"
                    "Comfyui_Minimax_h3_latent_Upscaler to be installed."
                ) from error
            upscaled_video = MinimaxH3LatentUpscaler3D.execute(
                latent=base_video,
                model_name=str(upscaler_model),
                mode={
                    "mode": UpscaleMode.TARGET_DIMENSIONS,
                    "width": int(final_width),
                    "height": int(final_height),
                },
                align=32,
                enable_temporal_chunking=bool(temporal_windowing),
                force_unload=True,
                device="cuda",
                precision="fp16",
            )[0]
        else:
            upscaled_video = base_video

        final_latent = LTXVConcatAVLatent.execute(
            upscaled_video, {"samples": assembled_audio}
        )[0]
        source_width = int(assembled_video.shape[-1]) * 16
        source_height = int(assembled_video.shape[-2]) * 16
        target = upscaled_video["samples"]
        target_width = int(target.shape[-1]) * 16
        target_height = int(target.shape[-2]) * 16
        status = (
            f"Assembled {len(segments)} base scenes into {latent_total_frames} latent "
            f"frames for {delivered_total} delivered frames "
            f"({expected_tokens} video tokens, {total_audio_ticks} audio ticks); "
            f"removed {removed_video_tokens} repeated/padded video tokens and "
            f"{removed_audio_ticks} repeated/padded audio ticks; "
            f"{source_width}x{source_height} -> {target_width}x{target_height}; "
            f"temporal windowing {'on' if temporal_windowing and latent_upscale else 'off'}; "
            "no second diffusion pass; original audio latent preserved."
        )
        return (final_latent, delivered_total, status)


class SimpleH3FinalTimelineTrim:
    """Remove only the final H3 temporal-grid padding after complete AV decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "delivered_frames": ("INT", {
                    "default": 1, "min": 1, "max": 1000000,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "status")
    FUNCTION = "trim"
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Trim the decoded windowed-upscale result to the exact manifest duration. "
        "Only terminal H3 grid padding is removed; scene boundaries are untouched."
    )

    def trim(self, images, audio, delivered_frames):
        frames = int(delivered_frames)
        available = int(images.shape[0])
        if available < frames:
            raise ValueError(
                f"Final timeline decode produced {available} frames; expected at least {frames}."
            )
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
        if waveform is None or sample_rate <= 0:
            raise ValueError("Final Timeline Trim requires a valid ComfyUI AUDIO value.")
        target_samples = int(round(frames / 24.0 * sample_rate))
        if int(waveform.shape[-1]) < target_samples:
            raise ValueError(
                f"Final audio decode produced {int(waveform.shape[-1])} samples; "
                f"expected at least {target_samples}."
            )
        trimmed_audio = dict(audio)
        trimmed_audio["waveform"] = waveform[..., :target_samples].contiguous()
        removed = available - frames
        return (
            images[:frames].contiguous(),
            trimmed_audio,
            f"Delivered exactly {frames} frames at 24 fps; removed {removed} final "
            "H3 grid-padding frames and matched the audio tail.",
        )


class SimpleH3FinalWindowedRefineAdvanced:
    """Advanced H3 refinement over an assembled latent using protected AV windows."""

    WINDOW_FRAMES = [90, 141, 192, 243, 294, 345, 396]

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "assembled_latent": ("LATENT",),
                "delivered_frames": ("INT", {
                    "default": 1, "min": 1, "max": 1000000,
                }),
                "refine": ("BOOLEAN", {"default": False}),
                "add_noise": (["enable", "disable"], {"default": "enable"}),
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0,
                    "step": 0.1, "round": 0.01,
                }),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "res_multistep"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "start_at_step": ("INT", {
                    "default": 6, "min": 0, "max": 10000,
                }),
                "end_at_step": ("INT", {
                    "default": 10000, "min": 0, "max": 10000,
                }),
                "return_with_leftover_noise": (["disable", "enable"], {
                    "default": "disable",
                }),
                "window_frames": (cls.WINDOW_FRAMES, {
                    "default": 243,
                    "tooltip": (
                        "Exact H3 AV-aligned temporal window. Adjacent windows share "
                        "39 protected frames; 243 is the recommended first test."
                    ),
                }),
                "audio_output": (["original", "refined"], {
                    "default": "original",
                    "tooltip": (
                        "original restores the complete audio latent from the base "
                        "generation. refined keeps the audio produced by every "
                        "refinement window and removes its repeated 39-frame overlap."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("refined_latent", "delivered_frames", "status")
    FUNCTION = "refine_windowed"
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Optional KSampler Advanced-style refinement after the complete base timeline "
        "has been assembled and upscaled. It processes exact AV-aligned windows, "
        "protects the previous refined 39-frame tail in every later window, joins "
        "latents without crossfade, and can preserve either the original complete "
        "audio or the audio refined inside the same temporal windows."
    )

    def refine_windowed(
        self, model, positive, negative, assembled_latent, delivered_frames,
        refine, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
        start_at_step, end_at_step, return_with_leftover_noise, window_frames,
        audio_output="original",
    ):
        if not bool(refine):
            return (
                assembled_latent,
                int(delivered_frames),
                "Windowed Advanced refine off; assembled upscale passed through unchanged.",
            )
        try:
            from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
            import nodes as comfy_nodes
        except Exception as error:
            raise RuntimeError(
                "Windowed Advanced refinement requires ComfyUI's native joint AV nodes."
            ) from error

        separated = LTXVSeparateAVLatent.execute(assembled_latent)
        full_video, full_audio = separated[0], separated[1]
        video = full_video["samples"]
        audio = full_audio["samples"]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        if video.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("Windowed Advanced refinement currently supports batch size 1.")

        window_tokens = _h3_video_tokens_for_total_frames(int(window_frames))
        overlap_frames = 39
        overlap_tokens = _h3_video_tokens_for_total_frames(overlap_frames)
        stride_tokens = window_tokens - overlap_tokens
        stride_frames = int(window_frames) - overlap_frames
        stride_audio = int(round(stride_frames / 24.0 * 40.0))
        window_audio = int(round(int(window_frames) / 24.0 * 40.0))
        if stride_tokens <= 0 or stride_audio <= 0:
            raise ValueError("The refinement window must be longer than 39 frames.")
        if abs(stride_frames / 24.0 * 40.0 - stride_audio) > 1e-6:
            raise ValueError(
                "The selected refinement window is not aligned to the shared H3 AV grid."
            )

        schedule_steps = max(1, int(steps))
        schedule_start = max(0, min(int(start_at_step), schedule_steps - 1))
        requested_end = int(end_at_step)
        schedule_end = (
            schedule_steps if requested_end >= 10000
            else max(schedule_start + 1, min(requested_end, schedule_steps))
        )
        total_tokens = int(video.shape[2])
        total_audio = int(audio.shape[-1])
        refined_video_parts = []
        refined_audio_parts = []
        previous_refined = None
        start_token = 0
        start_audio = 0
        window_index = 0

        def conditioning_for_refine(conditioning):
            """Keep text/image semantics but do not reapply generation keyframes."""
            if not isinstance(conditioning, (list, tuple)):
                return conditioning
            refined_conditioning = []
            for entry in conditioning:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    refined_conditioning.append(entry)
                    continue
                metadata = dict(entry[1])
                # Native I2V keyframes belong to the base-generation grid. The
                # assembled latent already contains their result; injecting them
                # again during refinement either causes a spatial mismatch after
                # upscale or produces a ghosted opening when naively resized.
                metadata.pop("minimax_keyframes", None)
                refined_conditioning.append([entry[0], metadata])
            return refined_conditioning

        while start_token < total_tokens:
            end_token = min(total_tokens, start_token + window_tokens)
            token_count = end_token - start_token
            from .masked_context import _pixel_frames
            current_frames = _pixel_frames(token_count)
            current_audio_count = int(round(current_frames / 24.0 * 40.0))
            end_audio = min(total_audio, start_audio + current_audio_count)
            available_audio_count = max(0, end_audio - start_audio)
            if available_audio_count <= 0:
                raise ValueError(
                    "The assembled audio latent has no samples for its final refinement window."
                )
            window_video = {
                "samples": video[:, :, start_token:end_token].contiguous()
            }
            audio_slice = audio[..., start_audio:end_audio].contiguous()
            if available_audio_count < current_audio_count:
                missing_audio_count = current_audio_count - available_audio_count
                repeated_tail = audio_slice[..., -1:].expand(
                    *audio_slice.shape[:-1], missing_audio_count
                )
                audio_slice = torch.cat(
                    (audio_slice, repeated_tail), dim=-1
                ).contiguous()
                _chain._LOG.info(
                    "Simple H3 final refine: padded final audio window by %d ticks "
                    "for AV alignment; padded ticks will not enter the output.",
                    missing_audio_count,
                )
            window_audio_latent = {
                "samples": audio_slice
            }
            current = LTXVConcatAVLatent.execute(
                window_video, window_audio_latent
            )[0]
            if previous_refined is not None:
                current, _ = apply_masked_video_continuation(
                    current, previous_refined, context_frames=overlap_frames
                )

            window_positive = conditioning_for_refine(positive)
            window_negative = conditioning_for_refine(negative)

            sampled = comfy_nodes.common_ksampler(
                model,
                (int(noise_seed) + window_index) & 0xffffffffffffffff,
                schedule_steps,
                float(cfg),
                str(sampler_name),
                str(scheduler),
                window_positive,
                window_negative,
                current,
                denoise=1.0,
                disable_noise=str(add_noise) == "disable",
                start_step=schedule_start,
                last_step=schedule_end,
                force_full_denoise=(
                    str(return_with_leftover_noise) == "disable"
                ),
            )[0]
            sampled_parts = LTXVSeparateAVLatent.execute(sampled)
            sampled_video = sampled_parts[0]["samples"]
            sampled_audio = sampled_parts[1]["samples"]
            if sampled_video.ndim == 4:
                sampled_video = sampled_video.unsqueeze(0)
            if sampled_audio.ndim == 3:
                sampled_audio = sampled_audio.unsqueeze(0)
            sampled_audio = sampled_audio[..., :available_audio_count]
            if previous_refined is None:
                refined_video_parts.append(sampled_video.detach().cpu().contiguous())
                refined_audio_parts.append(sampled_audio.detach().cpu().contiguous())
            else:
                refined_video_parts.append(
                    sampled_video[:, :, overlap_tokens:].detach().cpu().contiguous()
                )
                overlap_audio = int(round(overlap_frames / 24.0 * 40.0))
                refined_audio_parts.append(
                    sampled_audio[..., min(overlap_audio, available_audio_count):]
                    .detach().cpu().contiguous()
                )
            previous_refined = sampled
            window_index += 1
            if end_token >= total_tokens:
                break
            start_token += stride_tokens
            start_audio += stride_audio

        refined_video = torch.cat(refined_video_parts, dim=2)
        if int(refined_video.shape[2]) != total_tokens:
            raise RuntimeError(
                f"Windowed refinement reconstructed {int(refined_video.shape[2])} "
                f"video tokens; expected {total_tokens}."
            )
        selected_audio = audio.detach().cpu().contiguous()
        audio_note = "original complete audio latent restored"
        if str(audio_output) == "refined":
            selected_audio = torch.cat(refined_audio_parts, dim=-1).contiguous()
            if int(selected_audio.shape[-1]) != total_audio:
                raise RuntimeError(
                    f"Windowed refinement reconstructed {int(selected_audio.shape[-1])} "
                    f"audio ticks; expected {total_audio}."
                )
            audio_note = "refined window audio joined after removing 39-frame overlaps"
        result = LTXVConcatAVLatent.execute(
            {"samples": refined_video}, {"samples": selected_audio}
        )[0]
        _chain._LOG.info(
            "Simple H3 final window refine complete: %d windows; audio_output=%s; %s.",
            window_index, str(audio_output), audio_note,
        )
        return (
            result,
            int(delivered_frames),
            (
                f"Refined {window_index} overlapping AV windows of up to "
                f"{int(window_frames)} frames; protected 39 frames between windows; "
                f"advanced steps {schedule_start}->{schedule_end} of {schedule_steps}; "
                f"{audio_note}; no crossfade."
            ),
        )


def _publish_refined_window_preview(
    unique_id, transaction, index, count, path, has_audio,
):
    if (_chain.PromptServer is None or
            _chain.PromptServer.instance is None):
        return
    _chain.PromptServer.instance.send_sync(
        "simple_h3_chain_review",
        {
            "token": transaction,
            "node_id": str(unique_id),
            "execution_id": str(unique_id),
            "preview_kind": "refined_window",
            "clip_index": int(index),
            "clip_count": int(count),
            "shot_id": f"window_{int(index):04d}",
            "scene_prompt": "",
            "seed": "",
            "video": _chain._video_output_item(path),
            "has_audio": bool(has_audio),
            "warning": (
                f"Refined window {int(index)}/{int(count)} ready; "
                "the complete video will replace it after assembly."
            ),
            "preview_pending": False,
            "preview_revision": int(index),
            "auto_continue": True,
        },
        _chain.PromptServer.instance.client_id,
    )


class SimpleH3FinalLatentWindowDecodeAssemble:
    """Decode final video latents in overlapping windows and assemble one MP4."""

    WINDOW_FRAMES = [90, 141, 192, 243, 294, 345, 396]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (_chain.MANIFEST_TYPE,),
                "final_latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "delivered_frames": ("INT", {
                    "default": 1, "min": 1, "max": 1000000,
                }),
                "window_frames": (cls.WINDOW_FRAMES, {"default": 90}),
                "filename": ("STRING", {"default": "%date:yyyy-MM-dd%_refined"}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previews": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Show every decoded refined window with its synchronized audio. "
                        "Disable to show only the complete assembled final video."
                    ),
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "status")
    FUNCTION = "decode_and_assemble"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Decode the final H3 video latent through overlapping temporal windows, "
        "publish each isolated window as it becomes available, trim only the terminal "
        "grid padding, then stream-copy the parts and replace the preview with the "
        "complete video containing synchronized audio."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def decode_and_assemble(
        self, manifest, final_latent, video_vae, audio_vae, delivered_frames,
        window_frames, filename, save_output, previews=True,
        unique_id=None,
    ):
        try:
            from comfy_extras.nodes_lt import LTXVSeparateAVLatent
            from comfy_extras.nodes_audio import vae_decode_audio
        except Exception as error:
            raise RuntimeError(
                "Final Latent Window Decode requires ComfyUI's native H3 AV nodes."
            ) from error

        separated = LTXVSeparateAVLatent.execute(final_latent)
        video = separated[0]["samples"]
        audio = separated[1]["samples"]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        if video.ndim != 5 or audio.ndim != 4:
            raise ValueError(
                "Final Latent Window Decode received invalid H3 AV shapes: "
                f"video={tuple(video.shape)}, audio={tuple(audio.shape)}."
            )
        if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
            raise ValueError("Final Latent Window Decode currently supports batch size 1.")

        frames = int(delivered_frames)
        selected_window = int(window_frames)
        window_tokens = _h3_video_tokens_for_total_frames(selected_window)
        overlap_frames = 39
        overlap_tokens = _h3_video_tokens_for_total_frames(overlap_frames)
        stride_tokens = window_tokens - overlap_tokens
        stride_frames = selected_window - overlap_frames
        if stride_tokens <= 0 or stride_frames <= 0:
            raise ValueError("Final decode window must be longer than 39 frames.")

        from .masked_context import _pixel_frames
        latent_frames = _pixel_frames(int(video.shape[2]))
        if latent_frames < frames:
            raise ValueError(
                f"Final video latent decodes to {latent_frames} frames; "
                f"the manifest requires {frames}."
            )

        # Audio is tiny compared with decoded video frames. Decode it once so
        # native whole-track normalization remains stable and window joins do
        # not introduce gain changes or AAC priming gaps.
        decoded_audio = vae_decode_audio(
            audio_vae, {"samples": audio.contiguous()}
        )
        waveform = decoded_audio["waveform"]
        sample_rate = int(decoded_audio["sample_rate"])
        required_samples = int(round(frames / 24.0 * sample_rate))
        if int(waveform.shape[-1]) < required_samples:
            waveform = F.pad(
                waveform, (0, required_samples - int(waveform.shape[-1]))
            )
        else:
            waveform = waveform[..., :required_samples].contiguous()

        run_name = _safe_run_name(manifest.get("run_name", "h3_chain"))
        run_dir = _chain._absolute_output_path(run_name)
        preview_dir = os.path.join(run_dir, "previews", "refined")
        final_dir = os.path.join(run_dir, "final")
        os.makedirs(preview_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        transaction = uuid.uuid4().hex
        for old_name in os.listdir(preview_dir):
            old_path = os.path.join(preview_dir, old_name)
            if os.path.isfile(old_path):
                _chain._safe_unlink(old_path)

        final_name = _chain._safe_name(
            _expand_date_tokens(str(filename)), "refined_final"
        )
        final_path = _chain._versioned_path(
            os.path.join(final_dir, final_name + ".mp4"), transaction
        )
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Final Latent Window Decode requires ffmpeg.")

        window_paths = []
        preview_paths = []
        expected_windows = max(
            1,
            1 + int(math.ceil(
                max(0, frames - selected_window) / float(stride_frames)
            )),
        )
        concat_path = os.path.join(preview_dir, f".{transaction}.concat.txt")
        video_tmp = os.path.join(final_dir, f".{transaction}.video.mp4")
        wav_tmp = os.path.join(final_dir, f".{transaction}.wav")
        metadata_tmp = os.path.join(final_dir, f".{transaction}.metadata.txt")
        final_tmp = os.path.join(final_dir, f".{transaction}.tmp.mp4")
        emitted = 0
        start_token = 0
        window_index = 0
        completed = False
        try:
            while start_token < int(video.shape[2]) and emitted < frames:
                end_token = min(
                    int(video.shape[2]), start_token + window_tokens
                )
                token_count = end_token - start_token
                decoded_count = _pixel_frames(token_count)
                window_latent = video[:, :, start_token:end_token].contiguous()
                images = video_vae.decode(window_latent)
                if images.ndim == 5:
                    images = images.reshape(
                        -1, images.shape[-3], images.shape[-2], images.shape[-1]
                    )
                if int(images.shape[0]) < decoded_count:
                    raise ValueError(
                        f"Final VAE window {window_index + 1} returned "
                        f"{int(images.shape[0])} frames; expected {decoded_count}."
                    )
                images = images[:decoded_count]
                head = 0 if window_index == 0 else overlap_frames
                if int(images.shape[0]) <= head:
                    raise ValueError(
                        f"Final VAE window {window_index + 1} contains no new frames "
                        "after its 39-frame overlap."
                    )
                available = int(images.shape[0]) - head
                wanted = min(available, frames - emitted)
                delivered = images[head:head + wanted].contiguous()
                part_path = os.path.join(
                    preview_dir,
                    f"window_{window_index + 1:04d}.{transaction}.mp4",
                )
                _chain._write_segment_video(
                    delivered, part_path, 24, 20,
                    metadata={
                        "title": f"Refined decode window {window_index + 1}",
                        "comment": (
                            f"delivered frames {emitted + 1}-{emitted + wanted} "
                            f"of {frames}; decoded {decoded_count}; overlap {head}"
                        ),
                    },
                )
                window_paths.append(part_path)
                if bool(previews):
                    preview_path = os.path.join(
                        preview_dir,
                        f"window_{window_index + 1:04d}.preview.{transaction}.mp4",
                    )
                    window_wav = preview_path + ".wav"
                    audio_start = int(round(emitted / 24.0 * sample_rate))
                    audio_stop = int(round(
                        (emitted + wanted) / 24.0 * sample_rate
                    ))
                    try:
                        _chain._write_wav(
                            {
                                "waveform": waveform[..., audio_start:audio_stop],
                                "sample_rate": sample_rate,
                            },
                            window_wav,
                        )
                        _chain._run_ffmpeg([
                            ffmpeg, "-y", "-i", part_path, "-i", window_wav,
                            "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-t", f"{wanted / 24.0:.9f}",
                            "-movflags", "+faststart", preview_path,
                        ])
                    finally:
                        _chain._safe_unlink(window_wav)
                    preview_paths.append(preview_path)
                    _publish_refined_window_preview(
                        unique_id, transaction, window_index + 1,
                        expected_windows, preview_path, True,
                    )
                emitted += wanted
                window_index += 1
                del delivered, images, window_latent
                if emitted >= frames or end_token >= int(video.shape[2]):
                    break
                start_token += stride_tokens

            if emitted != frames:
                raise RuntimeError(
                    f"Windowed final decode emitted {emitted} frames; expected {frames}."
                )
            with open(concat_path, "w", encoding="utf-8") as handle:
                for path in window_paths:
                    escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
                    handle.write(f"file '{escaped}'\n")
            _chain._run_ffmpeg([
                ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                "-c", "copy", "-movflags", "+faststart", video_tmp,
            ])
            _chain._write_wav(
                {"waveform": waveform, "sample_rate": sample_rate}, wav_tmp
            )
            media_metadata = _chain._manifest_media_metadata(manifest)
            _chain._write_ffmetadata(metadata_tmp, media_metadata)
            _chain._run_ffmpeg([
                ffmpeg, "-y", "-i", video_tmp, "-i", wav_tmp,
                "-f", "ffmetadata", "-i", metadata_tmp,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "256k",
                "-t", f"{frames / 24.0:.9f}",
                "-map_metadata", "2",
                "-movflags", "use_metadata_tags+faststart", final_tmp,
            ])
            os.replace(final_tmp, final_path)
            completed = True

            status = (
                f"Decoded and released {window_index} overlapping latent windows; "
                f"assembled exactly {frames} frames with stable whole-track audio "
                f"(window previews {'on' if previews else 'off'}) "
                f"-> {final_path}"
            )
            if not bool(save_output):
                temporary = os.path.join(
                    preview_dir,
                    f"temporary_refined_preview.{uuid.uuid4().hex}.mp4",
                )
                os.replace(final_path, temporary)
                final_path = temporary
                SimpleH3ChainAssemble._remove_completed_run(
                    manifest, keep_path=final_path
                )
                status += "; published temporary final and removed recovery artifacts"

            videos = [_chain._video_output_item(final_path)]
            if (_chain.PromptServer is not None and
                    _chain.PromptServer.instance is not None):
                _chain.PromptServer.instance.send_sync(
                    "simple_h3_chain_review_resolved",
                    {
                        "token": transaction,
                        "node_id": str(unique_id),
                        "action": "final",
                        "status": status,
                        "final_video": _chain._video_output_item(final_path),
                    },
                    _chain.PromptServer.instance.client_id,
                )
            _chain._LOG.info("Simple H3 %s", status)
            return {
                "ui": {"videos": videos, "text": [status]},
                "result": (final_path, status),
            }
        finally:
            for path in (
                concat_path, video_tmp, wav_tmp, metadata_tmp, final_tmp,
            ):
                _chain._safe_unlink(path)
            if completed:
                for path in window_paths + preview_paths:
                    _chain._safe_unlink(path)


class SimpleH3I2VFinalLatentUpscale(SimpleH3LatentUpscaleRefine):
    """Direct single-clip I2V upscale without chain checkpoints or base decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_latent": ("LATENT",),
                "latent_upscale": ("BOOLEAN", {"default": True}),
                "upscaler_model": (cls._models(),),
                "final_width": ("INT", {
                    "default": 1280, "min": 64, "max": 4096, "step": 8,
                }),
                "final_height": ("INT", {
                    "default": 720, "min": 64, "max": 4096, "step": 8,
                }),
                "temporal_windowing": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("final_latent", "delivered_frames", "status")
    FUNCTION = "upscale_i2v"
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Direct final-only I2V path. It optionally applies the learned 3D latent "
        "upscaler to one completed AV latent, preserves its original audio, and "
        "performs no diffusion refinement or intermediate video decode."
    )

    def upscale_i2v(
        self, sampled_latent, latent_upscale, upscaler_model, final_width,
        final_height, temporal_windowing,
    ):
        try:
            from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
        except Exception as error:
            raise RuntimeError(
                "Direct I2V final upscale requires ComfyUI's native joint AV nodes."
            ) from error

        separated = LTXVSeparateAVLatent.execute(sampled_latent)
        video_latent, audio_latent = separated[0], separated[1]
        video = video_latent["samples"]
        if video.ndim == 4:
            video = video.unsqueeze(0)
            video_latent = {"samples": video}
        from .masked_context import _pixel_frames
        delivered_frames = _pixel_frames(int(video.shape[2]))
        source_width = int(video.shape[-1]) * 16
        source_height = int(video.shape[-2]) * 16

        if bool(latent_upscale):
            try:
                from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
                    MinimaxH3LatentUpscaler3D,
                    UpscaleMode,
                )
            except Exception as error:
                raise RuntimeError(
                    "Direct I2V final upscale requires LBH-123-AI/"
                    "Comfyui_Minimax_h3_latent_Upscaler to be installed."
                ) from error
            final_video = MinimaxH3LatentUpscaler3D.execute(
                latent=video_latent,
                model_name=str(upscaler_model),
                mode={
                    "mode": UpscaleMode.TARGET_DIMENSIONS,
                    "width": int(final_width),
                    "height": int(final_height),
                },
                align=32,
                enable_temporal_chunking=bool(temporal_windowing),
                force_unload=True,
                device="cuda",
                precision="fp16",
            )[0]
        else:
            final_video = video_latent

        final_latent = LTXVConcatAVLatent.execute(final_video, audio_latent)[0]
        target = final_video["samples"]
        target_width = int(target.shape[-1]) * 16
        target_height = int(target.shape[-2]) * 16
        status = (
            f"Direct I2V final latent: {delivered_frames} frames; "
            f"{source_width}x{source_height} -> {target_width}x{target_height}; "
            f"latent upscale {'on' if latent_upscale else 'off'}; "
            "no base decode and no diffusion refine; original audio preserved."
        )
        return (final_latent, delivered_frames, status)


class SimpleH3I2VFinalDecodeAssemble(SimpleH3FinalLatentWindowDecodeAssemble):
    """One final I2V windowed decode with no base-manifest dependency."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "delivered_frames": ("INT", {
                    "default": 1, "min": 1, "max": 1000000,
                }),
                "window_frames": (cls.WINDOW_FRAMES, {"default": 243}),
                "output_name": ("STRING", {"default": "i2v"}),
                "filename": ("STRING", {"default": "%date:yyyy-MM-dd%_i2v"}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previews": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show isolated decode windows before the final video.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    FUNCTION = "decode_i2v"
    DESCRIPTION = (
        "The only MP4 stage in the direct I2V path. It decodes the final AV latent "
        "in bounded windows and publishes one synchronized final video."
    )

    def decode_i2v(
        self, final_latent, video_vae, audio_vae, delivered_frames,
        window_frames, output_name, filename, save_output, previews=False,
        unique_id=None,
    ):
        manifest = {
            "format": "h3_i2v_final_v1",
            "run_name": _safe_run_name(output_name),
            "clip_count": 1,
            "total_delivered_frames": int(delivered_frames),
            "segments": [],
        }
        return super().decode_and_assemble(
            manifest, final_latent, video_vae, audio_vae,
            delivered_frames, window_frames, filename, save_output,
            previews=previews, unique_id=unique_id,
        )


class SimpleH3FinalWindowPreviewAssemble:
    """Persist small refined previews and publish one transactionally joined final."""

    WINDOW_FRAMES = [90, 141, 192, 243, 294, 345, 396]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (_chain.MANIFEST_TYPE,),
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "delivered_frames": ("INT", {"default": 1, "min": 1, "max": 1000000}),
                "window_frames": (cls.WINDOW_FRAMES, {"default": 90}),
                "filename": ("STRING", {"default": "%date:yyyy-MM-dd%_refined"}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "base_video_path": ("STRING", {
                    "tooltip": (
                        "Optional ordering dependency from the base assembler. "
                        "It guarantees the base final is complete before refined cleanup."
                    ),
                }),
                "save_base_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep or remove the completed base comparison video.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "status")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Simple Chain/Upscale"
    DESCRIPTION = (
        "Writes and publishes the refined result as isolated sequential MP4 windows, "
        "joins them into one final MP4, then replaces the last window with that complete "
        "video. It removes temporary previews and base recovery artifacts only after "
        "assembly when cleanup is enabled."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def assemble(self, manifest, images, audio, delivered_frames, window_frames,
                 filename, save_output, base_video_path=None,
                 save_base_output=True, unique_id=None):
        frames = int(delivered_frames)
        if int(images.shape[0]) < frames:
            raise ValueError(
                f"Refined preview received {int(images.shape[0])} frames; expected {frames}."
            )
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
        if waveform is None or sample_rate <= 0:
            raise ValueError("Refined Window Preview requires a valid ComfyUI AUDIO value.")
        images = images[:frames]
        required_samples = int(round(frames / 24.0 * sample_rate))
        if int(waveform.shape[-1]) < required_samples:
            raise ValueError("Refined Window Preview audio is shorter than the video timeline.")
        waveform = waveform[..., :required_samples]

        plan_run = str(manifest.get("run_name") or "h3_chain")
        run_dir = _chain._absolute_output_path(plan_run)
        preview_dir = os.path.join(run_dir, "previews", "refined")
        final_dir = os.path.join(run_dir, "final")
        os.makedirs(preview_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        transaction = uuid.uuid4().hex
        # Retain the current run's cards long enough for browser playback;
        # previews from the preceding execution are the safe ones to reap.
        for old_name in os.listdir(preview_dir):
            old_path = os.path.join(preview_dir, old_name)
            if os.path.isfile(old_path):
                _chain._safe_unlink(old_path)
        final_name = _chain._safe_name(
            _expand_date_tokens(str(filename)), "refined_final"
        )
        final_path = _chain._versioned_path(
            os.path.join(final_dir, final_name + ".mp4"), transaction
        )
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Refined Window Preview + Assemble requires ffmpeg.")

        window_paths = []
        expected_windows = int(math.ceil(frames / float(int(window_frames))))
        concat_path = os.path.join(preview_dir, f".{transaction}.concat.txt")
        video_tmp = os.path.join(final_dir, f".{transaction}.video.mp4")
        wav_tmp = os.path.join(final_dir, f".{transaction}.wav")
        metadata_tmp = os.path.join(final_dir, f".{transaction}.metadata.txt")
        final_tmp = os.path.join(final_dir, f".{transaction}.tmp.mp4")
        try:
            for index, start in enumerate(range(0, frames, int(window_frames)), 1):
                stop = min(frames, start + int(window_frames))
                path = os.path.join(
                    preview_dir, f"window_{index:04d}.{transaction}.mp4"
                )
                silent_path = path + ".silent.mp4"
                window_wav = path + ".wav"
                _chain._write_segment_video(
                    images[start:stop], silent_path, 24, 20,
                    metadata={
                        "title": f"Refined window {index}",
                        "comment": f"frames {start + 1}-{stop} of {frames}",
                    },
                )
                audio_start = int(round(start / 24.0 * sample_rate))
                audio_stop = int(round(stop / 24.0 * sample_rate))
                _chain._write_wav({
                    "waveform": waveform[..., audio_start:audio_stop],
                    "sample_rate": sample_rate,
                }, window_wav)
                _chain._run_ffmpeg([
                    ffmpeg, "-y", "-i", silent_path, "-i", window_wav,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{(stop - start) / 24.0:.9f}",
                    "-movflags", "+faststart", path,
                ])
                _chain._safe_unlink(silent_path)
                _chain._safe_unlink(window_wav)
                window_paths.append(path)
                _publish_refined_window_preview(
                    unique_id, transaction, index, expected_windows, path, True,
                )
            with open(concat_path, "w", encoding="utf-8") as handle:
                for path in window_paths:
                    escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
                    handle.write(f"file '{escaped}'\n")
            _chain._run_ffmpeg([
                ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                "-c", "copy", "-movflags", "+faststart", video_tmp,
            ])
            fitted_audio = {"waveform": waveform, "sample_rate": sample_rate}
            _chain._write_wav(fitted_audio, wav_tmp)
            media_metadata = _chain._manifest_media_metadata(manifest)
            _chain._write_ffmetadata(metadata_tmp, media_metadata)
            _chain._run_ffmpeg([
                ffmpeg, "-y", "-i", video_tmp, "-i", wav_tmp,
                "-f", "ffmetadata", "-i", metadata_tmp,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "256k", "-t", f"{frames / 24.0:.9f}",
                "-map_metadata", "2",
                "-movflags", "use_metadata_tags+faststart", final_tmp,
            ])
            os.replace(final_tmp, final_path)

            status = (
                f"Encoded {len(window_paths)} internal refined windows and assembled "
                f"the complete {frames}-frame final video -> {final_path}"
            )

            if not bool(save_output):
                preview_dir = os.path.join(run_dir, "previews", "refined")
                os.makedirs(preview_dir, exist_ok=True)
                for old_name in os.listdir(preview_dir):
                    if not old_name.startswith("temporary_refined_preview."):
                        continue
                    old_path = os.path.join(preview_dir, old_name)
                    if os.path.isfile(old_path):
                        _chain._safe_unlink(old_path)
                preview_path = os.path.join(
                    preview_dir,
                    f"temporary_refined_preview.{uuid.uuid4().hex}.mp4"
                )
                os.replace(final_path, preview_path)
                final_path = preview_path
                retained = [final_path]
                if base_video_path and bool(save_base_output):
                    retained.append(os.path.abspath(str(base_video_path)))
                SimpleH3ChainAssemble._remove_completed_run(
                    manifest, keep_paths=retained
                )
                status += (
                    "; temporary final preview published and base recovery artifacts removed"
                )
            if base_video_path and not bool(save_base_output):
                _chain._safe_unlink(os.path.abspath(str(base_video_path)))
                status += "; base comparison output removed"
            videos = [_chain._video_output_item(final_path)]
            if (_chain.PromptServer is not None and
                    _chain.PromptServer.instance is not None):
                _chain.PromptServer.instance.send_sync(
                    "simple_h3_chain_review_resolved", {
                        "token": transaction,
                        "node_id": str(unique_id),
                        "action": "final",
                        "status": status,
                        "final_video": _chain._video_output_item(final_path),
                    },
                    _chain.PromptServer.instance.client_id,
                )
            _chain._LOG.info("Simple H3 %s", status)
            return {"ui": {"videos": videos, "text": [status]},
                    "result": (final_path, status)}
        finally:
            for path in (
                concat_path, video_tmp, wav_tmp, metadata_tmp, final_tmp,
            ):
                _chain._safe_unlink(path)


from .storyboard_cuts_final import build_storyboard_cuts_node

SimpleH3StoryboardCutsUpscaleDecodeAssemble = build_storyboard_cuts_node({
    "upscale_base": SimpleH3LatentUpscaleRefine,
    "manifest_type": _chain.MANIFEST_TYPE,
    "chain": _chain,
    "safe_run_name": _safe_run_name,
    "expand_date_tokens": _expand_date_tokens,
    "publish_preview": _publish_refined_window_preview,
    "tokens_for_frames": _h3_video_tokens_for_total_frames,
})


NODE_CLASS_MAPPINGS = {
    "SimpleH3OptionalLoraLoader": SimpleH3OptionalLoraLoader,
    "SimpleH3LatentUpscaleResolution": SimpleH3LatentUpscaleResolution,
    "SimpleH3LatentUpscaleRefine": SimpleH3LatentUpscaleRefine,
    "SimpleH3LatentUpscaleRefineAdvanced": SimpleH3LatentUpscaleRefineAdvanced,
    "SimpleH3LatentUpscaleRefineMasked": SimpleH3LatentUpscaleRefineMasked,
    "SimpleH3FinalWindowedLatentUpscale": SimpleH3FinalWindowedLatentUpscale,
    "SimpleH3FinalTimelineTrim": SimpleH3FinalTimelineTrim,
    "SimpleH3FinalWindowedRefineAdvanced": SimpleH3FinalWindowedRefineAdvanced,
    "SimpleH3FinalLatentWindowDecodeAssemble": SimpleH3FinalLatentWindowDecodeAssemble,
    "SimpleH3I2VFinalLatentUpscale": SimpleH3I2VFinalLatentUpscale,
    "SimpleH3StoryboardCutsUpscaleDecodeAssemble": SimpleH3StoryboardCutsUpscaleDecodeAssemble,
    "SimpleH3I2VFinalDecodeAssemble": SimpleH3I2VFinalDecodeAssemble,
    "SimpleH3FinalWindowPreviewAssemble": SimpleH3FinalWindowPreviewAssemble,
    "SimpleH3ChainPlan": SimpleH3ChainPlan,
    "SimpleH3ChainLoopStart": SimpleH3ChainLoopStart,
    "SimpleH3ChainCurrent": SimpleH3ChainCurrent,
    "SimpleH3ChainContext": SimpleH3ChainContext,
    "SimpleH3BaseContextDecode": SimpleH3BaseContextDecode,
    "SimpleH3CutReferenceSheet": SimpleH3CutReferenceSheet,
    "SimpleH3SelectContinuityFrames": SimpleH3SelectContinuityFrames,
    "SimpleH3LoopTrim": SimpleH3LoopTrim,
    "SimpleH3ChainSegmentSave": SimpleH3ChainSegmentSave,
    "SimpleH3ChainReview": SimpleH3ChainReview,
    "SimpleH3BasePreview": SimpleH3BasePreview,
    "SimpleH3DirectEditPreview": SimpleH3DirectEditPreview,
    "SimpleH3ChainLoopEnd": SimpleH3ChainLoopEnd,
    "SimpleH3ChainAssemble": SimpleH3ChainAssemble,
    "SimpleH3BasePreviewAssemble": SimpleH3BasePreviewAssemble,
    "SimpleH3ChainManifestLoad": SimpleH3ChainManifestLoad,
    "SimpleH3StoryboardPlanSplit": SimpleH3StoryboardPlanSplit,
    "SimpleH3StoryboardSheetPrompt": SimpleH3StoryboardSheetPrompt,
    "SimpleH3StoryboardGridSheetPrompt": SimpleH3StoryboardGridSheetPrompt,
    "SimpleH3StoryboardGridConsistencyPrompt": SimpleH3StoryboardGridConsistencyPrompt,
    "SimpleH3StoryboardPromptList": SimpleH3StoryboardPromptList,
    "SimpleH3StoryboardContactSheet": SimpleH3StoryboardContactSheet,
    "SimpleH3FL2VAStoryboardPromptList": SimpleH3FL2VAStoryboardPromptList,
    "SimpleH3StoryboardCollect": SimpleH3StoryboardCollect,
    "SimpleH3StoryboardGridGuide": SimpleH3StoryboardGridGuide,
    "SimpleH3StoryboardSheetRefinePrepare": SimpleH3StoryboardSheetRefinePrepare,
    "SimpleH3StoryboardRefineDispatch": SimpleH3StoryboardRefineDispatch,
    "SimpleH3StoryboardSourceLatentInject": SimpleH3StoryboardSourceLatentInject,
    "SimpleH3StoryboardRefineCollect": SimpleH3StoryboardRefineCollect,
    "SimpleH3StoryboardSplit": SimpleH3StoryboardSplit,
    "SimpleH3CurrentStoryboardFrame": SimpleH3CurrentStoryboardFrame,
    "SimpleH3CurrentStoryboardGuide": SimpleH3CurrentStoryboardGuide,
    "SimpleH3StoryboardVideoReferenceRouter": SimpleH3StoryboardVideoReferenceRouter,
    "SimpleH3StoryboardBlockPlanList": SimpleH3StoryboardBlockPlanList,
    "SimpleH3StoryboardBlockCollect": SimpleH3StoryboardBlockCollect,
    "SimpleH3StoryboardBlockReferenceRouter": SimpleH3StoryboardBlockReferenceRouter,
    "SimpleH3StoryboardVideoIdentityRouter": SimpleH3StoryboardVideoIdentityRouter,
    "SimpleH3StoryboardFL2VAKeyframes": SimpleH3StoryboardFL2VAKeyframes,
}
NODE_CLASS_MAPPINGS.update(_IMAGE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_LONG_EDIT_NODE_CLASS_MAPPINGS)


NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3OptionalLoraLoader": "Simple H3 Load LoRA — Optional",
    "SimpleH3LatentUpscaleResolution": "Simple H3 Latent Upscale — Resolution",
    "SimpleH3LatentUpscaleRefine": "Simple H3 Latent Upscale + Refine — Optional",
    "SimpleH3LatentUpscaleRefineAdvanced": "Simple H3 Latent Upscale + Refine — Advanced",
    "SimpleH3LatentUpscaleRefineMasked": "Simple H3 Latent Upscale + Refine — Masked Continuity (Experimental)",
    "SimpleH3FinalWindowedLatentUpscale": "Simple H3 Final Latent Upscale — Windowed (Experimental)",
    "SimpleH3FinalTimelineTrim": "Simple H3 Final Timeline Trim",
    "SimpleH3FinalWindowedRefineAdvanced": "Simple H3 Final Latent Refine — Windowed Advanced (Experimental)",
    "SimpleH3FinalLatentWindowDecodeAssemble": "Simple H3 Final Latent — Window Decode + Assemble",
    "SimpleH3I2VFinalLatentUpscale": "Simple H3 I2V Final Latent Upscale — Optional",
    "SimpleH3StoryboardCutsUpscaleDecodeAssemble": "Simple H3 Storyboard Cuts — Per-Scene 3D Upscale + Final",
    "SimpleH3I2VFinalDecodeAssemble": "Simple H3 I2V Final — Window Decode + Save",
    "SimpleH3FinalWindowPreviewAssemble": "Simple H3 Refined Final Preview + Assemble",
    "SimpleH3ChainPlan": "Simple H3 Chain Plan",
    "SimpleH3ChainLoopStart": "Simple H3 Start / Resume",
    "SimpleH3ChainCurrent": "Simple H3 Current Scene — Prompt / Seed / Timing",
    "SimpleH3ChainContext": "Simple H3 Context — Masked AV / Masked Cut",
    "SimpleH3BaseContextDecode": "Simple H3 Base Context Decode — Full / Tail Only",
    "SimpleH3CutReferenceSheet": "Simple H3 Cut Reference Sheet",
    "SimpleH3SelectContinuityFrames": "Simple H3 Select Continuity Frames",
    "SimpleH3LoopTrim": "Simple H3 Trim + Lock Audio",
    "SimpleH3ChainSegmentSave": "Simple H3 Save Scene + Checkpoint",
    "SimpleH3ChainReview": "Simple H3 Review — Approve / Retry / Reroll / Stop",
    "SimpleH3BasePreview": "Simple H3 Base Preview — Auto Continue + Final",
    "SimpleH3DirectEditPreview": "Simple H3 Direct Edit Preview — Video + Source Audio",
    "SimpleH3ChainLoopEnd": "Simple H3 Loop Until Final Scene",
    "SimpleH3ChainAssemble": "Simple H3 Assemble Final Video",
    "SimpleH3BasePreviewAssemble": "Simple H3 Base Preview — Assemble Safely",
    "SimpleH3ChainManifestLoad": "Simple H3 Recover Chain",
    "SimpleH3StoryboardPlanSplit": "Simple H3 Storyboard Plan — Split Still / Ref2VA",
    "SimpleH3StoryboardSheetPrompt": "Simple H3 Storyboard Sheet Prompt",
    "SimpleH3StoryboardGridSheetPrompt": "Simple H3 Coherent Storyboard Grid — 2 / 4 / 6",
    "SimpleH3StoryboardGridConsistencyPrompt": "Simple H3 Storyboard Grid — Consistency Pass",
    "SimpleH3StoryboardPromptList": "Simple H3 Individual Storyboard Prompts",
    "SimpleH3StoryboardContactSheet": "Simple H3 Storyboard Preview Grid — Compact",
    "SimpleH3FL2VAStoryboardPromptList": "Simple H3 FL2VA Storyboard Prompts",
    "SimpleH3StoryboardCollect": "Simple H3 Collect Storyboard Images",
    "SimpleH3StoryboardGridGuide": "Simple H3 Storyboard — Neutral Geometry Grid Guide",
    "SimpleH3StoryboardSheetRefinePrepare": "Simple H3 Storyboard Sheet — High-Resolution Refine Prepare",
    "SimpleH3StoryboardBlockPlanList": "Simple H3 Multi-Storyboard — Build Blocks",
    "SimpleH3StoryboardBlockCollect": "Simple H3 Multi-Storyboard — Collect Sheets",
    "SimpleH3StoryboardBlockReferenceRouter": "Simple H3 Multi-Storyboard — Current Complete Sheet",
    "SimpleH3StoryboardRefineDispatch": "Simple H3 Storyboard Panels — Source-Latent Refine Dispatch",
    "SimpleH3StoryboardSourceLatentInject": "Simple H3 Storyboard Panel — Inject Real Source Latent",
    "SimpleH3StoryboardRefineCollect": "Simple H3 Storyboard Panels — Collect Refined",
    "SimpleH3StoryboardSplit": "Simple H3 Split Storyboard Panels",
    "SimpleH3CurrentStoryboardFrame": "Simple H3 Current Storyboard Frame",
    "SimpleH3CurrentStoryboardGuide": "Simple H3 Current Storyboard REF Guide",
    "SimpleH3StoryboardVideoReferenceRouter": "Simple H3 Storyboard Video Reference Router",
    "SimpleH3StoryboardVideoIdentityRouter": "Simple H3 Storyboard Video + Identity Router",
    "SimpleH3StoryboardFL2VAKeyframes": "Simple H3 Storyboard FL2VA Keyframes",
}
NODE_DISPLAY_NAME_MAPPINGS.update(_IMAGE_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_LONG_EDIT_NODE_DISPLAY_NAME_MAPPINGS)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
