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
import folder_paths
import comfy.sd
import comfy.utils
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

from .stable_engine import chain_nodes as _chain
from .stable_engine import nodes as _context
from .frame_gate import SimpleH3FrameGate
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
    if str(value).replace('\\', '/').startswith('Sexy AI Studio/'):
        return _chain._safe_name(value, 'h3_chain')
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
                "unique_id": "UNIQUE_ID",
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
        prompt=None, extra_pnginfo=None, unique_id=None,
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
        if bool(save_output):
            path = os.path.join(directory, base + ".mp4")
            version = 2
            while os.path.exists(path):
                path = os.path.join(directory, f"{base}_v{version}.mp4")
                version += 1
        else:
            node_key = _chain._safe_name(str(unique_id or "preview"), "preview")
            path = os.path.join(directory, f"preview_{node_key}.mp4")
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
                ui_result["videos"] = [{
                    "filename": os.path.basename(path),
                    "subfolder": subfolder,
                    "type": "output" if bool(save_output) else "temp",
                }]
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


from .ultimate_upscale import (
    SimpleH3UltimateUpscale, SimpleH3LatentUpscaleParams,
    SimpleH3LatentUpscaleWithModelParams, SimpleH3TemporalSplitParams,
    SimpleH3SpatialSplitParams,
)

NODE_CLASS_MAPPINGS = {
    "SimpleH3UltimateUpscale": SimpleH3UltimateUpscale,
    "SimpleH3LatentUpscaleParams": SimpleH3LatentUpscaleParams,
    "SimpleH3LatentUpscaleWithModelParams": SimpleH3LatentUpscaleWithModelParams,
    "SimpleH3TemporalSplitParams": SimpleH3TemporalSplitParams,
    "SimpleH3SpatialSplitParams": SimpleH3SpatialSplitParams,
    "SimpleH3DirectEditPreview": SimpleH3DirectEditPreview,
    "SimpleH3FrameGate": SimpleH3FrameGate,
    "SimpleH3OptionalLoraLoader": SimpleH3OptionalLoraLoader,
    "SimpleH3LatentUpscaleResolution": SimpleH3LatentUpscaleResolution,
    "SimpleH3LatentUpscaleRefine": SimpleH3LatentUpscaleRefine,
    "SimpleH3ChainPlan": SimpleH3ChainPlan,
    "SimpleH3ChainLoopStart": SimpleH3ChainLoopStart,
    "SimpleH3ChainCurrent": SimpleH3ChainCurrent,
    "SimpleH3ChainContext": SimpleH3ChainContext,
    "SimpleH3LoopTrim": SimpleH3LoopTrim,
    "SimpleH3ChainSegmentSave": SimpleH3ChainSegmentSave,
    "SimpleH3ChainLoopEnd": SimpleH3ChainLoopEnd,
    "SimpleH3ChainAssemble": SimpleH3ChainAssemble,
}
NODE_CLASS_MAPPINGS.update(_IMAGE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(_LONG_EDIT_NODE_CLASS_MAPPINGS)


NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3UltimateUpscale": "Simple H3 Ultimate Upscale",
    "SimpleH3LatentUpscaleParams": "Simple H3 Upscale Params — Interpolation",
    "SimpleH3LatentUpscaleWithModelParams": "Simple H3 Upscale Params — Learned Model",
    "SimpleH3TemporalSplitParams": "Simple H3 Upscale — Temporal Split",
    "SimpleH3SpatialSplitParams": "Simple H3 Upscale — Spatial Split",
    "SimpleH3DirectEditPreview": "Simple H3 Direct Edit Preview — Video + Source Audio",
    "SimpleH3FrameGate": "Simple H3 Frame Gate — First / Last Frame",
    "SimpleH3OptionalLoraLoader": "Simple H3 Load LoRA — Optional",
    "SimpleH3LatentUpscaleResolution": "Simple H3 Latent Upscale — Resolution",
    "SimpleH3LatentUpscaleRefine": "Simple H3 Latent Upscale + Refine — Optional",
    "SimpleH3ChainPlan": "Simple H3 Chain Plan",
    "SimpleH3ChainLoopStart": "Simple H3 Start / Resume",
    "SimpleH3ChainCurrent": "Simple H3 Current Scene — Prompt / Seed / Timing",
    "SimpleH3ChainContext": "Simple H3 Context — Masked AV / Masked Cut",
    "SimpleH3LoopTrim": "Simple H3 Trim + Lock Audio",
    "SimpleH3ChainSegmentSave": "Simple H3 Save Scene + Checkpoint",
    "SimpleH3ChainLoopEnd": "Simple H3 Loop Until Final Scene",
    "SimpleH3ChainAssemble": "Simple H3 Assemble Final Video",
}
NODE_DISPLAY_NAME_MAPPINGS.update(_IMAGE_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_LONG_EDIT_NODE_DISPLAY_NAME_MAPPINGS)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
