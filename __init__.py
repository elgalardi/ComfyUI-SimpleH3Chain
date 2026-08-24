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
from .masked_context import apply_masked_av_continuation


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
        _chain._output_root(), "h3_chains", run_name, "final"
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
        _chain.PLAN_TYPE, "STRING", "INT", "INT", "INT", "STRING"
    )
    RETURN_NAMES = (
        "plan", "summary", "clip_count", "width", "height", "plan_preview"
    )
    OUTPUT_TOOLTIPS = _chain.MiniMaxH3ChainPlan.OUTPUT_TOOLTIPS + (
        "Readable production plan with every scene, duration, steps, seed, and prompt.",
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
                    str(inputs.get("context_type", "video")),
                    str(inputs.get("audio_context_frames", "match_video")),
                    int(inputs.get("audio_feather_ticks", 8)),
                ))
        unique = list(dict.fromkeys(found))
        if len(unique) > 1:
            raise ValueError(
                "Simple H3 found Context nodes with different settings. Keep "
                "one Context mode per chain."
            )
        return unique[0] if unique else ("cut", "video", "match_video", 8)

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
        if context_type != "director_auto":
            return str(context_frames), str(context_type)
        mode = str(director_mode or "Continuous Story")
        if mode == "Cinematic Cuts":
            return "cut", "cut_reference"
        if mode == "Reference Edit":
            # A small still-state carry helps multi-scene reference edits while
            # allowing the connected source/reference to remain authoritative.
            frames = "5" if str(context_frames) == "cut" else str(context_frames)
            return frames, "images"
        if mode == "Edit":
            return "cut", "video"
        # Continuous Story keeps the established guide-based path as the safe
        # automatic default. masked_av remains an explicit A/B-test option
        # because it also requires routing this node's latent output.
        frames = "22" if str(context_frames) == "cut" else str(context_frames)
        return frames, "video"

    def build(
        self, plan_json_input, width, height, audio_mode, output_name,
        prompt=None,
    ):
        context_frames, context_type, audio_context_frames, audio_feather_ticks = (
            self._context_configuration(prompt)
        )
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
        cut_reference = context_type == "cut_reference"
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
            anchor_mode = "head" if context_type == "video" else "before"

        run_name = _safe_run_name(output_name)
        fingerprint = (
            "simple-h3-chain-v6-context-contract:"
            f"{width}x{height}:audio={audio_mode}:"
            f"frames={context_frames}:type={context_type}:"
            f"audio_context={audio_context_frames}:feather={audio_feather_ticks}"
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
                "frames" if context_type in ("images", "cut_reference")
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
        return result + (self._format_plan_preview(result[0], context_type),)

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
                    "default": "cut",
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
                    "director_auto", "video", "images", "cut_reference", "masked_av"
                ], {
                    "default": "director_auto",
                    "tooltip": (
                        "director_auto reads Director Mode from the connected plan: "
                        "Continuous Story -> video, Cinematic Cuts -> cut_reference, "
                        "Reference Edit -> images, Edit -> cut. video encodes the "
                        "selected tail as a temporal window. "
                        "images pins every selected frame independently. "
                        "cut_reference ignores the numeric window and uses only "
                        "the final 3 consecutive frames as visual state references, "
                        "without carrying camera motion. Audio continuity is selected "
                        "independently with audio_context_frames. masked_av is an "
                        "experimental direct latent-to-latent continuation for "
                        "any Director mode; it protects an exact 39-frame AV prefix "
                        "without decode/re-encode loss, but may resist a hard cut."
                    ),
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "Only for masked_av. Smoothly releases the final audio "
                        "context edge. 8 ticks equals 0.2 seconds at H3's 40 Hz "
                        "audio latent rate; 0 uses a hard boundary."
                    ),
                }),
            },
            "optional": {
                "audio_vae": ("VAE", {
                    "tooltip": "Optional H3 audio VAE for imported first-scene context.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "BOOLEAN", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "is_continuation", "latent")
    OUTPUT_TOOLTIPS = _chain.MiniMaxH3ChainContext.OUTPUT_TOOLTIPS + (
        "Latent ready for sampling. Required when context_type is masked_av; "
        "safe to use for every other mode as a pass-through.",
    )

    def apply(
        self, state, conditioning, vae, latent, context_frames,
        audio_context_frames, context_type, audio_feather_ticks=8, audio_vae=None,
    ):
        director_mode = str(
            state.get("plan", {}).get("director_mode") or "Continuous Story"
        )
        context_frames, context_type = SimpleH3ChainPlan._director_context(
            context_frames, context_type, director_mode
        )
        if context_type == "masked_av":
            context_frames = "39"
        cut_reference = context_type == "cut_reference"
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
            if director_mode != "Continuous Story":
                print(
                    "[Simple H3 Context] Experimental masked_av override active for "
                    f"{director_mode}; the protected 39-frame prefix may resist a hard cut."
                )
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
            return (_chain._prepare_native_guide_conditioning(conditioning), 0, False, latent)

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
                "video" if context_type == "video" and selected != 11
                else "frames"
            ),
            # Video follows Context Loop's tested head-overlap contract. Image
            # guides and cut references live before the delivered timeline.
            anchor_mode="head" if context_type == "video" else "before",
            crop=cfg["crop"],
            # Cut Reference intentionally keeps sound continuous even though its
            # three visual state frames do not continue the prior camera motion.
            audio_context_length=(effective_audio if effective_audio > 0 else -1),
            audio_mode="timeline",
            context_latent=previous_latent,
            audio_vae=audio_vae,
            context_audio=previous_audio,
        )
        if context_type == "video":
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
            },
        }

    def trim(self, images, trim_frames, audio=None):
        head_trim = int(getattr(trim_frames, "head", int(trim_frames)))
        tail_trim = int(getattr(trim_frames, "tail", 0))
        if tail_trim:
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
                head_samples = int(round(head_trim / 24.0 * sample_rate))
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
            raw_samples = int(round(int(images.shape[0]) / 24.0 * sample_rate))
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
            out_audio["_simple_h3_raw_frames"] = int(images.shape[0])
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
            },
        }

    @staticmethod
    def _temporary_preview_path():
        output_root = os.path.abspath(folder_paths.get_output_directory())
        preview_dir = os.path.abspath(os.path.join(
            output_root, "_simple_h3_temporary_preview"
        ))
        if os.path.commonpath([output_root, preview_dir]) != output_root:
            raise ValueError("Simple H3 temporary preview escaped the output folder.")
        os.makedirs(preview_dir, exist_ok=True)
        for name in os.listdir(preview_dir):
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
            preview_dir, "SimpleH3Preview.%s.mp4" % uuid.uuid4().hex
        )

    @staticmethod
    def _remove_completed_run(manifest):
        output_root = os.path.abspath(folder_paths.get_output_directory())
        chains_root = os.path.abspath(os.path.join(output_root, "h3_chains"))
        run_name = _safe_run_name(manifest.get("run_name", "h3_chain"))
        run_dir = os.path.abspath(os.path.join(chains_root, run_name))
        if (os.path.commonpath([chains_root, run_dir]) != chains_root
                or run_dir == chains_root):
            raise ValueError("Simple H3 refused to clean an unsafe run path.")
        def remove(attempt=0):
            if not os.path.isdir(run_dir):
                return
            try:
                shutil.rmtree(run_dir)
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

    def assemble(self, manifest, filename, save_output, source_audio=None):
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
        preview_path = self._temporary_preview_path()
        os.replace(final_path, preview_path)
        self._remove_completed_run(manifest)
        status = (
            "temporary final preview ready; removed this run's segments, "
            "checkpoints, prompts, reviews, manifest and permanent output"
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


STORYBOARD_LAYOUT_TYPE = "H3_STORYBOARD_LAYOUT"


def _storyboard_grid(scene_count):
    if scene_count <= 1:
        return 1, 1
    if scene_count <= 2:
        return 2, 1
    if scene_count <= 4:
        return 2, 2
    if scene_count <= 6:
        return 3, 2
    if scene_count <= 9:
        return 3, 3
    return 4, 3


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
                "sheet_megapixels": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 8.0, "step": 0.25,
                    "tooltip": "Total storyboard canvas area. 2 MP is a practical first test on a 4090.",
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

    def build(self, plan_json, scene_width, scene_height, sheet_megapixels):
        try:
            plan = json.loads(str(plan_json))
        except json.JSONDecodeError as error:
            raise ValueError("Storyboard Sheet Prompt requires valid Director plan JSON.") from error
        shots = plan.get("shots") or []
        count = len(shots)
        if not 1 <= count <= 12:
            raise ValueError("A single H3 storyboard sheet supports between 1 and 12 scenes.")
        cols, rows = _storyboard_grid(count)
        cell_count = cols * rows
        ratio = (cols * int(scene_width)) / float(rows * int(scene_height))
        area = float(sheet_megapixels) * 1024.0 * 1024.0
        sheet_width = max(32, round(math.sqrt(area * ratio) / 32) * 32)
        sheet_height = max(32, round(math.sqrt(area / ratio) / 32) * 32)

        prefix = str(
            plan.get("storyboard_prompt_prefix") or plan.get("prompt_prefix") or ""
        ).strip()
        lines = [
            "Create ONE single finished cinematic storyboard contact sheet.",
            f"The canvas must be a strict {cols} columns × {rows} rows grid with {cell_count} equal rectangular cells.",
            f"Exactly {count} cells contain scenes, read left-to-right and top-to-bottom.",
            "Every cell is one static first-frame composition, not a motion sequence.",
            "Panels must touch the exact grid boundaries. Keep all people and objects inside their own cell.",
            "No text, captions, numbers, labels, borders, speech bubbles, arrows, watermarks, or overlapping panels.",
            "Do not merge adjacent scenes and do not let any subject cross a cell boundary.",
        ]
        if cell_count > count:
            lines.append(
                f"Leave the final {cell_count - count} unused cell(s) plain solid black with no content."
            )
        if prefix:
            lines.extend((
                "Use the following only as identity, appearance, wardrobe, prop and visual-style rules; ignore video motion or audio instructions:",
                prefix,
            ))
        for index, shot in enumerate(shots, 1):
            static_prompt = str(
                shot.get("storyboard_prompt") or shot.get("prompt") or ""
            ).strip()
            if not static_prompt:
                raise ValueError(f"Scene {index} has no storyboard prompt.")
            lines.append(f"CELL {index}: {static_prompt}")
        layout = {
            "version": 1,
            "scene_count": count,
            "columns": cols,
            "rows": rows,
            "scene_width": int(scene_width),
            "scene_height": int(scene_height),
            "sheet_width": sheet_width,
            "sheet_height": sheet_height,
        }
        summary = (
            f"{count} scenes · {cols}×{rows} grid · {sheet_width}×{sheet_height} sheet "
            f"· panels normalized to {int(scene_width)}×{int(scene_height)}"
        )
        return "\n\n".join(lines), layout, sheet_width, sheet_height, count, summary


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
                    "identity_and_wardrobe",
                    "strict_story_continuity",
                    "identity_only",
                ], {
                    "default": "identity_and_wardrobe",
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
                "This is still a deliberate clean cut with independent framing, not a continuation frame."
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
        for index, shot in indexed_shots:
            static_prompt = str(
                shot.get("storyboard_prompt") or shot.get("prompt") or ""
            ).strip()
            if not static_prompt:
                raise ValueError(f"Scene {index} has no storyboard prompt.")
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
            if prefix:
                parts.extend((
                    "Apply these global identity, appearance, wardrobe, prop, continuity, and visual-style rules:",
                    prefix,
                ))
            parts.extend(("Scene composition:", static_prompt))
            if continuity_mode == "strict_story_continuity" and index > 1:
                previous = str(
                    shots[index - 2].get("storyboard_prompt")
                    or shots[index - 2].get("prompt") or ""
                ).strip()
                if previous:
                    parts.extend((
                        "Previous scene state (continuity facts only; do not copy its framing):",
                        previous,
                    ))
            prompts.append("\n\n".join(parts))
            seeds.append(int((int(base_seed) + index - 1) & 0xffffffffffffffff))
            scene_numbers.append(index)

        generated_count = len(prompts)
        scope_label = "all scenes" if generation_scope == "all_scenes" else f"scene {int(selected_scene)} only"
        return (
            prompts,
            seeds,
            scene_numbers,
            generated_count,
            f"prepared {scope_label} as independent full-resolution images · {continuity_mode}",
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


class SimpleH3StoryboardSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_sheet": ("IMAGE",),
                "layout": (STORYBOARD_LAYOUT_TYPE,),
                "inset_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.25,
                    "tooltip": "Crop this percentage from every edge of each panel before resizing.",
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
        return frames, f"split {count} ordered panels from {cols}×{rows} storyboard grid"


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


class SimpleH3StoryboardFL2VAKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": (_chain.STATE_TYPE,),
            "storyboard_frames": ("IMAGE",),
            "prompt": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "bridge_prompt", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain/Storyboard"
    DESCRIPTION = (
        "Use the current storyboard image as FL2VA first_frame and the next image "
        "as last_frame. The final scene leaves last_frame empty for a free ending."
    )

    def select(self, state, storyboard_frames, prompt):
        index = int(state.get("index", 1)) - 1
        count = int(storyboard_frames.shape[0])
        if index < 0 or index >= count:
            raise ValueError(f"Scene {index + 1} has no storyboard image; only {count} exist.")
        first = storyboard_frames[index:index + 1]
        if index + 1 < count:
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
            direction = (
                "The connected first_frame is the exact approved opening composition. This is the final scene: "
                "develop the described action freely toward a coherent, deliberate ending without introducing "
                "a new unresolved event. No last_frame is imposed."
            )
            status = f"scene {index + 1}/{count}: first frame anchored, final ending free"
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


NODE_CLASS_MAPPINGS = {
    "SimpleH3OptionalLoraLoader": SimpleH3OptionalLoraLoader,
    "SimpleH3ChainPlan": SimpleH3ChainPlan,
    "SimpleH3ChainLoopStart": SimpleH3ChainLoopStart,
    "SimpleH3ChainCurrent": SimpleH3ChainCurrent,
    "SimpleH3ChainContext": SimpleH3ChainContext,
    "SimpleH3CutReferenceSheet": SimpleH3CutReferenceSheet,
    "SimpleH3SelectContinuityFrames": SimpleH3SelectContinuityFrames,
    "SimpleH3LoopTrim": SimpleH3LoopTrim,
    "SimpleH3ChainSegmentSave": SimpleH3ChainSegmentSave,
    "SimpleH3ChainReview": SimpleH3ChainReview,
    "SimpleH3ChainLoopEnd": SimpleH3ChainLoopEnd,
    "SimpleH3ChainAssemble": SimpleH3ChainAssemble,
    "SimpleH3ChainManifestLoad": SimpleH3ChainManifestLoad,
    "SimpleH3StoryboardSheetPrompt": SimpleH3StoryboardSheetPrompt,
    "SimpleH3StoryboardPromptList": SimpleH3StoryboardPromptList,
    "SimpleH3FL2VAStoryboardPromptList": SimpleH3FL2VAStoryboardPromptList,
    "SimpleH3StoryboardCollect": SimpleH3StoryboardCollect,
    "SimpleH3StoryboardSplit": SimpleH3StoryboardSplit,
    "SimpleH3CurrentStoryboardFrame": SimpleH3CurrentStoryboardFrame,
    "SimpleH3CurrentStoryboardGuide": SimpleH3CurrentStoryboardGuide,
    "SimpleH3StoryboardVideoReferenceRouter": SimpleH3StoryboardVideoReferenceRouter,
    "SimpleH3StoryboardFL2VAKeyframes": SimpleH3StoryboardFL2VAKeyframes,
}
NODE_CLASS_MAPPINGS.update(_IMAGE_NODE_CLASS_MAPPINGS)


NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3OptionalLoraLoader": "Simple H3 Load LoRA — Optional",
    "SimpleH3ChainPlan": "Simple H3 Chain Plan",
    "SimpleH3ChainLoopStart": "Simple H3 Start / Resume",
    "SimpleH3ChainCurrent": "Simple H3 Current Scene — Prompt / Seed / Timing",
    "SimpleH3ChainContext": "Simple H3 Context — Video / Images / Cut Reference / Masked AV",
    "SimpleH3CutReferenceSheet": "Simple H3 Cut Reference Sheet",
    "SimpleH3SelectContinuityFrames": "Simple H3 Select Continuity Frames",
    "SimpleH3LoopTrim": "Simple H3 Trim + Lock Audio",
    "SimpleH3ChainSegmentSave": "Simple H3 Save Scene + Checkpoint",
    "SimpleH3ChainReview": "Simple H3 Review — Approve / Retry / Reroll / Stop",
    "SimpleH3ChainLoopEnd": "Simple H3 Loop Until Final Scene",
    "SimpleH3ChainAssemble": "Simple H3 Assemble Final Video",
    "SimpleH3ChainManifestLoad": "Simple H3 Recover Chain",
    "SimpleH3StoryboardSheetPrompt": "Simple H3 Storyboard Sheet Prompt",
    "SimpleH3StoryboardPromptList": "Simple H3 Individual Storyboard Prompts",
    "SimpleH3FL2VAStoryboardPromptList": "Simple H3 FL2VA Storyboard Prompts",
    "SimpleH3StoryboardCollect": "Simple H3 Collect Storyboard Images",
    "SimpleH3StoryboardSplit": "Simple H3 Split Storyboard Panels",
    "SimpleH3CurrentStoryboardFrame": "Simple H3 Current Storyboard Frame",
    "SimpleH3CurrentStoryboardGuide": "Simple H3 Current Storyboard REF Guide",
    "SimpleH3StoryboardVideoReferenceRouter": "Simple H3 Storyboard Video Reference Router",
    "SimpleH3StoryboardFL2VAKeyframes": "Simple H3 Storyboard FL2VA Keyframes",
}
NODE_DISPLAY_NAME_MAPPINGS.update(_IMAGE_NODE_DISPLAY_NAME_MAPPINGS)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
