"""Focused MiniMax H3 scene chaining nodes for ComfyUI.

This first release deliberately reuses the pinned, locally tested Context Loop
runtime while presenting a smaller and stable public interface.  It does not
modify the native MiniMax H3 sampling chain.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import torch
import torch.nn.functional as F

from .stable_engine import chain_nodes as _chain
from .stable_engine import nodes as _context


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
                "context_frames": (list(_chain.H3_CONTEXT_LENGTHS), {
                    "default": 22,
                    "tooltip": (
                        "Frames retained in disk checkpoints for recovery. Clean "
                        "Cut mode does not inject them as motion context."
                    ),
                }),
                "audio_mode": (list(_chain.AUDIO_MODES), {
                    "default": "generated_audio",
                }),
                "output_name": ("STRING", {
                    "default": "h3_chain",
                    "tooltip": "Folder and final-chain name. Use a new name for a new production.",
                }),
            }
        }

    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Turn a Story Director JSON plan into independent H3 shots joined by "
        "clean cuts and visual reference continuity."
    )

    def build(self, plan_json_input, width, height, context_frames,
              audio_mode, output_name):
        run_name = _safe_run_name(output_name)
        fingerprint = (
            f"simple-h3-chain-v2-clean-cuts:{width}x{height}:"
            f"ctx={context_frames}:audio={audio_mode}"
        )
        result = super().build(
            plan_json=plan_json_input,
            run_name=run_name,
            generation_fingerprint=fingerprint,
            width=width,
            height=height,
            context_length=context_frames,
            encode_mode="video",
            anchor_mode="head",
            crop="disabled",
            audio_mode=audio_mode,
            audio_context_length=0,
            default_duration_seconds=15.0,
            default_steps=5,
            base_seed=0,
            segment_crf=18,
        )
        return result + (self._format_plan_preview(result[0]),)

    @staticmethod
    def _format_plan_preview(plan):
        compatibility = plan.get("compatibility", {})
        width = compatibility.get("width", "?")
        height = compatibility.get("height", "?")
        lines = [
            "SIMPLE H3 CHAIN PLAN",
            "=" * 58,
            str(plan.get("summary", "")),
            f"Resolution: {width} × {height}",
            f"Output: {plan.get('run_name', 'h3_chain')}",
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
                    "tooltip": "Full source song when the selected audio mode uses one.",
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
        "Keep every scene independent. Previous latents are deliberately not "
        "inserted; continuity comes from the Cut Reference Sheet instead."
    )

    def apply(self, state, conditioning, vae, latent, audio_vae=None):
        return (_chain._prepare_native_guide_conditioning(conditioning), 0, False)


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
                ], {"default": "<Picture 2>"}),
                "panel_size": ([384, 512, 640, 768], {"default": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("continuity_sheet", "augmented_prompt", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Build a Ref2VA continuity picture from two isolated frames carried "
        "from the previous accepted scene and append its exact tag to the prompt."
    )

    def build(self, state, prompt, identity_image, continuity_tag, panel_size):
        size = int(panel_size)
        previous = state.get("previous_frames")
        panels = []
        if torch.is_tensor(previous) and previous.ndim == 4 and previous.shape[0]:
            panels.append(_square_panel(previous[0:1], size))
            if previous.shape[0] > 1:
                panels.append(_square_panel(previous[-1:], size))
            else:
                panels.append(_square_panel(previous[0:1], size))
        else:
            panels.append(_square_panel(identity_image, size))
        sheet = torch.cat(panels, dim=2)
        scene = int(state.get("index", 1))
        if len(panels) == 1:
            instruction = (
                f"{continuity_tag} is the clean identity fallback for the first "
                "scene; use the primary reference pictures for exact identity."
            )
            status = f"scene {scene}: clean fallback; no predecessor"
        else:
            instruction = (
                f"{continuity_tag} contains two isolated frames from the "
                "previously accepted scene. Preserve the current wardrobe, "
                "hairstyle, accessories, physical changes, and persistent props "
                "shown there, while treating this scene as a clean cinematic cut "
                "with a new camera setup. Do not continue the previous camera "
                "motion or copy its background unless explicitly requested."
            )
            status = (
                f"scene {scene}: {len(panels)} isolated predecessor frames as "
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
        "Select two isolated continuity frames. Connect frames_for_loop_end to "
        "Loop Until Final Scene; keep the original images connected to Save Scene."
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
        selected = torch.cat((frame_1, frame_2), dim=0)
        return (
            selected, frame_1, frame_2,
            f"selected frames {first + 1} and {second + 1} of {count}",
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
        return super().trim(
            images=images,
            trim_frames=trim_frames,
            audio=audio,
            fps=24.0,
            match_tail=True,
        )


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
            },
            "optional": {
                "source_audio": ("AUDIO", {
                    "tooltip": "Connect the full source song only when the plan uses source audio.",
                }),
            },
        }

    def assemble(self, manifest, filename, source_audio=None):
        final_name = _versioned_final_name(manifest, filename)
        return super().assemble(
            manifest=manifest,
            audio_source="plan",
            filename=final_name,
            audio_bitrate=256,
            source_audio=source_audio,
        )

    DESCRIPTION = "Join every accepted scene and automatically follow the plan's audio mode."


class SimpleH3ChainManifestLoad(_chain.MiniMaxH3ChainManifestLoad):
    CATEGORY = "MiniMax H3/Simple Chain"


NODE_CLASS_MAPPINGS = {
    "SimpleH3ChainPlan": SimpleH3ChainPlan,
    "SimpleH3ChainLoopStart": SimpleH3ChainLoopStart,
    "SimpleH3ChainCurrent": SimpleH3ChainCurrent,
    "SimpleH3ChainContext": SimpleH3ChainContext,
    "SimpleH3CutReferenceSheet": SimpleH3CutReferenceSheet,
    "SimpleH3SelectContinuityFrames": SimpleH3SelectContinuityFrames,
    "SimpleH3LoopTrim": SimpleH3LoopTrim,
    "SimpleH3ChainSegmentSave": SimpleH3ChainSegmentSave,
    "SimpleH3ChainLoopEnd": SimpleH3ChainLoopEnd,
    "SimpleH3ChainAssemble": SimpleH3ChainAssemble,
    "SimpleH3ChainManifestLoad": SimpleH3ChainManifestLoad,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3ChainPlan": "Simple H3 Chain Plan",
    "SimpleH3ChainLoopStart": "Simple H3 Start / Resume",
    "SimpleH3ChainCurrent": "Simple H3 Current Scene — Prompt / Seed / Timing",
    "SimpleH3ChainContext": "Simple H3 Clean Cut - No Latent Carry",
    "SimpleH3CutReferenceSheet": "Simple H3 Cut Reference Sheet",
    "SimpleH3SelectContinuityFrames": "Simple H3 Select Continuity Frames",
    "SimpleH3LoopTrim": "Simple H3 Trim + Lock Audio",
    "SimpleH3ChainSegmentSave": "Simple H3 Save Scene + Checkpoint",
    "SimpleH3ChainLoopEnd": "Simple H3 Loop Until Final Scene",
    "SimpleH3ChainAssemble": "Simple H3 Assemble Final Video",
    "SimpleH3ChainManifestLoad": "Simple H3 Recover Chain",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
