"""Focused MiniMax H3 scene chaining nodes for ComfyUI.

This first release deliberately reuses the pinned, locally tested Context Loop
runtime while presenting a smaller and stable public interface.  It does not
modify the native MiniMax H3 sampling chain.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path


_CUSTOM_NODES_DIR = str(Path(__file__).resolve().parent.parent)
if _CUSTOM_NODES_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_NODES_DIR)

_chain = importlib.import_module(
    "ComfyUI-MiniMaxH3-Contex-Loop.chain_nodes"
)
_context = importlib.import_module(
    "ComfyUI-MiniMaxH3-Contex-Loop.nodes"
)


def _safe_run_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "h3_chain"))
    value = value.strip("._-")
    return value or "h3_chain"


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
                    "tooltip": "Frames inherited from the previous scene. 22 is the stable default.",
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
        "Turn a Story Director JSON plan into a stable H3 scene chain. "
        "Advanced continuity settings are intentionally fixed to tested values."
    )

    def build(self, plan_json_input, width, height, context_frames,
              audio_mode, output_name):
        run_name = _safe_run_name(output_name)
        fingerprint = (
            f"simple-h3-chain-v1:{width}x{height}:"
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
        return super().assemble(
            manifest=manifest,
            audio_source="plan",
            filename=filename,
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
    "SimpleH3ChainContext": "Simple H3 Auto Context",
    "SimpleH3LoopTrim": "Simple H3 Trim + Lock Audio",
    "SimpleH3ChainSegmentSave": "Simple H3 Save Scene + Checkpoint",
    "SimpleH3ChainLoopEnd": "Simple H3 Loop Until Final Scene",
    "SimpleH3ChainAssemble": "Simple H3 Assemble Final Video",
    "SimpleH3ChainManifestLoad": "Simple H3 Recover Chain",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
