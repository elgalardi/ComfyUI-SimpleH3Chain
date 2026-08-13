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
        return super().build(
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


class SimpleH3ChainLoopStart(_chain.MiniMaxH3ChainLoopStart):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainCurrent(_chain.MiniMaxH3ChainCurrent):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainContext(_chain.MiniMaxH3ChainContext):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3LoopTrim(_context.MiniMaxH3LoopTrim):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainSegmentSave(_chain.MiniMaxH3ChainSegmentSave):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainLoopEnd(_chain.MiniMaxH3ChainLoopEnd):
    CATEGORY = "MiniMax H3/Simple Chain"


class SimpleH3ChainAssemble(_chain.MiniMaxH3ChainAssemble):
    CATEGORY = "MiniMax H3/Simple Chain"


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
    "SimpleH3ChainLoopStart": "Simple H3 Chain Start",
    "SimpleH3ChainCurrent": "Simple H3 Current Scene",
    "SimpleH3ChainContext": "Simple H3 Auto Context",
    "SimpleH3LoopTrim": "Simple H3 Trim Overlap",
    "SimpleH3ChainSegmentSave": "Simple H3 Save Scene",
    "SimpleH3ChainLoopEnd": "Simple H3 Chain End",
    "SimpleH3ChainAssemble": "Simple H3 Assemble Video",
    "SimpleH3ChainManifestLoad": "Simple H3 Recover Chain",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
