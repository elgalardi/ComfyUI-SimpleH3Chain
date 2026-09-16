from __future__ import annotations

import json
import math


class SimpleH3CompactContinuousPlanJSON:
    """Repeat one compact prompt for an exact number of generated continuations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "scene_count": ("INT", {
                    "forceInput": True, "min": 1, "max": 12,
                }),
                "seconds_per_scene": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Nominal seconds per scene, including decimals. H3 frame-grid alignment and continuation context can change the delivered duration.",
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "scene_count", "output_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"

    def build(self, prompt, scene_count, seconds_per_scene, seed=0, steps=6):
        prompt_payload = str(prompt or "").strip()
        if not prompt_payload:
            raise ValueError("Compact continuous mode received an empty prompt.")
        count = max(1, min(12, int(scene_count)))
        scene_prompts = None
        try:
            parsed = json.loads(prompt_payload)
            if isinstance(parsed, dict) and isinstance(
                parsed.get("scene_prompts"), list
            ):
                scene_prompts = [
                    str(item or "").strip() for item in parsed["scene_prompts"]
                ]
        except json.JSONDecodeError:
            pass
        if scene_prompts is None:
            # Backwards-compatible manual/direct use: one plain prompt repeats.
            scene_prompts = [prompt_payload] * count
        if len(scene_prompts) != count or any(not item for item in scene_prompts):
            raise ValueError(
                f"Compact continuous director returned {len(scene_prompts)} prompts; "
                f"scene_count requires exactly {count}. Disable Hold and regenerate."
            )
        seconds = float(seconds_per_scene)
        if not math.isfinite(seconds) or not 1.0 <= seconds <= 15.0:
            raise ValueError("Seconds per scene must be a finite number between 1 and 15.")
        requested_frames = math.ceil(seconds * 24)

        def valid_at_or_above(value):
            value = int(value)
            return value + (5 - value % 17) % 17

        def valid_at_or_below(value):
            value = int(value)
            candidate = value - ((value - 5) % 17)
            return candidate if candidate >= 5 else 0

        first_length = valid_at_or_above(requested_frames)
        continuation_length = valid_at_or_below(requested_frames + 39)
        if continuation_length <= 39:
            raise ValueError("The selected continuation duration is too short.")
        lengths = [first_length] + [continuation_length] * (count - 1)
        delivered = first_length + sum(length - 39 for length in lengths[1:])
        shots = [
            {
                "id": f"continuous_scene_{index:03d}",
                "length": int(length),
                "steps": int(steps),
                "seed": int(seed),
                "prompt": scene_prompts[index - 1],
            }
            for index, length in enumerate(lengths, 1)
        ]
        payload = {
            "summary": (
                f"{count} compact chronological continuous scene prompts."
            ),
            "director_mode": "Continuous Story",
            "prompt_prefix": "",
            "shots": shots,
        }
        status = (
            f"{count} continuous scene(s) × {seconds_per_scene}s nominal -> "
            f"{delivered} delivered frames / {delivered / 24.0:.3f}s; "
            "one compact clean prompt assigned to each scene"
        )
        return (
            json.dumps(payload, ensure_ascii=False), count,
            delivered / 24.0, status,
        )


NODE_CLASS_MAPPINGS = {
    "SimpleH3CompactContinuousPlanJSON": SimpleH3CompactContinuousPlanJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3CompactContinuousPlanJSON": "Simple H3 Compact Continuous — Scene Count",
}
