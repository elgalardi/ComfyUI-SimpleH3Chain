"""First/last frame routing for Simple H3 scene chains."""


class SimpleH3FrameGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": ("H3_CHAIN_STATE",),
                "image": ("IMAGE",),
            },
            "optional": {"last_frame": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "IMAGE")
    RETURN_NAMES = ("first_frame", "is_first_scene", "status", "last_frame")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = (
        "Pass the original opening image only on scene 1. "
        "The optional last frame passes through on every scene. "
        "No resizing, copying or image processing is performed."
    )

    def select(self, state, image, last_frame=None):
        scene = int(state["index"])
        first = scene == 1
        status = (
            f"Scene {scene}: first frame {'connected' if first else 'omitted'}; "
            f"last frame {'connected' if last_frame is not None else 'omitted'}"
        )
        return image if first else None, first, status, last_frame
