"""CPU-only checks; no ComfyUI, torch, images or server are loaded."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "simple_h3_frame_gate", Path(__file__).resolve().parents[1] / "frame_gate.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrameGateTests(unittest.TestCase):
    def test_frames_are_passed_by_identity(self):
        gate = module.SimpleH3FrameGate()
        image = object()
        last = object()
        for index in (1, 2, 12):
            for target in (None, last):
                result = gate.select({"index": index}, image, target)
                self.assertIs(result[0], image if index == 1 else None)
                self.assertIs(result[1], index == 1)
                self.assertIsInstance(result[2], str)
                self.assertIs(result[3], target)

    def test_last_frame_is_optional(self):
        image = object()
        result = module.SimpleH3FrameGate().select({"index": 1}, image)
        self.assertIs(result[0], image)
        self.assertIsNone(result[3])

    def test_socket_order(self):
        self.assertEqual(module.SimpleH3FrameGate.RETURN_TYPES,
                         ("IMAGE", "BOOLEAN", "STRING", "IMAGE"))


if __name__ == "__main__":
    unittest.main()
