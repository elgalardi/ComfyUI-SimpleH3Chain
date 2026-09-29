"""CPU tests for still tensor shapes and grouped decode, without real models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch


class Nested:
    is_nested = True
    def __init__(self, tensors):
        self.tensors = tensors
    def unbind(self):
        return self.tensors


def load_nodes():
    path = Path(__file__).resolve().parents[1] / "still_nodes.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    calls = []
    scope = dict(torch=torch, comfy=SimpleNamespace(
        nested_tensor=SimpleNamespace(NestedTensor=Nested),
        model_management=SimpleNamespace(intermediate_device=lambda: "cpu",
                                        load_models_gpu=lambda *a, **kw: calls.append(kw))))
    exec(compile(tree, str(path), "exec"), scope)
    return scope, calls


class StillTests(unittest.TestCase):
    def test_latent_and_audio_template(self):
        scope, _ = load_nodes()
        for batch in (1, 2):
            video, audio = scope["SimpleH3StillLatent"]().make(64, 96, batch)[0]["samples"].unbind()
            self.assertEqual(tuple(video.shape), (batch, 24, 1, 6, 4))
            self.assertEqual(tuple(audio.shape), (batch, 32, 2, 2))
            self.assertEqual(video.count_nonzero(), 0)
            self.assertEqual(audio.count_nonzero(), 0)

    def test_group_decode_batch_and_frame_selection(self):
        scope, calls = load_nodes()
        seen = []
        def adaptive(z):
            seen.append(z.clone())
            return torch.arange(20.).view(1, 1, 20, 1, 1).expand(1, 3, 20, 4, 4)
        fsm = SimpleNamespace(latents_mean=torch.ones(24), latents_std=torch.ones(24) * 2,
                              _adaptive_decode=adaptive, _finalize_pixels=lambda x: x)
        vae = SimpleNamespace(first_stage_model=fsm, device="cpu", vae_dtype=torch.float32,
                              patcher=object(), memory_used_decode=lambda *a: 100)
        latent = torch.ones(2, 24, 1, 2, 2)
        out = scope["SimpleH3StillDecode"]().decode(vae, {"samples": Nested((latent, torch.zeros(2, 32, 2, 2)))})[0]
        self.assertEqual(tuple(out.shape), (2, 4, 4, 3))
        self.assertTrue(torch.all(out == 3))
        self.assertEqual(tuple(seen[0].shape), (1, 24, 5, 2, 2))
        self.assertTrue(torch.all(seen[0] == 3))
        self.assertEqual(len(calls), 1)

    def test_video_uses_native_decode(self):
        scope, calls = load_nodes()
        expected = torch.zeros(1, 5, 4, 4, 3)
        vae = SimpleNamespace(first_stage_model=object(), decode=lambda x: expected)
        out = scope["SimpleH3StillDecode"]().decode(vae, {"samples": torch.zeros(1, 24, 2, 2, 2)})[0]
        self.assertEqual(tuple(out.shape), (5, 4, 4, 3))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
