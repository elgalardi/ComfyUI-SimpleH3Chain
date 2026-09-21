"""CPU-only contract checks; no model loading, denoising, or GPU allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.argv = [sys.argv[0], '--cpu']
spec = importlib.util.spec_from_file_location('simple_h3_upscale_test', ROOT / 'ultimate_upscale.py')
up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)


class UpscaleTests(unittest.TestCase):
    def test_schema(self):
        schema = up.SimpleH3UltimateUpscale.define_schema()
        self.assertEqual(schema.node_id, 'SimpleH3UltimateUpscale')
        flag = next(i for i in schema.inputs if i.id == 'keep_model_loaded')
        self.assertTrue(flag.default)

    def test_explicit_unload_policy_and_audio(self):
        video = up.torch.zeros(1, 24, 1, 2, 2)
        audio = up.torch.zeros(1, 32, 2, 2)
        latent = {'samples': up.comfy.nested_tensor.NestedTensor((video, audio))}
        for keep in (True, False):
            with self.subTest(keep=keep), \
                 patch.object(up, 'sample_piece', side_effect=lambda piece, *args: piece['samples']), \
                 patch.object(up.comfy.model_management, 'unload_model_and_clones') as unload, \
                 patch.object(up.comfy.model_management, 'soft_empty_cache') as empty:
                result = up.SimpleH3UltimateUpscale.execute(
                    latent=latent, conditioning=[], model=SimpleNamespace(clone_base_uuid='test'),
                    noise=None, sampler=None, sigmas=up.torch.tensor([1., 0.]),
                    latent_upscale_param={'method': 'bicubic', 'width': 64, 'height': 64},
                    keep_model_loaded=keep,
                )
                self.assertEqual(unload.call_count, 0 if keep else 1)
                self.assertEqual(empty.call_count, 0 if keep else 1)
                samples = result.result[0]['samples']
                self.assertEqual(tuple(samples.tensors[0].shape), (1, 24, 1, 4, 4))
                self.assertTrue(up.torch.equal(samples.tensors[1], audio))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
