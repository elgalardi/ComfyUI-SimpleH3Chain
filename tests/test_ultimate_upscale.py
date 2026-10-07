"""CPU-only contract checks; no model loading, denoising, or GPU allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

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
        self.assertEqual(schema.outputs[-1].id, 'context_latent')

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

    def test_refined_prefix_survives_tiles_steps_and_temporal_stitch(self):
        torch = up.torch
        video = torch.zeros(1, 24, 72, 4, 4)
        audio = torch.arange(405, dtype=torch.float32).reshape(1, 1, 1, -1).expand(1, 32, 2, -1).clone()
        latent = {'samples': up.comfy.nested_tensor.NestedTensor((video, audio))}
        previous = torch.full((1, 24, 12, 8, 8), 7.0)
        state = {'index': 2, 'plan': {'compatibility': {
            'context_length': 39, 'generation_fingerprint': 'type=masked_av'}},
            'previous_refined_latent': {'samples': [previous, torch.zeros(1, 32, 2, 65)]}}
        spatial = dict(tile_width=96, tile_height=96, tile_size_mode='rows_cols',
                       grid_rows=2, grid_cols=2, spatial_w_overlap=32, spatial_h_overlap=32,
                       min_tile_size=32, fade_width=32, fade_height=32,
                       overlap_mode='earlier', overlap_blend='smoothstep')
        calls = []

        def corrupt(piece, *args):
            mask = piece.get('noise_mask')
            if mask is not None:
                calls.append(mask.tensors[0].clone())
            v, a = piece['samples'].tensors
            return up.comfy.nested_tensor.NestedTensor((v + 0.125, a + 100))

        for joint in (True, False, None):
            with self.subTest(joint=joint), patch.object(up, 'sample_piece', side_effect=corrupt):
                params = None if joint is None else dict(spatial, joint_steps=joint)
                result = up.SimpleH3UltimateUpscale.execute(
                    latent=latent, conditioning=[], model=SimpleNamespace(),
                    noise=SimpleNamespace(seed=0), sampler=None, sigmas=torch.tensor([1., .5, 0.]),
                    latent_upscale_param={'method': 'bicubic', 'width': 128, 'height': 128},
                    temporal_split_param={'chunk_length': 119, 'temporal_overlap': 17, 'anchor_strength': .999},
                    spatial_split_param=params, state=state)
                output = result.result[0]['samples']
                self.assertTrue(torch.equal(output.tensors[0][:, :, :12], previous))
                self.assertTrue(torch.equal(output.tensors[1], audio))
                self.assertTrue(torch.equal(video, torch.zeros_like(video)))
                context = result.result[3]
                self.assertIs(context['samples'], latent['samples'])
                tail = context['_simple_h3_refined_latent']['samples'].tensors
                self.assertEqual(tail[0].shape[2], 12)
                self.assertEqual(tail[1].shape[-1], 65)
                self.assertNotEqual(tail[0].untyped_storage().data_ptr(), output.tensors[0].untyped_storage().data_ptr())
                self.assertGreater(len(result.result[1]), 1)
        self.assertTrue(any(torch.amin(mask).item() == 0 for mask in calls))

    def test_learned_model_is_loaded_once_per_execution_cache(self):
        class Resize(up.torch.nn.Module):
            def forward(self, x, scale, target_size):
                return up.F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)

        video = up.torch.zeros(1, 24, 12, 4, 4)
        params = {'model_name': 'mock.safetensors', 'width': 128, 'height': 128,
                  'device': 'cpu', 'precision': 'fp32'}
        cache = {}
        with patch.object(up, 'load_upscale_model', return_value=Resize()) as loader:
            first = up.upscale_video(video, params, model_cache=cache)[0]
            second = up.upscale_video(video, params, model_cache=cache)[0]
        loader.assert_called_once()
        self.assertTrue(up.torch.equal(first, second))
        self.assertEqual(len(cache), 1)

    def test_old_checkpoint_cannot_silently_drop_refined_protection(self):
        latent = {'samples': up.comfy.nested_tensor.NestedTensor((
            up.torch.zeros(1, 24, 72, 2, 2), up.torch.zeros(1, 32, 2, 405)))}
        state = {'index': 2, 'plan': {'compatibility': {
            'context_length': 39, 'generation_fingerprint': 'type=masked_av'}}}
        with self.assertRaisesRegex(ValueError, 'checkpoint is missing'):
            up.SimpleH3UltimateUpscale.execute(
                latent=latent, conditioning=[], model=None, noise=None, sampler=None,
                sigmas=up.torch.tensor([1., 0.]), state=state)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
