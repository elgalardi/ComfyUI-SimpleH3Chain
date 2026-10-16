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
    def test_advanced_schema_keeps_original_sockets_and_outputs(self):
        original = up.SimpleH3UltimateUpscale.define_schema()
        advanced = up.SimpleH3UltimateUpscaleAdvanced.define_schema()
        self.assertEqual(advanced.node_id, 'SimpleH3UltimateUpscaleAdvanced')
        ids = {i.id for i in advanced.inputs}
        self.assertTrue({'steps', 'cfg', 'sampler_name', 'scheduler', 'noise_seed',
            'start_at_step', 'end_at_step', 'add_noise', 'return_with_leftover_noise',
            'state', 'latent_upscale_param', 'spatial_split_param', 'temporal_split_param'} <= ids)
        self.assertFalse({'noise', 'sampler', 'sigmas'} & ids)
        self.assertEqual([o.id for o in original.outputs], [o.id for o in advanced.outputs])
        self.assertEqual(original.node_id, 'SimpleH3UltimateUpscale')

    def test_advanced_selects_native_schedule_range_and_noise(self):
        full = up.torch.tensor([1., .9, .8, .7, .6, .5, .4, .3, 0.])
        for start, end, leftover, expected in (
            (2, 6, 'disable', [.8, .7, .6, .5, 0.]),
            (2, 6, 'enable', [.8, .7, .6, .5, .4]),
            (6, 10000, 'disable', [.4, .3, 0.]),
            (9, 10000, 'disable', [0.]),
            (7, 3, 'enable', [.7]),
            (0, 0, 'disable', [0.]),
        ):
            for add_noise in ('enable', 'disable'):
                with self.subTest(start=start, end=end, leftover=leftover, noise=add_noise), \
                     patch.object(up.comfy.samplers, 'KSampler', return_value=SimpleNamespace(sigmas=full)) as schedule, \
                     patch.object(up.comfy.samplers, 'sampler_object', return_value='sampler') as sampler, \
                     patch.object(up.SimpleH3UltimateUpscale, 'execute', return_value='result') as run:
                    model = object()
                    result = up.SimpleH3UltimateUpscaleAdvanced.execute(
                        latent='latent', conditioning='positive', model=model,
                        add_noise=add_noise, noise_seed=123, steps=8, cfg=2.5,
                        sampler_name='dpm_2', scheduler='karras', start_at_step=start,
                        end_at_step=end, return_with_leftover_noise=leftover,
                        negative='negative', state='state', latent_upscale_param='upscale',
                        temporal_split_param='temporal', spatial_split_param='spatial',
                        keep_model_loaded=False)
                    self.assertEqual(result, 'result')
                    schedule.assert_called_once_with(model, 8, device='cpu', sampler='dpm_2', scheduler='karras', denoise=1.0)
                    sampler.assert_called_once_with('dpm_2')
                    args = run.call_args.kwargs
                    up.torch.testing.assert_close(args['sigmas'], up.torch.tensor(expected))
                    self.assertEqual(args['noise'].seed, 123)
                    self.assertIsInstance(args['noise'], up.Noise_RandomNoise if add_noise == 'enable' else up._ZeroNoise)
                    self.assertEqual(args['cfg'], 2.5)
                    self.assertEqual(args['negative'], 'negative')
                    self.assertEqual(args['state'], 'state')
                    self.assertEqual(args['latent_upscale_param'], 'upscale')
                    self.assertEqual(args['temporal_split_param'], 'temporal')
                    self.assertEqual(args['spatial_split_param'], 'spatial')
                    self.assertFalse(args['keep_model_loaded'])
        up.torch.testing.assert_close(full, up.torch.tensor([1., .9, .8, .7, .6, .5, .4, .3, 0.]))

    def test_disabled_noise_is_honored_by_both_spatial_paths(self):
        video = up.torch.zeros(1, 24, 2, 4, 4)
        audio = up.torch.zeros(1, 32, 2, 10)
        spatial = dict(tile_width=64, tile_height=64, tile_size_mode='rows_cols',
            grid_rows=1, grid_cols=1, spatial_w_overlap=0, spatial_h_overlap=0,
            min_tile_size=32, fade_width=32, fade_height=32,
            overlap_mode='earlier', overlap_blend='smoothstep')
        calls = []
        def sample(piece, cond, model, noise, *args):
            calls.append(noise.generate_noise(piece))
            return piece['samples']
        for process in (up.spatial_process, up.spatial_process_joint):
            with self.subTest(process=process.__name__), \
                 patch.object(up, 'sample_piece', side_effect=sample), \
                 patch.object(up.comfy.sample, 'prepare_noise', side_effect=AssertionError('random noise requested')):
                process(video, audio, [], spatial, None, up._ZeroNoise(123), None,
                    up.torch.tensor([.5, .25, 0.]), None, 1.)
        self.assertTrue(calls)
        self.assertTrue(all(up.torch.count_nonzero(t).item() == 0 for n in calls for t in n.tensors))

    def test_empty_step_range_does_not_load_sampler(self):
        piece = {'samples': object()}
        with patch.object(up, 'build_guider') as guider:
            output = up.sample_piece(piece, [], None, None, None, up.torch.tensor([0.]), None, 1.)
        self.assertIs(output, piece['samples'])
        guider.assert_not_called()

    def test_progress_accumulates_chunks_tiles_and_sampler_steps(self):
        latent = {'samples': up.comfy.nested_tensor.NestedTensor((
            up.torch.zeros(1, 24, 72, 8, 8), up.torch.zeros(1, 32, 2, 405)))}
        spatial = dict(tile_width=96, tile_height=96, tile_size_mode='rows_cols',
                       grid_rows=2, grid_cols=2, spatial_w_overlap=32, spatial_h_overlap=32,
                       min_tile_size=32, fade_width=32, fade_height=32,
                       overlap_mode='earlier', overlap_blend='smoothstep')

        def sample(piece, cond, model, noise, sampler, sigmas, negative, cfg, progress):
            steps = len(sigmas) - 1
            for step in range(steps):
                progress.update(step + 1)
            progress.advance(steps)
            return piece['samples']

        for joint in (True, False, None):
            values = []
            def make_bar(total):
                return SimpleNamespace(total=total,
                    update_absolute=lambda value, **kwargs: values.append(value))
            with self.subTest(joint=joint), \
                 patch.object(up.comfy.utils, 'ProgressBar', side_effect=make_bar) as bars, \
                 patch.object(up, 'sample_piece', side_effect=sample):
                result = up.SimpleH3UltimateUpscale.execute(
                    latent=latent, conditioning=[], model=SimpleNamespace(),
                    noise=up.Noise_RandomNoise(0), sampler=None,
                    sigmas=up.torch.tensor([1., .5, 0.]),
                    temporal_split_param={'chunk_length': 119, 'temporal_overlap': 17, 'anchor_strength': .999},
                    spatial_split_param=None if joint is None else dict(spatial, joint_steps=joint))
            bars.assert_called_once()
            total = bars.call_args.args[0]
            self.assertEqual(total, len(result.result[1]) * 1000)
            self.assertEqual(values, sorted(values))
            self.assertEqual(values[0], 0)
            self.assertEqual(values[-1], total)
            self.assertTrue(all(value < total for value in values[:-1]))

    def test_sampler_callback_uses_shared_bar_and_preserves_preview(self):
        video = up.torch.zeros(1, 24, 1, 2, 2)
        audio = up.torch.zeros(1, 32, 2, 2)
        samples = up.comfy.nested_tensor.NestedTensor((video, audio))
        progress = Mock()
        previewer = Mock()
        previewer.decode_latent_to_preview_image.return_value = ('JPEG', 'preview', 512)
        guider = SimpleNamespace(model_patcher=SimpleNamespace(
            load_device='cpu', model=SimpleNamespace(latent_format=object())))

        def sample(*args, **kwargs):
            for step in range(2):
                kwargs['callback'](step, samples, samples, 2)
            return samples

        guider.sample = sample
        with patch.object(up, 'build_guider', return_value=guider), \
             patch.object(up.comfy.sample, 'fix_empty_latent_channels', side_effect=lambda model, value, *args: value), \
             patch.object(up.latent_preview, 'get_previewer', return_value=previewer), \
             patch.object(up.latent_preview, 'prepare_callback') as local_callback:
            up.sample_piece({'samples': samples}, [], None,
                SimpleNamespace(seed=0, generate_noise=lambda latent: samples), None,
                up.torch.tensor([1., .5, 0.]), None, 1., progress)
        local_callback.assert_not_called()
        self.assertEqual([call.args[0] for call in progress.update.call_args_list], [1, 2])
        self.assertEqual(progress.update.call_args.args[1], ('JPEG', 'preview', 512))
        progress.advance.assert_called_once_with(2)
        self.assertEqual(previewer.decode_latent_to_preview_image.call_args.args[0], 'JPEG')

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

        for mode, joint, advanced in ((m, j, a) for m in ('masked_av', 'transfer_av')
                                      for j in (True, False, None) for a in (False, True)):
            state['plan']['compatibility']['generation_fingerprint'] = 'type=' + mode
            with self.subTest(mode=mode, joint=joint, advanced=advanced), \
                 patch.object(up, 'sample_piece', side_effect=corrupt), \
                 patch.object(up.comfy.samplers, 'KSampler', return_value=SimpleNamespace(sigmas=torch.tensor([1., .5, 0.]))):
                params = None if joint is None else dict(spatial, joint_steps=joint)
                args = dict(
                    latent=latent, conditioning=[], model=SimpleNamespace(),
                    latent_upscale_param={'method': 'bicubic', 'width': 128, 'height': 128},
                    temporal_split_param={'chunk_length': 119, 'temporal_overlap': 17, 'anchor_strength': .999},
                    spatial_split_param=params, state=state)
                if advanced:
                    result = up.SimpleH3UltimateUpscaleAdvanced.execute(**args,
                        add_noise='disable', noise_seed=123, steps=2, cfg=1.,
                        sampler_name='lcm', scheduler='beta', start_at_step=1, end_at_step=2,
                        return_with_leftover_noise='disable')
                else:
                    result = up.SimpleH3UltimateUpscale.execute(**args,
                        noise=up.Noise_RandomNoise(0), sampler=None, sigmas=torch.tensor([1., .5, 0.]))
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
