"""Block audio routing and native guide delegation, without GPU/model loads."""
import ast
import importlib.util
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('block_audio_nodes', ROOT / 'long_edit_nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class BlockAudioGuideTests(unittest.TestCase):
    def test_three_real_blocks_share_exact_vhs_audio_and_image_origin(self):
        raw = json.loads(nodes._source_plan(720, 'source', 'Transfer', 10, 0, 6,
                                           transfer_av=True, max_blocks=3)[0])
        shots = [dict(s, raw_frames=s['length'], scene_prompt=s['prompt']) for s in raw['shots']]
        timeline = torch.arange(720, dtype=torch.float32)
        delivered = []
        for index in range(1, 4):
            state = {'index': index, 'plan': {'shots': shots, 'compatibility': {
                'anchor_mode': 'head', 'context_length': 39, 'audio_context_length': 39,
                'transfer_av': True}}}
            cap, skip = nodes.SimpleH3SourceVideoVHSParams().parameters(state)
            result = nodes.SimpleH3SourceVideoLoadedAVBlock().prepare(
                timeline[skip:skip + cap].reshape(-1, 1, 1, 1), state,
                {'waveform': timeline[skip:skip + cap].reshape(1, 1, -1), 'sample_rate': 24}, True)
            audio = result[6]['waveform'][0, 0]
            self.assertEqual(audio.shape[-1], result[2])
            self.assertTrue(torch.equal(audio[:cap], timeline[skip:skip + cap]))
            self.assertEqual(result[1][0].item(), audio[0].item())
            delivered.append(result[0].flatten())
        self.assertTrue(torch.equal(torch.cat(delivered), timeline))

    def test_disabled_never_materializes_lazy_waveform(self):
        source = Mock()
        outputs = ('real', 'encoded', 243, 10, 'request', 'status')
        with patch.object(nodes.SimpleH3SourceVideoLoadedBlock, 'prepare', return_value=outputs):
            result = nodes.SimpleH3SourceVideoLoadedAVBlock().prepare(None, {}, source, False)
        self.assertEqual(result, (*outputs, None))
        source.get.assert_not_called()

    def test_already_windowed_preroll_is_not_skipped_again(self):
        # The VHS second window already starts 39 frames before the next block.
        waveform = torch.arange(27900, dtype=torch.float32).reshape(1, 1, -1)
        outputs = (None, None, 277, 10, '', '')
        with patch.object(nodes.SimpleH3SourceVideoLoadedBlock, 'prepare', return_value=outputs):
            audio = nodes.SimpleH3SourceVideoLoadedAVBlock().prepare(
                None, {}, {'waveform': waveform, 'sample_rate': 2400}, True)[6]
        self.assertEqual(audio['waveform'].shape[-1], 27700)
        self.assertEqual(audio['waveform'][0, 0, 0].item(), 0)
        self.assertEqual(audio['waveform'][0, 0, -1].item(), 27699)
        self.assertNotEqual(audio['waveform'].data_ptr(), waveform.data_ptr())

    def test_native_grid_tail_is_silence_not_repeated_audio(self):
        outputs = (None, None, 243, 10, '', '')
        with patch.object(nodes.SimpleH3SourceVideoLoadedBlock, 'prepare', return_value=outputs):
            audio = nodes.SimpleH3SourceVideoLoadedAVBlock().prepare(None, {},
                {'waveform': torch.ones(1, 2, 24000), 'sample_rate': 2400}, True)[6]
        self.assertEqual(audio['waveform'].shape[-1], 24300)
        self.assertEqual(audio['waveform'][..., :24000].min().item(), 1)
        self.assertEqual(audio['waveform'][..., 24000:].max().item(), 0)

    def test_no_audio_marker_bypasses_guidance(self):
        with patch.object(nodes.SimpleH3SourceVideoLoadedBlock, 'prepare', return_value=(0, 0, 243, 0, '', '')):
            self.assertIsNone(nodes.SimpleH3SourceVideoLoadedAVBlock().prepare(
                None, {}, {'_vhs_no_audio': True}, True)[6])

    def test_lazy_input_only_requested_when_enabled(self):
        node = nodes.SimpleH3SourceVideoLoadedAVBlock()
        self.assertEqual(node.check_lazy_status(None, {}, use_video_audio=False), [])
        self.assertEqual(node.check_lazy_status(None, {}, use_video_audio=True), ['source_audio'])

    def test_hq_off_is_exact_passthrough_on_uses_native_add_guide(self):
        positive, latent, audio, vae = object(), object(), object(), object()
        native = Mock(return_value=('guided',))
        module = SimpleNamespace(MiniMaxH3AddGuide=SimpleNamespace(execute=native))
        with patch.dict(sys.modules, {'comfy_extras.nodes_minimax_h3': module}):
            node = nodes.SimpleH3OptionalAudioGuide()
            self.assertIs(node.apply(positive, latent, vae)[0], positive)
            native.assert_not_called()
            self.assertEqual(node.apply(positive, latent, vae, audio), ('guided',))
        native.assert_called_once_with(positive=positive, latent=latent, frame_idx=0,
                                       audio_vae=vae, audio=audio)

    def test_upscale_audio_window_uses_40hz_not_inverse_ratio(self):
        tree = ast.parse((ROOT / 'ultimate_upscale.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trim_keyframe')
        scope = {'math': math, 'FRAME_RESCALE': 5 / 3}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'audio-window', 'exec'), scope)
        waveform = torch.arange(400).reshape(1, 1, -1)
        keyframe = {'resolved_frame_index': 0, 'audio_latent': waveform}
        first = scope['trim_keyframe'](keyframe, 0, 120)
        second = scope['trim_keyframe'](keyframe, 120, 240)
        self.assertTrue(torch.equal(first['audio_latent'], waveform[..., :200]))
        self.assertTrue(torch.equal(second['audio_latent'], waveform[..., 200:400]))
        self.assertEqual(second['resolved_frame_index'], 0)


if __name__ == '__main__':
    unittest.main()
