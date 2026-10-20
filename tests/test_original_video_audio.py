"""Original-video soundtrack selection; CPU/FFmpeg only, no model or API calls."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_audio_nodes', ROOT / 'long_edit_nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class OriginalVideoAudioTests(unittest.TestCase):
    def assemble_node(self, audio):
        class Parent:
            def assemble(self, **kwargs):
                self.called = kwargs
                return {'result': ('final.mp4',)}

        chain = SimpleNamespace(MiniMaxH3ChainAssemble=Parent, FPS=24, MANIFEST_TYPE='MANIFEST',
            _audio_fingerprint=Mock(return_value='original-waveform'), _LOG=Mock())
        load = Mock(return_value=audio)
        node = next(n for n in ast.parse((ROOT / '__init__.py').read_text(encoding='utf-8')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'SimpleH3ChainAssemble')
        scope = {'_chain': chain, '_source_audio_from_vhs': load,
                 '_versioned_final_name': lambda manifest, filename: filename}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'assembly-selector', 'exec'), scope)
        return scope['SimpleH3ChainAssemble'](), chain, load

    def test_default_does_not_read_source_or_change_plan_audio(self):
        instance, chain, load = self.assemble_node(None)
        manifest = {'total_delivered_frames': 1440, 'compatibility': {'audio_mode': 'generated_audio'}}
        instance.assemble(manifest, 'final', True)
        self.assertEqual(instance.called['audio_source'], 'plan')
        self.assertIs(instance.called['manifest'], manifest)
        load.assert_not_called()
        self.assertFalse(instance.INPUT_TYPES()['optional']['use_video_audio'][1]['default'])

    def test_original_audio_is_final_only_and_does_not_mutate_manifest(self):
        audio = {'waveform': torch.ones(1, 2, 480), 'sample_rate': 48000}
        instance, chain, load = self.assemble_node(audio)
        manifest = {'total_delivered_frames': 1440, 'compatibility': {'audio_mode': 'generated_audio'}}
        instance.assemble(manifest, 'final', True, use_video_audio=True, prompt={'graph': 1}, unique_id='final')
        load.assert_called_once_with({'graph': 1}, 'final', 60.0)
        self.assertEqual(instance.called['audio_source'], 'source')
        self.assertIs(instance.called['source_audio'], audio)
        self.assertEqual(instance.called['manifest']['compatibility']['source_audio_hash'], 'original-waveform')
        self.assertNotIn('source_audio_hash', manifest['compatibility'])
        self.assertEqual(manifest['compatibility']['audio_mode'], 'generated_audio')

    def test_original_without_soundtrack_is_silent_not_generated(self):
        instance, chain, load = self.assemble_node(None)
        instance.assemble({'total_delivered_frames': 240, 'compatibility': {}}, 'final', True,
                          use_video_audio=True, prompt={}, unique_id='final')
        self.assertEqual(instance.called['audio_source'], 'none')
        chain._LOG.warning.assert_called_once()

    def test_source_audio_reads_from_start_not_current_vhs_block(self):
        ffmpeg, _ = nodes._video_tools()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.mkv'
            subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                'color=size=16x16:rate=24:duration=2', '-f', 'lavfi', '-i',
                'sine=frequency=440:sample_rate=48000:duration=0.25', '-c:v', 'ffv1', '-c:a', 'pcm_f32le', str(path)],
                check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            prompt = {'loader': {'class_type': 'VHS_LoadVideo', 'inputs': {
                'video': 'source.mkv', 'frame_load_cap': 240, 'skip_first_frames': 480}},
                'end': {'class_type': 'SimpleH3ChainLoopEnd', 'inputs': {'frames': ['loader', 0]}},
                'final': {'class_type': 'SimpleH3ChainAssemble', 'inputs': {'manifest': ['end', 0]}}}
            with patch.object(nodes.folder_paths, 'get_input_directory', return_value=temporary), \
                 patch.object(nodes.folder_paths, 'get_annotated_filepath', return_value=str(path)), \
                 patch.object(nodes, '_decode_video_window', side_effect=AssertionError('No frame decoding')):
                audio = nodes._source_audio_from_vhs(prompt, 'final', 1.5)
            self.assertEqual(audio['sample_rate'], 48000)
            self.assertEqual(tuple(audio['waveform'].shape), (1, 2, 72000))
            self.assertGreater(audio['waveform'][..., :6000].abs().max().item(), 0.01)
            self.assertEqual(audio['waveform'][..., 24000:].abs().max().item(), 0.0)

    def test_silent_video_and_ambiguous_loader(self):
        ffmpeg, _ = nodes._video_tools()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'silent.mkv'
            subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                'color=size=16x16:rate=24:duration=1', '-c:v', 'ffv1', str(path)],
                check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            prompt = {'loader': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'silent.mkv'}},
                      'final': {'class_type': 'SimpleH3ChainAssemble', 'inputs': {'manifest': ['loader', 0]}}}
            with patch.object(nodes.folder_paths, 'get_input_directory', return_value=temporary), \
                 patch.object(nodes.folder_paths, 'get_annotated_filepath', return_value=str(path)):
                self.assertIsNone(nodes._source_audio_from_vhs(prompt, 'final', 1.0))
            prompt['other'] = dict(prompt['loader'])
            prompt['final']['inputs']['extra'] = ['other', 0]
            with self.assertRaisesRegex(ValueError, 'one upstream VHS'):
                nodes._source_audio_from_vhs(prompt, 'final', 1.0)

    def test_source_track_late_start_keeps_leading_silence(self):
        ffmpeg, _ = nodes._video_tools()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'delayed.mkv'
            subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                'color=size=16x16:rate=24:duration=1', '-itsoffset', '0.25', '-f', 'lavfi', '-i',
                'sine=frequency=440:sample_rate=48000:duration=0.25', '-c:v', 'ffv1', '-c:a', 'pcm_f32le', str(path)],
                check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            prompt = {'loader': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'delayed.mkv'}},
                      'final': {'class_type': 'SimpleH3ChainAssemble', 'inputs': {'manifest': ['loader', 0]}}}
            with patch.object(nodes.folder_paths, 'get_input_directory', return_value=temporary), \
                 patch.object(nodes.folder_paths, 'get_annotated_filepath', return_value=str(path)):
                audio = nodes._source_audio_from_vhs(prompt, 'final', 0.75)
            self.assertEqual(audio['waveform'].shape[-1], 36000)
            self.assertEqual(audio['waveform'][..., :10000].abs().max().item(), 0.0)
            self.assertGreater(audio['waveform'][..., 14000:22000].abs().max().item(), 0.01)


if __name__ == '__main__':
    unittest.main()
