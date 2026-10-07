"""Offline dual-resolution contracts; no ComfyUI execution or model loading."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch
import uuid
import torch

ROOT = Path(__file__).resolve().parents[1]


def definition(path, name):
    return next(n for n in ast.parse((ROOT / path).read_text(encoding='utf-8')).body
                if getattr(n, 'name', None) == name)


def load(node, scope):
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'offline-contract', 'exec'), scope)
    return scope[node.name]


class TwoPassTests(unittest.TestCase):
    def test_base_decoder_only_receives_the_native_context_tail(self):
        chain = SimpleNamespace(STATE_TYPE='STATE', _streams_from_latent=lambda value: value['samples'])
        import math
        cls = load(definition('__init__.py', 'SimpleH3BaseContextDecode'), dict(_chain=chain, math=math))
        video = torch.zeros(1, 24, 72, 4, 4)
        vae = SimpleNamespace(decode=Mock(return_value=torch.zeros(1, 39, 32, 32, 3)))
        state = {'plan': {'base_preview': False, 'compatibility': {'context_length': 39}}}
        images, status = cls().decode(state, {'samples': [video]}, vae)
        self.assertEqual(vae.decode.call_args.args[0].shape[2], 12)
        self.assertEqual(images.shape[0], 39)
        self.assertIn('tail only', status)

    def test_refined_trim_does_not_treat_delivery_as_a_base_tail(self):
        class Parent:
            def trim(self, images, trim_frames, **kwargs):
                return images[trim_frames:], None

        cls = load(definition('__init__.py', 'SimpleH3LoopTrim'),
                   dict(_context=SimpleNamespace(MiniMaxH3LoopTrim=Parent),
                        _chain=SimpleNamespace(STATE_TYPE='STATE'), F=torch.nn.functional))
        state = {'index': 2, 'plan': {'base_preview': False,
                 'compatibility': {'context_length': 39},
                 'shots': [{}, {'raw_frames': 123, 'delivered_frames': 84}]}}
        frames = torch.zeros(123, 8, 8, 3)
        out, _ = cls().trim(frames, 39, state=state, delivery_frames=True)
        self.assertEqual(out.shape[0], 84)
        tail = frames[-39:]
        out, _ = cls().trim(tail, 39, state=state)
        self.assertIs(out, tail)

    def test_plan_records_delivery_without_replacing_base_canvas(self):
        class Parent:
            OUTPUT_TOOLTIPS = ()

            def build(self, **kwargs):
                plan = {'compatibility': {'width': kwargs['width'], 'height': kwargs['height']},
                        'plan_hash': kwargs['generation_fingerprint']}
                return plan, '', 1, kwargs['width'], kwargs['height']

        chain = SimpleNamespace(MiniMaxH3ChainPlan=Parent, PLAN_TYPE='PLAN', AUDIO_MODES=['generated_audio'])
        cls = load(definition('__init__.py', 'SimpleH3ChainPlan'),
                   dict(_chain=chain, json=json, hashlib=hashlib, _safe_run_name=lambda x: x))
        cls._format_plan_preview = staticmethod(lambda *args: '')
        args = dict(plan_json_input='{}', width=608, height=352, audio_mode='generated_audio',
                    output_name='test', base_preview=True)
        old = cls().build(**args)[0]
        new = cls().build(**args, delivery_width=832, delivery_height=448)[0]
        self.assertEqual(new['compatibility']['width'], 608)
        self.assertEqual(new['compatibility']['delivery_width'], 832)
        self.assertNotEqual(new['plan_hash'], old['plan_hash'])
        self.assertNotIn(':delivery=', old['plan_hash'])
        for bad in ({'delivery_width': 832}, {'delivery_height': 448},
                    {'delivery_width': 0, 'delivery_height': 448}):
            with self.assertRaises(ValueError):
                cls().build(**args, **bad)
        self.assertIn('delivery_width', cls.INPUT_TYPES()['optional'])
        prompt = {'235': {'class_type': 'SimpleH3UltimateUpscale', 'inputs': {'state': ['3503', 0]}}}
        protected = cls().build(**args, delivery_width=832, delivery_height=448, prompt=prompt)[0]
        self.assertIn(':refined_prefix=v1', protected['plan_hash'])
        self.assertNotEqual(protected['plan_hash'], new['plan_hash'])

    def test_existing_video_preserves_base_tail_and_plan(self):
        calls = []

        class Parent:
            def prepare(self, plan, **kwargs):
                calls.append((plan, kwargs))
                width = plan['compatibility']['width']
                context = {'frames': ('tail', width), 'plan_hash': plan['plan_hash']}
                if kwargs['prepend_original']:
                    context['prelude'] = {'width': width, 'height': plan['compatibility']['height']}
                return context, 'original will not be prepended'

        chain = SimpleNamespace(MiniMaxH3ChainExternalVideo=Parent,
                                _resolve_video_inputs=Mock(return_value=('frames', 'audio', 24, 'route')))
        cls = load(definition('__init__.py', 'SimpleH3ExistingVideoContext'), dict(_chain=chain))
        plan = {'plan_hash': 'BASE_HASH', 'compatibility': {
            'width': 608, 'height': 352, 'delivery_width': 832, 'delivery_height': 448}}
        original = json.dumps(plan, sort_keys=True)
        context, status = cls().prepare(plan, source_frames='frames', prepend_original=True)
        self.assertEqual(context['frames'], ('tail', 608))
        self.assertEqual(context['plan_hash'], 'BASE_HASH')
        self.assertEqual(context['prelude'], {'width': 832, 'height': 448})
        self.assertEqual(json.dumps(plan, sort_keys=True), original)
        self.assertEqual(len(calls), 2)
        self.assertNotIn('not be prepended', status)
        chain._resolve_video_inputs.assert_called_once()
        context, _ = cls().prepare(plan, source_frames='frames', prepend_original=False)
        self.assertNotIn('prelude', context)

    def test_segment_saves_refined_video_but_base_checkpoint(self):
        node = definition('stable_engine/chain_nodes.py', 'MiniMaxH3ChainSegmentSave')
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('save', '_encode_scene_preview', 'INPUT_TYPES')]
        writer, saver = Mock(), Mock()
        scope = dict(Any=Any, os=os, json=json, uuid=uuid, FPS=24, STATE_TYPE='STATE',
                     _st_save=saver, _write_segment_video=writer, _LOG=Mock(),
                     _compact_latent=lambda x: x, _tensor_cpu_clone=lambda x: x,
                     _artifact_paths=lambda *x: {'segment': 'out/scene.mp4',
                         'checkpoint': 'out/base.safetensors', 'metadata': 'out/scene.json'},
                     _write_run_archives=lambda *x: {}, _archive_media_metadata=lambda x: {},
                     _versioned_path=lambda path, version: path, _atomic_text=Mock(),
                     _history_hash=lambda *x: 'HASH', _relative_output_path=lambda x: x,
                     _prompt_fields=lambda *x: {}, _file_sha256=lambda x: 'SHA',
                     _atomic_json=Mock(), _safe_unlink=Mock(), _cleanup_previous_artifacts=Mock())
        scope['_video_output_item'] = lambda path: {'filename': path, 'type': 'output'}
        cls = load(node, scope)

        class Frames:
            shape = (24, 352, 608, 3)

            def __getitem__(self, key):
                return 'BASE_TAIL'

        refined = SimpleNamespace(shape=(24, 448, 832, 3))
        shot = dict(delivered_frames=24, raw_frames=24, id='one', prompt='test',
                    prompt_hash='PROMPT', seed=1, steps=4)
        plan = dict(shots=[shot], compatibility={'audio_mode': 'silent', 'context_length': 5},
                    base_preview=True, segment_crf=18, run_name='test', plan_hash='PLAN')
        with patch('os.makedirs'), patch('os.path.isfile', return_value=False), patch('os.replace'):
            cls().save({'plan': plan, 'index': 1}, Frames(),
                       {'samples': ['BASE_VIDEO', 'BASE_AUDIO']}, delivery_images=refined,
                       show_preview=True)
        self.assertIs(writer.call_args.args[0], refined)
        writer.assert_called_once()
        tensors = saver.call_args.args[0]
        self.assertEqual(tensors['context_frames'], 'BASE_TAIL')
        self.assertEqual(tensors['video'], 'BASE_VIDEO')
        self.assertIn('delivery_images', cls.INPUT_TYPES()['optional'])
        with self.assertRaisesRegex(ValueError, 'Refined delivery frames'):
            cls().save({'plan': plan, 'index': 1}, Frames(), {},
                       delivery_images=SimpleNamespace(shape=(23, 448, 832, 3)))
        mux = Mock()
        scope['_pyav_mux_audio'] = mux
        scope['_validate_audio'] = lambda audio, *args, **kwargs: (audio['waveform'], audio['sample_rate'])
        audio = {'waveform': torch.zeros(1, 2, 24000), 'sample_rate': 24000}
        writer.reset_mock()
        with patch('os.makedirs'), patch('os.path.isfile', return_value=False), patch('os.replace'):
            result = cls().save({'plan': plan, 'index': 1}, Frames(),
                {'samples': ['BASE_VIDEO', 'BASE_AUDIO'],
                 '_simple_h3_refined_latent': {'samples': ['HQ_TAIL', 'HQ_AUDIO_TAIL']}},
                delivery_images=refined, show_preview=True, audio=audio)
        mux.assert_called_once()
        writer.assert_called_once()
        self.assertEqual(result['ui']['videos'][0]['filename'], 'out/scene.mp4')
        self.assertTrue(result['result'][0]['embedded_audio'])
        self.assertEqual(saver.call_args.args[0]['refined_video'], 'HQ_TAIL')

    def test_original_video_validation_uses_delivery_canvas(self):
        scope = dict(Any=Any, FPS=24, os=os, _absolute_output_path=lambda x: x,
                     _file_sha256=lambda x: 'SHA')
        validate = load(definition('stable_engine/chain_nodes.py', '_validate_prelude'), scope)
        prelude = dict(prepend=True, frame_count=48, fps=24, width=832, height=448,
                       video='original.mp4', video_sha256='SHA')
        manifest = {'prelude': prelude, 'compatibility': {
            'width': 608, 'height': 352, 'delivery_width': 832, 'delivery_height': 448}}
        with patch('os.path.isfile', return_value=True):
            self.assertIs(validate(manifest), prelude)
            prelude['width'] = 608
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                validate(manifest)
            manifest['compatibility'].pop('delivery_width')
            manifest['compatibility'].pop('delivery_height')
            prelude['height'] = 352
            self.assertIs(validate(manifest), prelude)


if __name__ == '__main__':
    unittest.main()
