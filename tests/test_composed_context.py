"""CPU-only composed-context contracts; no decoding, model loads or sampling."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('composed_masked_context', ROOT / 'masked_context.py')
masked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masked)


def load_class(name, scope):
    tree = ast.parse((ROOT / '__init__.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'context-contract', 'exec'), scope)
    return scope[name]


class Nested:
    def __init__(self, values):
        self.values = values

    def unbind(self):
        return self.values


class ComposedContextTests(unittest.TestCase):
    def test_scene_one_memory_is_loaded_as_a_small_copied_window_with_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'first.safetensors'
            video = self.source()['samples'][0]
            save_file({'video': video}, str(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            chain = SimpleNamespace(MiniMaxH3ChainContext=type('Parent', (), {'OUTPUT_TOOLTIPS': ()}),
                STATE_TYPE='STATE', _LOG=SimpleNamespace(warning=lambda *args: None),
                _absolute_output_path=lambda value: value,
                _file_sha256=lambda value: hashlib.sha256(Path(value).read_bytes()).hexdigest())
            scope = dict(_chain=chain, safe_open=safe_open, composed_memory_start=masked.composed_memory_start)
            cls = load_class('SimpleH3ChainContext', scope)
            segment = dict(index=1, checkpoint=str(path), checkpoint_sha256=digest,
                           raw_frames=243, delivered_frames=243)
            state = dict(index=3, segments=[segment])
            memory = cls._first_scene_memory(state)
            start = masked.composed_memory_start(72)
            self.assertTrue(torch.equal(memory, video[:, :, start:start + 2]))
            self.assertEqual(memory.shape[2], 2)
            # The same saved segment gives identical context after resume, not an in-memory cache.
            resumed = cls._first_scene_memory(json.loads(json.dumps(state)))
            self.assertTrue(torch.equal(memory, resumed))
            previous = self.source()
            previous['samples'][0] += 1000
            result = masked.compose_previous_av_context(previous, 39, memory)
            self.assertTrue(torch.equal(result['samples'][0][:, :, :2], memory))
            self.assertTrue(torch.equal(result['samples'][0][:, :, 2:], previous['samples'][0][:, :, -10:]))
            self.assertIs(result['samples'][1], previous['samples'][1])
            segment['checkpoint_sha256'] = 'invalid'
            with self.assertRaisesRegex(ValueError, 'integrity'):
                cls._first_scene_memory(state)
            self.assertIsNone(cls._first_scene_memory(dict(segments=[])))

    def source(self, steps=72):
        return {'samples': [torch.arange(steps, dtype=torch.float32).reshape(1, 1, steps, 1, 1),
                            torch.arange(500, dtype=torch.float32).reshape(1, 1, 1, 500)]}

    def test_native_windows_earlier_picture_and_exact_recent_tail(self):
        for steps, head in ((72, 0), (72, 39), (82, 39), (17, 0)):
            with self.subTest(steps=steps, head=head):
                source = self.source(steps)
                original = source['samples'][0].clone()
                result = masked.compose_previous_av_context(source, head)
                video, audio = result['samples']
                self.assertEqual(video.shape[2], 12)
                start = int(video[0, 0, 0, 0, 0])
                self.assertEqual(start % 5, 0)
                self.assertGreaterEqual(masked._pixel_frames(start), head)
                self.assertLessEqual(start + 2, steps - 10)
                self.assertTrue(torch.equal(video[:, :, :2], original[:, :, start:start + 2]))
                self.assertTrue(torch.equal(video[:, :, 2:], original[:, :, -10:]))
                self.assertEqual(masked._pixel_frames(12) - masked._pixel_frames(2), 34)
                self.assertIs(audio, source['samples'][1])
                self.assertTrue(torch.equal(source['samples'][0], original))
                self.assertNotEqual(video.untyped_storage().data_ptr(), source['samples'][0].untyped_storage().data_ptr())

    def test_short_history_falls_back_and_invalid_temporal_phase_is_rejected(self):
        source = self.source(12)
        self.assertIs(masked.compose_previous_av_context(source), source)
        short = self.source(17)
        self.assertIs(masked.compose_previous_av_context(short, 39), short)
        with self.assertRaisesRegex(ValueError, 'complete native H3'):
            masked.compose_previous_av_context(self.source(71))

    def test_masks_trim_and_audio_match_ordinary_masked_av(self):
        comfy = ModuleType('comfy')
        comfy.nested_tensor = SimpleNamespace(NestedTensor=Nested)
        model_base = ModuleType('comfy.model_base')
        model_base.MiniMaxH3 = SimpleNamespace(scale_latent_inpaint=lambda: None)
        target = {'samples': [torch.zeros(1, 1, 72, 1, 1), torch.zeros(1, 1, 1, 500)]}
        source = self.source()
        before_audio = source['samples'][1].clone()
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.nested_tensor': comfy.nested_tensor,
                                     'comfy.model_base': model_base}), \
             patch.object(masked, '_ensure_av_mask_payload_compat'):
            baseline, trim = masked.apply_masked_av_continuation(target, source)
            composed, composed_trim = masked.apply_masked_av_continuation(
                target, masked.compose_previous_av_context(source))
        self.assertEqual((trim, composed_trim), (39, 39))
        self.assertTrue(torch.equal(baseline['samples'].values[1], composed['samples'].values[1]))
        for left, right in zip(baseline['noise_mask'].values, composed['noise_mask'].values):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(composed['samples'].values[0][:, :, 2:12], source['samples'][0][:, :, -10:]))
        self.assertTrue(torch.equal(before_audio, source['samples'][1]))
        self.assertEqual(torch.count_nonzero(target['samples'][0]).item(), 0)

    def test_plan_cache_and_masked_contract_distinguish_new_mode(self):
        class Parent:
            OUTPUT_TOOLTIPS = ()

            def build(self, **kwargs):
                self.received = kwargs
                return {'compatibility': {}}, '', 2, 320, 320

        scope = dict(_chain=SimpleNamespace(MiniMaxH3ChainPlan=Parent, PLAN_TYPE='PLAN', AUDIO_MODES=[]),
                     hashlib=hashlib, json=json, _safe_run_name=lambda value: value)
        cls = load_class('SimpleH3ChainPlan', scope)
        cls._format_plan_preview = staticmethod(lambda *args: '')
        args = dict(plan_json_input='{}', width=320, height=320, audio_mode='generated_audio',
                    output_name='composed_test', base_preview=False)
        fingerprints = []
        for mode in ('masked_av', 'composed_av'):
            prompt = {'context': {'class_type': 'SimpleH3ChainContext', 'inputs': {'context_type': mode}}}
            node = cls()
            node.build(**args, prompt=prompt)
            fingerprints.append(cls.IS_CHANGED(prompt=prompt))
            self.assertEqual(node.received['context_length'], 39)
            self.assertEqual(node.received['anchor_mode'], 'head')
            self.assertTrue(node.received['_compensate_final_overlap_loss'])
            self.assertFalse(node.received['_preserve_delivered_duration'])
            self.assertIn('type=' + mode, node.received['generation_fingerprint'])
        self.assertNotEqual(*fingerprints)
        baseline = cls()
        prompt = {'context': {'class_type': 'SimpleH3ChainContext', 'inputs': {
            'context_type': 'composed_av', 'composed_memory': 'First Scene'}}}
        baseline.build(**args, prompt=prompt)
        self.assertIn('composed_memory=first_scene_v1', baseline.received['generation_fingerprint'])
        self.assertNotEqual(cls.IS_CHANGED(prompt=prompt), fingerprints[-1])

    def test_context_composes_only_on_generated_continuations(self):
        plan_class = SimpleNamespace(_director_context=lambda frames, mode, director: (frames, mode),
            _audio_context_value=lambda *args: 39)
        compose_calls = []
        apply_calls = []

        def compose(value, head, *memory):
            compose_calls.append((value, head, *memory))
            return 'composed'

        def apply(target, source, *args):
            apply_calls.append(source)
            return target, 39

        scope = dict(_chain=SimpleNamespace(MiniMaxH3ChainContext=type('Parent', (), {'OUTPUT_TOOLTIPS': ()}),
                     STATE_TYPE='STATE', _prepare_native_guide_conditioning=lambda value: value),
            SimpleH3ChainPlan=plan_class, json=json, _SplitTrim=lambda head, tail: head + tail,
            compose_previous_av_context=compose, apply_masked_av_continuation=apply)
        cls = load_class('SimpleH3ChainContext', scope)
        plan = {'shots': [dict(raw_frames=243, delivered_frames=243),
                         dict(raw_frames=243, delivered_frames=204)],
                'compatibility': {'audio_mode': 'generated_audio'}}
        args = dict(conditioning='positive', vae=None, latent='target', context_frames='39',
                    audio_context_frames='match_video', context_type='composed_av')
        first = cls().apply(state=dict(index=1, plan=plan), **args)
        self.assertFalse(first[2])
        state = dict(index=2, plan=plan, previous_frames='tail', previous_latent='previous')
        result = cls().apply(state=state, **args)
        self.assertEqual(compose_calls, [('previous', 0)])
        self.assertEqual(apply_calls, ['composed'])
        self.assertEqual(result, ('positive', 39, True, 'target'))
        args['context_type'] = 'masked_av'
        cls().apply(state=state, **args)
        self.assertEqual(len(compose_calls), 1)
        self.assertEqual(apply_calls[-1], 'previous')
        args.update(context_type='composed_av', composed_memory='First Scene')
        cls._first_scene_memory = staticmethod(lambda value: 'first-scene-window')
        cls().apply(state=state, **args)
        self.assertEqual(compose_calls[-1], ('previous', 0))
        state['index'] = 3
        plan['shots'].append(dict(raw_frames=243, delivered_frames=204))
        cls().apply(state=state, **args)
        self.assertEqual(compose_calls[-1], ('previous', 39, 'first-scene-window'))


if __name__ == '__main__':
    unittest.main()
