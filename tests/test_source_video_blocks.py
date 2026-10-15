"""Frame-exact source blocks; no model loads, HTTP calls or GPU sampling."""
import importlib.util
import ast
import hashlib
import json
import math
import re
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import subprocess
from typing import Any

import torch
import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('source_blocks', Path(__file__).resolve().parents[1] / 'long_edit_nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class SourceBlockTests(unittest.TestCase):
    def test_continuation_uses_accepted_first_and_previous_prompts(self):
        plan = json.loads(nodes._source_plan(720, 'source', 'Transfer', 10, 0, 6, masked_av=True)[0])
        state = self.state(plan, 3)
        state['plan']['compatibility'].update(anchor_mode='head', context_length=39, audio_context_length=39)
        state['segments'] = [{'index': 1, 'scene_prompt': 'Established blue suit and red hair'},
                             {'index': 2, 'prompt': 'Ends standing beside the goal'}]
        result = nodes.SimpleH3SourceVideoLoadedBlock().prepare(torch.zeros(240, 1, 1, 3), state)
        request = result[4]
        self.assertIn('FIRST ACCEPTED BLOCK:\nEstablished blue suit and red hair', request)
        self.assertIn('PREVIOUS ACCEPTED BLOCK:\nEnds standing beside the goal', request)
        self.assertIn('SOURCE BLOCK 3/3: 20.000..30.000s', request)
        self.assertIn('Keep subject_definitions consistent', request)
        self.assertIn('design already chosen for active Direction Controls', request)
        self.assertIn('Do not redesign that treatment per block', request)
        self.assertIn('continue a selected orbit without restarting it', request)
        self.assertIn('camera/setting outside explicit user or active Direction Control edits', request)
        state['index'] = 2
        second = nodes.SimpleH3SourceVideoLoadedBlock().prepare(torch.zeros(240, 1, 1, 3), state)[4]
        self.assertEqual(second.count('Established blue suit and red hair'), 1)
        self.assertNotIn('Ends standing beside the goal', second)
        state['index'] = 1
        first = nodes.SimpleH3SourceVideoLoadedBlock().prepare(torch.zeros(240, 1, 1, 3), state)[4]
        self.assertNotIn('CONTINUITY MEMORY', first)
        state['index'] = 3
        state['plan']['compatibility'].update(anchor_mode='before', context_length=1, audio_context_length=0)
        cut = nodes.SimpleH3SourceVideoLoadedBlock().prepare(torch.zeros(240, 1, 1, 3), state)[4]
        self.assertNotIn('CONTINUITY MEMORY', cut)

    def test_metadata_only_plan_never_decodes_a_canvas_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / 'source.mkv'
            subprocess.run([nodes._video_tools()[0], '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                'testsrc=size=32x16:rate=24', '-frames:v', '310', '-c:v', 'ffv1', str(video)], check=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            prompt = {'source': {'class_type':'SimpleH3SourceVideoVHSMetadataPlan','inputs':{}},
                'project': {'class_type':'SimpleH3ChainPlan','inputs':{'plan_json_input':['source',0]}},
                'start': {'class_type':'SimpleH3ChainLoopStart','inputs':{'plan':['project',0]}},
                'current': {'class_type':'SimpleH3ChainCurrent','inputs':{'state':['start',1]}},
                'params': {'class_type':'SimpleH3SourceVideoVHSParams','inputs':{'state':['current',0]}},
                'vhs': {'class_type':'VHS_LoadVideo','inputs':{'video':'source.mkv','frame_load_cap':['params',0],'skip_first_frames':['params',1]}},
                'context': {'class_type':'SimpleH3ChainContext','inputs':{'state':['current',0],'context_type':'masked_av'}}}
            with patch.object(nodes.folder_paths,'get_input_directory',return_value=temporary), \
                 patch.object(nodes.folder_paths,'get_annotated_filepath',return_value=str(video)), \
                 patch.object(nodes.subprocess,'run',wraps=nodes.subprocess.run) as run, \
                 patch.object(nodes,'_decode_video_window',side_effect=AssertionError('No frame decoding')):
                result=nodes.SimpleH3SourceVideoVHSMetadataPlan().build(prompt=prompt,unique_id='source')
                plan=json.loads(result[0])
                self.assertEqual([s['delivered_frames'] for s in plan['shots']],[240,70])
                self.assertEqual(len(run.call_args_list),1)
                self.assertIn('ffprobe',Path(run.call_args.args[0][0]).name)
                factor=math.sqrt(420*420/(32*16))
                self.assertEqual(result[-2:],(round(32*factor)//32*32,round(16*factor)//32*32))

    def test_connected_context_overrides_a_stale_masked_av_widget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.mp4'
            path.write_bytes(b'source')
            prompt = {'source': {'class_type': 'SimpleH3SourceVideoVHSPlan', 'inputs': {}},
                'project': {'class_type': 'SimpleH3ChainPlan', 'inputs': {'plan_json_input': ['source', 0]}},
                'start': {'class_type': 'SimpleH3ChainLoopStart', 'inputs': {'plan': ['project', 0]}},
                'current': {'class_type': 'SimpleH3ChainCurrent', 'inputs': {'state': ['start', 1]}},
                'context': {'class_type': 'SimpleH3ChainContext', 'inputs': {'state': ['current', 0], 'context_type': 'masked_av'}},
                'unrelated_context': {'class_type': 'SimpleH3ChainContext', 'inputs': {'context_type': 'source_control_cut'}}}
            info = {'loaded_fps': 24, 'source_duration': 310 / 24}
            source = nodes.SimpleH3SourceVideoVHSPlan()
            result = source.build(str(path), info, masked_av=False, prompt=prompt, unique_id='source')
            plan = json.loads(result[0])
            self.assertEqual(plan['source_timeline'], 'masked_av')
            self.assertEqual([s['delivered_frames'] for s in plan['shots']], [240, 70])
            self.assertEqual([s['length'] for s in plan['shots']], [243, 124])
            first_key = source.IS_CHANGED(video=str(path), prompt=prompt, unique_id='source', masked_av=False)
            prompt['context']['inputs']['context_type'] = 'source_control_cut'
            result = source.build(str(path), info, masked_av=True, prompt=prompt, unique_id='source')
            self.assertNotIn('source_timeline', json.loads(result[0]))
            self.assertNotEqual(first_key, source.IS_CHANGED(video=str(path), prompt=prompt, unique_id='source', masked_av=False))

    def test_actual_vhs_generator_blocks_match_full_timeline(self):
        vhs_path = Path(__file__).resolve().parents[2] / 'comfyui-videohelpersuite/videohelpersuite/load_video_nodes.py'
        if not vhs_path.is_file():
            self.skipTest('VHS is not installed')
        tree = ast.parse(vhs_path.read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cv_frame_generator')
        class Progress:
            def __init__(self, count): pass
            def update_absolute(self, *args): pass
        scope = {'cv2': cv2, 'np': np, 'torch': torch, 'ProgressBar': Progress}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(vhs_path), 'exec'), scope)
        generate = scope['cv_frame_generator']
        def load(path, cap, skip=0):
            gen = generate(str(path), 24, cap, skip, 1)
            info = next(gen)
            return torch.from_numpy(np.stack(list(gen))), {'loaded_fps': 24, 'source_duration': info[3]}
        for rate in (24, 25, 30):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'source.mkv'
                subprocess.run([nodes._video_tools()[0], '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                    f'testsrc=size=16x16:rate={rate}', '-frames:v', str(rate * 25 + 1), '-c:v', 'ffv1', str(path)],
                    check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                first, info = load(path, 1)
                self.assertEqual(len(first), 1)
                raw, count, _, _, _ = nodes.SimpleH3SourceVideoVHSPlan().build(str(path), info, 30, 'Transfer')
                plan = json.loads(raw)
                blocks = []
                for index in range(1, count + 1):
                    state = self.state(plan, index)
                    cap, skip = nodes.SimpleH3SourceVideoVHSParams().parameters(state)
                    images, _ = load(path, cap, skip)
                    blocks.append(nodes.SimpleH3SourceVideoLoadedBlock().prepare(images, state)[0])
                reference, _ = load(path, sum(len(b) for b in blocks))
                self.assertTrue(torch.equal(torch.cat(blocks), reference))

    def test_vhs_plan_uses_source_duration_not_single_loaded_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / 'source.mp4'
            video.write_bytes(b'test metadata source')
            for duration, expected in [(30, [240, 240, 240]), (25.04, [240, 240, 120]), (5, [120])]:
                info = {'loaded_fps': 24, 'loaded_frame_count': 1, 'source_duration': duration}
                raw, count, seconds, _, _ = nodes.SimpleH3SourceVideoVHSPlan().build(str(video), info, 30, 'Transfer')
                plan = json.loads(raw)
                self.assertEqual([s['delivered_frames'] for s in plan['shots']], expected)
                self.assertEqual((count, seconds), (len(expected), sum(expected) / 24))
                for index, length in enumerate(expected, 1):
                    state = self.state(plan, index)
                    cap, skip = nodes.SimpleH3SourceVideoVHSParams().parameters(state)
                    self.assertEqual((cap, skip), (length, sum(expected[:index - 1])))
                    frames = torch.full((cap, 1, 1, 3), float(index))
                    real, encoded, raw_count, _, request, _ = nodes.SimpleH3SourceVideoLoadedBlock().prepare(frames, state)
                    self.assertTrue(torch.equal(real, frames))
                    self.assertEqual(len(encoded), raw_count)
                    self.assertIn(f'SOURCE BLOCK {index}/{count}', request)
                    self.assertTrue(torch.equal(encoded[-1], frames[-1]))
            info['loaded_fps'] = 30
            with self.assertRaisesRegex(ValueError, 'force_rate'):
                nodes.SimpleH3SourceVideoVHSPlan().build(str(video), info)

    def test_vhs_loaded_block_rejects_wrong_cap_or_format_truncation(self):
        plan = json.loads(nodes._source_plan(480, 'source', 'Transfer', 10, 0, 6)[0])
        with self.assertRaisesRegex(ValueError, 'VHS loaded 239'):
            nodes.SimpleH3SourceVideoLoadedBlock().prepare(torch.zeros(239, 1, 1, 3), self.state(plan, 2))

    def test_vhs_upload_selection_is_shared_without_a_path_widget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'selected.mp4'
            path.write_bytes(b'selected video')
            prompt = {'plan': {'inputs': {'video_info': ['upload', 3]}},
                      'upload': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'selected.mp4'}}}
            info = {'loaded_fps': 24, 'source_duration': 25, 'loaded_frame_count': 1}
            with patch.object(nodes.folder_paths, 'get_input_directory', return_value=temporary), \
                 patch.object(nodes.folder_paths, 'get_annotated_filepath', return_value=str(path)):
                result = nodes.SimpleH3SourceVideoVHSPlan().build(video_info=info, prompt=prompt, unique_id='plan')
                self.assertEqual(result[-1], str(path.resolve()))
                self.assertEqual(result[1], 3)
                original = nodes.SimpleH3SourceVideoVHSPlan.IS_CHANGED(prompt=prompt, unique_id='plan')
                path.write_bytes(b'changed selected video')
                self.assertNotEqual(original, nodes.SimpleH3SourceVideoVHSPlan.IS_CHANGED(prompt=prompt, unique_id='plan'))
                prompt['upload']['inputs']['video'] = ['primitive', 0]
                with self.assertRaisesRegex(ValueError, 'directly'):
                    nodes.SimpleH3SourceVideoVHSPlan().build(video_info=info, prompt=prompt, unique_id='plan')

    def test_file_blocks_match_24fps_resampling_of_other_source_rates(self):
        for rate in (25, 30):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'source.mkv'
                ffmpeg, _ = nodes._video_tools()
                subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                    f'testsrc=size=16x16:rate={rate}', '-frames:v', '601', '-c:v', 'ffv1', str(path)], check=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                raw, count, _, _, _, source = nodes.SimpleH3SourceVideoFilePlan().build(str(path), 0, 'Transfer', 10)
                plan = json.loads(raw)
                blocks = [nodes.SimpleH3SourceVideoFileBlock().load(source, self.state(plan, index))[0]
                    for index in range(1, count + 1)]
                self.assertTrue(torch.equal(torch.cat(blocks), nodes._decode_video_window(source, 0, source['frames'])))

    def test_file_plan_decodes_one_frame_and_each_block_seeks_its_own_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.mkv'
            ffmpeg, _ = nodes._video_tools()
            subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                'testsrc=size=16x16:rate=24', '-frames:v', '601', '-c:v', 'ffv1', str(path)], check=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with patch.object(nodes, '_decode_video_window', wraps=nodes._decode_video_window) as decode:
                raw, total, seconds, _, first, source = nodes.SimpleH3SourceVideoFilePlan().build(str(path), 0, 'Transfer', 10)
                self.assertEqual((total, seconds, first.shape[0]), (3, 601 / 24, 1))
                self.assertEqual([(call.args[1], call.args[2]) for call in decode.call_args_list], [(0, 1)])
                plan = json.loads(raw)
                actual = []
                for index in range(1, 4):
                    frames, padded, raw_count, _, _, _ = nodes.SimpleH3SourceVideoFileBlock().load(source, self.state(plan, index))
                    actual.append(frames)
                    self.assertEqual(len(padded), raw_count)
                self.assertEqual([(call.args[1], call.args[2]) for call in decode.call_args_list],
                    [(0, 1), (0, 240), (240, 240), (480, 121)])
            # A tiny synthetic full decode is only a test oracle, never the production file-plan path.
            expected = nodes._decode_video_window(source, 0, 601)
            self.assertTrue(torch.equal(torch.cat(actual), expected))
            limited = nodes.SimpleH3SourceVideoFilePlan().build(str(path), 10, 'Transfer', 10)
            self.assertEqual((limited[1], limited[2]), (1, 10))
            capped = nodes.SimpleH3SourceVideoFilePlan().build(str(path), 30, 'Transfer', 10)
            self.assertEqual(capped[:3], (raw, total, seconds))
            capped_plan = json.loads(capped[0])
            self.assertEqual([shot['delivered_frames'] for shot in capped_plan['shots']], [240, 240, 121])
            final = nodes.SimpleH3SourceVideoFileBlock().load(capped[-1], self.state(capped_plan, capped[1]))
            self.assertEqual(len(final[0]), 121)
            self.assertEqual(final[3], 121 / 24)
            changed = dict(source, signature='old-file-signature')
            with self.assertRaisesRegex(ValueError, 'changed after planning'):
                nodes._decode_video_window(changed, 0, 1)
            with self.assertRaisesRegex(ValueError, 'another source'):
                nodes.SimpleH3SourceVideoFileBlock().load(changed, self.state(plan, 1))

    def state(self, raw, index):
        shots = [dict(shot, raw_frames=shot['length'], scene_prompt=shot['prompt']) for shot in raw['shots']]
        return {'index': index, 'plan': {'shots': shots, 'compatibility': {
            'anchor_mode': 'before', 'context_length': 1, 'audio_context_length': 0}}}

    def test_thirty_seconds_is_three_exact_ten_second_blocks(self):
        images = torch.arange(720).reshape(720, 1, 1, 1)
        raw, count, seconds, _ = nodes.SimpleH3SourceVideoPlan().build(images, 'Transfer identity', 10, -1, 6)
        plan = json.loads(raw)
        self.assertEqual((count, seconds), (3, 30))
        self.assertEqual([shot['delivered_frames'] for shot in plan['shots']], [240] * 3)
        self.assertEqual([shot['length'] for shot in plan['shots']], [243] * 3)
        self.assertEqual([shot['seed'] for shot in plan['shots']], [(1 << 64) - 1, 0, 1])
        blocks = []
        for index in range(1, 4):
            frames, encoded, length, duration, request, _ = nodes.SimpleH3SourceVideoBlock().slice(images, self.state(plan, index))
            blocks.append(frames)
            self.assertEqual((len(frames), len(encoded), length, duration), (240, 243, 243, 10))
            self.assertTrue(torch.equal(encoded[:240], frames))
            self.assertTrue(torch.equal(encoded[-1], frames[-1]))
            self.assertIn(f'SOURCE BLOCK {index}/3', request)
        self.assertTrue(torch.equal(torch.cat(blocks), images))

    def test_partial_last_block_and_one_frame_video(self):
        for count in (1, 241, 601, 720, 721):
            images = torch.zeros(count, 2, 2, 3)
            raw, total, seconds, _ = nodes.SimpleH3SourceVideoPlan().build(images, 'Request', 10)
            plan = json.loads(raw)
            self.assertEqual(sum(s['delivered_frames'] for s in plan['shots']), count)
            self.assertEqual(seconds, count / 24)
            frames, encoded, length, _, _, _ = nodes.SimpleH3SourceVideoBlock().slice(images, self.state(plan, total))
            self.assertEqual(len(frames), count - (total - 1) * 240)
            self.assertEqual(len(encoded), length)
            self.assertEqual(length % 17, 5)

    def test_prompt_records_actual_text_without_mutating_input_state(self):
        raw = json.loads(nodes.SimpleH3SourceVideoPlan().build(torch.zeros(480, 1, 1, 3), 'Request')[0])
        state = self.state(raw, 1)
        updated, prompt = nodes.SimpleH3SourceBlockPrompt().apply(state, '{"scene_prompts": ["Actual visual prompt"]}')
        self.assertEqual(prompt, 'Actual visual prompt')
        self.assertEqual(updated['plan']['shots'][0]['prompt'], prompt)
        self.assertEqual(state['plan']['shots'][0]['prompt'], 'Request')
        self.assertEqual(updated['plan']['shots'][1], state['plan']['shots'][1])
        with self.assertRaises(ValueError):
            nodes.SimpleH3SourceBlockPrompt().apply(state, '{"scene_prompts": ["One", "Two"]}')

    def test_masked_overlap_is_not_silently_applied_to_source_windows(self):
        raw = json.loads(nodes.SimpleH3SourceVideoPlan().build(torch.zeros(240, 1, 1, 3), 'Request')[0])
        state = self.state(raw, 1)
        state['plan']['compatibility'].update(anchor_mode='head', context_length=39)
        with self.assertRaisesRegex(ValueError, 'source_control_cut'):
            nodes.SimpleH3SourceVideoBlock().slice(torch.zeros(240, 1, 1, 3), state)

    def test_source_masked_av_keeps_each_delivered_window_and_pads_a_prefix(self):
        images = torch.arange(601).reshape(601, 1, 1, 1)
        plan = json.loads(nodes._source_plan(601, 'source', 'Transfer', 10, 0, 6, masked_av=True)[0])
        self.assertEqual([s['length'] for s in plan['shots']], [243, 294, 175])
        self.assertEqual([s['delivered_frames'] for s in plan['shots']], [240, 240, 121])
        for index in range(1, 4):
            state = self.state(plan, index)
            state['plan']['compatibility'].update(anchor_mode='head', context_length=39, audio_context_length=39)
            cap, skip = nodes.SimpleH3SourceVideoVHSParams().parameters(state)
            current = images[skip:skip + cap]
            real, encoded, length, _, request, _ = nodes.SimpleH3SourceVideoLoadedBlock().prepare(current, state)
            prefix = 39 if index > 1 else 0
            self.assertTrue(torch.equal(real, current))
            self.assertTrue(torch.equal(encoded[prefix:prefix + cap], current))
            self.assertEqual(len(encoded), length)
            self.assertIn('same speaker voices', request)

    def test_native_plan_keeps_exact_delivery_and_resume_identity(self):
        path = Path(__file__).resolve().parents[1] / 'stable_engine' / 'chain_nodes.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'_canonical_json', '_fingerprint', '_safe_name', '_prompt_text',
            '_h3_frame_length_at_least', '_h3_frame_length', '_validate_h3_length',
            '_derived_seed', '_normalize_plan', '_history_contract', '_history_hash'}
        scope = dict(json=json, hashlib=hashlib, math=math, re=re, Any=Any,
            FPS=24, PLAN_VERSION=2, MAX_SHOTS=128, MAX_SEED=(1 << 64) - 1,
            MAX_H3_FRAMES=3592, H3_CONTEXT_LENGTHS=(1, 5, 11, 22, 39),
            AUDIO_MODES=('generated_audio',))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'native-plan', 'exec'), scope)
        masked = nodes._source_plan(601, 'source', 'Transfer', 10, 0, 6, masked_av=True)[0]
        normalized = scope['_normalize_plan'](masked, 'Sexy AI Studio/Tests/SourceAV',
            320, 320, 39, 'video', 'head', 'disabled', 'generated_audio', 39, 10, 6, 0, 18,
            compensate_final_overlap_loss=True)
        self.assertEqual(normalized['total_delivered_frames'], 601)
        self.assertEqual([s['raw_frames'] for s in normalized['shots']], [243, 294, 175])
        self.assertEqual([s['generation_start_frame'] for s in normalized['shots']], [0, 201, 441])
        for index in range(1, 4):
            state = {'index': index, 'plan': normalized}
            cap, skip = nodes.SimpleH3SourceVideoVHSParams().parameters(state)
            self.assertEqual((cap, skip), [(240, 0), (240, 240), (121, 480)][index - 1])
        short = nodes._source_plan(310, 'source', 'Transfer', 10, 0, 6, masked_av=True)[0]
        repaired = scope['_normalize_plan'](short, 'Sexy AI Studio/Tests/SourceAV',
            320, 320, 39, 'video', 'head', 'disabled', 'generated_audio', 39, 10, 6, 0, 18,
            compensate_final_overlap_loss=True)
        self.assertEqual(repaired['total_delivered_frames'], 310)
        self.assertEqual(nodes.SimpleH3SourceVideoVHSParams().parameters({'index':2, 'plan':repaired}), (70, 240))
        real, encoded, length, _, _, _ = nodes.SimpleH3SourceVideoLoadedBlock().prepare(
            torch.zeros(70, 2, 2, 3), {'index':2, 'plan':repaired})
        self.assertEqual((len(real), len(encoded), length), (70, 124, 124))
        for count in (1, 601, 720, 721):
            images = torch.zeros(count, 2, 2, 3)
            raw = nodes.SimpleH3SourceVideoPlan().build(images, 'Transfer identity')[0]
            plan = scope['_normalize_plan'](raw, 'Sexy AI Studio/Tests/SourceBlocks',
                320, 320, 1, 'video', 'before', 'disabled', 'generated_audio', 0, 10, 6, 0, 18)
            self.assertEqual(plan['total_delivered_frames'], count)
            self.assertEqual([s['audio_start_seconds'] for s in plan['shots']],
                [i * 10 for i in range(len(plan['shots']))])
            for index in range(1, len(plan['shots']) + 1):
                state = {'index': index, 'plan': plan}
                frames, encoded, _, _, _, _ = nodes.SimpleH3SourceVideoBlock().slice(images, state)
                self.assertEqual(len(frames), plan['shots'][index - 1]['delivered_frames'])
                self.assertEqual(len(encoded), plan['shots'][index - 1]['raw_frames'])
                before = scope['_history_hash'](plan, index)
                updated, _ = nodes.SimpleH3SourceBlockPrompt().apply(state, '{"scene_prompts": ["Generated prompt"]}')
                self.assertEqual(scope['_history_hash'](updated['plan'], index), before)
        changed = images.clone()
        changed[0, 0, 0, 0] = 1
        self.assertNotEqual(json.loads(nodes.SimpleH3SourceVideoPlan().build(images, 'Request')[0])['shots'][0]['id'],
            json.loads(nodes.SimpleH3SourceVideoPlan().build(changed, 'Request')[0])['shots'][0]['id'])


if __name__ == '__main__':
    unittest.main()
