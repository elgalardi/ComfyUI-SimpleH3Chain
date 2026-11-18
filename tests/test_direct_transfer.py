import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch


spec = importlib.util.spec_from_file_location('direct_transfer', Path(__file__).resolve().parents[1] / 'direct_transfer_nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class DirectTransferTests(unittest.TestCase):
    def test_exact_duration_and_native_grid(self):
        for count in (1, 4, 5, 22, 70, 240, 480, 720, 960):
            frames = torch.arange(count).float().reshape(-1, 1, 1, 1)
            padded, length, actual, seconds, audio, _ = nodes.SimpleH3DirectTransferPrepare().prepare(frames, False)
            self.assertEqual(length % 17, 5)
            self.assertEqual(actual, count)
            self.assertLess(length - count, 17)
            self.assertTrue(torch.equal(padded[:count], frames))
            if length > count:
                self.assertTrue(torch.all(padded[count:] == count - 1))
            output, _ = nodes.SimpleH3DirectTransferOutput().finish(padded, count, None, False)
            self.assertTrue(torch.equal(output, frames))
            self.assertIsNone(audio)

    def test_same_instances_do_not_carry_previous_run(self):
        prepare = nodes.SimpleH3DirectTransferPrepare()
        prompt = nodes.SimpleH3DirectTransferPrompt()
        finish = nodes.SimpleH3DirectTransferOutput()
        for label, count, value in [('couple', 480, 1), ('girl and cat', 70, 2), ('new clip', 240, 3)]:
            frames = torch.full((count, 1, 1, 1), float(value))
            audio = {'waveform': torch.full((1, 1, count * 100), float(value)), 'sample_rate': 2400}
            result = prepare.prepare(frames, True, audio)
            self.assertEqual(prompt.extract(json.dumps({'scene_prompts': [label]})), (label,))
            images, delivered = finish.finish(result[0], count, None, True, result[4])
            self.assertEqual(len(images), count)
            self.assertTrue(torch.all(images == value))
            self.assertTrue(torch.all(delivered['waveform'] == value))
        self.assertEqual(vars(prepare), {})
        self.assertEqual(vars(prompt), {})
        self.assertEqual(vars(finish), {})

    def test_audio_off_does_not_touch_source(self):
        source = Mock()
        result = nodes.SimpleH3DirectTransferPrepare().prepare(torch.ones(240, 1, 1, 1), False, source)
        self.assertIsNone(result[4])
        source.get.assert_not_called()
        generated = {'waveform': torch.ones(1, 1, 24300), 'sample_rate': 2400}
        _, audio = nodes.SimpleH3DirectTransferOutput().finish(result[0], 240, generated, False, source)
        self.assertEqual(audio['waveform'].shape[-1], 24000)
        source.get.assert_not_called()

    def test_missing_soundtrack_is_not_a_guide(self):
        result = nodes.SimpleH3DirectTransferPrepare().prepare(torch.ones(70, 1, 1, 1), True, {'_vhs_no_audio': True})
        self.assertIsNone(result[4])
        _, audio = nodes.SimpleH3DirectTransferOutput().finish(result[0], 70, None, True, result[4])
        self.assertIsNone(audio)

    def test_audio_tail_is_silent_and_input_not_modified(self):
        waveform = torch.ones(1, 2, 24000)
        result = nodes.SimpleH3DirectTransferPrepare().prepare(torch.ones(240, 1, 1, 1), True,
            {'waveform': waveform, 'sample_rate': 2400})
        audio = result[4]['waveform']
        self.assertEqual(audio.shape[-1], result[1] * 100)
        self.assertTrue(torch.all(audio[..., 24000:] == 0))
        audio[..., :24000].zero_()
        self.assertTrue(torch.all(waveform == 1))

    def test_prompt_accepts_plain_and_both_director_contracts(self):
        p = nodes.SimpleH3DirectTransferPrompt()
        for value in ['new prompt', '{"scene_prompts":["new prompt"]}', '{"edit_prompt":"new prompt"}']:
            self.assertEqual(p.extract(value), ('new prompt',))
        for value in ['', '{"scene_prompts":[]}', '{"scene_prompts":["one","two"]}', '{}']:
            with self.assertRaises(ValueError):
                p.extract(value)

    def test_lazy_audio(self):
        p = nodes.SimpleH3DirectTransferPrepare()
        self.assertEqual(p.check_lazy_status(None, False), [])
        self.assertEqual(p.check_lazy_status(None, True), ['source_audio'])


if __name__ == '__main__':
    unittest.main()
