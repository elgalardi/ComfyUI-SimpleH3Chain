import importlib.util
import json
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('continuous_duration_nodes', Path(__file__).resolve().parents[1] / 'long_edit_nodes.py')
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class ContinuousPlanDurationTests(unittest.TestCase):
    def test_widget_accepts_thirty_seconds(self):
        settings = nodes.SimpleH3CompactContinuousPlanJSON.INPUT_TYPES()['required']['seconds_per_scene'][1]
        self.assertEqual(settings['max'], 30.0)

    def test_single_scene_twenty_and_thirty_seconds(self):
        for seconds in (1.0, 5.0, 15.0, 20.0, 30.0):
            output = nodes.SimpleH3CompactContinuousPlanJSON().build('A woman walks.', 1, seconds)
            shot = json.loads(output[0])['shots'][0]
            self.assertEqual(shot['length'] % 17, 5)
            self.assertGreaterEqual(shot['length'], math.ceil(seconds * 24))
            self.assertLess(shot['length'] - math.ceil(seconds * 24), 17)
            self.assertEqual(output[2], shot['length'] / 24)

    def test_continuations_keep_existing_overlap_contract(self):
        output = nodes.SimpleH3CompactContinuousPlanJSON().build(
            json.dumps({'scene_prompts': ['first', 'second', 'third']}), 3, 30.0)
        shots = json.loads(output[0])['shots']
        self.assertEqual(len(shots), 3)
        self.assertTrue(all(s['length'] % 17 == 5 for s in shots))
        self.assertEqual(output[2], (shots[0]['length'] + sum(s['length'] - 39 for s in shots[1:])) / 24)

    def test_invalid_durations_still_rejected(self):
        for seconds in (0.0, 0.99, 30.01, float('inf'), float('nan')):
            with self.assertRaisesRegex(ValueError, 'between 1 and 30'):
                nodes.SimpleH3CompactContinuousPlanJSON().build('A woman walks.', 1, seconds)


if __name__ == '__main__':
    unittest.main()
