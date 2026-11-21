import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.argv = [sys.argv[0], '--cpu']
pkg = ModuleType('simple_2d_test_package')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
spec = importlib.util.spec_from_file_location(pkg.__name__ + '.latent_upscale_2d', ROOT / 'latent_upscale_2d.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class Learned2DTests(unittest.TestCase):
    def run_node(self, latent):
        def fake(data, *args, **kwargs):
            v = data['samples']
            shape = (*v.shape[:-2], v.shape[-2] * 2, v.shape[-1] * 2)
            return m.io.NodeOutput({'samples': torch.ones(shape)})
        with patch.object(m._Backend, 'execute', side_effect=fake):
            return m.SimpleH3LatentUpscale2D.execute(latent, 'test', {'mode': 'target dimensions', 'width': 128, 'height': 128}).result[0]

    def test_own_schema(self):
        schema = m.SimpleH3LatentUpscale2D.define_schema()
        self.assertEqual(schema.node_id, 'SimpleH3LatentUpscale2D')
        self.assertEqual(schema.inputs[0].id, 'latent')

    def test_joint_audio_and_metadata_preserved(self):
        video = torch.zeros(1, 24, 2, 4, 4)
        audio = torch.randn(1, 32, 2, 10)
        source = {'samples': m.comfy.nested_tensor.NestedTensor((video, audio)), 'batch_index': [3]}
        output = self.run_node(source)
        self.assertEqual(output['samples'].tensors[0].shape[-2:], (8, 8))
        self.assertIs(output['samples'].tensors[1], audio)
        self.assertEqual(output['batch_index'], [3])
        self.assertEqual(source['samples'].tensors[0].shape[-2:], (4, 4))

    def test_masks_keep_protected_prefix(self):
        video = torch.zeros(1, 24, 2, 4, 4)
        audio = torch.zeros(1, 32, 2, 10)
        vm = torch.ones(1, 1, 2, 4, 4)
        vm[:, :, 0] = 0
        am = torch.zeros_like(audio)
        result = self.run_node({'samples': m.comfy.nested_tensor.NestedTensor((video, audio)),
            'noise_mask': m.comfy.nested_tensor.NestedTensor((vm, am))})
        mask = result['noise_mask'].tensors
        self.assertEqual(mask[0].shape, (1, 1, 2, 8, 8))
        self.assertTrue(torch.all(mask[0][:, :, 0] == 0))
        self.assertTrue(torch.all(mask[0][:, :, 1] == 1))
        self.assertIs(mask[1], am)

    def test_plain_video(self):
        output = self.run_node({'samples': torch.zeros(1, 24, 2, 4, 4)})
        self.assertEqual(output['samples'].shape, (1, 24, 2, 8, 8))

    def test_wrong_latent_rejected(self):
        with self.assertRaisesRegex(ValueError, '24-channel'):
            self.run_node({'samples': torch.zeros(1, 4, 4, 4)})
