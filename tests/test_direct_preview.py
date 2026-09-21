"""Offline preview contracts: mocked encoding, no ComfyUI or model execution."""
import ast
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch
import uuid


class PreviewTests(unittest.TestCase):
    def test_temporary_preview_is_reused_and_correctly_addressed(self):
        source = (Path(__file__).resolve().parents[1] / '__init__.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                    and n.name == 'SimpleH3DirectEditPreview')
        chain = SimpleNamespace(_safe_name=lambda value, fallback: value or fallback,
                                _write_segment_video=Mock(), _run_ffmpeg=Mock(),
                                _safe_unlink=Mock())
        scope = dict(torch=SimpleNamespace(is_tensor=lambda value: True),
                     os=os, uuid=uuid, json=json, shutil=shutil, _chain=chain,
                     _expand_date_tokens=lambda value: value,
                     folder_paths=SimpleNamespace(get_temp_directory=lambda: 'temp',
                                                  get_output_directory=lambda: 'output'))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'preview', 'exec'), scope)
        cls = scope['SimpleH3DirectEditPreview']
        frames = SimpleNamespace(ndim=4, shape=(24, 720, 1280, 3))
        with patch('os.makedirs'), patch('os.path.exists', return_value=False), \
             patch('os.replace'), patch('shutil.which', return_value='ffmpeg'), \
             patch('builtins.open', mock_open()):
            first = cls().preview(frames, 24, 'first-name', False, True, unique_id='3604')
            second = cls().preview(frames, 24, 'another-name', False, True, unique_id='3604')
            other = cls().preview(frames, 24, 'first-name', False, True, unique_id='3605')
            disabled = cls().preview(frames, 24, 'unused', False, False)
        media = first['ui']['videos'][0]
        self.assertEqual(media['type'], 'temp')
        self.assertEqual(media['filename'], 'preview_3604.mp4')
        self.assertEqual(media, second['ui']['videos'][0])
        self.assertNotEqual(media['filename'], other['ui']['videos'][0]['filename'])
        self.assertIs(first['result'][0], frames)
        self.assertNotIn('videos', disabled['ui'])
        self.assertEqual(cls.RETURN_TYPES, ('IMAGE', 'STRING', 'STRING'))


if __name__ == '__main__':
    unittest.main()
