"""Stateless single-clip transfer preparation and delivery. No checkpoint IO."""

import json

import torch
import torch.nn.functional as F


def _audio_length(audio, frames):
    if audio is None or audio.get('_vhs_no_audio', False):
        return None
    rate = int(audio['sample_rate'])
    length = round(frames * rate / 24)
    waveform = audio['waveform'][..., :length].clone()
    if waveform.shape[-1] < length:
        waveform = F.pad(waveform, (0, length - waveform.shape[-1]))
    return {'waveform': waveform, 'sample_rate': rate}


class SimpleH3DirectTransferPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'use_video_audio': ('BOOLEAN', {'default': True})},
                'optional': {'source_audio': ('AUDIO', {'lazy': True})}}

    RETURN_TYPES = ('IMAGE', 'INT', 'INT', 'FLOAT', 'AUDIO', 'STRING')
    RETURN_NAMES = ('padded_frames', 'native_length', 'source_frames', 'director_seconds', 'guide_audio', 'status')
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax H3/Direct Transfer'
    DESCRIPTION = 'Prepare only the currently loaded clip. Pad the H3 frame grid at the tail, without history, loops or checkpoints.'

    def check_lazy_status(self, images, use_video_audio=True, source_audio=None):
        return ['source_audio'] if use_video_audio and source_audio is None else []

    def prepare(self, images, use_video_audio=True, source_audio=None):
        count = int(images.shape[0])
        if count < 1:
            raise ValueError('Direct transfer needs at least one loaded source frame.')
        length = max(5, count)
        length += (5 - length % 17) % 17
        padded = images
        if length > count:
            padded = torch.cat((images, images[-1:].expand(length - count, -1, -1, -1)))
        audio = _audio_length(source_audio, length) if use_video_audio else None
        status = (f'Direct clip: {count} frames / {count / 24:.3f}s; '
                  f'{length - count} temporary tail frames; no continuation state.')
        # The director currently advertises a 30-second input limit. Sampling
        # uses the full actual frame count even if a later test loads more.
        return padded, length, count, min(30.0, max(1.0, count / 24)), audio, status


class SimpleH3DirectTransferPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'director_prompt': ('STRING', {'forceInput': True})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('prompt',)
    FUNCTION = 'extract'
    CATEGORY = 'MiniMax H3/Direct Transfer'
    DESCRIPTION = 'Extract the current director response, with no saved prompt or chain state.'

    def extract(self, director_prompt):
        prompt = str(director_prompt or '').strip()
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if 'scene_prompts' in parsed:
                values = parsed['scene_prompts']
                if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
                    raise ValueError('Direct transfer expects one director scene. Set Scenes to 1.')
                prompt = values[0].strip()
            elif isinstance(parsed.get('edit_prompt'), str):
                prompt = parsed['edit_prompt'].strip()
            else:
                raise ValueError('The current director response contains no scene prompt.')
        if not prompt:
            raise ValueError('The current director response is empty.')
        return (prompt,)


class SimpleH3DirectTransferOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'source_frames': ('INT', {'forceInput': True}),
                             'generated_audio': ('AUDIO',), 'use_video_audio': ('BOOLEAN', {'default': True})},
                'optional': {'source_audio': ('AUDIO', {'lazy': True})}}

    RETURN_TYPES = ('IMAGE', 'AUDIO')
    RETURN_NAMES = ('images', 'audio')
    FUNCTION = 'finish'
    CATEGORY = 'MiniMax H3/Direct Transfer'
    DESCRIPTION = 'Remove temporary H3 tail padding and select original or generated audio. Never reads earlier runs.'

    def check_lazy_status(self, images, source_frames, generated_audio, use_video_audio=True, source_audio=None):
        return ['source_audio'] if use_video_audio and source_audio is None else []

    def finish(self, images, source_frames, generated_audio, use_video_audio=True, source_audio=None):
        count = int(source_frames)
        if count < 1 or count > len(images):
            raise ValueError('Decoded direct clip is shorter than its current source frame count.')
        delivered = images[:count].clone() if count < len(images) else images
        audio = _audio_length(source_audio if use_video_audio else generated_audio, count)
        return delivered, audio


NODE_CLASS_MAPPINGS = {
    'SimpleH3DirectTransferPrepare': SimpleH3DirectTransferPrepare,
    'SimpleH3DirectTransferPrompt': SimpleH3DirectTransferPrompt,
    'SimpleH3DirectTransferOutput': SimpleH3DirectTransferOutput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'SimpleH3DirectTransferPrepare': 'Simple H3 Direct Transfer — Prepare Clip (Stateless)',
    'SimpleH3DirectTransferPrompt': 'Simple H3 Direct Transfer — Current Prompt',
    'SimpleH3DirectTransferOutput': 'Simple H3 Direct Transfer — Exact Duration / Audio',
}
