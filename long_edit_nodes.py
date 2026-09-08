from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
from comfy_extras.nodes_audio import load as _load_audio_file


def _decode_video_window(path, skip_frames, frame_count, fps=24):
    """Decode only one exact CFR window through ffmpeg."""
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", path,
    ], capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError("Could not inspect source video: " + probe.stderr[-1000:])
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", f"{int(skip_frames)/float(fps):.9f}",
        "-i", path, "-an", "-vf", f"fps={int(fps)}",
        "-frames:v", str(int(frame_count)), "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True)
    if result.returncode:
        raise RuntimeError("Could not decode source video window: " +
                           result.stderr.decode("utf-8", "replace")[-1500:])
    frame_bytes = width * height * 3
    count = len(result.stdout) // frame_bytes
    if count < 1:
        raise ValueError("The requested source-video window contains no frames.")
    array = np.frombuffer(result.stdout[:count * frame_bytes], dtype=np.uint8).copy()
    images = torch.from_numpy(array.reshape(count, height, width, 3)).float().div_(255.0)
    if count < int(frame_count):
        images = torch.cat([images, images[-1:].repeat(int(frame_count)-count,1,1,1)], 0)
    return images


class SimpleH3UnifiedVideoSource:
    """Select a source once; expose director sample, full audio and descriptor."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["video"])
        return {"required": {
            "video": (sorted(files),),
            "source_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0,
                                                  "max": 86400.0, "step": 0.1}),
            "director_sample_frames": ("INT", {"default": 24, "min": 1, "max": 120}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "H3_VIDEO_SOURCE", "STRING")
    RETURN_NAMES = ("director_sample", "full_audio", "source", "status")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/Simple Chain"

    def load(self, video, source_start_seconds, director_sample_frames):
        path = folder_paths.get_annotated_filepath(video)
        start = int(round(float(source_start_seconds) * 24.0))
        sample = _decode_video_window(path, start, int(director_sample_frames), 24)
        waveform, rate = _load_audio_file(path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": int(rate)}
        source = {"path": path, "audio": audio,
                  "source_start_seconds": float(source_start_seconds)}
        return sample, audio, source, (
            f"{video}: selected once; director sample={int(director_sample_frames)} "
            f"frames; recursive windows at 24 fps"
        )

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(path)


class SimpleH3RecursiveVideoWindow:
    """Decode the current plan scene from one unified video source."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": ("H3_CHAIN_STATE",),
            "source": ("H3_VIDEO_SOURCE",),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "frame_load_cap", "skip_first_frames", "status")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/Simple Chain"

    def load(self, state, source):
        plan, index = state["plan"], int(state["index"])
        shot = plan["shots"][index - 1]
        raw = int(shot["raw_frames"])
        delivered_before = sum(int(x["delivered_frames"])
                               for x in plan["shots"][:index-1])
        overlap = 0 if index == 1 else 39
        base = int(round(float(source.get("source_start_seconds", 0.0)) * 24.0))
        skip = base + max(0, delivered_before - overlap)
        images = _decode_video_window(source["path"], skip, raw, 24)
        full = source["audio"]
        rate, waveform = int(full["sample_rate"]), full["waveform"]
        a0, count = int(round(skip/24.0*rate)), int(round(raw/24.0*rate))
        audio = waveform[..., a0:min(a0+count, int(waveform.shape[-1]))]
        if int(audio.shape[-1]) < count:
            audio = F.pad(audio, (0, count-int(audio.shape[-1])))
        return images, {"waveform": audio.contiguous(), "sample_rate": rate}, raw, skip, (
            f"Scene {index}/{len(plan['shots'])}: decoded only source "
            f"[{skip}:{skip+raw}) at 24 fps"
        )


class SimpleH3LongEditPlanJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE",),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "block_seconds": (["5", "10", "15"], {"default": "10"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "clip_count", "source_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"

    def build(self, source_frames, edit_prompt, block_seconds):
        if not torch.is_tensor(source_frames) or source_frames.ndim != 4:
            raise ValueError("Long edit requires a decoded IMAGE frame batch.")
        total_frames = int(source_frames.shape[0])
        total_seconds = total_frames / 24.0
        requested = int(block_seconds)
        if total_seconds < 5.0:
            raise ValueError("Long edit requires at least 5 seconds at 24 fps.")

        # H3 accepts only 17k+5 frame lengths.  Use explicit lengths so the
        # chain does not round every duration upward and demand more source
        # audio than the loader actually supplied.
        requested_frames = requested * 24
        target_length = requested_frames + (5 - requested_frames % 17) % 17

        def valid_at_or_below(value):
            value = int(value)
            candidate = value - ((value - 5) % 17)
            return candidate if candidate >= 5 else 0

        # The first sampler length is delivered in full. Every later sampler
        # receives a 39-frame visual head from the preceding source interval;
        # that head is trimmed from delivery. Build RAW lengths accordingly so
        # audio/video windows remain entirely inside the original source.
        lengths = []
        delivered_total = 0
        first = valid_at_or_below(min(target_length, total_frames))
        if first:
            lengths.append(first)
            delivered_total = first
        while total_frames - delivered_total > 39:
            wanted_delivery = min(requested_frames, total_frames - delivered_total)
            raw = valid_at_or_below(wanted_delivery + 39)
            if raw <= 39:
                break
            delivered = raw - 39
            if delivered_total + delivered > total_frames:
                break
            lengths.append(raw)
            delivered_total += delivered
        used_frames = delivered_total
        if not lengths or used_frames < 5:
            raise ValueError("The selected source range has no H3-valid block.")

        prompt = str(edit_prompt or "").strip()
        if not prompt:
            raise ValueError("Long edit received an empty director prompt.")
        shots = [
            {
                "id": f"source_block_{i:03d}",
                "length": int(length),
                "steps": 6,
                "seed": i - 1,
                "prompt": prompt,
            }
            for i, length in enumerate(lengths, 1)
        ]
        payload = {
            "summary": (
                f"Long direct edit of one continuous source take in "
                f"{len(shots)} blocks; identical edit contract in every block."
            ),
            "director_mode": "Continuous Story",
            # Keep continuity as a chain/latent concern.  H3 must receive the
            # director's edit contract verbatim in every block, without extra
            # plan prose being prepended by the chain normalizer.
            "prompt_prefix": "",
            "shots": shots,
        }
        status = (
            f"{total_frames} source frames / {total_seconds:.3f}s -> "
            f"{used_frames} delivered frames / {used_frames / 24.0:.3f}s in "
            f"{len(shots)} H3-valid contiguous blocks; raw sampler lengths: "
            + ", ".join(f"{x}f" for x in lengths)
        )
        return json.dumps(payload, ensure_ascii=False), len(shots), total_seconds, status


class SimpleH3LongControlEditPlanJSON:
    """Split a tracked source into independent ControlNet-guided blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_frames": ("IMAGE",),
            "edit_prompt": ("STRING", {"forceInput": True}),
            "block_seconds": (["5", "10", "15"], {"default": "5"}),
        }}

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "clip_count", "source_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"

    def build(self, source_frames, edit_prompt, block_seconds):
        if not torch.is_tensor(source_frames) or source_frames.ndim != 4:
            raise ValueError("Long ControlNet edit requires a decoded IMAGE frame batch.")
        total_frames = int(source_frames.shape[0])
        prompt = str(edit_prompt or "").strip()
        if total_frames < 24 or not prompt:
            raise ValueError("Long ControlNet edit requires source frames and a prompt.")
        nominal = int(block_seconds) * 24
        lengths = []
        remaining = total_frames
        while remaining:
            delivered = min(nominal, remaining)
            overlap = 0 if not lengths else 5
            needed = delivered + overlap
            raw = needed + (5 - needed % 17) % 17
            lengths.append((raw, delivered, overlap))
            remaining -= delivered
        shots = [{
            "id": f"control_block_{i:03d}", "length": int(raw),
            "delivered_frames": int(delivered),
            "audio_start_frame": int(sum(x[1] for x in lengths[:i - 1])),
            # A fixed seed reduces block-to-block environment restyling. The
            # weak cut-reference guide supplies the latest generated appearance.
            "steps": 6, "seed": 0, "prompt": prompt,
        } for i, (raw, delivered, overlap) in enumerate(lengths, 1)]
        payload = {
            "summary": f"Long ControlNet edit in {len(shots)} independent source blocks.",
            "director_mode": "Source Control Cuts", "prompt_prefix": "", "shots": shots,
        }
        status = (
            f"{total_frames} source frames / {total_frames / 24.0:.3f}s -> "
            f"{len(shots)} independent ControlNet block(s); no generated-frame carry"
        )
        return json.dumps(payload, ensure_ascii=False), len(shots), total_frames / 24.0, status


class SimpleH3ImageContinuationPlanJSON:
    """Build a repeated-prompt H3 chain without any source-video dependency."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_prompt": ("STRING", {"forceInput": True}),
                "total_seconds": ("INT", {
                    "default": 30, "min": 5, "max": 600, "step": 5,
                }),
                "block_seconds": (["5", "10", "15"], {"default": "10"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "clip_count", "output_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"

    def build(self, edit_prompt, total_seconds, block_seconds):
        prompt = str(edit_prompt or "").strip()
        if not prompt:
            raise ValueError("Image continuation received an empty director prompt.")

        total_frames = int(total_seconds) * 24
        requested_frames = int(block_seconds) * 24

        def valid_at_or_above(value):
            value = int(value)
            return value + (5 - value % 17) % 17

        def valid_at_or_below(value):
            value = int(value)
            candidate = value - ((value - 5) % 17)
            return candidate if candidate >= 5 else 0

        # Scene 1 is delivered in full. Later scenes carry a protected 39-frame
        # AV head from the preceding generated scene; that head is trimmed.
        lengths = []
        delivered_total = 0
        # Match the native H3 duration convention for the opening block: a
        # nominal 5/10/15-second request rounds up to 124/243/362 frames.
        first = valid_at_or_above(min(requested_frames, total_frames))
        if first:
            lengths.append(first)
            delivered_total = first
        while total_frames - delivered_total > 39:
            wanted_delivery = min(
                requested_frames, total_frames - delivered_total)
            raw = valid_at_or_below(wanted_delivery + 39)
            if raw <= 39:
                break
            delivered = raw - 39
            if delivered_total + delivered > total_frames:
                break
            lengths.append(raw)
            delivered_total += delivered

        if not lengths:
            raise ValueError("The requested duration has no H3-valid block.")

        shots = [
            {
                "id": f"image_continuation_{index:03d}",
                "length": int(length),
                "steps": 6,
                "seed": index - 1,
                "prompt": prompt,
            }
            for index, length in enumerate(lengths, 1)
        ]
        payload = {
            "summary": (
                f"Image-started continuous generation in {len(shots)} blocks; "
                "the same clean prompt is used verbatim in every block."
            ),
            "director_mode": "Continuous Story",
            "prompt_prefix": "",
            "shots": shots,
        }
        status = (
            f"image-only chain -> {delivered_total} delivered frames / "
            f"{delivered_total / 24.0:.3f}s in {len(shots)} blocks; "
            "no source video or source audio"
        )
        return (
            json.dumps(payload, ensure_ascii=False), len(shots),
            delivered_total / 24.0, status,
        )


class SimpleH3CompactContinuousPlanJSON:
    """Repeat one compact prompt for an exact number of generated continuations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "scene_count": ("INT", {
                    "forceInput": True, "min": 1, "max": 12,
                }),
                "seconds_per_scene": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Nominal seconds per scene, including decimals. H3 frame-grid alignment and continuation context can change the delivered duration.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "scene_count", "output_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"

    def build(self, prompt, scene_count, seconds_per_scene):
        prompt_payload = str(prompt or "").strip()
        if not prompt_payload:
            raise ValueError("Compact continuous mode received an empty prompt.")
        count = max(1, min(12, int(scene_count)))
        scene_prompts = None
        try:
            parsed = json.loads(prompt_payload)
            if isinstance(parsed, dict) and isinstance(
                parsed.get("scene_prompts"), list
            ):
                scene_prompts = [
                    str(item or "").strip() for item in parsed["scene_prompts"]
                ]
        except json.JSONDecodeError:
            pass
        if scene_prompts is None:
            # Backwards-compatible manual/direct use: one plain prompt repeats.
            scene_prompts = [prompt_payload] * count
        if len(scene_prompts) != count or any(not item for item in scene_prompts):
            raise ValueError(
                f"Compact continuous director returned {len(scene_prompts)} prompts; "
                f"scene_count requires exactly {count}. Disable Hold and regenerate."
            )
        seconds = float(seconds_per_scene)
        if not math.isfinite(seconds) or not 1.0 <= seconds <= 15.0:
            raise ValueError("Seconds per scene must be a finite number between 1 and 15.")
        requested_frames = math.ceil(seconds * 24)

        def valid_at_or_above(value):
            value = int(value)
            return value + (5 - value % 17) % 17

        def valid_at_or_below(value):
            value = int(value)
            candidate = value - ((value - 5) % 17)
            return candidate if candidate >= 5 else 0

        first_length = valid_at_or_above(requested_frames)
        continuation_length = valid_at_or_below(requested_frames + 39)
        if continuation_length <= 39:
            raise ValueError("The selected continuation duration is too short.")
        lengths = [first_length] + [continuation_length] * (count - 1)
        delivered = first_length + sum(length - 39 for length in lengths[1:])
        shots = [
            {
                "id": f"continuous_scene_{index:03d}",
                "length": int(length),
                "steps": 6,
                "seed": 0,
                "prompt": scene_prompts[index - 1],
            }
            for index, length in enumerate(lengths, 1)
        ]
        payload = {
            "summary": (
                f"{count} compact chronological continuous scene prompts."
            ),
            "director_mode": "Continuous Story",
            "prompt_prefix": "",
            "shots": shots,
        }
        status = (
            f"{count} continuous scene(s) × {seconds_per_scene}s nominal -> "
            f"{delivered} delivered frames / {delivered / 24.0:.3f}s; "
            "one compact clean prompt assigned to each scene"
        )
        return (
            json.dumps(payload, ensure_ascii=False), count,
            delivered / 24.0, status,
        )


class SimpleH3LongEditSourceSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": ("H3_CHAIN_STATE",),
                "source_frames": ("IMAGE",),
            },
            "optional": {"source_audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("block_frames", "block_audio", "length", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"

    def select(self, state, source_frames, source_audio=None):
        plan = state["plan"]
        index = int(state["index"])
        shot = plan["shots"][index - 1]
        raw = int(shot["raw_frames"])
        delivered_before = sum(
            int(item["delivered_frames"]) for item in plan["shots"][:index - 1]
        )
        overlap = 0 if index == 1 else 39
        start = max(0, delivered_before - overlap)
        end = start + raw
        available = int(source_frames.shape[0])
        selected = source_frames[start:min(end, available)]
        if int(selected.shape[0]) < raw:
            selected = torch.cat(
                [selected, selected[-1:].repeat(raw - int(selected.shape[0]), 1, 1, 1)],
                dim=0,
            )

        audio_out = source_audio
        if source_audio is not None:
            waveform = source_audio["waveform"]
            rate = int(source_audio["sample_rate"])
            a0 = int(round(start / 24.0 * rate))
            count = int(round(raw / 24.0 * rate))
            part = waveform[..., a0:min(a0 + count, int(waveform.shape[-1]))]
            if int(part.shape[-1]) < count:
                part = torch.nn.functional.pad(part, (0, count - int(part.shape[-1])))
            audio_out = {"waveform": part, "sample_rate": rate}
        status = (
            f"Block {index}/{len(plan['shots'])}: source [{start}:{end}) -> "
            f"{raw} H3 frames; protected head={overlap}."
        )
        return selected.contiguous(), audio_out, raw, status


class SimpleH3RecursiveLoaderWindow:
    """Calculate the exact VHS window required by the current H3 scene."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": ("H3_CHAIN_STATE",),
            "source_start_seconds": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
            }),
        }}

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frame_load_cap", "skip_first_frames", "force_rate", "status")
    FUNCTION = "calculate"
    CATEGORY = "MiniMax H3/Simple Chain"

    def calculate(self, state, source_start_seconds):
        plan = state["plan"]
        index = int(state["index"])
        shot = plan["shots"][index - 1]
        raw = int(shot["raw_frames"])
        delivered_before = sum(
            int(item["delivered_frames"]) for item in plan["shots"][:index - 1]
        )
        overlap = 0 if index == 1 else 39
        base = int(round(float(source_start_seconds) * 24.0))
        skip = base + max(0, delivered_before - overlap)
        status = (
            f"Scene {index}/{len(plan['shots'])}: VHS skip={skip}, cap={raw}, "
            f"24 fps; nominal scene duration comes from Stage 0."
        )
        return raw, skip, 24.0, status


class SimpleH3AudioWindow:
    """Select one continuous source soundtrack without decoding all video frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1}),
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 86400.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"

    def select(self, audio, start_seconds, duration_seconds):
        waveform = audio["waveform"]
        rate = int(audio["sample_rate"])
        start = max(0, int(round(float(start_seconds) * rate)))
        count = max(1, int(round(float(duration_seconds) * rate)))
        part = waveform[..., start:min(start + count, int(waveform.shape[-1]))]
        if int(part.shape[-1]) < count:
            part = F.pad(part, (0, count - int(part.shape[-1])))
        return ({"waveform": part.contiguous(), "sample_rate": rate},
                f"source audio {float(start_seconds):.3f}s.."
                f"{float(start_seconds) + float(duration_seconds):.3f}s")


def _fit_audio_to_h3_frames(audio, target_frames):
    if audio is None:
        return None
    waveform = audio.get("waveform")
    rate = int(audio.get("sample_rate", 0))
    if not torch.is_tensor(waveform) or rate <= 0:
        raise ValueError("Free Mode received an invalid AUDIO object.")
    samples = int(round(int(target_frames) / 24.0 * rate))
    waveform = waveform[..., :samples]
    if int(waveform.shape[-1]) < samples:
        waveform = F.pad(waveform, (0, samples - int(waveform.shape[-1])))
    return {"waveform": waveform.contiguous(), "sample_rate": rate}


class SimpleH3LongControlSourceSlice(SimpleH3LongEditSourceSlice):
    """Read each independent ControlNet block at its exact source position."""

    def select(self, state, source_frames, source_audio=None):
        plan = state["plan"]
        index = int(state["index"])
        shot = plan["shots"][index - 1]
        raw = int(shot["raw_frames"])
        delivered_start = sum(
            int(x["delivered_frames"]) for x in plan["shots"][:index - 1]
        )
        overlap = 0 if index == 1 else 5
        start = max(0, delivered_start - overlap)
        end = start + raw
        selected = source_frames[start:min(end, int(source_frames.shape[0]))]
        if int(selected.shape[0]) == 0:
            selected = source_frames[-1:]
        if int(selected.shape[0]) < raw:
            selected = torch.cat([
                selected, selected[-1:].repeat(raw - int(selected.shape[0]), 1, 1, 1)
            ], dim=0)
        audio_out = source_audio
        if source_audio is not None:
            waveform, rate = source_audio["waveform"], int(source_audio["sample_rate"])
            a0 = int(round(delivered_start / 24.0 * rate))
            delivered = int(shot["delivered_frames"])
            count = int(round(delivered / 24.0 * rate))
            part = waveform[..., a0:min(a0 + count, int(waveform.shape[-1]))]
            if int(part.shape[-1]) < count:
                part = F.pad(part, (0, count - int(part.shape[-1])))
            audio_out = {
                "waveform": part.contiguous(), "sample_rate": rate,
                "_simple_h3_source_delivered": True,
            }
        status = (
            f"Control block {index}/{len(plan['shots'])}: source [{start}:{end}); "
            f"generated-context head={overlap}."
        )
        return selected.contiguous(), audio_out, raw, status


class SimpleH3OptionalSourceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_frames": ("INT", {"forceInput": True}),
                "use_source_audio": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {"source_audio": ("AUDIO", {"lazy": True})},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("reference_audio", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def check_lazy_status(
        cls, target_frames, use_source_audio, source_audio=None,
    ):
        if bool(use_source_audio) and source_audio is None:
            return ["source_audio"]
        return []

    def select(self, target_frames, use_source_audio, source_audio=None):
        if not bool(use_source_audio):
            return None, "Source audio disabled; H3 will generate new audio."
        if source_audio is None:
            raise ValueError(
                "Use source audio is enabled, but no source video audio is connected."
            )
        fitted = _fit_audio_to_h3_frames(source_audio, target_frames)
        return fitted, f"Source audio enabled and fitted to {int(target_frames)} H3 frames."


class SimpleH3AudioOutputSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_audio": ("AUDIO",),
                "target_frames": ("INT", {"forceInput": True}),
                "use_source_audio": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {"source_audio": ("AUDIO", {"lazy": True})},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def check_lazy_status(
        cls, generated_audio, target_frames, use_source_audio,
        source_audio=None,
    ):
        if bool(use_source_audio) and source_audio is None:
            return ["source_audio"]
        return []

    def select(
        self, generated_audio, target_frames, use_source_audio,
        source_audio=None,
    ):
        if bool(use_source_audio):
            if source_audio is None:
                raise ValueError(
                    "Use source audio is enabled, but no source video audio is connected."
                )
            return (
                _fit_audio_to_h3_frames(source_audio, target_frames),
                "Output uses the time-fitted source-video audio.",
            )
        return generated_audio, "Output uses H3 generated audio."


class SimpleH3OptionalSourceVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_source_video": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
                "source_frames": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("reference_video", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def check_lazy_status(cls, use_source_video, source_frames=None):
        if bool(use_source_video) and source_frames is None:
            return ["source_frames"]
        return []

    def select(self, use_source_video, source_frames=None):
        if not bool(use_source_video):
            return None, "Image/reference-only mode; source video branch disabled."
        if source_frames is None:
            raise ValueError("Use source video is enabled, but no video is connected.")
        return source_frames, "Source video enabled as temporal/editing reference."


class SimpleH3FreeLatentSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_source_video": ("BOOLEAN", {"forceInput": True}),
                "reference_latent": ("LATENT", {"lazy": True}),
                "video_edit_latent": ("LATENT", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Simple Chain"

    @classmethod
    def check_lazy_status(
        cls, use_source_video, reference_latent=None, video_edit_latent=None,
    ):
        wanted = "video_edit_latent" if bool(use_source_video) else "reference_latent"
        value = video_edit_latent if bool(use_source_video) else reference_latent
        return [wanted] if value is None else []

    def select(self, use_source_video, reference_latent, video_edit_latent):
        if bool(use_source_video):
            return video_edit_latent, "Using encoded source-video AV latent."
        return reference_latent, "Using native empty Ref2VA latent for image-only generation."


NODE_CLASS_MAPPINGS = {
    "SimpleH3LongEditPlanJSON": SimpleH3LongEditPlanJSON,
    "SimpleH3LongControlEditPlanJSON": SimpleH3LongControlEditPlanJSON,
    "SimpleH3ImageContinuationPlanJSON": SimpleH3ImageContinuationPlanJSON,
    "SimpleH3CompactContinuousPlanJSON": SimpleH3CompactContinuousPlanJSON,
    "SimpleH3LongEditSourceSlice": SimpleH3LongEditSourceSlice,
    "SimpleH3RecursiveLoaderWindow": SimpleH3RecursiveLoaderWindow,
    "SimpleH3AudioWindow": SimpleH3AudioWindow,
    "SimpleH3LongControlSourceSlice": SimpleH3LongControlSourceSlice,
    "SimpleH3OptionalSourceAudio": SimpleH3OptionalSourceAudio,
    "SimpleH3AudioOutputSelect": SimpleH3AudioOutputSelect,
    "SimpleH3OptionalSourceVideo": SimpleH3OptionalSourceVideo,
    "SimpleH3FreeLatentSwitch": SimpleH3FreeLatentSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3LongEditPlanJSON": "Simple H3 Long Edit — Build Source Blocks",
    "SimpleH3LongControlEditPlanJSON": "Simple H3 Long Control Edit — Independent Blocks",
    "SimpleH3ImageContinuationPlanJSON": "Simple H3 Image Start — Repeated Prompt Blocks",
    "SimpleH3CompactContinuousPlanJSON": "Simple H3 Compact Continuous — Scene Count",
    "SimpleH3LongEditSourceSlice": "Simple H3 Long Edit — Current Source Block",
    "SimpleH3RecursiveLoaderWindow": "Simple H3 Recursive VHS Window — Current Scene",
    "SimpleH3AudioWindow": "Simple H3 Source Audio — Start / Duration",
    "SimpleH3LongControlSourceSlice": "Simple H3 Long Control Edit — Current Source Block",
    "SimpleH3OptionalSourceAudio": "Simple H3 Free Mode — Optional Source Audio",
    "SimpleH3AudioOutputSelect": "Simple H3 Free Mode — Output Audio",
    "SimpleH3OptionalSourceVideo": "Simple H3 Free Mode — Optional Source Video",
    "SimpleH3FreeLatentSwitch": "Simple H3 Free Mode — Latent Route",
}
