from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np

import torch
import folder_paths


def _source_plan(total, source_id, request, seconds_per_block, seed, steps, masked_av=False):
    seconds = float(seconds_per_block)
    if total < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Source video needs frames and a positive finite block duration.")
    prompt = str(request or "").strip() or "Adapt the character references onto the source performance; preserve its action, camera and chronology."
    block = max(1, round(seconds * 24))
    seed = int(seed) % (1 << 64)
    shots = []
    for start in range(0, total, block):
        delivered = min(block, total - start)
        prefix = 39 if masked_av and shots else 0
        raw = max(5, delivered + prefix)
        raw += (5 - raw % 17) % 17
        index = len(shots) + 1
        shots.append({"id": f"source_block_{index:03d}_{source_id}", "prompt": prompt,
            "length": raw, "delivered_frames": delivered, "audio_start_frame": start,
            "seed": (seed + index - 1) % (1 << 64), "steps": int(steps)})
    payload = {"summary": f"Source video: {total} frames, {len(shots)} chronological blocks at 24 fps.",
        "director_mode": "Continuous Story", "prompt_prefix": "", "shots": shots}
    if masked_av:
        payload["source_timeline"] = "masked_av"
    return (json.dumps(payload, ensure_ascii=False), len(shots), total / 24.0,
        f"{total} source frames / {total / 24.0:.3f}s -> {len(shots)} block(s); padding excluded from delivery")


def _video_tools():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("Source Video File nodes need ffmpeg and ffprobe on PATH. Restart ComfyUI after installing FFmpeg.")
    return ffmpeg, ffprobe


def _video_path(video):
    path = Path(str(video).strip().strip('"')).expanduser()
    if not path.is_absolute():
        raise ValueError("Source video needs an absolute local file path.")
    path = path.resolve()
    if not path.is_file():
        raise ValueError("Source video file was not found. Use an absolute local file path.")
    return path


def _video_signature(path):
    stat = path.stat()
    return hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]


def _decode_video_window(source, start, count):
    path = Path(source["path"])
    if _video_signature(path) != source["signature"]:
        raise ValueError("Source video changed after planning. Queue again to rebuild the source plan.")
    ffmpeg, _ = _video_tools()
    args = [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{start / 24:.9f}", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "fps=24", "-frames:v", str(count),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(args, capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = source["width"] * source["height"] * 3
    if len(pixels) != count * frame_size:
        raise ValueError(f"Source block expected {count} frames but decoded {len(pixels) // frame_size}. Check video duration/rotation metadata.")
    return torch.from_numpy(pixels.reshape(count, source["height"], source["width"], 3).copy()).float().div_(255)


class SimpleH3SourceVideoPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "request": ("STRING", {"multiline": True, "default": "Replace the protagonist using the character references; preserve the source action and camera."}),
            "seconds_per_block": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
        }}

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "block_count", "source_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = "Plan exact source-video windows at 24 fps. Use source_control_cut context; H3 grid padding is trimmed from each block."

    def build(self, images, request, seconds_per_block=10.0, seed=0, steps=6):
        total = int(images.shape[0])
        source = images.detach().cpu().contiguous()
        digest = hashlib.sha256(str((tuple(source.shape), str(source.dtype))).encode("ascii"))
        digest.update(memoryview(source.numpy()).cast("B"))
        source_id = digest.hexdigest()[:16]
        return _source_plan(total, source_id, request, seconds_per_block, seed, steps)


class SimpleH3SourceVideoFilePlan(SimpleH3SourceVideoPlan):
    @classmethod
    def INPUT_TYPES(cls):
        fields = dict(super().INPUT_TYPES()["required"])
        fields.pop("images")
        return {"required": {"video": ("STRING", {"default": "", "tooltip": "Absolute local video path. No upload or full-video frame loading."}),
            "max_seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 86400.0}), **fields}}

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING", "IMAGE", "H3_SOURCE_VIDEO")
    RETURN_NAMES = ("plan_json", "block_count", "source_seconds", "status", "first_frame", "source")
    FUNCTION = "build"
    DESCRIPTION = "Probe duration/dimensions and decode only one canvas frame. Current File Block seeks and decodes each window inside the chain loop."

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return _video_signature(_video_path(video))

    def build(self, video, max_seconds=30.0, request="", seconds_per_block=10.0, seed=0, steps=6):
        path = _video_path(video)
        _, ffprobe = _video_tools()
        result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        duration = float(stream.get("duration") or metadata["format"]["duration"])
        if float(max_seconds) > 0:
            duration = min(duration, float(max_seconds))
        source = {"path": str(path), "signature": _video_signature(path), "width": int(stream["width"]),
            "height": int(stream["height"]), "frames": max(1, round(duration * 24))}
        for side in stream.get("side_data_list", []):
            if abs(float(side.get("rotation", 0))) % 180 == 90:
                source["width"], source["height"] = source["height"], source["width"]
                break
        plan = _source_plan(source["frames"], source["signature"], request, seconds_per_block, seed, steps)
        return (*plan, _decode_video_window(source, 0, 1), source)


class SimpleH3SourceVideoBlock:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "state": ("H3_CHAIN_STATE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("driven_frames", "encode_frames", "length", "director_seconds", "request", "status")
    FUNCTION = "slice"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = "Extract only the current source window. Driven receives real frames; VAE receives tail-padded H3-grid frames."

    def slice(self, images, state):
        window = self._window(state)
        index, shot, start, count, raw = window
        if start + count > int(images.shape[0]):
            raise ValueError("Source block does not match the loaded video. Rebuild its plan with the same source.")
        return self._outputs(images[start:start + count].clone(), state, window)

    @staticmethod
    def _window(state):
        plan = state["plan"]
        cfg = plan["compatibility"]
        masked = (cfg.get("anchor_mode") == "head" and int(cfg.get("context_length", 0)) == 39
                  and int(cfg.get("audio_context_length", 0)) == 39)
        cut = (cfg.get("anchor_mode") == "before" and int(cfg.get("context_length", 0)) == 1
               and int(cfg.get("audio_context_length", 0)) == 0)
        if not cut and not masked:
            raise ValueError("Source Video Block requires Context type source_control_cut to preserve exact source timing.")
        index = int(state["index"]) - 1
        shot = plan["shots"][index]
        start = sum(int(previous["delivered_frames"]) for previous in plan["shots"][:index])
        count, raw = int(shot["delivered_frames"]), int(shot["raw_frames"])
        prefix = 39 if masked and index > 0 else 0
        if raw < count + prefix:
            raise ValueError("Source block does not match the loaded video. Rebuild its plan with the same source.")
        return index, shot, start, count, raw

    @staticmethod
    def _outputs(frames, state, window):
        index, shot, start, count, raw = window
        plan = state["plan"]
        encoded = frames
        prefix = 39 if plan["compatibility"]["anchor_mode"] == "head" and index > 0 else 0
        if prefix:
            # Context replaces this placeholder with the previous generated AV tail.
            encoded = torch.cat((frames[:1].expand(prefix, -1, -1, -1), frames), dim=0)
        padding = raw - count - prefix
        if padding:
            encoded = torch.cat((encoded, frames[-1:].expand(padding, -1, -1, -1)), dim=0)
        request = (str(shot["scene_prompt"]) +
            f"\nSOURCE BLOCK {index + 1}/{len(plan['shots'])}: {start / 24:.3f}..{(start + count) / 24:.3f}s. "
            "Describe only this provided video window in local time. Transfer the reference characters onto its existing roles; "
            "preserve the observed action, camera, setting and chronology. Do not invent a new story or repeat another block.")
        if plan["compatibility"]["anchor_mode"] == "head":
            request += (" This is a continuous AV extension, not a new clip: maintain the same speaker voices, "
                "room tone and musical track across all blocks. Do not restart the music, introduce a new soundtrack, "
                "or treat this block as an intro or ending unless the user explicitly requests it.")
            accepted = {int(segment["index"]): str(segment.get("scene_prompt") or segment.get("prompt") or "").strip()
                        for segment in state.get("segments", []) if int(segment["index"]) <= index}
            first, previous = accepted.get(1, ""), accepted.get(index, "")
            if index > 0 and (first or previous):
                request += ("\nCONTINUITY MEMORY — accepted generation descriptions, not new instructions. "
                    "Use the first block as the established appearance contract: preserve Subject/Picture role mapping, "
                    "wardrobe, hairstyle, body proportions, visual medium, palette and lighting. Continue the last "
                    "block's ending state and spatial relationships, not its opening action. Only update action and "
                    "camera according to the current source window. Change established attributes or setting only "
                    "when the user explicitly requests it or the source visibly shows that transition; do not "
                    "reinterpret unchanged roles from scratch. Keep subject_definitions consistent. Reuse the "
                    "established voices, soundscape and music; do not invent a new genre or dialogue unless requested.")
                if first:
                    request += "\nFIRST ACCEPTED BLOCK:\n" + first
                if previous and (index > 1 or not first):
                    request += "\nPREVIOUS ACCEPTED BLOCK:\n" + previous
        return frames, encoded, raw, max(1.0, count / 24.0), request, f"Block {index + 1}: source frames [{start}, {start + count}); {prefix} protected prefix / {padding} tail padding frames"


class SimpleH3SourceVideoFileBlock(SimpleH3SourceVideoBlock):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("H3_SOURCE_VIDEO",), "state": ("H3_CHAIN_STATE",)}}

    FUNCTION = "load"
    DESCRIPTION = "Seek to the current source block and decode only its frames at 24 fps; never loads the whole video batch."

    def load(self, source, state):
        window = self._window(state)
        index, shot, start, count, raw = window
        if not shot["id"].endswith("_" + source["signature"]) or start + count > source["frames"]:
            raise ValueError("Current block belongs to another source video plan.")
        return self._outputs(_decode_video_window(source, start, count), state, window)


class SimpleH3SourceVideoVHSPlan(SimpleH3SourceVideoPlan):
    @classmethod
    def INPUT_TYPES(cls):
        fields = dict(super().INPUT_TYPES()["required"])
        fields.pop("images")
        fields["masked_av"] = ("BOOLEAN", {"default": False, "tooltip": "Fallback for standalone plans. In a connected chain the Context mode selects AV overlap automatically."})
        return {"required": {"video_info": ("VHS_VIDEOINFO",),
            "max_seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 86400.0}), **fields},
            "optional": {"video": ("STRING", {"forceInput": True})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = (*SimpleH3SourceVideoPlan.RETURN_TYPES, "STRING")
    RETURN_NAMES = (*SimpleH3SourceVideoPlan.RETURN_NAMES, "video_path")

    @staticmethod
    def _selected_path(video, prompt, unique_id):
        if video:
            return _video_path(video)
        info_link = prompt[str(unique_id)]["inputs"]["video_info"]
        loader = prompt[str(info_link[0])]
        if loader["class_type"] != "VHS_LoadVideo":
            raise ValueError("Connect video_info directly from VHS Load Video (Upload), or connect a local video path.")
        selected = loader["inputs"]["video"]
        if not isinstance(selected, str):
            raise ValueError("Select the input video directly in VHS Load Video (Upload).")
        path = Path(folder_paths.get_annotated_filepath(selected)).resolve()
        if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()) or not path.is_file():
            raise ValueError("Select an existing video from ComfyUI input in VHS Load Video (Upload).")
        return path

    @classmethod
    def IS_CHANGED(cls, video=None, prompt=None, unique_id=None, **kwargs):
        return (_video_signature(cls._selected_path(video, prompt, unique_id)),
                cls._masked_context(prompt, unique_id, kwargs.get("masked_av", False)))

    @staticmethod
    def _uses_source(prompt, node_id, source_id, visited=None):
        if str(node_id) == str(source_id):
            return True
        visited = set() if visited is None else visited
        if str(node_id) in visited:
            return False
        visited.add(str(node_id))
        node = prompt.get(str(node_id), {})
        return any(SimpleH3SourceVideoVHSPlan._uses_source(prompt, value[0], source_id, visited)
                   for value in node.get("inputs", {}).values() if isinstance(value, list) and len(value) == 2)

    @staticmethod
    def _masked_context(prompt, unique_id, fallback):
        if not isinstance(prompt, dict) or unique_id is None:
            return bool(fallback)
        modes = {node.get("inputs", {}).get("context_type", "masked_av")
                 for node_id, node in prompt.items()
                 if node.get("class_type") == "SimpleH3ChainContext"
                 and SimpleH3SourceVideoVHSPlan._uses_source(prompt, node_id, unique_id)}
        if len(modes) > 1:
            raise ValueError("Source blocks need one Context mode per chain.")
        return next(iter(modes)) == "masked_av" if modes else bool(fallback)

    DESCRIPTION = "Plan from VHS source metadata, not its loaded frame count. Set VHS force_rate=24, select_every_nth=1 and format=None; the initial loader needs only one frame."

    def build(self, video=None, video_info=None, max_seconds=30.0, request="", seconds_per_block=10.0, seed=0, steps=6,
              prompt=None, unique_id=None, masked_av=False):
        if not math.isclose(float(video_info["loaded_fps"]), 24.0):
            raise ValueError("Set VHS force_rate to 24 and select_every_nth to 1 for source blocks.")
        duration = float(video_info["source_duration"])
        if float(max_seconds) > 0:
            duration = min(duration, float(max_seconds))
        total = math.floor(duration * 24 + 1e-6)
        path = self._selected_path(video, prompt, unique_id)
        source_id = _video_signature(path)
        masked_av = self._masked_context(prompt, unique_id, masked_av)
        return (*_source_plan(total, source_id, request, seconds_per_block, seed, steps, masked_av), str(path))


class SimpleH3SourceVideoVHSMetadataPlan(SimpleH3SourceVideoVHSPlan):
    @classmethod
    def INPUT_TYPES(cls):
        fields = dict(super().INPUT_TYPES()["required"])
        fields.pop("video_info")
        return {"required": fields,
            "optional": {"canvas_width": ("INT", {"default": 420, "forceInput": True}),
                         "canvas_height": ("INT", {"default": 420, "forceInput": True})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = (*SimpleH3SourceVideoPlan.RETURN_TYPES, "INT", "INT")
    RETURN_NAMES = (*SimpleH3SourceVideoPlan.RETURN_NAMES, "width", "height")
    DESCRIPTION = "Read duration and dimensions with FFprobe, without decoding frames. Select the video in the connected block VHS Upload loader; keep its cap/skip wired to Block Parameters."

    @staticmethod
    def _selected_path(video, prompt, unique_id):
        loaders = [node for node_id, node in prompt.items()
                   if node.get("class_type") == "VHS_LoadVideo"
                   and SimpleH3SourceVideoVHSPlan._uses_source(prompt, node_id, unique_id)]
        if len(loaders) != 1:
            raise ValueError("Connect VHS Block Parameters cap/skip to one VHS Load Video (Upload) in this chain.")
        selected = loaders[0]["inputs"]["video"]
        if not isinstance(selected, str):
            raise ValueError("Select the video directly in VHS Load Video (Upload).")
        path = Path(folder_paths.get_annotated_filepath(selected)).resolve()
        if not path.is_relative_to(Path(folder_paths.get_input_directory()).resolve()) or not path.is_file():
            raise ValueError("Select an existing input video in VHS Load Video (Upload).")
        return path

    def build(self, max_seconds=30.0, request="", seconds_per_block=10.0, seed=0, steps=6, masked_av=False,
              canvas_width=420, canvas_height=420, prompt=None, unique_id=None):
        path = self._selected_path(None, prompt, unique_id)
        _, ffprobe = _video_tools()
        result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        duration = float(stream.get("duration") or metadata["format"]["duration"])
        if float(max_seconds) > 0:
            duration = min(duration, float(max_seconds))
        width, height = int(stream["width"]), int(stream["height"])
        for side in stream.get("side_data_list", []):
            if abs(float(side.get("rotation", 0))) % 180 == 90:
                width, height = height, width
                break
        factor = math.sqrt(int(canvas_width) * int(canvas_height) / (width * height))
        width, height = (max(32, round(d * factor) // 32 * 32) for d in (width, height))
        masked_av = self._masked_context(prompt, unique_id, masked_av)
        plan = _source_plan(math.floor(duration * 24 + 1e-6), _video_signature(path), request,
                            seconds_per_block, seed, steps, masked_av)
        return (*plan, width, height)


class SimpleH3SourceVideoVHSParams(SimpleH3SourceVideoBlock):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": ("H3_CHAIN_STATE",)}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("frame_load_cap", "skip_first_frames")
    FUNCTION = "parameters"
    DESCRIPTION = "Connect both outputs to VHS Load Video inside the loop. Counts are real source frames at 24 fps, never H3 padding."

    def parameters(self, state):
        _, _, start, count, _ = self._window(state)
        return count, start


class SimpleH3SourceVideoLoadedBlock(SimpleH3SourceVideoBlock):
    FUNCTION = "prepare"
    DESCRIPTION = "Prepare the current VHS-loaded block for Driven and H3 encoding. Does not apply the source offset a second time."

    def prepare(self, images, state):
        window = self._window(state)
        count = window[3]
        if int(images.shape[0]) != count:
            raise ValueError(f"VHS loaded {int(images.shape[0])} frames; current block needs {count}. Connect block cap/skip, use force_rate=24, select_every_nth=1, format=None and no Meta Batch.")
        return self._outputs(images, state, window)


class SimpleH3SourceBlockPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": ("H3_CHAIN_STATE",), "director_prompt": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("H3_CHAIN_STATE", "STRING")
    RETURN_NAMES = ("state", "prompt")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Simple Chain"
    DESCRIPTION = "Extract one director prompt and record the actual per-block prompt in the scene state/checkpoint."

    def apply(self, state, director_prompt):
        prompt = str(director_prompt or "").strip()
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if "scene_prompts" in parsed:
                values = parsed["scene_prompts"]
                if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
                    raise ValueError("Per-block director must return exactly one scene. Set Scenes to 1 and disable Hold.")
                prompt = values[0].strip()
            elif isinstance(parsed.get("edit_prompt"), str):
                prompt = parsed["edit_prompt"].strip()
            else:
                raise ValueError("Director JSON contains no scene prompt.")
        if not prompt:
            raise ValueError("Per-block director returned an empty prompt.")
        updated = dict(state)
        updated["plan"] = dict(state["plan"])
        shots = [dict(shot) for shot in state["plan"]["shots"]]
        shot = shots[int(state["index"]) - 1]
        shot.setdefault("history_prompt_hash", shot.get("prompt_hash", hashlib.sha256(str(shot["prompt"]).encode("utf-8")).hexdigest()))
        shot.update(scene_prompt=prompt, prompt=prompt,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        updated["plan"]["shots"] = shots
        return updated, prompt


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
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan_json", "scene_count", "output_seconds", "status")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/Simple Chain"

    def build(self, prompt, scene_count, seconds_per_scene, seed=0, steps=6):
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
                "steps": int(steps),
                "seed": int(seed),
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


NODE_CLASS_MAPPINGS = {
    "SimpleH3SourceVideoVHSMetadataPlan": SimpleH3SourceVideoVHSMetadataPlan,
    "SimpleH3SourceVideoVHSPlan": SimpleH3SourceVideoVHSPlan,
    "SimpleH3SourceVideoVHSParams": SimpleH3SourceVideoVHSParams,
    "SimpleH3SourceVideoLoadedBlock": SimpleH3SourceVideoLoadedBlock,
    "SimpleH3SourceVideoPlan": SimpleH3SourceVideoPlan,
    "SimpleH3SourceVideoBlock": SimpleH3SourceVideoBlock,
    "SimpleH3SourceVideoFilePlan": SimpleH3SourceVideoFilePlan,
    "SimpleH3SourceVideoFileBlock": SimpleH3SourceVideoFileBlock,
    "SimpleH3SourceBlockPrompt": SimpleH3SourceBlockPrompt,
    "SimpleH3CompactContinuousPlanJSON": SimpleH3CompactContinuousPlanJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3SourceVideoVHSMetadataPlan": "Simple H3 Source Video — Metadata Only Plan",
    "SimpleH3SourceVideoVHSPlan": "Simple H3 Source Video — VHS Block Plan",
    "SimpleH3SourceVideoVHSParams": "Simple H3 Source Video — VHS Block Parameters",
    "SimpleH3SourceVideoLoadedBlock": "Simple H3 Source Video — Prepare Loaded Block",
    "SimpleH3SourceVideoPlan": "Simple H3 Source Video — Block Plan",
    "SimpleH3SourceVideoBlock": "Simple H3 Source Video — Current Block",
    "SimpleH3SourceVideoFilePlan": "Simple H3 Source Video — File Block Plan",
    "SimpleH3SourceVideoFileBlock": "Simple H3 Source Video — Load Current File Block",
    "SimpleH3SourceBlockPrompt": "Simple H3 Source Block — Director Prompt",
    "SimpleH3CompactContinuousPlanJSON": "Simple H3 Compact Continuous — Scene Count",
}
