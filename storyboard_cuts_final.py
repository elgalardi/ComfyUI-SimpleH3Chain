import os
import shutil
import uuid

import torch
import torch.nn.functional as F


def build_storyboard_cuts_node(api):
    """Build the node after SimpleH3Chain's shared helpers are initialized."""

    class SimpleH3StoryboardCutsUpscaleDecodeAssemble(api["upscale_base"]):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "manifest": (api["manifest_type"],),
                    "video_vae": ("VAE",),
                    "audio_vae": ("VAE",),
                    "latent_upscale": ("BOOLEAN", {"default": True}),
                    "upscaler_model": (cls._models(),),
                    "final_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                    "final_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                    "decode_window_frames": ([90, 141, 192, 243, 294, 345, 396], {"default": 90}),
                    "filename": ("STRING", {"default": "%date:yyyy-MM-dd%_storyboard_cuts"}),
                    "save_output": ("BOOLEAN", {"default": True}),
                    "scene_previews": ("BOOLEAN", {"default": False}),
                },
                "hidden": {"unique_id": "UNIQUE_ID"},
            }

        RETURN_TYPES = ("STRING", "STRING")
        RETURN_NAMES = ("video_path", "status")
        FUNCTION = "process_cuts"
        OUTPUT_NODE = True
        CATEGORY = "MiniMax H3/Simple Chain/Upscale"
        DESCRIPTION = (
            "Final path for independent storyboard cuts. It loads, 3D-upscales, "
            "decodes and releases one complete scene at a time, then concatenates "
            "the scene MP4s. No temporal operation crosses a cut."
        )

        @classmethod
        def IS_CHANGED(cls, *args, **kwargs):
            return float("NaN")

        def process_cuts(
            self, manifest, video_vae, audio_vae, latent_upscale, upscaler_model,
            final_width, final_height, decode_window_frames, filename, save_output,
            scene_previews=False, unique_id=None,
        ):
            chain = api["chain"]
            if not isinstance(manifest, dict) or manifest.get("format") != "h3_chain_manifest_v3":
                raise ValueError("Storyboard Cuts final requires a completed Simple H3 manifest.")
            segments = list(manifest.get("segments") or [])
            if not segments:
                raise ValueError("Storyboard Cuts final received an empty manifest.")
            fingerprint = str((manifest.get("compatibility") or {}).get("generation_fingerprint") or "")
            if "type=masked_av" in fingerprint:
                raise ValueError(
                    "This node is exclusively for independent storyboard cuts. "
                    "Use the continuity final nodes for Masked AV chains."
                )
            try:
                from safetensors.torch import load_file as load_safetensors
                from comfy_extras.nodes_audio import vae_decode_audio
                if bool(latent_upscale):
                    from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
                        MinimaxH3LatentUpscaler3D, UpscaleMode,
                    )
            except Exception as error:
                raise RuntimeError("Storyboard Cuts final dependencies are unavailable.") from error
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("Storyboard Cuts final requires ffmpeg.")

            run_name = api["safe_run_name"](manifest.get("run_name", "h3_chain"))
            run_dir = chain._absolute_output_path(run_name)
            work_dir = os.path.join(run_dir, "previews", "refined_scenes")
            final_dir = os.path.join(run_dir, "final")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            transaction = uuid.uuid4().hex
            final_name = chain._safe_name(api["expand_date_tokens"](str(filename)), "storyboard_cuts_final")
            final_path = chain._versioned_path(os.path.join(final_dir, final_name + ".mp4"), transaction)
            final_tmp = os.path.join(final_dir, f".{transaction}.tmp.mp4")
            concat_path = os.path.join(work_dir, f".{transaction}.concat.txt")
            clips, scratch = [], []
            total_frames = 0
            window_tokens = api["tokens_for_frames"](int(decode_window_frames))
            overlap_frames = 39
            overlap_tokens = api["tokens_for_frames"](overlap_frames)
            stride_tokens = window_tokens - overlap_tokens
            if stride_tokens <= 0:
                raise ValueError("Scene decode window must be longer than 39 frames.")
            try:
                for position, segment in enumerate(segments):
                    index = int(segment.get("index", position + 1))
                    chain._verify_segment_artifacts(segment, index)
                    tensors = load_safetensors(
                        chain._absolute_output_path(segment["checkpoint"]), device="cpu"
                    )
                    video = tensors["video"].detach().cpu().contiguous()
                    audio = tensors["audio"].detach().cpu().contiguous()
                    if video.ndim == 4:
                        video = video.unsqueeze(0)
                    if audio.ndim == 3:
                        audio = audio.unsqueeze(0)
                    delivered = int(segment.get("delivered_frames", 0))
                    if delivered <= 0:
                        raise ValueError(f"Scene {index} has no delivered frame count.")
                    if bool(latent_upscale):
                        final_video = MinimaxH3LatentUpscaler3D.execute(
                            latent={"samples": video}, model_name=str(upscaler_model),
                            mode={"mode": UpscaleMode.TARGET_DIMENSIONS,
                                  "width": int(final_width), "height": int(final_height)},
                            align=32, enable_temporal_chunking=True, force_unload=True,
                            device="cuda", precision="fp16",
                        )[0]["samples"]
                    else:
                        final_video = video
                    decoded_parts = []
                    emitted = 0
                    start_token = 0
                    window_index = 0
                    while start_token < int(final_video.shape[2]) and emitted < delivered:
                        end_token = min(int(final_video.shape[2]), start_token + window_tokens)
                        window = video_vae.decode(
                            final_video[:, :, start_token:end_token].contiguous()
                        )
                        if window.ndim == 5:
                            window = window.reshape(-1, *window.shape[-3:])
                        head = 0 if window_index == 0 else overlap_frames
                        wanted = min(int(window.shape[0]) - head, delivered - emitted)
                        if wanted <= 0:
                            raise ValueError(
                                f"Scene {index} decode window {window_index + 1} produced no new frames."
                            )
                        decoded_parts.append(window[head:head + wanted].cpu().contiguous())
                        emitted += wanted
                        window_index += 1
                        if emitted >= delivered or end_token >= int(final_video.shape[2]):
                            break
                        start_token += stride_tokens
                    if emitted != delivered:
                        raise ValueError(
                            f"Scene {index} decoded {emitted} frames; expected {delivered}."
                        )
                    images = torch.cat(decoded_parts, dim=0)
                    decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})
                    waveform = decoded_audio["waveform"]
                    sample_rate = int(decoded_audio["sample_rate"])
                    samples = int(round(delivered / 24.0 * sample_rate))
                    waveform = (F.pad(waveform, (0, samples - int(waveform.shape[-1])))
                                if int(waveform.shape[-1]) < samples else waveform[..., :samples].contiguous())
                    raw_video = os.path.join(work_dir, f".scene_{index:04d}.{transaction}.video.mp4")
                    raw_audio = os.path.join(work_dir, f".scene_{index:04d}.{transaction}.wav")
                    clip_path = os.path.join(work_dir, f"scene_{index:04d}.{transaction}.mp4")
                    scratch.extend((raw_video, raw_audio))
                    chain._write_segment_video(images, raw_video, 24, 20)
                    chain._write_wav({"waveform": waveform, "sample_rate": sample_rate}, raw_audio)
                    chain._run_ffmpeg([
                        ffmpeg, "-y", "-i", raw_video, "-i", raw_audio,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-t", f"{delivered / 24.0:.9f}",
                        "-movflags", "+faststart", clip_path,
                    ])
                    clips.append(clip_path)
                    total_frames += delivered
                    if bool(scene_previews):
                        api["publish_preview"](
                            unique_id, transaction, index, len(segments), clip_path, True
                        )
                    chain._LOG.info(
                        "Simple H3 cut %d/%d completed: %d frames; 3D upscale %s.",
                        index, len(segments), delivered, "on" if latent_upscale else "off",
                    )
                    del tensors, video, audio, final_video, images, decoded_parts, waveform

                with open(concat_path, "w", encoding="utf-8") as handle:
                    for path in clips:
                        escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
                        handle.write(f"file '{escaped}'\n")
                chain._run_ffmpeg([
                    ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                    "-c", "copy", "-movflags", "use_metadata_tags+faststart", final_tmp,
                ])
                os.replace(final_tmp, final_path)
                status = (
                    f"Processed {len(segments)} storyboard scenes independently; "
                    f"assembled {total_frames} frames; 3D upscale "
                    f"{'on' if latent_upscale else 'off'}; no cross-scene windows "
                    f"and no diffusion refine -> {final_path}"
                )
                if not bool(save_output):
                    temporary = os.path.join(
                        work_dir, f"temporary_storyboard_final.{uuid.uuid4().hex}.mp4"
                    )
                    os.replace(final_path, temporary)
                    final_path = temporary
                    status += "; temporary final only"
                return {
                    "ui": {"videos": [chain._video_output_item(final_path)], "text": [status]},
                    "result": (final_path, status),
                }
            finally:
                for path in scratch + [concat_path, final_tmp]:
                    chain._safe_unlink(path)
                if not bool(scene_previews):
                    for path in clips:
                        chain._safe_unlink(path)

    return SimpleH3StoryboardCutsUpscaleDecodeAssemble
