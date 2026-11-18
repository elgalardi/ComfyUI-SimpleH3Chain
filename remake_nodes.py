"""Explicit canvas and extended duration for Viggle Remake conditioning."""

import logging

import torch

import comfy.utils
import comfy.nested_tensor
import nodes
import node_helpers


class SimpleH3SourceDetailFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "count": ([1, 3, 5, 10], {"default": 3})},
                "optional": {"bypass": ("BOOLEAN", {"default": False, "tooltip": "Send no detail frames or frame indices."})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("detail_frames", "frame_indices")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Remake"
    DESCRIPTION = "Selects evenly spaced frames from the original clip, before reducing its resolution for motion guidance."

    def select(self, images, count, bypass=False):
        if bypass:
            return None, None
        total = len(images)
        if total == 0:
            raise ValueError("Connect a non-empty video frame batch.")
        count = min(int(count), total)
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
        return images[indices], ", ".join(map(str, indices))


def _resize_reference(image, scale):
    height, width = image.shape[1:3]
    size = [max(32, round(axis * scale / 32) * 32) for axis in (width, height)]
    return comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), *size, "lanczos", "disabled").movedim(1, -1)


class _RemakeDetailClip:
    """Present detail frames to Qwen as the visual samples of Video 1."""

    def __init__(self, clip, video_item, remake_prompt=None):
        self.clip = clip
        self.video_item = video_item
        self.remake_prompt = remake_prompt

    def tokenize(self, text, **kwargs):
        if self.remake_prompt is None:
            items = kwargs.pop("minimax_ref_items", None) or []
            kwargs["minimax_ref_items"] = [self.video_item, *items]
        return self.clip.tokenize(text, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        if self.remake_prompt is not None:
            tokens = self.clip.tokenize(self.remake_prompt, minimax_ref_items=[self.video_item])
        return self.clip.encode_from_tokens_scheduled(tokens)


class SimpleH3RemakeConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True}),
                "frames": ("INT", {"default": 124, "min": 5, "max": 2147483647,
                                   "step": 1, "tooltip": "Requested source frames. Limited by the loaded clip, then padded to 17n+5. Long clips are experimental."}),
                "width": ("INT", {"default": 832, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
            },
            "optional": {
                "source": ("IMAGE", {"tooltip": "Source frames at 24 fps. Disconnect to use native picture references and an empty AV latent with Add Guide."}),
                "ref_1": ("IMAGE",),
                "source_detail_frames": ("IMAGE", {"tooltip": "Up to 10 sharp samples of <Video 1>. Picture references are separate."}),
                "source_detail_frame_indices": ("STRING", {"forceInput": True, "tooltip": "Connect frame_indices from Simple H3 Source Detail Frames to label the samples with their original times at 24 fps. Does not pin output frames."}),
                "audio": ("AUDIO",),
                "ref_2": ("IMAGE",),
                "ref_3": ("IMAGE",),
                "ref_4": ("IMAGE",),
                "source_detail_max_edge": ([720, 768, 960, 1024, 1536, 2048], {"default": 1024, "tooltip": "Maximum long edge of source detail frames, independent of output resolution. Dimensions are aligned to multiples of 32. Larger references use more memory."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "latent", "frames", "source_frames", "status")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Remake"
    DESCRIPTION = "Uses Remake with a source clip, or native H3 picture conditioning and an empty AV latent for Add Guide when source is disconnected."

    def run(self, clip, vae, audio_vae, prompt, frames, width, height,
            audio=None, ref_1=None, ref_2=None, ref_3=None, ref_4=None, source=None,
            source_detail_frames=None, source_detail_frame_indices=None, source_detail_max_edge=1024):
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError("Width and height must be positive multiples of 32.")
        if frames < 5:
            raise ValueError("Request at least 5 frames.")
        details = None
        if source_detail_frames is not None:
            if source_detail_frames.ndim != 4 or not 1 <= len(source_detail_frames) <= 10:
                raise ValueError("Connect a batch of 1 to 10 source detail frames.")
            detail_scale = min(1.0, int(source_detail_max_edge) / max(source_detail_frames.shape[1:3]))
            details = _resize_reference(source_detail_frames, detail_scale)

        regular_refs = [ref for ref in (ref_1, ref_2, ref_3, ref_4) if ref is not None]
        detail_status = ""
        if details is not None:
            video_item = {"type": "video", "data": details}
            if source_detail_frame_indices is not None:
                try:
                    indices = [int(value) for value in source_detail_frame_indices.replace(",", " ").split()]
                except ValueError as error:
                    raise ValueError("Source detail frame indices must be integers.") from error
                if len(indices) != len(details) or any(index < 0 or index >= frames for index in indices):
                    raise ValueError("Connect indices for the same detail frames, within the requested clip length.")
                if any(a >= b for a, b in zip(indices, indices[1:])):
                    raise ValueError("Source detail frame indices must be in ascending order, without duplicates.")
                video_item["timestamps"] = [index / 24.0 for index in indices]
            prompt = str(prompt) + (
                "\nThe visual samples of <Video 1> show the original performer. Use them for source appearance traits "
                "that the edit asks to preserve. Apply all requested changes."
            )
            detail_height, detail_width = details.shape[1:3]
            detail_status = f" | <Video 1> detail: {len(details)} frame(s), {detail_width}x{detail_height}"
        if source is None:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, align_frame_count

            if details is not None:
                clip = _RemakeDetailClip(clip, video_item)
            references = {f"ref_image_{index}": image for index, image in enumerate(regular_refs)}
            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip, prompt=prompt, width=width, height=height, length=frames,
                vae=vae, audio_vae=audio_vae, ref_images=references,
            )
            positive, latent = result.args
            if details is not None:
                # Give the sparse appearance reference its own valid VAE span.
                # The external Add Guide still supplies the complete performance.
                count = align_frame_count(len(details))
                indices = [round(i * (len(details) - 1) / (count - 1)) for i in range(count)]
                encoded = vae.encode(details[indices])
                video_ref = {"kind": "video", "latent_t": encoded.shape[2],
                             "latent_h": encoded.shape[-2], "latent_w": encoded.shape[-1],
                             "ref_audio_t": 0, "latent": encoded, "audio_latent": None}
                refs = positive[0][1].get("minimax_refs", [])
                positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": [video_ref, *refs]})
            actual_frames = align_frame_count(frames)
            status = f"Native H3 / Add Guide | {width}x{height} | {actual_frames} frames at 24 fps" + detail_status
            if audio is not None:
                status += " | route guide audio through Add Guide"
            logging.info("[Simple H3 Remake] %s", status)
            return positive, latent, actual_frames, 0, status

        upstream = nodes.NODE_CLASS_MAPPINGS.get("RemakeConditioning")
        if upstream is None:
            raise RuntimeError("Install ComfyUI-Viggle-Remake and restart ComfyUI.")
        if source.ndim != 4 or source.shape[0] == 0:
            raise ValueError("Connect a non-empty source image batch at 24 fps.")

        source_frames = min(int(frames), int(source.shape[0]))
        actual_frames = 5 + 17 * max(0, (source_frames - 5 + 16) // 17)
        prepared = source[:source_frames, ..., :3]
        if tuple(prepared.shape[1:3]) != (height, width):
            prepared = comfy.utils.common_upscale(
                prepared.movedim(-1, 1), width, height, "bicubic", "center"
            ).movedim(1, -1)
        padding = actual_frames - source_frames
        if padding:
            prepared = torch.cat((prepared, prepared[-1:].expand(padding, -1, -1, -1)), dim=0)

        if audio is not None and audio.get("_vhs_no_audio", False):
            audio = None
        if details is not None:
            picture_labels = "".join(f"<Picture {i + 1}>: " for i in range(len(regular_refs)))
            clip = _RemakeDetailClip(clip, video_item, picture_labels + prompt)
        positive, latent = upstream().run(
            clip=clip, vae=vae, audio_vae=audio_vae, source=prepared,
            prompt=prompt, ref_1=ref_1, pins="", frames=actual_frames,
            short_edge=min(width, height), audio=audio,
            ref_2=ref_2, ref_3=ref_3, ref_4=ref_4,
        )
        if audio is None:
            # Official Remake freezes encoded silence. Without source audio,
            # start an empty audio stream and allow the sampler to generate it.
            video, encoded_audio = latent["samples"].unbind()
            video_mask, _ = latent["noise_mask"].unbind()
            latent = dict(latent)
            latent["samples"] = comfy.nested_tensor.NestedTensor((video, torch.zeros_like(encoded_audio)))
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, torch.ones_like(encoded_audio)))
        status = (
            f"Remake {width}x{height} | requested {frames} | source used {source_frames} "
            f"| generated {actual_frames} frames at 24 fps | repeated tail {padding} "
            f"| audio {'generated' if audio is None else 'source locked'}"
        ) + detail_status
        logging.info("[Simple H3 Remake] %s", status)
        return positive, latent, actual_frames, source_frames, status


class SimpleH3ReferenceNoiseSource:
    """CPU noise ordering compatible with Viggle Remake (Apache-2.0)."""

    def __init__(self, seed, reference_shapes):
        self.seed = seed
        self.reference_shapes = reference_shapes

    def generate_noise(self, input_latent):
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        # Keep each complete draw: splitting draws changes PyTorch's RNG sequence.
        for shape in self.reference_shapes:
            torch.randn(shape, generator=generator)
        samples = input_latent["samples"]
        if not isinstance(samples, comfy.nested_tensor.NestedTensor):
            raise ValueError("Simple H3 Reference Noise expects a MiniMax H3 AV latent.")
        return comfy.nested_tensor.NestedTensor([
            torch.randn(stream.shape, generator=generator) for stream in samples.unbind()
        ])


class SimpleH3ReferenceNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
        }}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Remake"
    DESCRIPTION = "Reference-aware AV noise. Preserves Remake's seed ordering for native references and pinned frames."

    def run(self, positive, seed):
        metadata = positive[0][1]
        references = list(metadata.get("minimax_refs", []))
        references.extend(metadata.get("minimax_keyframes", []))
        shapes = tuple(tuple(block["latent"].shape) for block in references if "latent" in block)
        return (SimpleH3ReferenceNoiseSource(int(seed), shapes),)


NODE_CLASS_MAPPINGS = {"SimpleH3RemakeConditioning": SimpleH3RemakeConditioning, "SexyAIRemakeConditioning": SimpleH3RemakeConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {key: "Simple H3 Remake Conditioning — Custom Size / Frames" for key in NODE_CLASS_MAPPINGS}
NODE_CLASS_MAPPINGS["SimpleH3ReferenceNoise"] = SimpleH3ReferenceNoise
NODE_DISPLAY_NAME_MAPPINGS["SimpleH3ReferenceNoise"] = "Simple H3 Reference Noise"
NODE_CLASS_MAPPINGS["SimpleH3SourceDetailFrames"] = SimpleH3SourceDetailFrames
NODE_DISPLAY_NAME_MAPPINGS["SimpleH3SourceDetailFrames"] = "Simple H3 Source Detail Frames"
