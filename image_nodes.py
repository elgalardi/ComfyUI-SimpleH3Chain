"""Focused still-image generation nodes for MiniMax H3.

The graph surface is ours; low-level tensor conventions come directly from
ComfyUI's native MiniMax H3 implementation.
"""

import math

import torch
import torch.nn.functional as F

import comfy.model_sampling
import comfy.samplers
import comfy.utils
import node_helpers
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _resize,
)


CATEGORY = "MiniMax H3/Simple Chain/Image"


def _still_prompt(prompt, fidelity, reference_count):
    tags = ", ".join(f"<Picture {i}>" for i in range(1, reference_count + 1))
    if fidelity >= 0.85:
        preserve = "Preserve exact facial identity, body proportions, hair, wardrobe materials, and distinctive details."
    elif fidelity >= 0.55:
        preserve = "Preserve recognizable identity, hairstyle, wardrobe, and important visual details."
    else:
        preserve = "Use the references for identity while allowing broader visual reinterpretation."
    return (
        "Create one single finished high-detail cinematic still image. One full frame only; no grid, collage, "
        "contact sheet, captions, borders, or duplicate subjects. Use a locked instant in time with coherent anatomy, "
        "sharp eyes, natural skin detail, detailed hands, and clean spatial construction. "
        f"Reference order: {tags}. {preserve}\n\n{str(prompt).strip()}"
    )


class SimpleH3ImagePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "forceInput": True}),
                "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 32}),
                "identity_fidelity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "reference_resolution": (["match_output_area", "maximum_identity_2048"], {"default": "match_output_area"}),
            },
            "optional": {
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "final_prompt", "info")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, clip, vae, reference_image_1, prompt, width, height,
                identity_fidelity, reference_resolution,
                reference_image_2=None, reference_image_3=None):
        width = max(CANVAS_MULTIPLE, round(int(width) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        height = max(CANVAS_MULTIPLE, round(int(height) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        references = [x for x in (reference_image_1, reference_image_2, reference_image_3) if x is not None]
        final_prompt = _still_prompt(prompt, float(identity_fidelity), len(references))
        latent, _ = _empty_av_latent(width, height, 5)
        latent["h3_context_frames"] = 5
        latent["h3_requested_frames"] = 5

        ref_items, ref_blocks, sizes = [], [], []
        for image in references:
            h, w = int(image.shape[1]), int(image.shape[2])
            if reference_resolution == "maximum_identity_2048":
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / max(1, min(w, h)))
            else:
                scale = min(1.0, math.sqrt((width * height) / max(1, w * h)))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(image[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                "latent": vae.encode(resized),
            })
            sizes.append(f"{tw}x{th}")
        tokens = clip.tokenize(final_prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
        info = f"our H3 still path · output {width}x{height} · 5-frame packet · refs {', '.join(sizes)}"
        return conditioning, latent, final_prompt, info


def _reference_board(images, width, height):
    """Fit ordered references into one clean canvas used as the FL2VA source frame."""
    count = len(images)
    panel_widths = [width // count] * count
    panel_widths[-1] += width - sum(panel_widths)
    panels = []
    for image, panel_width in zip(images, panel_widths):
        frame = image[:1, ..., :3]
        h, w = int(frame.shape[1]), int(frame.shape[2])
        scale = min(panel_width / max(1, w), height / max(1, h))
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = F.interpolate(
            frame.movedim(-1, 1).float(), size=(rh, rw), mode="bilinear",
            align_corners=False, antialias=True,
        ).movedim(1, -1)
        panel = torch.zeros((1, height, panel_width, 3), dtype=resized.dtype, device=resized.device)
        x, y = (panel_width - rw) // 2, (height - rh) // 2
        panel[:, y:y + rh, x:x + rw] = resized
        panels.append(panel)
    return torch.cat(panels, dim=2).contiguous()


class SimpleH3FL2VAStoryboardPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image_1": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 32}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25,
                    "tooltip": (
                        "Final storyboard area in 1024² megapixels. Width and height define only the aspect ratio; "
                        "both resolved axes are aligned to MiniMax H3's 32-pixel grid."
                    ),
                }),
                "candidate_frames": ([5, 22], {
                    "default": 5,
                    "tooltip": (
                        "5 is the fast default. 22 produces more still candidates but does not anchor "
                        "the reference layout into the generated scene."
                    ),
                }),
            },
            "optional": {
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "reference_board", "final_prompt", "info")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, clip, vae, reference_image_1, prompt, width, height,
                megapixels, candidate_frames, reference_image_2=None, reference_image_3=None):
        requested_width = max(1, int(width))
        requested_height = max(1, int(height))
        ratio = requested_width / requested_height
        target_area = max(0.25, float(megapixels)) * 1024.0 * 1024.0
        width = max(32, round(math.sqrt(target_area * ratio) / 32) * 32)
        height = max(32, round(math.sqrt(target_area / ratio) / 32) * 32)
        references = [x for x in (reference_image_1, reference_image_2, reference_image_3) if x is not None]
        board = _reference_board(references, width, height)
        labels = ", ".join(f"reference image {i} depicts subject S{i}" for i in range(1, len(references) + 1))
        final_prompt = (
            "Generate a completely new single finished cinematic scene image from noise. The connected images are "
            f"identity references only: {labels}. Use them only for each subject's recognizable face, hair, skin, "
            "body proportions, and other identity-defining traits. Do not copy their composition, pose, crop, "
            "background, lighting, camera angle, borders, spacing, or image layout. Do not reproduce a reference "
            "sheet, montage, collage, split screen, before-and-after image, duplicate person, or separate panel. "
            "The storyboard description below is the sole authority for wardrobe, action, environment, framing, "
            "lens, lighting, character placement, and final composition. Produce one unified full-frame scene with "
            "coherent anatomy, sharp faces, detailed hands, and no captions or labels.\n\n" + str(prompt).strip()
        )
        frames = int(candidate_frames)
        latent, natural_frames = _empty_av_latent(width, height, frames)
        latent["h3_context_frames"] = natural_frames
        latent["h3_requested_frames"] = natural_frames
        # Supply every identity separately to Qwen's vision encoder. Crucially,
        # none is VAE-encoded as a keyframe: the diffusion canvas starts from
        # noise, so the source layout cannot be geometrically preserved.
        vision_references = [image[:1, ..., :3] for image in references]
        tokens = clip.tokenize(final_prompt, images=vision_references)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (
            conditioning, latent, board, final_prompt,
            f"FL2VA vision-only storyboard generation · {len(references)} separate identity references · ratio source "
            f"{requested_width}x{requested_height} · {float(megapixels):g} MP -> {width}x{height} "
            f"(multiple of 32) · {natural_frames} candidates · no reference keyframe anchor",
        )


class SimpleH3ImageBatchPrepare:
    """Prepare one to four independent still-image samples without storyboard logic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 32}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "candidate_frames": ([5, 22], {"default": 5}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "final_prompt", "info")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, clip, vae, prompt, width, height,
                batch_count, candidate_frames, reference_image_1=None,
                reference_image_2=None, reference_image_3=None,
                reference_image_4=None):
        width = max(32, round(int(width) / 32) * 32)
        height = max(32, round(int(height) / 32) * 32)
        batch_count = max(1, min(4, int(batch_count)))
        references = [
            image for image in (
                reference_image_1, reference_image_2,
                reference_image_3, reference_image_4,
            ) if image is not None
        ]
        if references:
            labels = ", ".join(
                f"<Picture {index}> is identity reference S{index}"
                for index in range(1, len(references) + 1)
            )
            reference_rule = (
                f"Connected identity references: {labels}. Preserve recognizable identity "
                "while following the requested composition. Do not copy the reference layout, "
                "background, crop, borders, or pose unless explicitly requested. "
            )
        else:
            reference_rule = "No identity reference is connected. "
        final_prompt = (
            "Create one single finished high-detail image per batch sample. Each output must be "
            "an independent full-frame composition, never a storyboard, grid, collage, contact "
            "sheet, split screen, sequence, captioned panel, or duplicated layout. "
            f"{reference_rule}\n\n{str(prompt).strip()}"
        )
        frames = int(candidate_frames)
        latent, natural_frames = _empty_av_latent(
            width, height, frames, batch_size=batch_count
        )
        latent["h3_context_frames"] = natural_frames
        latent["h3_requested_frames"] = natural_frames
        latent["h3_image_batch_count"] = batch_count
        latent["h3_frames_per_batch"] = natural_frames
        vision_references = [image[:1, ..., :3] for image in references]
        tokens = clip.tokenize(final_prompt, images=vision_references)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (
            conditioning,
            latent,
            final_prompt,
            f"H3 independent image batch · {batch_count} outputs · {len(references)} refs · "
            f"{width}x{height} · {natural_frames} candidates per output",
        )


class SimpleH3ImageSampling:
    PROFILES = {
        "LightX balanced | ER-SDE 6 steps": ("er_sde", 6),
        "LightX fastest | ER-SDE 4 steps": ("er_sde", 4),
        "LightX studio balanced | ER-SDE 8 steps": ("er_sde", 8),
        "LightX studio quality | ER-SDE 12 steps": ("er_sde", 12),
        "Base balanced | RES 12 steps": ("res_multistep", 12),
        "Base quality | RES 20 steps": ("res_multistep", 20),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "profile": (list(cls.PROFILES), {"default": "LightX balanced | ER-SDE 6 steps"}),
            "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.1}),
            "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("MODEL", "SAMPLER", "SIGMAS", "STRING")
    RETURN_NAMES = ("model", "sampler", "sigmas", "info")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, model, profile, shift_video, shift_audio):
        sampler_name, steps = self.PROFILES[profile]
        patched = model.clone()

        class H3Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass

        original = patched.get_model_object("model_sampling")
        sampling = H3Sampling(patched.model.model_config)
        sampling.set_parameters(
            shift=float(shift_video), audio_shift=float(shift_audio),
            multiplier=getattr(original, "multiplier", 1000),
        )
        if hasattr(original, "noise_scale"):
            sampling.set_noise_scale(original.noise_scale)
        patched.add_object_patch("model_sampling", sampling)
        options = patched.model_options.get("transformer_options", {}).copy()
        options["minimax_h3_sigma_shift_video"] = float(shift_video)
        options["minimax_h3_sigma_shift_audio"] = float(shift_audio)
        patched.model_options["transformer_options"] = options
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(sampling, "simple", steps).cpu()
        return patched, sampler, sigmas, f"{profile} · shifts {shift_video:g}/{shift_audio:g}"


class SimpleH3ImageDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "INT")
    RETURN_NAMES = ("candidate_frames", "frame_count", "info", "batch_count")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, samples, vae):
        batch_count = max(1, int(samples.get("h3_image_batch_count", 1)))
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        elif images.ndim != 4:
            raise ValueError("MiniMax H3 VAE returned an unsupported image tensor.")
        frames_per_batch = max(1, int(samples.get(
            "h3_frames_per_batch", samples.get("h3_requested_frames", images.shape[0])
        )))
        keep = min(int(images.shape[0]), frames_per_batch * batch_count)
        images = images[:keep].contiguous()
        return (
            images,
            int(images.shape[0]),
            f"decoded {int(images.shape[0])} candidates across {batch_count} image batches",
            batch_count,
        )


class SimpleH3ImageSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "candidate_frames": ("IMAGE",),
            "selection": (["stable_quality", "sharpest", "first", "middle", "last"], {"default": "stable_quality"}),
            "skip_first_frames": ("INT", {"default": 5, "min": 0, "max": 38, "step": 1}),
        }, "optional": {
            "batch_count": ("INT", {"default": 1, "min": 1, "max": 4, "forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("selected_image", "all_candidates", "selected_index", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(self, candidate_frames, selection, skip_first_frames, batch_count=1):
        frames = candidate_frames
        if frames.ndim != 4 or frames.shape[0] < 1:
            raise ValueError("No H3 still candidates were supplied.")
        count = int(frames.shape[0])
        batch_count = max(1, min(int(batch_count), count))
        if batch_count > 1:
            per_batch = count // batch_count
            selected, reports = [], []
            for batch_index in range(batch_count):
                start = batch_index * per_batch
                end = count if batch_index == batch_count - 1 else start + per_batch
                chosen, _, chosen_index, report = self.select(
                    frames[start:end], selection, skip_first_frames, 1
                )
                selected.append(chosen)
                reports.append(f"batch {batch_index + 1}: {report}")
            return (
                torch.cat(selected, dim=0).contiguous(),
                frames,
                -1,
                f"selected {batch_count} independent images · " + " · ".join(reports),
            )
        start = min(count - 1, max(0, int(skip_first_frames)))
        fixed = {"first": start, "middle": start + (count - start) // 2, "last": count - 1}
        if selection in fixed:
            index = fixed[selection]
            return frames[index:index + 1].clone(), frames, index, f"selected {selection} frame {index}/{count - 1}"

        # Metrics run on small CPU/GPU-friendly previews, not full-resolution candidates.
        candidate_view = frames[start:]
        x = candidate_view[..., :3].movedim(-1, 1)
        h, w = x.shape[-2:]
        scale = min(1.0, 384.0 / max(h, w))
        if scale < 1.0:
            x = F.interpolate(x.float(), scale_factor=scale, mode="bilinear", align_corners=False, antialias=True)
        else:
            x = x.float()
        gray = 0.2126 * x[:, :1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        kernel = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=x.device).view(1,1,3,3)
        sharp = torch.log1p(F.conv2d(gray, kernel, padding=1).var(dim=(1,2,3)) * 1000)
        if selection == "sharpest" or count == 1:
            scores = sharp
        else:
            sharp_n = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)
            delta = torch.zeros_like(sharp_n)
            if count > 1:
                delta[0] = (x[0] - x[1]).abs().mean()
                delta[-1] = (x[-1] - x[-2]).abs().mean()
                if count > 2:
                    delta[1:-1] = ((x[1:-1]-x[:-2]).abs().mean((1,2,3)) + (x[1:-1]-x[2:]).abs().mean((1,2,3))) / 2
            stability = 1 - (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
            scores = 0.82 * sharp_n + 0.18 * stability
        index = start + int(torch.argmax(scores).item())
        return frames[index:index + 1].clone(), frames, index, f"selected frame {index}/{count - 1} by {selection}"


NODE_CLASS_MAPPINGS = {
    "SimpleH3ImagePrepare": SimpleH3ImagePrepare,
    "SimpleH3FL2VAStoryboardPrepare": SimpleH3FL2VAStoryboardPrepare,
    "SimpleH3ImageBatchPrepare": SimpleH3ImageBatchPrepare,
    "SimpleH3ImageSampling": SimpleH3ImageSampling,
    "SimpleH3ImageDecode": SimpleH3ImageDecode,
    "SimpleH3ImageSelect": SimpleH3ImageSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleH3ImagePrepare": "Simple H3 Image Prepare",
    "SimpleH3FL2VAStoryboardPrepare": "Simple H3 FL2VA Storyboard Image Prepare",
    "SimpleH3ImageBatchPrepare": "Simple H3 Image Batch Prepare (1–4)",
    "SimpleH3ImageSampling": "Simple H3 Image Sampling",
    "SimpleH3ImageDecode": "Simple H3 Image Decode",
    "SimpleH3ImageSelect": "Simple H3 Best Image",
}
