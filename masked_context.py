"""Native masked-latent continuation for Simple H3 Chain.

This implementation deliberately has no dependency on external custom-node
packs.  It uses ComfyUI's native MiniMax H3 nested AV denoise-mask contract:
zero preserves an existing latent token and one generates a new token.
"""

from __future__ import annotations

import logging
import functools

import torch


_LOG = logging.getLogger("simple_h3_chain.masked_context")
FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_PAYLOAD_COMPAT_MARKER = "_simple_h3_av_mask_payload_compat"


def _ensure_av_mask_payload_compat():
    """Expose packed video/audio masks to MiniMax H3 on transitional ComfyUI builds.

    Recent ComfyUI versions contain the H3 mask engine and inpaint scaling, but
    some builds do not yet unpack the nested AV denoise mask in ``extra_conds``.
    Without these two conditions the workflow runs normally while silently
    treating every continuation as a fresh shot.
    """
    import comfy.conds
    import comfy.utils as comfy_utils
    from comfy.model_base import MiniMaxH3

    current = getattr(MiniMaxH3, "extra_conds", None)
    if not callable(current):
        raise RuntimeError("MiniMaxH3.extra_conds is unavailable in this ComfyUI build.")
    if getattr(current, _PAYLOAD_COMPAT_MARKER, False):
        return

    # Native implementations already emit both conditions.  Detect this from
    # the function code without executing it with synthetic model state.
    code = getattr(current, "__code__", None)
    names = set(getattr(code, "co_names", ()) or ())
    constants = {value for value in (getattr(code, "co_consts", ()) or ()) if isinstance(value, str)}
    if {"denoise_mask", "audio_denoise_mask"}.issubset(names | constants):
        return

    @functools.wraps(current, updated=())
    def extra_conds_with_av_masks(self, **kwargs):
        out = current(self, **kwargs)
        if not isinstance(out, dict):
            return out
        if "denoise_mask" in out and "audio_denoise_mask" in out:
            return out

        packed_mask = kwargs.get("denoise_mask")
        latent_shapes = kwargs.get("latent_shapes")
        if packed_mask is None or latent_shapes is None or len(latent_shapes) < 2:
            return out
        masks = comfy_utils.unpack_latents(packed_mask, latent_shapes)
        if len(masks) < 2:
            return out
        if "denoise_mask" not in out and torch.amin(masks[0]).item() < 1.0 - 1e-3:
            out["denoise_mask"] = comfy.conds.CONDRegular(masks[0][:, :1].clone())
        if "audio_denoise_mask" not in out and torch.amin(masks[1]).item() < 1.0 - 1e-3:
            out["audio_denoise_mask"] = comfy.conds.CONDRegular(masks[1][:, :1].clone())
        return out

    setattr(extra_conds_with_av_masks, _PAYLOAD_COMPAT_MARKER, True)
    MiniMaxH3.extra_conds = extra_conds_with_av_masks
    _LOG.info(
        "Simple H3 enabled MiniMax H3 AV-mask payload compatibility for masked continuation."
    )


def _pixel_frames(latent_steps: int) -> int:
    return sum(FRAME_PER_TOKEN[index % 5] for index in range(int(latent_steps)))


def _streams(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError("Simple H3 masked context requires a joint H3 AV latent.")
    if len(parts) < 2:
        raise ValueError("Simple H3 masked context requires video and audio latent streams.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "Simple H3 masked context received invalid AV shapes: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}."
        )
    return video, audio


def _exact_av_context(requested: int, source_frames: int, target_frames: int) -> int:
    """Snap to a shared H3 24-fps video / 40-Hz audio boundary.

    Shared boundaries are 39, 90, 141, ... frames.  Simple H3 currently
    exposes at most 39, so this normally selects 39 or refuses clearly.
    """
    cap = min(int(requested), int(source_frames), int(target_frames) - 1)
    candidates = []
    value = 39
    while value <= cap:
        candidates.append(value)
        value += 51
    if not candidates:
        raise ValueError(
            "Masked AV Continuation needs at least 39 source frames and a target "
            "longer than 39 frames. Use Video, Images, or Cut Reference for shorter clips."
        )
    selected = candidates[-1]
    if selected != int(requested):
        _LOG.warning(
            "Simple H3 masked context snapped %d requested frames to the exact AV boundary %d.",
            int(requested), selected,
        )
    return selected


def _audio_feather(mask, audio_steps: int, feather_ticks: int):
    """Protect the AV prefix while releasing its final audio edge smoothly."""
    feather = max(0, min(int(feather_ticks), int(audio_steps)))
    hard = int(audio_steps) - feather
    if hard:
        mask[..., :hard] = 0.0
    if feather:
        positions = torch.arange(
            1, feather + 1, device=mask.device, dtype=mask.dtype
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * positions / float(feather))
        shape = [1] * mask.ndim
        shape[-1] = feather
        mask[..., hard:audio_steps] = ramp.view(*shape)


def apply_masked_av_continuation(
    target_latent, source_latent, context_frames=39, audio_feather_ticks=8
):
    """Copy the previous generated AV tail into a protected target prefix."""
    try:
        import comfy.nested_tensor
        from comfy.model_base import MiniMaxH3
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "This ComfyUI build lacks native MiniMax H3 AV-mask support. "
            "Update ComfyUI or use the existing Video context mode."
        ) from error

    if not callable(getattr(MiniMaxH3, "scale_latent_inpaint", None)):
        raise RuntimeError(
            "This ComfyUI build cannot apply nested MiniMax H3 AV masks. "
            "Update ComfyUI or use the existing Video context mode."
        )

    _ensure_av_mask_payload_compat()

    target_video, target_audio = _streams(target_latent)
    source_video, source_audio = _streams(source_latent)
    if target_video.shape[0] != 1 or target_audio.shape[0] != 1:
        raise ValueError("Masked AV Continuation currently supports target batch size 1.")
    if source_video.shape[0] != 1 or source_audio.shape[0] != 1:
        raise ValueError("Masked AV Continuation currently supports source batch size 1.")

    target_frames = _pixel_frames(target_video.shape[2])
    source_frames = _pixel_frames(source_video.shape[2])
    preserved = _exact_av_context(context_frames, source_frames, target_frames)
    video_steps = 2 + 5 * ((preserved - 5) // 17)
    audio_steps = int(round(preserved / FPS * AUDIO_HZ))

    if _pixel_frames(video_steps) != preserved:
        raise RuntimeError("Internal H3 temporal-grid calculation failed.")
    if video_steps >= target_video.shape[2] or audio_steps >= target_audio.shape[-1]:
        raise ValueError("Masked AV context would consume the complete target clip.")
    if source_video.shape[2] < video_steps or source_audio.shape[-1] < audio_steps:
        raise ValueError("The previous generated AV latent is shorter than the context window.")
    if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ):
        raise ValueError(
            "Masked AV Continuation requires identical source/target resolution."
        )
    if tuple(source_audio.shape[1:3]) != tuple(target_audio.shape[1:3]):
        raise ValueError("Masked AV Continuation requires matching H3 audio geometry.")

    out_video = target_video.clone()
    out_audio = target_audio.clone()
    out_video[:, :, :video_steps] = source_video[:, :, -video_steps:].to(
        device=out_video.device, dtype=out_video.dtype
    )
    out_audio[..., :audio_steps] = source_audio[..., -audio_steps:].to(
        device=out_audio.device, dtype=out_audio.dtype
    )

    video_mask = torch.ones(
        (1, 1, out_video.shape[2], out_video.shape[3], out_video.shape[4]),
        device=out_video.device, dtype=torch.float32,
    )
    audio_mask = torch.ones(
        (1, 1, out_audio.shape[2], out_audio.shape[3]),
        device=out_audio.device, dtype=torch.float32,
    )
    video_mask[:, :, :video_steps] = 0.0
    _audio_feather(audio_mask, audio_steps, audio_feather_ticks)

    result = target_latent.copy()
    result["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
    result["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    _LOG.info(
        "Simple H3 masked AV continuation: %d frames, %d video steps, %d audio "
        "steps, %d feather ticks.",
        preserved, video_steps, audio_steps,
        max(0, min(int(audio_feather_ticks), audio_steps)),
    )
    return result, preserved


def apply_masked_video_continuation(
    target_latent, source_latent, context_frames=39
):
    """Protect a previous refined video tail without conditioning the audio stream."""
    try:
        import comfy.nested_tensor
        from comfy.model_base import MiniMaxH3
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "This ComfyUI build lacks native MiniMax H3 mask support."
        ) from error

    if not callable(getattr(MiniMaxH3, "scale_latent_inpaint", None)):
        raise RuntimeError(
            "This ComfyUI build cannot apply MiniMax H3 video masks."
        )

    _ensure_av_mask_payload_compat()
    target_video, target_audio = _streams(target_latent)
    source_video, _source_audio = _streams(source_latent)
    if target_video.shape[0] != 1 or source_video.shape[0] != 1:
        raise ValueError("Refined Masked continuity currently supports batch size 1.")

    target_frames = _pixel_frames(target_video.shape[2])
    source_frames = _pixel_frames(source_video.shape[2])
    preserved = _exact_av_context(context_frames, source_frames, target_frames)
    video_steps = 2 + 5 * ((preserved - 5) // 17)
    if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ):
        raise ValueError(
            "Refined Masked continuity requires identical upscale resolution."
        )

    out_video = target_video.clone()
    out_video[:, :, :video_steps] = source_video[:, :, -video_steps:].to(
        device=out_video.device, dtype=out_video.dtype
    )
    video_mask = torch.ones(
        (1, 1, out_video.shape[2], out_video.shape[3], out_video.shape[4]),
        device=out_video.device, dtype=torch.float32,
    )
    video_mask[:, :, :video_steps] = 0.0
    audio_mask = torch.ones(
        (1, 1, target_audio.shape[2], target_audio.shape[3]),
        device=target_audio.device, dtype=torch.float32,
    )

    result = target_latent.copy()
    result["samples"] = comfy.nested_tensor.NestedTensor((out_video, target_audio))
    result["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    _LOG.info(
        "Simple H3 refined masked video continuity: %d frames, %d protected video "
        "steps; audio unmasked and available to the refinement sampler.",
        preserved, video_steps,
    )
    return result, preserved
