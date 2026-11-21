"""Simple H3 learned 2D upscale, independent of the external LBH package."""
import torch
import torch.nn.functional as F
import comfy.nested_tensor
from comfy_api.latest import io
from .third_party.lbh_latent_2d import MinimaxH3LatentUpscalerNode2D as _Backend


def _resize_mask(mask, video):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (4, 5):
        raise ValueError("Expected a 4D or 5D video noise mask.")
    target = video.shape[-3:] if video.ndim == 5 else video.shape[-2:]
    if mask.ndim == 4 and video.ndim == 5:
        mask = mask.unsqueeze(2)
    return F.interpolate(mask.float(), size=target, mode="nearest").to(mask.dtype)


class SimpleH3LatentUpscale2D(_Backend):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "SimpleH3LatentUpscale2D"
        schema.display_name = "Simple H3 Latent Upscale — Learned 2D"
        schema.category = "Simple H3 Chain/Upscale"
        schema.inputs[0] = io.Latent.Input("latent", tooltip="H3 video or joint AV latent. Only video is upscaled; audio is preserved exactly.")
        schema.outputs = [io.Latent.Output("latent")]
        schema.description = "Learned LBH-compatible 2D/temporal upscaling. Supports video and joint AV latents, preserves audio and metadata, and resizes masks. No separate LBH node package required."
        return schema

    @classmethod
    def execute(cls, latent, model_name, mode, align=32, force_unload=True,
                device="cuda", precision="fp16", sharpness=1, temporal_chunking=None):
        samples = latent.get("samples")
        joint = isinstance(samples, comfy.nested_tensor.NestedTensor)
        if joint:
            if len(samples.tensors) != 2:
                raise ValueError("Expected video and audio in the H3 joint latent.")
            video, audio = samples.tensors
        else:
            video, audio = samples, None
        if not isinstance(video, torch.Tensor) or video.ndim not in (4, 5) or video.shape[1] != 24:
            raise ValueError("Expected a 24-channel H3 video latent.")
        result = _Backend.execute(
            {"samples": video}, model_name, mode, align, force_unload, device,
            precision, sharpness=sharpness, temporal_chunking=temporal_chunking)
        upscaled = result.result[0]["samples"]
        output = dict(latent)
        output["samples"] = comfy.nested_tensor.NestedTensor((upscaled, audio)) if joint else upscaled
        mask = latent.get("noise_mask")
        if isinstance(mask, comfy.nested_tensor.NestedTensor):
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((_resize_mask(mask.tensors[0], upscaled), mask.tensors[1]))
        elif mask is not None:
            output["noise_mask"] = _resize_mask(mask, upscaled)
        return io.NodeOutput(output)
