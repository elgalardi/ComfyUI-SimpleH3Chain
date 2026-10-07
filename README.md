# ComfyUI Simple H3 Chain

Focused orchestration for the current Ref2VA, FL2VA and still-image workflows.
Native MiniMax H3 model loading, conditioning, sampling and decoding remain
unchanged. Context Loop does not need to be installed separately.

## Included nodes (26)

- Scene chain: `SimpleH3ChainPlan`, `SimpleH3ChainLoopStart`,
  `SimpleH3ChainCurrent`, `SimpleH3ChainContext`, `SimpleH3LoopTrim`,
  `SimpleH3ChainSegmentSave`, `SimpleH3ChainLoopEnd`, `SimpleH3ChainAssemble`.
- Prompt plan: `SimpleH3CompactContinuousPlanJSON`.
- Base context decode: `SimpleH3BaseContextDecode` decodes the aligned tail only
  when Plan's `base_preview=false`; it displays no video and avoids a full base decode.
- Existing-video continuation: `SimpleH3ExistingVideoContext` prepares a base-size
  tail for Start / Resume. When `prepend_original=true`, the original is
  normalized to 24 fps and the optional delivery canvas before final assembly.
- Frame routing: `SimpleH3FrameGate`.
- Local video preview: `SimpleH3DirectEditPreview`. With `save_output=false`,
  one temporary MP4 per node is overwritten rather than creating saved copies.
  Set `preview_output_subfolder=Sexy AI Studio/Previews` to keep these
  overwrite-only files under the output directory instead of the temp directory.
- Optional model LoRA: `SimpleH3OptionalLoraLoader`.
- Optional upscale: `SimpleH3LatentUpscaleResolution`, `SimpleH3LatentUpscaleRefine`.
- Ultimate upscale: `SimpleH3UltimateUpscale`, `SimpleH3LatentUpscaleParams`,
  `SimpleH3LatentUpscaleWithModelParams`, `SimpleH3TemporalSplitParams`,
  `SimpleH3SpatialSplitParams`. See [ULTIMATE_UPSCALE.md](ULTIMATE_UPSCALE.md)
  for the default `keep_model_loaded` behavior and memory tradeoffs.
- Still image: `SimpleH3ImageBatchPrepare`, `SimpleH3ImageSampling`,
  `SimpleH3ImageDecode`, `SimpleH3ImageSelect`.
- Canvas calculator: `SimpleH3DimensionsScale` takes numeric `width` and `height`,
  scales their pixel area to the selected decimal megapixels, and rounds both
  outputs to the nearest positive `multiple` (default 32). Aspect ratio and pixel
  area are approximate after rounding. It does not resize an image or latent and
  works independently of the generation model.

## Two-pass scene delivery

For base generation followed by Ultimate Upscale, connect the raw **base**
sampler output and trimmed **base** images to Save Scene's checkpoint inputs
and Loop End. Connect trimmed **refined** images to Save Scene's optional
`delivery_images`, and refined trimmed audio to its audio input. Use the same
Context trim contract on both branches. With a regular full-scene base decoder,
keep `base_preview=true`. For tail-only operation, use `SimpleH3BaseContextDecode`,
set `base_preview=false`, and connect that tail directly to checkpoint/Loop End
image inputs. Set the refined Trim's `delivery_frames=true` so it still trims
the complete HQ decode. See [ULTIMATE_UPSCALE.md](ULTIMATE_UPSCALE.md) for refined
prefix protection and checkpoint wiring.

Connect final `delivery_width` and `delivery_height` to Chain Plan together.
These dimensions participate in the generation fingerprint and imported-video
validation, without changing the base sampling/context canvas. The final
assembly uses refined MP4 segments; resume checkpoints retain base latents.
Older graphs without these optional inputs keep their existing behavior.
Save Scene's `show_preview=true` displays its saved MP4 directly. Connected
audio is muxed without re-encoding the video; a separate Direct Edit Preview
is not needed. There is no approval pause.

`SimpleH3ExistingVideoContext` accepts either native VIDEO or decoded
IMAGE/AUDIO, never both routes. Imported videos must contain enough frames for
the plan's context (39 frames for Masked AV). It does not provide a streaming
decoder: loading a long video can require substantial system RAM.

## First / last frame routing

`SimpleH3FrameGate` passes the original first frame only in scene 1 and omits it
for continuations. The optional last frame passes through in every scene.
Output slots are first frame (0), first-scene flag (1), status (2), last frame (3).
It does not resize or copy images. Replace `MiniMaxH3ChainFirstSceneImage` with
this node when migrating FL2VA from Context Loop.

## Shared seed and output

Connect one `PrimitiveInt` to the director, sampler, and compact plan's `seed`.
The plan records the supplied seed on each scene. Connect/assign the sampler
steps to the plan's optional `steps` input when changing the default of 6.

The Studio workflows use `Sexy AI Studio/Video`, `Sexy AI Studio/I2V` and
`Sexy AI Studio/Images` under ComfyUI's output directory. Scene files, checkpoints
and previews belong to the configured run folder. Still-image dimensions support
up to 6144 pixels per axis, aligned to 32; high resolutions require sufficient VRAM.

The optional learned upscale requires `Comfyui_Minimax_h3_latent_Upscaler` and
its compatible model only when enabled. ComfyUI provides the native H3 nodes;
other workflow dependencies such as KJNodes are separate packages.

## Install and compatibility

Clone this repository into ComfyUI/custom_nodes and restart ComfyUI. Update
existing installations with `git pull`. The September 2026 cleanup removes
experimental storyboard, review and alternate refinement node IDs. Older
workflows using those IDs require a prior revision.

`stable_engine` is retained internal runtime code required by the scene chain,
not another installed node pack. See THIRD_PARTY_NOTICES.md and LICENSE.

Offline frame-routing checks: `python -B tests/test_frame_gate.py`.
These checks do not load models or execute ComfyUI generations.
