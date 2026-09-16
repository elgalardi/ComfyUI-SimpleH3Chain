# ComfyUI Simple H3 Chain

Focused orchestration for the current Ref2VA, FL2VA and still-image workflows.
Native MiniMax H3 model loading, conditioning, sampling and decoding remain
unchanged. Context Loop does not need to be installed separately.

## Included nodes (17)

- Scene chain: `SimpleH3ChainPlan`, `SimpleH3ChainLoopStart`,
  `SimpleH3ChainCurrent`, `SimpleH3ChainContext`, `SimpleH3LoopTrim`,
  `SimpleH3ChainSegmentSave`, `SimpleH3ChainLoopEnd`, `SimpleH3ChainAssemble`.
- Prompt plan: `SimpleH3CompactContinuousPlanJSON`.
- Frame routing: `SimpleH3FrameGate`.
- Optional model LoRA: `SimpleH3OptionalLoraLoader`.
- Optional upscale: `SimpleH3LatentUpscaleResolution`, `SimpleH3LatentUpscaleRefine`.
- Still image: `SimpleH3ImageBatchPrepare`, `SimpleH3ImageSampling`,
  `SimpleH3ImageDecode`, `SimpleH3ImageSelect`.

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
