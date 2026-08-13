# ComfyUI Simple H3 Chain

A focused MiniMax H3 scene-chain interface for the workflow used by SexyAI.

The native H3 model, LoRA, attention, Ref2Video, sampler and decode chain stays
unchanged. These nodes replace only the orchestration layer around it.

## First test release

This version uses the locally pinned stable Context Loop engine and provides
new, separate node IDs. This lets both implementations coexist for A/B tests
without changing or overwriting the original package.

`Simple H3 Chain Plan` intentionally exposes only:

- Story Director JSON
- width and height
- inherited context frames
- audio mode
- output name

The tested continuity defaults are fixed internally: video context, head
anchor, no crop, automatic audio context, CRF 18, and deterministic scene
seeds. Scene duration and sampling steps continue to come from the JSON plan.

## Dependency

For this first local test, keep `ComfyUI-MiniMaxH3-Contex-Loop` installed and
pinned to the tested stable version. A later standalone release can vendor the
small runtime subset after the new interface is validated.
