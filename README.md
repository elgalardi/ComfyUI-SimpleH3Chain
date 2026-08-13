# ComfyUI Simple H3 Chain

A focused MiniMax H3 scene-chain interface for the workflow used by SexyAI.

The native H3 model, LoRA, attention, Ref2Video, sampler and decode chain stays
unchanged. These nodes replace only the orchestration layer around it.

The workflow title `H3 REF2VA — ONE REFERENCE SHEET` is the native ComfyUI
`MiniMaxH3ReferenceToVideo` node with a custom title. It intentionally remains
native so reference autogrow inputs and future official H3 fixes keep working.

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

Additional simplifications:

- Start / Resume accepts only the plan, starting scene, and optional source song.
- Trim always uses 24 fps and frame-locks the audio tail.
- Assemble follows the plan's audio mode and uses AAC 256 kbps automatically.
- Current Scene, Auto Context, Save Scene, and Loop End expose only their wired
  data ports; they have no unnecessary configuration widgets.

## Dependency

For this first local test, keep `ComfyUI-MiniMaxH3-Contex-Loop` installed and
pinned to the tested stable version. A later standalone release can vendor the
small runtime subset after the new interface is validated.
