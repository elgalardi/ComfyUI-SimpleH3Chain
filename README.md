# ComfyUI Simple H3 Chain

A focused MiniMax H3 clean-cut scene interface for the workflow used by SexyAI.

The native H3 model, LoRA, attention, Ref2Video, sampler and decode chain stays
unchanged. These nodes replace only the orchestration layer around it.

## Clean-cut continuity mode

Simple H3 Chain no longer feeds the previous scene latent into the next scene.
Every scene is a clean Ref2VA generation and therefore behaves like a cinematic
cut instead of a continuous camera take.

Use the two continuity nodes inside the recursive body:

1. Keep the original subject images in their existing Ref2VA slots. Connect
   `Simple H3 Cut Reference Sheet.continuity_sheet` to the next free reference
   slot and select that same tag on the node (`<Picture 2>`, `<Picture 3>`, etc.).
   Connect `augmented_prompt` to Ref2VA's prompt input and one clean subject
   image to `identity_image` as the scene-1 fallback.
2. Keep the full trimmed scene connected to `Simple H3 Save Scene + Checkpoint`.
3. Also connect the full trimmed scene to `Simple H3 Select Continuity Frames`.
4. Connect `frames_for_loop_end` from that selector to `Simple H3 Loop Until
   Final Scene.images` instead of connecting the full scene directly.

On scene 1 the additional reference contains only the clean fallback image. For
every later scene it contains two isolated frames selected at 30% and 80% of
the previous accepted clip. The augmented prompt automatically describes the
selected Picture tag as continuity evidence and asks H3 to preserve wardrobe,
hairstyle, accessories, physical changes, and persistent props while allowing
a new camera and environment. Original Story Director Picture tags remain
unchanged.

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
- checkpoint context frames (used for recovery, not latent continuation)
- audio mode
- output name

The tested storage defaults remain fixed internally: head checkpoints, no crop,
automatic audio bookkeeping, CRF 18, and deterministic scene seeds. Scene
duration and sampling steps continue to come from the JSON plan. The Clean Cut
node deliberately bypasses motion-context injection.

Additional simplifications:

- Start / Resume accepts only the plan, starting scene, and optional source song.
- Trim always uses 24 fps and frame-locks the audio tail.
- Assemble follows the plan's audio mode and uses AAC 256 kbps automatically.
- Assemble expands date tokens such as `%date:yyyy-MM-dd%` and preserves every
  final render by adding `_v2`, `_v3`, and so on. Scene segments and their
  transactional checkpoints keep their normal replace-by-scene behavior.
- Current Scene, Auto Context, Save Scene, and Loop End expose only their wired
  data ports; they have no unnecessary configuration widgets.

## Dependency

For this first local test, keep `ComfyUI-MiniMaxH3-Contex-Loop` installed and
pinned to the tested stable version. A later standalone release can vendor the
small runtime subset after the new interface is validated.
