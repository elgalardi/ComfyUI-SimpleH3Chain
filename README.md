# ComfyUI Simple H3 Chain

A focused MiniMax H3 clean-cut scene interface for the workflow used by SexyAI.

The native H3 model, LoRA, attention, Ref2Video, sampler and decode chain stays
unchanged. These nodes replace only the orchestration layer around it.

## Clean-cut continuity mode

Simple H3 Chain no longer feeds the previous scene latent into the next scene.
Every scene is a clean Ref2VA generation and therefore behaves like a cinematic
cut instead of a continuous camera take when `Simple H3 Context` is set to
`cut`. The same node can instead carry 1, 5, 11, 22, or 39 previous tail frames.
Choose `video` for Context Loop's tested head-overlap contract: the selected
frames are reproduced at the beginning of every later scene and Trim removes
that exact overlap. The 11-frame option is a hybrid midpoint represented as
eleven consecutive frame guides because H3's native temporal VAE grid jumps
from 5 directly to 22. Choose `images` to pin the selected frames independently
before the delivered timeline. Choose `cut_reference` to ignore the numeric
visual window and use only the final three consecutive frames as visual-state
references; its `match_video` audio setting retains the established 22-frame
generated-audio continuity. The Plan
reads this Context widget automatically so timing, trimming, checkpoints and
final assembly use the same contract.

`audio_context_frames` is independent from picture continuity. Use
`match_video`, `off`, `1`, `5`, `11`, `22`, or `39`. For example, visual `5`
with audio `22` keeps a lighter picture carry while preserving a longer voice,
music, and ambience tail. Visual `cut` with audio `22` performs a hard camera
cut while generated sound remains continuous. Source-track mode already reads
from one continuous timeline, so this selector primarily affects
`generated_audio` and `source_plus_timeline`.

For a short audio lead followed by newly generated H3 sound, select
`source_intro_generated` in `Simple H3 Chain Plan`. Connect the audio from VHS
to `source_audio` on `Simple H3 Start / Resume` and connect the MiniMax H3 audio
VAE to `Simple H3 Context`. The lead length is automatic: the first scene's
planned duration minus two seconds. If the incoming audio is shorter, all of
the available audio is used and H3 begins generating from that point. The
Context node anchors
the excerpt at frame zero automatically; do not also connect
`source_audio_slice` as a Ref2VA audio reference in this mode. Only scene 1
receives that exact opening excerpt. H3 generates the rest of scene 1 and later
scenes continue the generated audio through the configured audio-context frames.
Final assembly automatically uses the checkpointed generated soundtrack. VHS
lazy AUDIO values are accepted directly.

`Simple H3 Assemble Final Video.save_output` controls retention. Enabled keeps
the normal final video and recovery artifacts. Disabled assembles one temporary
preview, then removes that completed run's segments, checkpoints, prompts,
review media, manifest and permanent final. The temporary MP4 remains playable
until another temporary run finishes, at which point it is replaced. Failed
assemblies are never cleaned, so their checkpoints remain recoverable.

Use the two continuity nodes inside the recursive body:

1. Keep the original subject images in their existing Ref2VA slots. Connect
   `Simple H3 Cut Reference Sheet.continuity_sheet` to the next free reference
   slot and select that same tag on the node (`<Picture 2>`, `<Picture 3>`, etc.).
   Connect `augmented_prompt` to Ref2VA's prompt input and one clean subject
   image to `identity_image` as the scene-1 fallback.
2. Keep the full trimmed scene connected to `Simple H3 Save Scene + Checkpoint`.
3. Connect the saved segment and the trimmed audio to `Simple H3 Review — Approve / Retry / Reroll / Stop`, then connect its approved segment to Loop End. The review gate deliberately has no checkpoint-resume panel. Its synchronized preview automatically trims or pads sub-frame audio rounding differences without changing the saved scene.
   Enable `Continue` to display each completed segment without pausing for approval; the player receives the assembled final video when the chain finishes.
4. Also connect the full trimmed scene to `Simple H3 Select Continuity Frames`.
5. Connect `frames_for_loop_end` from that selector to `Simple H3 Loop Until
   Final Scene.images` instead of connecting the full scene directly.

On scene 1 the additional reference contains only the clean fallback image. For
every later scene it contains the final three consecutive frames of
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
