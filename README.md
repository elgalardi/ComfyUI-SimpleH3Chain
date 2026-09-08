# ComfyUI Simple H3 Chain

A focused MiniMax H3 clean-cut scene interface for the workflow used by SexyAI.

The native H3 model, LoRA, attention, Ref2Video, sampler and decode chain stays
unchanged. These nodes replace only the orchestration layer around it.

## Optional model LoRA loader

`Simple H3 Load LoRA — Optional` is a model-only loader intended for fixed API
workflows. Its `lora_name` menu begins with `None`; selecting it, or setting
`strength_model` to zero, returns the exact input `MODEL` unchanged. This lets a
workflow keep up to four LoRA loaders permanently connected while the frontend
activates only the requested slots. No CLIP or text-encoder patch is applied.

## Optional learned latent upscale

`Simple H3 Latent Upscale — Resolution` keeps the requested delivery size but,
when enabled, calculates an aligned first pass at approximately half width and
half height (25% of the final pixels). `Simple H3 Latent Upscale + Refine —
Optional` then applies LBH-123-AI's learned MiniMax H3 3D latent upscaler and a
short partial-denoise pass at the requested resolution. When disabled, both
nodes are exact pass-throughs and the original workflow is preserved.

For multi-scene chains, decode the `delivery_latent` but route
`context_latent` to the checkpoint and Loop End. This keeps Masked AV on one
stable low-resolution grid between scenes. The generated audio latent is never
refined: it is copied unchanged into the final-resolution delivery latent.

The two refinement contracts are intentionally separate nodes. `Simple H3
Latent Upscale + Refine — Optional` retains its bounded automatic 2–6-step
denoise pass. `Simple H3 Latent Upscale + Refine — Advanced` has no denoise
widget and instead exposes the full global schedule, `start_at_step`,
`end_at_step`, `add_noise`, and `return_with_leftover_noise` with native
KSampler Advanced semantics. For an 8-step plan, starting at step 3 evaluates
five high-resolution steps. Leave `end_at_step` at 10000 to finish the schedule
and remove leftover noise.

`Simple H3 Latent Upscale + Refine — Masked Continuity (Experimental)` keeps
the original Advanced node untouched. Connect `state` from Simple H3 Current
Scene. The first clip initializes a separate refined-resolution history; later
clips protect the previous refined 39-frame video tail during refinement. Audio
is not copied or masked by this second continuity layer. `audio_output` selects
either the untouched base audio (`original`) or the audio produced by the extra
sampling pass (`refined`). Continue routing
`delivery_latent` to decode and `context_latent` to Segment Save and Loop End.

`Simple H3 Final Latent Upscale — Windowed (Experimental)` is a separate
post-chain experiment. Connect the completed manifest from Loop End. It loads
the saved base-latent checkpoints only after every scene has finished, removes
the repeated 39-frame Masked AV heads and any H3 grid padding, rebuilds one
exact delivered AV timeline, and applies the learned 3D upscaler with temporal
windowing. It deliberately performs no second diffusion pass and preserves the
original audio latent. This keeps low-resolution Masked AV continuity entirely
independent from the final upscale and avoids per-scene upscale resets.
Because an arbitrary delivered duration does not always end on an H3 latent
token boundary, the upscale retains the smallest required terminal grid
padding. The recommended final sink, `Simple H3 Final Latent — Window Decode +
Assemble`, removes that padding while decoding instead of materializing the
complete movie as one IMAGE tensor.

The final sink decodes overlapping latent windows sequentially, drops the
repeated 39-frame head from every later window, writes temporary silent window
MP4s, and releases each decoded tensor before moving to the next window. It
decodes the much smaller audio timeline once to preserve whole-track level and
normalization, then concatenates the video parts and muxes that audio into the
final movie. The embedded player receives only the completed result.
`save_output=false` publishes a temporary final preview and cleans base recovery
artifacts after successful verification. Window parts are removed after a
successful assembly and retained on failure for diagnosis. This route replaces
the old full video decode, full audio decode, terminal trim, and VHS Combine
chain, so long outputs no longer require all decoded frames in memory at once.

`Simple H3 Base Preview — Auto Continue + Final` is the matching simplified
monitor for the low-resolution base chain. It has no approval/retry controls:
each saved scene appears automatically, execution continues, and the downstream
base assembler returns the complete base movie to the same player. Its filename
and `save_output` outputs directly control that assembler.

`Simple H3 Chain Plan.base_preview` is the master switch for the complete base
branch. Disabled, it decodes only the exact 39-frame Masked AV visual tail,
writes transactional latent checkpoints and prompt metadata, skips every base
MP4, and drives the scene-preview monitor in bypass mode. Enabled, it decodes
and displays each complete low-resolution scene. The refined workflow connects
the completed manifest directly to final windowed upscale/refinement; a separate
base-video assembler is not required.

`Simple H3 Final Latent Refine — Windowed Advanced (Experimental)` optionally
sits between the final windowed upscaler and the decoders. It exposes native
Advanced sampler controls (`add_noise`, schedule steps, `start_at_step`,
`end_at_step`, and leftover noise), processes AV-aligned temporal windows, and
protects the preceding refined 39-frame tail in every later window. The windows
are concatenated in latent space without a crossfade. `audio_output=original`
restores the untouched complete base-audio latent; `audio_output=refined` keeps
the audio sampled in every refinement window and removes its repeated 39-frame
span before joining. Start with a 243-frame window and a late 6→8 refinement
schedule.

Required optional dependency:
[`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler),
with a compatible model in `ComfyUI/models/latent_upscale_models`.

## Output layout

The plan's `output_name` is the run folder. New runs do not add an intermediate
`h3_chains` directory. For example, `output_name=sexyai` keeps the complete run
under `ComfyUI/output/sexyai`:

```text
output/sexyai/
|-- checkpoints/          # transactional latent and metadata checkpoints
|-- previews/
|   |-- base/             # optional low-resolution scene/base previews
|   `-- refined/          # internal refined windows or temporary refined final
|-- final/                # retained base/refined final videos
|-- frames/               # optional PNG export sequences
|-- source/               # copied source media when required by the plan
|-- reviews/              # review media when the interactive gate is used
|-- partial/              # optional partial manifests
|-- plan.json
|-- workflow.json
|-- api_prompt.json
`-- manifest.json
```

Folders that are not used by a workflow are not created. Scene preview files
replace their matching scene on reruns, while final videos keep versioned names.
The legacy `output/h3_chains` tree is left untouched and remains readable by the
gallery and checkpoint recovery endpoints.

## Clean-cut continuity mode

By default, Simple H3 Chain does not feed the previous scene latent into the
next scene. Every scene is a clean Ref2VA generation and therefore behaves like
a cinematic cut instead of a continuous camera take when `Simple H3 Context`
is set to `cut`. The opt-in `masked_av` mode described below is the sole direct
latent-continuation path. The same node can otherwise carry 1, 5, 11, 22, or 39
previous tail frames.
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

### Masked AV continuation (experimental)

`masked_av` is a lossless continuation path for **Continuous Story**. It copies
the final exact 39-frame video/audio run from the previous generated H3 latent
into the new latent, protects that prefix from denoising, and generates only
the future. This avoids both the decoded-frame guide reinterpretation and the
audio decode/re-encode round trip. `audio_feather_ticks=8` gives the audio edge
a short 0.2-second half-cosine release; use `0` only to compare a hard seam.

Connect the Context node's new `latent` output to the sampler latent input.
That output is a normal pass-through in `video`, `images`, and `cut_reference`,
so one connection works for every mode. The original conditioning output stays
connected as before. Scene 1 remains unchanged; imported scene-1 media uses the
established VAE guide because no original sampler latent exists yet.

Mode ownership is intentionally strict:

- `Continuous Story`: may use `masked_av` for the strongest same-take AV seam.
- `Cinematic Cuts`: `cut_reference` remains the automatic recommendation.
  `masked_av` may be selected manually as an experimental A/B test; its protected
  prefix can produce a strong seam or resist the requested camera change.
- `Reference Edit`: use `images` or the normal reference path so the supplied
  reference remains authoritative.
- `Edit`: use `cut`; the locked source plate remains authoritative.

For normal use, leave `context_type` on `director_auto`. The existing
Story Director → Chain Plan connection already carries `director_mode`, so no
additional cable is needed. It resolves to `video` for Continuous Story,
`cut_reference` for Cinematic Cuts, `images` for Reference Edit, and a clean
`cut` for Edit. Manual choices remain available for controlled A/B tests.

The implementation uses current ComfyUI native MiniMax H3 nested AV masks and
does not install or import any external Motion Context node pack. It requires
matching resolution, batch size 1, a target longer than 39 frames, and a recent
ComfyUI build with native MiniMax H3 AV-mask support.

The direct latent-tail technique and exact shared AV-boundary analysis were
informed by seitanism's GPL-3.0
[ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef).
Simple H3 keeps its own review, retry, checkpoint, per-scene reference routing,
and final-assembly engine; that external node pack is neither installed nor
required.

Scene durations are delivered durations for the normal head-overlap guide mode.
`masked_av` deliberately keeps its original lossless contract instead: the
selected duration is the raw H3 sampler length and every later scene removes
only its exact 39-frame protected prefix. This makes later delivered scenes
39 frames shorter, but avoids skipping newly generated bridge motion at a scene
boundary. Larger context windows in normal guide mode generate more raw frames
and take proportionally longer.

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
