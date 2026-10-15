# Simple H3 Ultimate Upscale

Local H3-only adaptation of PlagueKind's MMH3 Ultimate Upscale (MIT; attribution
in `licenses/PlagueKind-UltimateUpscale-MIT.txt`). It does not import or modify
the PlagueKind package, and contains no LTX nodes.

Replace the original main node with **Simple H3 Ultimate Upscale** and reconnect
the same inputs/outputs. Existing MMH3 parameter nodes remain compatible through
the same H3 parameter socket types. Independent Simple H3 parameter nodes are
also included for interpolation, learned upscale, temporal split and spatial split.

`keep_model_loaded` defaults to `true`: omit explicit H3 unloading before learned
GPU upscaling and after refinement. `false` restores the original H3 unload
policy. This does not pin H3 in VRAM or disable ComfyUI memory management: later
VAE decoding and memory pressure can still evict it. Keeping both H3 and a learned
upscaler resident can increase peak VRAM; turn this option off if necessary.

The learned upscaler itself is execution-local. Its CPU weights are reused across
temporal chunks within one node execution, avoiding repeated disk reads and model
construction, then released when the execution finishes. It is moved back to CPU
after each learned upscale so it is not pinned in VRAM alongside H3 sampling;
the retention option concerns the H3 diffusion model, not the auxiliary upscaler.
No global memory-manager patches or permanent model caches are installed.

Restart ComfyUI to register the new nodes. No existing workflows are rewritten.
CPU contract tests mock sampling; they do not measure GPU speed or image quality.

The node progress bar accumulates all temporal chunks, spatial tiles and sampling
steps instead of restarting per tile. It reserves progress for upscale and stitching
and reaches 100% only when outputs are ready. This measures work stages, not elapsed
time or an ETA; model initialization and learned upscale can hold the bar still.
Native latent previews remain enabled according to ComfyUI's preview setting.

## Advanced sampling controls

`Simple H3 Ultimate Upscale — Advanced` is a separate node; the original node's
schema and wired noise/sampler/sigmas inputs are unchanged. Advanced replaces
those three wires with `noise_seed`, `add_noise`, `steps`, `sampler_name`,
`scheduler`, `cfg`, `start_at_step`, `end_at_step` and `return_with_leftover_noise`.
Reconnect model, conditioning, latent, optional negative, parameter nodes and
state as before. Output indices, audio preservation and protected HQ prefix are
the same. CFG requires negative conditioning, as in the original upscale node.

The native KSampler builds the full schedule, including sampler-specific sigma
handling. Start/end select an interval in that schedule, not a denoise fraction.
Defaults are 8 total steps, start 6, end 10000 (the schedule's end), LCM/beta,
CFG 1, noise enabled and leftover noise disabled. These defaults are a test
starting point, not an automatic conversion of the existing wired sigma recipe.
For comparison, keep total steps fixed and change only start, e.g. 5, 6, 7.
An empty interval performs upscale/prefix protection but no sampling.

With leftover noise disabled, an early end is forced to sigma zero, like
KSampler Advanced. Enable preserves the selected ending sigma: the result and
saved HQ tail can still be noisy, so disable it for final decode/continuation.
Noise disable is honored in full-frame, joint tiled and sequential tiled paths.
Existing joint tiling still samples each sigma pair separately; multistep solver
history does not span those calls. Use no spatial split or `joint_steps=false`
when testing a sampler's complete trajectory. No GPU quality benchmark is implied.

## Protected refined scene continuity

Connect Current Scene `state` to Ultimate Upscale's optional `state` input in a
Masked AV chain. Scene 1 initializes the refined context. On later scenes the
previous refined 39-frame tail replaces the beginning of the upscaled latent.
That prefix is masked during both joint and sequential spatial sampling and
restored after every joint step, tile result, and temporal stitch. Audio remains
the original base audio latent; refinement does not generate a new soundtrack.

The original three outputs keep their indices. The appended `context_latent`
output carries the original base latent plus a cloned compact refined tail
(12 video tokens and 65 audio ticks for 39 frames). Connect it to both Segment
Save and Loop End's `sampled_latent` inputs. Continue decoding the first output
at delivery resolution. Checkpoints persist the HQ tail for resume, not a second
full HQ latent sequence.

The chain plan fingerprints the connected refined-context contract. Use a new
project folder rather than mixing old checkpoints that have no refined tail.
Keep dimensions and model recipe unchanged when resuming.

The HQ decoder receives the protected overlap before the repeated head frames
are trimmed. Native VAE temporal windows remain active. Exact latent-prefix
preservation does not guarantee an imperceptible generated-motion or decoded
pixel boundary; real two-scene render comparisons are still required.
