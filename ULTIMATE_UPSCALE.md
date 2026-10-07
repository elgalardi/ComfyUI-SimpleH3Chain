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
