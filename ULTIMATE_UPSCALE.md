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

The learned upscaler itself is execution-local and released after each upscale;
the retention option concerns the H3 diffusion model, not the auxiliary upscaler.
No global memory-manager patches or permanent model caches are installed.

Restart ComfyUI to register the new nodes. No existing workflows are rewritten.
CPU contract tests mock sampling; they do not measure GPU speed or image quality.
