# Single-frame still nodes

`SimpleH3 Still Latent (single frame)` creates a one-frame H3 audiovisual latent.
It replaces `FizgigH3StillLatent`, including workflow nodes titled
`STILL AUDIO TEMPLATE — no extra sampling`. Width, height, batch size and the
LATENT output are unchanged. A 64×64 latent can supply only the audio template
through Separate AV Latent; no sampler or model is run by this node.

`SimpleH3 Still Decode` replaces `FizgigH3StillDecode`. For a single H3 latent
frame it replicates the denormalized latent into five temporal positions, uses
the VAE adaptive decode and keeps pixel frame index 3. Multiple-frame latents
and other VAEs use native decode. No extra learned model or persistent tensor
cache is added. ComfyUI owns inference mode and model memory management.

These are adaptations of Peter Neill's Fizgig H3 Still code (MIT, upstream
commit 10d5171). See `licenses/Fizgig-H3-Still-MIT.txt`. They do not import or
require the Fizgig package. Existing Fizgig workflows keep their original node
IDs until explicitly migrated; both packages can remain installed.

CPU checks: `python -m unittest discover -s tests -p test_still_nodes.py`.
Real image quality and GPU performance still require a generation test.
