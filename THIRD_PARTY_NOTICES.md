# Runtime attribution

The internal `stable_engine` contains adapted MiniMax H3 Context Loop runtime
and shared patch code from ComfyUI-MiniMaxH3-Contex-Loop, continuing the work of
NikoDemon80's ComfyUI-H3-Motion-Context. Upstream source:
https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop

Upstream copyright notice: Copyright (C) 2026 NikoDemon80.
The runtime is distributed under GNU GPL version 3; the full text is in LICENSE.
Local changes adapt the runtime to Simple H3 scene orchestration, output paths
and preview handling. It is bundled here and does not import the separate
Context Loop installation.

## Learned 2D latent upscaler

`third_party/lbh_latent_2d.py` bundles the MIT-licensed LBH-123-AI inference implementation from https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler at commit f34cadb19f10fee7e36091b690cac39b4eb40573. Copyright (c) 2026 LBH-123-AI. License: licenses/LBH-Latent-Upscaler-MIT.txt. The standalone registration is removed. Simple H3 supplies its own public node, AV/mask handling and metadata preservation. Model weights are separate and are not included.
