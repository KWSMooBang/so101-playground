#!/usr/bin/env bash
# FLUX 3 Action (black-forest-labs/flux-action) env + weights for T004, after so101_bench
# scripts/FLUX3_fine_tuning/pod/setup_env.sh, trimmed to inference on one 24 GB GPU:
#   flux-action @ e2dd1d8 venv (Python 3.12, torch 2.10 cu128, NATTEN 0.21.6) + fastapi/uvicorn/lpips,
#   base VAE + text encoder (no trunk), the SO-101 package, the bench LoRA (raw update 3000) merged into it.
# Both black-forest-labs repos are gated: accept the licenses on the Hub with the account whose token is used.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$here"
[[ -d flux-action ]] || git clone https://github.com/black-forest-labs/flux-action.git
git -C flux-action checkout -q e2dd1d8dbc5977b54315d61f7548c63c043d6d4f
(cd flux-action && uv sync --locked --extra encoders --extra data --extra serve)
uv pip install --python flux-action/.venv/bin/python 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org/
uv pip install --python flux-action/.venv/bin/python fastapi uvicorn lpips
hf=flux-action/.venv/bin/hf
$hf download black-forest-labs/flux-3-action-base --revision 62878e2925e59b7a89ec14463ce89932624c490d \
    --include 'video_vae.safetensors' --include 'text_encoder/*' --include config.json --local-dir weights/base
$hf download black-forest-labs/flux-3-action-so101 --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849 \
    --exclude 'assets/*' --local-dir weights/so101
$hf download 5hadytru/so101_bench_flux-3-action-so101 --revision 07a7632e8f21ce0cc613233ae833a6b362748e6b \
    --include 'adapter_*' --include 'config.json' --include 'policy_*' --include README.md --include train_config.json \
    --local-dir weights/lora_raw3000
flux-action/.venv/bin/python "$repo/src/so101_playground/merge_flux3_lora.py" weights/lora_raw3000 weights/so101 weights/so101_bench_raw3000
