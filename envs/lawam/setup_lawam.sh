#!/usr/bin/env bash
# LaWAM env + weights for T004 (src/so101_playground/lawam_probe_server.py).
# The shared lerobot env lacks the `lawam` extra (diffusers), so LaWAM gets its own venv:
# Python 3.12, third_party/lerobot (v0.7.0, has lerobot.policies.lawam) with [lawam].
#   LEROBOT_SRC defaults to this checkout's submodule; pass another checkout's if the submodule is not initialized.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_SRC="${LEROBOT_SRC:-$here/../../third_party/lerobot}"
cd "$here"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "$LEROBOT_SRC[lawam]"
hf=.venv/bin/hf
# Bench LaWAM full fine-tune, update 45000 (public; 21/39 on the bench's validation set).
$hf download 5hadytru/so101_bench_LaWAM_full_fine_tune --revision 8df458f038f78e0eb02d050fff783f24ed4dabda \
    --local-dir weights/so101_bench_LaWAM
# The VLM is built from the Hub id before the checkpoint overwrites it; prefetch it into HF_HOME.
$hf download Qwen/Qwen3-VL-2B-Instruct
