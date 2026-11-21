#!/usr/bin/env bash
# InternW0-Δ inference env + weights for T006 (src/so101_playground/internw0_probe_server.py).
# Inference only: no training extras, no dataset, no InternW0-Delta-Base, no Wan2.2 DiT weights (the task
# config sets skip_dit_load_from_pretrain=true; the SO-101 checkpoint holds the whole video + action expert).
# Pins follow the bench's scripts/InternW0_fine_tuning/pod/setup_env.sh.
#   envs/internw0/setup_internw0.sh            (~31 GB of weights under envs/internw0/weights)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
REPO_REV=90801baa3bdc7829c1e4989edfe3d0fb2410ea6a
CKPT_REV=75e60ae81e9037df52d1ca0c00eb22d0471017e5   # 5hadytru/so101_bench_InternW0-Delta (official step 18200 at root)
WAN22_REV=921dbaf3f1674a56f47e83fb80a34bac8a8f203e
WAN21_REV=37ec512624d61f7aa208f7ea8140a131f93afc9a
RYNN_REV=36069aa841fc58f500d307316cbfdf379777ce8a
CK=$here/weights/checkpoints   # = WAM_CHECKPOINT_ROOT; the repo resolves model ids under it

if [ ! -d InternW0-Delta ]; then
    curl -sL "https://github.com/InternRobotics/InternW0-Delta/archive/$REPO_REV.tar.gz" | tar xz
    mv "InternW0-Delta-$REPO_REV" InternW0-Delta
fi
if [ ! -x InternW0-Delta/.venv/bin/python ]; then
    uv venv --python 3.11 InternW0-Delta/.venv
    uv pip install --python InternW0-Delta/.venv/bin/python torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128
    (cd InternW0-Delta && uv pip install --python .venv/bin/python -e '.[modelscope]' fastapi uvicorn)
fi
PY=$here/InternW0-Delta/.venv/bin/python
HF=$here/InternW0-Delta/.venv/bin/hf

# SO-101 fine-tune: weights + the run's resolved config + normalization stats (12.4 GB).
$HF download 5hadytru/so101_bench_InternW0-Delta --revision $CKPT_REV \
    --include model.pt --include config.yaml --include dataset_stats.json --local-dir "$here/weights/so101_bench_InternW0-Delta"
# Wan2.2: config JSONs only (no DiT weights).
$HF download Wan-AI/Wan2.2-TI2V-5B --revision $WAN22_REV --include '*.json' --local-dir "$CK/Wan-AI/Wan2.2-TI2V-5B"
# Converted T5 encoder + Wan2.2 VAE (ModelScope only; 11.4 + 1.4 GB).
$PY -c "from modelscope import snapshot_download; snapshot_download('DiffSynth-Studio/Wan-Series-Converted-Safetensors', allow_patterns=['models_t5_umt5-xxl-enc-bf16.safetensors', 'Wan2.2_VAE.safetensors'], local_dir='$CK/DiffSynth-Studio/Wan-Series-Converted-Safetensors')"
# UMT5 tokenizer only.
$HF download Wan-AI/Wan2.1-T2V-1.3B --revision $WAN21_REV --include 'google/umt5-xxl/*' --local-dir "$CK/Wan-AI/Wan2.1-T2V-1.3B"
# RynnBrain VLM (4.4 GB).
$HF download Alibaba-DAMO-Academy/RynnBrain1.1-2B --revision $RYNN_REV --local-dir "$CK/RynnBrain1.1-2B"
du -sh "$here/weights"/* "$CK"/*
