#!/usr/bin/env bash
# Start the InternW0-Δ probe server (T006) on one GPU.
#   GPU=0 scripts/run_internw0_server.sh <port> <log_dir> [extra args, e.g. --save_latents_dir DIR]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
W=$here/envs/internw0
port=$1 log_dir=$2
shift 2
mkdir -p "$log_dir"
cd "$here"
CUDA_VISIBLE_DEVICES="${GPU:-0}" WAM_CHECKPOINT_ROOT=$W/weights/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 \
    exec "$W/InternW0-Delta/.venv/bin/python" src/so101_playground/internw0_probe_server.py \
    --repo "$W/InternW0-Delta" --checkpoint "$W/weights/so101_bench_InternW0-Delta/model.pt" \
    --train_config "$W/weights/so101_bench_InternW0-Delta/config.yaml" \
    --stats "$W/weights/so101_bench_InternW0-Delta/dataset_stats.json" \
    --port "$port" --log_dir "$log_dir" --device cuda:0 "$@"
