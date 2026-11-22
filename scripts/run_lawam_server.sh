#!/usr/bin/env bash
# Start the LaWAM probe server (T004) on one GPU.
#   GPU=0 LAWAM_ENV=<dir with .venv and weights/> scripts/run_lawam_server.sh <port> <log_dir>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
L=${LAWAM_ENV:-$here/envs/lawam}
port=$1 log_dir=$2
shift 2
mkdir -p "$log_dir"
cd "$here"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${GPU:-0}" exec "$L/.venv/bin/python" src/so101_playground/lawam_probe_server.py \
    --repo_id "$L/weights/so101_bench_LaWAM" --port "$port" --log_dir "$log_dir" "$@"
