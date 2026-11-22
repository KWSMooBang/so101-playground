#!/usr/bin/env bash
# Run scripts/wam_gap_eval.py from the isaac env on one GPU.
#   ISAAC_ENV=<checkout with envs/isaac/.venv> GPU=1 scripts/run_t004_eval.sh <model> <condition> <out_dir> <port> [extra args...]
# Episodes: tasks/T004_bin1.jsonl (val_set single-object Bin rows, rigid single-body objects) with the
# bench's fixed val_set layouts. Override with EPISODES=<jsonl> LAYOUTS=<jsonl> (T007: tasks/home_objects.jsonl).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-/workspace/so101-playground/.claude/worktrees/scaffold-env}"
model=$1 cond=$2 out=$3 port=$4
shift 4
# shellcheck disable=SC1091
source "$ISAAC_ENV/envs/isaac/env.sh"
mkdir -p "$out"
cd "$here"
CUDA_VISIBLE_DEVICES="${GPU:-1}" exec python scripts/wam_gap_eval.py --headless --model "$model" --condition "$cond" \
    --episodes_jsonl "${EPISODES:-tasks/T004_bin1.jsonl}" \
    --layouts_jsonl "${LAYOUTS:-$SO101_BENCH_ROOT/tasks/layouts/val_set_layouts.jsonl}" \
    --out_dir "$out" --policy_port "$port" "$@"
