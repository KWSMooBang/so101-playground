#!/usr/bin/env bash
# T009: one Isaac worker (scripts/t009_worker.py) on GPU ${GPU:-1}, serving the file queue <root>/queue.
#   GPU=1 scripts/run_t009_worker.sh <port> <root> <worker id> [extra args, e.g. --max_ticks 120]
# A failed Isaac startup (no PhysX scene / CUDA bad state) is retried up to 3 times; jobs a dead worker held
# are requeued by the next start with the same worker id.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-/workspace/so101-playground/.claude/worktrees/scaffold-env}"
port=$1 root=$2 wid=$3
shift 3
# shellcheck disable=SC1091
source "$ISAAC_ENV/envs/isaac/env.sh"
mkdir -p "$root/logs"
cd "$here"
until curl -sf "http://127.0.0.1:$port/act" > /dev/null; do sleep 10; done  # server still loading
for attempt in 1 2 3 4 5; do
    echo "$(date +%T) [$wid] start (attempt $attempt)" >> "$root/logs/$wid.log"
    CUDA_VISIBLE_DEVICES="${GPU:-1}" python scripts/t009_worker.py --headless --queue "$root/queue" \
        --out_dir "$root/episodes" --worker "$wid" --policy_port "$port" \
        --t004_layouts "$SO101_BENCH_ROOT/tasks/layouts/val_set_layouts.jsonl" "$@" >> "$root/logs/$wid.log" 2>&1
    rc=$?
    echo "$(date +%T) [$wid] exit rc=$rc" >> "$root/logs/$wid.log"
    [ -e "$root/queue/STOP" ] && [ -z "$(ls -A "$root/queue/pending" 2>/dev/null)" ] && break
    sleep 30
done
