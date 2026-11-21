#!/usr/bin/env bash
# T004 video pass (LaWAM): 1 episode per condition, first --max_ticks ticks, replay at every query, frames and
# LAM features saved; then one mp4 per condition and the observed-vs-ideal feature distance (e_exec).
#   GPU=1 scripts/run_t004_videos.sh <port> <root> "<cond1,...>" [max_ticks]
# Needs a lawam_probe_server.py that supports return_features.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
port=$1 root=$2 conds=$3 ticks=${4:-360}
py="${VIDEO_PY:-$here/envs/lawam/.venv/bin/python}"   # numpy + PIL + av (envs/lawam/setup_lawam.sh installs av)
vids="$root/lawam_videos"
mkdir -p "$vids"
for cond in ${conds//,/ }; do
    out="$vids/$cond"
    echo "$(date +%T) [videos] $cond"
    "$here/scripts/run_t004_eval.sh" lawam "$cond" "$out" "$port" --run_tag "${cond}_vid" --num_episodes 1 \
        --replay_episodes 1 --replay_every 1 --save_frames --max_ticks "$ticks" > "$vids/$cond.log" 2>&1 \
        || echo "eval failed for $cond"
    "$py" "$here/src/so101_playground/t004_videos.py" "$out" "$vids/T004-$cond.mp4" \
        --scores "$root/lawam/lam_scores.jsonl" --tag "${cond}_vid" || echo "video failed for $cond"
done
"$py" "$here/src/so101_playground/t004_videos.py" --exec_summary "$vids" > "$vids/e_exec.json"
cat "$vids/e_exec.json"
