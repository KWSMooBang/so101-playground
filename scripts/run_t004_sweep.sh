#!/usr/bin/env bash
# One T004 sweep: the same episodes under each condition, sequentially, against a running policy server.
#   GPU=1 scripts/run_t004_sweep.sh <model> <port> <root> "<cond1,cond2,...>" <num_episodes> <replay_episodes> [extra eval args]
# Outputs <root>/<model>/<cond>/ (episodes.jsonl, ep*.npz, frames, predicted/); logs <root>/<model>/<cond>.log.
# A condition whose episodes.jsonl already lists every episode is skipped (resumable).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
model=$1 port=$2 root=$3 conds=$4 n=$5 replay=$6
failed=0
shift 6
for cond in ${conds//,/ }; do
    out="$root/$model/$cond"
    mkdir -p "$out"
    echo "$(date +%T) [sweep] $model $cond start"
    "$here/run_t004_eval.sh" "$model" "$cond" "$out" "$port" --num_episodes "$n" --replay_episodes "$replay" "$@" \
        > "$root/$model/$cond.log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || failed=$((failed + 1))
    echo "$(date +%T) [sweep] $model $cond rc=$rc $(grep -c . "$out/episodes.jsonl" 2>/dev/null) episodes, $(grep -c '"success": true' "$out/episodes.jsonl" 2>/dev/null) success"
done
echo "$(date +%T) [sweep] done, $failed failed condition(s)"
exit $(( failed > 0 ))
