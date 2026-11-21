#!/usr/bin/env bash
# T005 oracle-correction sweep: N parallel Isaac clients pull settings "<cond>:<alpha>" from a shared queue
# and run them against one LaWAM probe server (no replay; raw e_pred only). On a 24 GB card next to the server
# (~7.7 GB) only 2 clients fit (~6.5-7 GB each); a 3rd fails at Isaac startup.
#   GPU=1 scripts/run_t005_sweep.sh <port> <root> <workers> <num_episodes> <setting> [<setting> ...]
# setting = none | <gap>:<alpha>   e.g. gain2:0.5. Outputs <root>/lawam/<gap>-a<alpha>/ (none -> none/), logs
# <root>/lawam/<tag>.log. A setting whose episodes.jsonl already lists every episode skips them (resumable).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
port=$1 root=$2 workers=$3 n=$4
shift 4
queue="$root/lawam/.queue.$$"
mkdir -p "$root/lawam"
printf '%s\n' "$@" > "$queue"

next() {  # pop the first line of the queue under a lock
    exec 9>"$queue.lock"
    flock 9
    local line
    line=$(head -n1 "$queue")
    [ -n "$line" ] && sed -i 1d "$queue"
    flock -u 9
    echo "$line"
}

worker() {
    local w=$1 s cond alpha tag rc
    while s=$(next) && [ -n "$s" ]; do
        cond=${s%%:*}
        alpha=0
        tag=$cond
        if [ "$s" != "$cond" ]; then alpha=${s#*:}; tag="$cond-a$alpha"; fi
        for attempt in 1 2 3; do  # a failed Isaac startup (no PhysX scene / CUDA bad state) is retried
            echo "$(date +%T) [w$w] $tag start (attempt $attempt)"
            "$here/run_t004_eval.sh" lawam "$cond" "$root/lawam/$tag" "$port" --num_episodes "$n" \
                --oracle_alpha "$alpha" --run_tag "$tag" --token_stats >> "$root/lawam/$tag.log" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] && break
            grep -q "Failed to create simulation view\|_root_physx_view\|CUDA being in bad state" "$root/lawam/$tag.log" || break
            sleep 60
        done
        echo "$(date +%T) [w$w] $tag rc=$rc $(grep -c . "$root/lawam/$tag/episodes.jsonl" 2>/dev/null) episodes," \
             "$(grep -c '"success": true' "$root/lawam/$tag/episodes.jsonl" 2>/dev/null) success"
    done
}

for w in $(seq 1 "$workers"); do
    worker "$w" &
    sleep 20  # stagger Isaac startups
done
wait
rm -f "$queue" "$queue.lock"
echo "$(date +%T) [sweep] done"
