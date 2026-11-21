#!/usr/bin/env bash
# T006 sweep: N parallel Isaac clients pull settings "<cond>[:<alpha>]" from a shared queue and run them against
# one InternW0 probe server (src/so101_playground/internw0_probe_server.py). Sessions on the server are keyed by
# the client's out dir, so parallel clients do not share visual memory.
#   GPU=1 EXTRA="--replay_episodes 8 --replay_every 2 --replay_mode nominal" \
#       scripts/run_t006_sweep.sh <port> <root> <workers> <num_episodes> <setting> [<setting> ...]
# Outputs <root>/internw0/<tag>/ (tag = cond, or cond-a<alpha>), logs <root>/internw0/<tag>.log. A setting whose
# episodes.jsonl already lists every episode skips them (resumable).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
port=$1 root=$2 workers=$3 n=$4
shift 4
EXTRA=${EXTRA:-}
queue="$root/internw0/.queue.$$"
mkdir -p "$root/internw0"
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
            # shellcheck disable=SC2086
            "$here/run_t004_eval.sh" internw0 "$cond" "$root/internw0/$tag" "$port" --num_episodes "$n" \
                --oracle_alpha "$alpha" --run_tag "$tag" $EXTRA >> "$root/internw0/$tag.log" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] && break
            grep -q "Failed to create simulation view\|_root_physx_view\|CUDA being in bad state" "$root/internw0/$tag.log" || break
            sleep 60
        done
        echo "$(date +%T) [w$w] $tag rc=$rc $(grep -c . "$root/internw0/$tag/episodes.jsonl" 2>/dev/null) episodes," \
             "$(grep -c '"success": true' "$root/internw0/$tag/episodes.jsonl" 2>/dev/null) success"
    done
}

for w in $(seq 1 "$workers"); do
    worker "$w" &
    sleep 20  # stagger Isaac startups
done
wait
rm -f "$queue" "$queue.lock"
echo "$(date +%T) [sweep] done"
