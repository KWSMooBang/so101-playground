#!/usr/bin/env bash
# T007 sweep: N parallel Isaac clients (GPU ${GPU:-1}) pull jobs from a queue and run them against one policy
# server. A job is "<cond>[:<alpha>]@<offset>+<count>": episodes offset..offset+count-1 of
# tasks/home_objects.jsonl, written to <root>/<model>/<tag>/ (tag = cond or cond-a<alpha>); jobs of one tag
# cover disjoint episodes and append to the same episodes.jsonl. Episodes already listed there are skipped,
# so a sweep can be resumed.
#   GPU=1 EXTRA="--replay_episodes 99 --replay_every 2 --replay_mode nominal" \
#       scripts/run_t007_sweep.sh <model> <port> <root> <workers> <job> [<job> ...]
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
model=$1 port=$2 root=$3 workers=$4
shift 4
EXTRA=${EXTRA:-}
export EPISODES=${EPISODES:-$here/../tasks/home_objects.jsonl}
export LAYOUTS=${LAYOUTS:-$here/../tasks/layouts/home_objects_layouts.jsonl}
queue="$root/$model/.queue.$$"
mkdir -p "$root/$model"
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
    local w=$1 job s range off cnt cond alpha tag rc log
    while job=$(next) && [ -n "$job" ]; do
        s=${job%%@*} range=${job#*@}
        off=${range%%+*} cnt=${range#*+}
        cond=${s%%:*} alpha=0 tag=$cond
        if [ "$s" != "$cond" ]; then alpha=${s#*:}; tag="$cond-a$alpha"; fi
        log="$root/$model/$tag.o$off.log"
        for attempt in 1 2 3; do  # a failed Isaac startup (no PhysX scene / CUDA bad state) is retried
            echo "$(date +%T) [w$w] $tag @$off+$cnt start (attempt $attempt)"
            # shellcheck disable=SC2086
            "$here/run_t004_eval.sh" "$model" "$cond" "$root/$model/$tag" "$port" --episode_offset "$off" \
                --num_episodes "$cnt" --oracle_alpha "$alpha" --run_tag "$tag" $EXTRA >> "$log" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] && break
            grep -q "Failed to create simulation view\|_root_physx_view\|CUDA being in bad state" "$log" || break
            sleep 60
        done
        echo "$(date +%T) [w$w] $tag @$off+$cnt rc=$rc; $tag total $(grep -c . "$root/$model/$tag/episodes.jsonl" 2>/dev/null)" \
             "episodes, $(grep -c '"success": true' "$root/$model/$tag/episodes.jsonl" 2>/dev/null) success"
    done
}

for w in $(seq 1 "$workers"); do
    worker "$w" &
    sleep 20  # stagger Isaac startups
done
wait
rm -f "$queue" "$queue.lock"
echo "$(date +%T) [sweep] done"
