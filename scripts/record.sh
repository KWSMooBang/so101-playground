#!/usr/bin/env bash
# Record teleoperated episodes (leader -> follower, wrist + overhead cameras) as a LeRobot dataset in the
# SO-101 Bench datasets' format: LeRobot v3.0, 30 fps, AV1 480x640, observation.images.{wrist,overhead},
# 6-D .pos state/action with use_degrees=false.
#
# Usage: scripts/record.sh NAME "TASK" [extra lerobot-record args]
#   e.g. scripts/record.sh pilot "Place the white pen in the bin" --dataset.num_episodes=3
#
# Writes outputs/datasets/NAME (local only, no Hub upload). If NAME exists, new episodes are appended to it
# (--resume=true), so a dataset can be recorded over several sessions and with several tasks.
#
# During recording (global keys, any window): Right arrow = end the episode (or the reset) early,
# Left arrow = discard and re-record the episode, Esc = stop and save. Each episode is followed by a reset
# phase (teleop on, nothing recorded) except the last. Start and end each episode in the rest pose;
# on exit the follower's torque turns off and it goes limp.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
NAME=$1
TASK=$2
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/envs/lerobot/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/so101_devices.sh"

DATASET="$ROOT/outputs/datasets/$NAME"
RESUME=false
if [ -e "$DATASET/meta/info.json" ]; then
    RESUME=true
fi

# lerobot parses CLI values as YAML ("a: b" becomes a dict, "yes" becomes True), so pass the task as a
# double-quoted YAML string.
TASK_YAML=${TASK//\\/\\\\}
TASK_YAML="\"${TASK_YAML//\"/\\\"}\""

echo "dataset: $DATASET (resume=$RESUME)"
echo "task:    $TASK"
if [ -t 0 ]; then
    read -r -p "Leader in the rest pose? Enter starts episode recording right away (Ctrl+C cancels) "
fi

# The rerun viewer freezes the loop within about a minute (README), so no live display.
exec lerobot-record \
    "${ARM_ARGS[@]}" \
    --dataset.repo_id="local/$NAME" \
    --dataset.root="$DATASET" \
    --dataset.no_stamp=true \
    --dataset.single_task="$TASK_YAML" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=false \
    --resume="$RESUME" \
    "$@"
