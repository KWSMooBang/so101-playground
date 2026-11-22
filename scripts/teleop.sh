#!/usr/bin/env bash
# Teleoperate the home-desktop SO-101: leader arm -> follower arm, with the wrist and overhead cameras.
# Usage: scripts/teleop.sh [extra lerobot-teleoperate args]     e.g.  scripts/teleop.sh --teleop_time_s=60
#
# Ctrl+C stops. On exit the follower's torque turns off and it goes limp, so put the leader in a resting
# pose (or hold the follower) before stopping.
#
# Joint units: use_degrees=false on both arms, i.e. LeRobot's RANGE_M100_100 (gripper 0..100), the same
# normalization the SO-101 Bench datasets were recorded with (before their fixed-calibration mapping).
# Device paths: scripts/so101_devices.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/envs/lerobot/env.sh"

# shellcheck disable=SC1091
source "$ROOT/scripts/so101_devices.sh"

exec lerobot-teleoperate \
    "${ARM_ARGS[@]}" \
    --fps=30 \
    --display_data=true \
    "$@"
