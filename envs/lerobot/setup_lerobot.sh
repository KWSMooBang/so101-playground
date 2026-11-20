#!/usr/bin/env bash
# Build the LeRobot environment (Python 3.12): lerobot (editable, pinned submodule) + SO-101 Bench plugin.
# Usage: envs/lerobot/setup_lerobot.sh    (from anywhere; idempotent)
# Result: envs/lerobot/.venv (gitignored)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv"

# feetech      SO-101 motors (serial bus)
# remote       async / remote inference (lerobot-policy-server; Zenoh + msgpack)
# core_scripts lerobot-record / -teleoperate / -calibrate / -replay (dataset + hardware + viz)
# evaluation   lerobot-eval (used by --env.type=so101_bench)
# training     lerobot-train
# smolvla      SmolVLA policy deps (ACT needs no extra)
LEROBOT_EXTRAS="feetech,remote,core_scripts,evaluation,training,smolvla"

[ -d "$VENV" ] || uv venv --python 3.12 --seed "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install -e "$ROOT/third_party/lerobot[${LEROBOT_EXTRAS}]"
uv pip install -e "$ROOT/third_party/so101_bench/integrations/lerobot"
uv pip install -e "$ROOT"

echo "[setup_lerobot] done. Next:  source envs/lerobot/env.sh"
