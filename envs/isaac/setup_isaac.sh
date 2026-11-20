#!/usr/bin/env bash
# Build the Isaac environment (Python 3.11): Isaac Sim 5.1 (pip) + Isaac Lab v2.3.2 + so101_bench.
# Usage: envs/isaac/setup_isaac.sh        (from anywhere; idempotent)
# Result: envs/isaac/.venv, envs/isaac/IsaacLab (both gitignored)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv"
ISAACLAB="$HERE/IsaacLab"

ISAACSIM_VERSION="5.1.0"
# Isaac Lab v2.3.2 ships the `isaaclab` extension 0.54.2, the version the bench README tests.
ISAACLAB_TAG="v2.3.2"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"

export OMNI_KIT_ACCEPT_EULA=YES   # NVIDIA Omniverse EULA; needed for non-interactive runs

[ -d "$VENV" ] || uv venv --python 3.11 --seed "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com
uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

if [ ! -d "$ISAACLAB/.git" ]; then
    git clone https://github.com/isaac-sim/IsaacLab "$ISAACLAB"
fi
git -C "$ISAACLAB" fetch --tags --quiet
git -C "$ISAACLAB" checkout --quiet "$ISAACLAB_TAG"

# Isaac Lab extensions only; no RL frameworks (rl_games, rsl_rl, skrl, sb3).
"$ISAACLAB/isaaclab.sh" --install none

uv pip install -e "$ROOT/third_party/so101_bench/source/so101_bench"

echo "[setup_isaac] done. Next:"
echo "  source envs/isaac/env.sh"
echo "  python third_party/so101_bench/scripts/download_assets.py   # ~2 GB"
