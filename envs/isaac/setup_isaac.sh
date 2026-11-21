#!/usr/bin/env bash
# Build the Isaac environment (Python 3.11): Isaac Sim 5.1 (pip) + Isaac Lab v2.3.2 + so101_bench.
# Usage: envs/isaac/setup_isaac.sh        (from anywhere; idempotent)
# Result: envs/isaac/.venv (~16 GB), envs/isaac/IsaacLab (both gitignored)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv"
ISAACLAB="$HERE/IsaacLab"

ISAACSIM_VERSION="5.1.0"
# Isaac Lab v2.3.2 ships the `isaaclab` extension 0.54.2, the version the bench README tests.
ISAACLAB_TAG="v2.3.2"

export OMNI_KIT_ACCEPT_EULA=YES   # NVIDIA Omniverse EULA; needed for non-interactive runs

[ -d "$VENV" ] || uv venv --python 3.11 --seed "$VENV"
export VIRTUAL_ENV="$VENV"

# Brings its own torch 2.7.0 (CUDA 12) and numpy 1.26.0.
uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com

if [ ! -d "$ISAACLAB/.git" ]; then
    git clone https://github.com/isaac-sim/IsaacLab "$ISAACLAB"
fi
git -C "$ISAACLAB" fetch --tags --quiet
git -C "$ISAACLAB" checkout --quiet "$ISAACLAB_TAG"

# Install the Isaac Lab extensions directly instead of `isaaclab.sh --install`, which
#   - fails to build flatdict==4.0.1 with current setuptools (no pkg_resources) and still exits 0,
#     leaving `isaaclab` itself uninstalled, and
#   - reinstalls torch and pulls numpy 2.x, which Isaac Sim 5.1 does not support.
# constraints.txt keeps Isaac Sim's torch/numpy; build-constraints.txt pins setuptools for sdists.
# No RL frameworks (rl_games, rsl_rl, skrl, sb3) are installed.
SRC="$ISAACLAB/source"
uv pip install --build-constraint "$HERE/build-constraints.txt" -c "$HERE/constraints.txt" \
    -e "$SRC/isaaclab" -e "$SRC/isaaclab_assets" -e "$SRC/isaaclab_tasks" \
    -e "$SRC/isaaclab_rl" -e "$SRC/isaaclab_mimic" -e "$SRC/isaaclab_contrib" \
    -e "$ROOT/third_party/so101_bench/source/so101_bench"

# Check the install without importing the extensions: isaaclab_tasks and so101_bench import omni/pxr
# modules that exist only inside a running Kit app. scripts/isaac_smoke.py is the runtime check.
"$VENV/bin/python" -c "
import importlib.util, numpy, torch
missing = [m for m in ('isaacsim', 'isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'so101_bench')
           if importlib.util.find_spec(m) is None]
assert not missing, f'not installed: {missing}'
assert numpy.__version__.startswith('1.'), numpy.__version__
print('torch', torch.__version__, 'numpy', numpy.__version__)"

echo "[setup_isaac] done. Next:"
echo "  source envs/isaac/env.sh"
echo "  python third_party/so101_bench/scripts/download_assets.py   # ~2 GB"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/isaac_smoke.py --headless   # runtime check"
