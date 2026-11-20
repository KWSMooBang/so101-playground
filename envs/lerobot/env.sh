# Source this to use the LeRobot environment:  source envs/lerobot/env.sh
# Works in bash and zsh. Also exports what the so101_bench plugin needs to start the simulator.
if [ -n "${BASH_SOURCE:-}" ]; then _self="${BASH_SOURCE[0]}"; else _self="${(%):-%x}"; fi
_here="$(cd "$(dirname "$_self")" && pwd)"

source "$_here/.venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export SO101_BENCH_ROOT="$(cd "$_here/../../third_party/so101_bench" && pwd)"
export ISAACLAB_SH="$(cd "$_here/../isaac" && pwd)/IsaacLab/isaaclab.sh"
export SO101_BENCH_ISAAC_VENV="$(cd "$_here/../isaac" && pwd)/.venv"
unset _self _here
if [ -n "${BASH_SOURCE:-}" ]; then _self="${BASH_SOURCE[0]}"; else _self="${(%):-%x}"; fi
. "$(cd "$(dirname "$_self")/../isaac" && pwd)/vulkan.sh"; unset _self
