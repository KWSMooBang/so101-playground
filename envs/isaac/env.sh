# Source this to use the Isaac environment:  source envs/isaac/env.sh
# Works in bash and zsh.
if [ -n "${BASH_SOURCE:-}" ]; then _self="${BASH_SOURCE[0]}"; else _self="${(%):-%x}"; fi
_here="$(cd "$(dirname "$_self")" && pwd)"

if ! source "$_here/.venv/bin/activate"; then
    echo "env.sh: no venv at $_here/.venv (run $_here/setup_isaac.sh)" >&2
    unset _self _here
    return 1
fi
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH="$_here/IsaacLab"
export ISAACLAB_SH="$_here/IsaacLab/isaaclab.sh"
export SO101_BENCH_ROOT="$(cd "$_here/../../third_party/so101_bench" && pwd)"
unset _self _here
if [ -n "${BASH_SOURCE:-}" ]; then _self="${BASH_SOURCE[0]}"; else _self="${(%):-%x}"; fi
. "$(cd "$(dirname "$_self")" && pwd)/vulkan.sh"; unset _self
