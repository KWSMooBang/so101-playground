# Sourced by envs/isaac/env.sh and envs/lerobot/env.sh.
# Our container mounts the NVIDIA driver libraries (libEGL_nvidia, rtcore, optix) but not the Vulkan
# ICD manifest, so the Vulkan loader only sees mesa and Isaac Sim finds no RTX GPU. If the system has
# no NVIDIA manifest, point the loader at ours (no root, no driver install).
if [ -n "${BASH_SOURCE:-}" ]; then _vk_self="${BASH_SOURCE[0]}"; else _vk_self="${(%):-%x}"; fi
_vk_here="$(cd "$(dirname "$_vk_self")" && pwd)"
if [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ ! -e /etc/vulkan/icd.d/nvidia_icd.json ] \
    && [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
    export VK_DRIVER_FILES="$_vk_here/nvidia_icd.json"
    export VK_ICD_FILENAMES="$_vk_here/nvidia_icd.json"   # name used by older loaders
fi
unset _vk_self _vk_here
