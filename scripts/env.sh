#!/usr/bin/env bash
# Source this file; do not install or replace the hardware-specific PyTorch build.
HYRCAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYRCAP_RUNTIME_DIR="${HYRCAP_RUNTIME_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/hyrcap}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${HYRCAP_ROOT}:${HYRCAP_RUNTIME_DIR}/native${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -d /usr/local/PPU_SDK ]]; then
  export PPU_SDK=/usr/local/PPU_SDK
  export PPU_HOME=/usr/local/PPU_SDK
  export LD_LIBRARY_PATH="${PPU_SDK}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
