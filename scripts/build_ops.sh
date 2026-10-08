#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export MAX_JOBS="${MAX_JOBS:-4}"
TASK_BUILD="${HYRCAP_RUNTIME_DIR}/build-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${HYRCAP_RUNTIME_DIR}/native" "${TASK_BUILD}"
cd "${HYRCAP_ROOT}/util"
python3 -B setup.py build_ext --build-lib "${HYRCAP_RUNTIME_DIR}/native" --build-temp "${TASK_BUILD}/nms"
cd "${HYRCAP_ROOT}/models/hyrcap/ops"
python3 -B setup.py build_ext --build-lib "${HYRCAP_RUNTIME_DIR}/native" --build-temp "${TASK_BUILD}/attention"
cd "${HYRCAP_ROOT}"
python3 -B -c 'from models.hyrcap.ops.functions.ms_deform_attn_func import MSDA; from util import nms; print("Compiled:", MSDA.__file__, nms.nms_1d_cpu.__file__)'
