#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYRCAP_DATA_ROOT="${HYRCAP_DATA_ROOT:-${ROOT}/data/thumos14}"
export HYRCAP_OUTPUT_ROOT="${HYRCAP_OUTPUT_ROOT:-${ROOT}/../hyrcap-run-outputs}"

source "${ROOT}/scripts/env.sh"
python3 -B "${ROOT}/scripts/run_pipeline.py" --root "${ROOT}"
