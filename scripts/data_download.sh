#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYRCAP_DATA_ROOT="${HYRCAP_DATA_ROOT:-${ROOT}/data/thumos14}"
export HF_ENDPOINT="https://huggingface.co"
DATA_REPO="${HYRCAP_DATA_REPO:-iurnehc/thumos14-hyrcap-features}"

python3 -B "${ROOT}/scripts/prepare_data.py" \
  --data-root "${HYRCAP_DATA_ROOT}" \
  --repo "${DATA_REPO}"
