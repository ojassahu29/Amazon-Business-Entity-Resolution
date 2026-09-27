#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT}/../6ab10eb3b23ba_student_resource/student_resource/dataset}"
PYTHON_BIN="${PYTHON:-python3}"

exec "${PYTHON_BIN}" "${ROOT}/code/business_entity_resolution/src/autoresearch_bench.py" \
  --data-dir "${DATA_DIR}" \
  --seed 42 \
  --sample-size 5000
