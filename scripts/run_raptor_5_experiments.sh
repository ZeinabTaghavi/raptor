#!/usr/bin/env bash
set -euo pipefail

# Run top-5 RAPTOR experiments. Use --dataset to run one dataset at a time:
#   scripts/run_raptor_5_experiments.sh --dataset novelhopqa
#   scripts/run_raptor_5_experiments.sh --dataset qasper --resume

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

exec scripts/run_all_qwen_experiments.sh \
  --top-k 5 \
  --output-root raptor_5_runs \
  "$@"
