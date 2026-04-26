#!/usr/bin/env bash
set -euo pipefail

# Run top-10 RAPTOR experiments. This is the named companion to the top-5
# wrapper so the two scenarios can live side by side.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

exec scripts/run_all_qwen_experiments.sh \
  --top-k 10 \
  --output-root raptor_10_runs \
  "$@"
