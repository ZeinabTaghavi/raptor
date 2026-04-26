#!/usr/bin/env bash
set -euo pipefail

# Run top-5 RAPTOR experiments. Use --dataset to run one or more datasets:
#   scripts/run_raptor_5_experiments.sh --dataset novelhopqa
#   scripts/run_raptor_5_experiments.sh --dataset qasper loogle quality --resume

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=2,3

DATASETS=()
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "--dataset requires at least one dataset name" >&2
        exit 2
      fi

      while [[ $# -gt 0 && "$1" != --* ]]; do
        IFS=',' read -r -a dataset_items <<< "$1"
        for dataset in "${dataset_items[@]}"; do
          if [[ -n "${dataset}" ]]; then
            DATASETS+=("${dataset}")
          fi
        done
        shift
      done
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  exec scripts/run_all_qwen_experiments.sh \
    --top-k 5 \
    --output-root raptor_5_runs \
    "${FORWARDED_ARGS[@]}"
fi

for dataset in "${DATASETS[@]}"; do
  scripts/run_all_qwen_experiments.sh \
    --top-k 5 \
    --output-root raptor_5_runs \
    --dataset "${dataset}" \
    "${FORWARDED_ARGS[@]}"
done
