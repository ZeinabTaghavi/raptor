#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-/mnt/cache/taghavi}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NOVELHOPQA_BOOKS_ROOT="${NOVELHOPQA_BOOKS_ROOT:-../passing_meta_tag/novelhopqa/book-corpus-root}"
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"

declare -A DATASET_CONFIGS=(
  [qasper]="configs/raptor/qasper_retrieval_ablation.yaml"
  [loogle]="configs/raptor/loogle_retrieval_ablation.yaml"
  [narrativeqa]="configs/raptor/nqa_retrieval_ablation.yaml"
  [quality]="configs/experiments/quality_retrieval_ablation.yaml"
  [novelhopqa]="configs/experiments/novelhopqa_retrieval_ablation.yaml"
)

DEFAULT_DATASETS=(
  qasper
  loogle
  narrativeqa
  quality
  novelhopqa
)

run_one() {
  local dataset_name="$1"
  local default_yaml="$2"

  if [[ ! -f "${default_yaml}" ]]; then
    echo "Missing config for ${dataset_name}: ${default_yaml}" >&2
    exit 2
  fi

  echo "Running RAPTOR with Qwen config for dataset=${dataset_name}"
  echo "  config=${default_yaml}"

  python scripts/run_raptor_experiment.py \
    --dataset-name "${dataset_name}" \
    --default-yaml "${default_yaml}" \
    "${@:3}"
}

if [[ $# -gt 0 && "$1" == "--dataset" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--dataset requires a dataset name" >&2
    exit 2
  fi

  dataset_name="$2"
  shift 2

  if [[ -z "${DATASET_CONFIGS[${dataset_name}]:-}" ]]; then
    echo "Unsupported dataset: ${dataset_name}" >&2
    echo "Supported datasets: ${DEFAULT_DATASETS[*]}" >&2
    exit 2
  fi

  run_one "${dataset_name}" "${DATASET_CONFIGS[${dataset_name}]}" "$@"
  exit 0
fi

for dataset_name in "${DEFAULT_DATASETS[@]}"; do
  run_one "${dataset_name}" "${DATASET_CONFIGS[${dataset_name}]}" "$@"
done
