#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-/mnt/cache/taghavi}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NOVELHOPQA_BOOKS_ROOT="${NOVELHOPQA_BOOKS_ROOT:-../passing_meta_tag/novelhopqa/book-corpus-root}"
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"

TOP_K="${TOP_K:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-raptor_${TOP_K}_runs}"
export NOVELHOPQA_REPORT_DIR="${NOVELHOPQA_REPORT_DIR:-${OUTPUT_ROOT}/novelhopqa/_loader_reports}"

DEFAULT_DATASETS="novelhopqa" # qasper,loogle,narrativeqa,quality,
DATASETS_CSV="${RAPTOR_DATASETS_CSV:-$DEFAULT_DATASETS}"

config_for_dataset() {
  case "$1" in
    qasper) echo "configs/raptor/qasper_retrieval_ablation.yaml" ;;
    loogle) echo "configs/raptor/loogle_retrieval_ablation.yaml" ;;
    narrativeqa) echo "configs/raptor/nqa_retrieval_ablation.yaml" ;;
    quality) echo "configs/experiments/quality_retrieval_ablation.yaml" ;;
    novelhopqa) echo "configs/experiments/novelhopqa_retrieval_ablation.yaml" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local dataset_name="$1"
  local default_yaml="$2"

  if [[ ! -f "${default_yaml}" ]]; then
    echo "Missing config for ${dataset_name}: ${default_yaml}" >&2
    exit 2
  fi

  echo "Running RAPTOR dataset=${dataset_name} top_k=${TOP_K}"
  echo "  config=${default_yaml}"
  echo "  output_root=${OUTPUT_ROOT}"

  python scripts/run_raptor_experiment.py \
    --dataset-name "${dataset_name}" \
    --default-yaml "${default_yaml}" \
    --output-root "${OUTPUT_ROOT}" \
    --retrieval-top-k "${TOP_K}"
}

if [ "$#" -gt 0 ]; then
  python scripts/run_raptor_experiment.py "$@"
  exit 0
fi

IFS=',' read -r -a requested_datasets <<<"${DATASETS_CSV}"
for dataset_name in "${requested_datasets[@]}"; do
  dataset_name="$(echo "${dataset_name}" | xargs)"
  if [[ -z "${dataset_name}" ]]; then
    continue
  fi
  if ! default_yaml="$(config_for_dataset "${dataset_name}")"; then
    echo "Unsupported dataset in RAPTOR_DATASETS_CSV: ${dataset_name}" >&2
    echo "Supported datasets: ${DEFAULT_DATASETS}" >&2
    exit 2
  fi
  run_one "${dataset_name}" "${default_yaml}"
done
