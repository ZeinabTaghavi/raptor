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

DEFAULT_DATASETS=(
  qasper
  loogle
  narrativeqa
  quality
  novelhopqa
)

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

usage() {
  cat <<'USAGE' >&2
Usage:
  scripts/run_all_qwen_experiments.sh [--top-k K] [--dataset DATASET] [extra run_raptor_experiment.py args]

Examples:
  scripts/run_all_qwen_experiments.sh --top-k 10
  scripts/run_all_qwen_experiments.sh --top-k 5 --dataset novelhopqa
  scripts/run_all_qwen_experiments.sh --top-k 5 --dataset qasper --resume

Outputs default to raptor_<top-k>_runs/<dataset>/<run-name>/.
USAGE
}

run_one() {
  local dataset_name="$1"
  local default_yaml="$2"

  if [[ ! -f "${default_yaml}" ]]; then
    echo "Missing config for ${dataset_name}: ${default_yaml}" >&2
    exit 2
  fi

  echo "Running RAPTOR with Qwen config for dataset=${dataset_name} top_k=${TOP_K}"
  echo "  config=${default_yaml}"
  echo "  output_root=${OUTPUT_ROOT}"

  python scripts/run_raptor_experiment.py \
    --dataset-name "${dataset_name}" \
    --default-yaml "${default_yaml}" \
    --output-root "${OUTPUT_ROOT}" \
    --retrieval-top-k "${TOP_K}" \
    "${@:3}"
}

TOP_K="${TOP_K:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DATASET_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --top-k)
      if [[ $# -lt 2 ]]; then
        echo "--top-k requires a value" >&2
        exit 2
      fi
      TOP_K="$2"
      shift 2
      ;;
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "--dataset requires a dataset name" >&2
        exit 2
      fi
      DATASET_NAME="$2"
      shift 2
      ;;
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "--output-root requires a value" >&2
        exit 2
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$TOP_K" =~ ^[0-9]+$ || "$TOP_K" -lt 1 ]]; then
  echo "--top-k must be a positive integer" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-raptor_${TOP_K}_runs}"
export NOVELHOPQA_REPORT_DIR="${NOVELHOPQA_REPORT_DIR:-${OUTPUT_ROOT}/novelhopqa/_loader_reports}"

if [[ -n "${DATASET_NAME}" ]]; then
  if ! default_yaml="$(config_for_dataset "${DATASET_NAME}")"; then
    echo "Unsupported dataset: ${DATASET_NAME}" >&2
    echo "Supported datasets: ${DEFAULT_DATASETS[*]}" >&2
    exit 2
  fi

  run_one "${DATASET_NAME}" "${default_yaml}" "${EXTRA_ARGS[@]}"
  exit 0
fi

for dataset_name in "${DEFAULT_DATASETS[@]}"; do
  default_yaml="$(config_for_dataset "${dataset_name}")"
  run_one "${dataset_name}" "${default_yaml}" "${EXTRA_ARGS[@]}"
done
