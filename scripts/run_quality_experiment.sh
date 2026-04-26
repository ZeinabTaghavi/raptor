#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/cache/taghavi}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

TOP_K="${TOP_K:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-raptor_${TOP_K}_runs}"

python scripts/run_raptor_experiment.py \
  --dataset-name quality \
  --default-yaml configs/experiments/quality_retrieval_ablation.yaml \
  --output-root "${OUTPUT_ROOT}" \
  --retrieval-top-k "${TOP_K}" \
  "$@"
