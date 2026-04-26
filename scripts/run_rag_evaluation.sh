#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'USAGE' >&2
Usage:
  scripts/run_rag_evaluation.sh RUN_DIR DATASET_NAME SPLIT METHOD_NAME [extra evaluate_rag_run.py args]

Example:
  scripts/run_rag_evaluation.sh \
    raptor_runs/qasper/qasper_retrieval_ablation_raptor \
    qasper \
    test \
    raptor \
    --disable-bert-score
USAGE
  exit 2
fi

RUN_DIR="$1"
DATASET_NAME="$2"
SPLIT="$3"
METHOD_NAME="$4"
shift 4

python scripts/evaluate_rag_run.py \
  --run-dir "$RUN_DIR" \
  --dataset-name "$DATASET_NAME" \
  --split "$SPLIT" \
  --method-name "$METHOD_NAME" \
  --output-dir "Raptor Evaluations" \
  --ks 5 10 \
  --generation-top-k 10 \
  "$@"
