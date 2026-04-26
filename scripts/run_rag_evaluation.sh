#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  # Evaluate every completed run under raptor_runs/<dataset>/<run>/
  scripts/run_rag_evaluation.sh [extra evaluate_rag_run.py args]

  # Explicit all-runs mode, useful when extra args might look positional
  scripts/run_rag_evaluation.sh --all [extra evaluate_rag_run.py args]

  # Evaluate one run
  scripts/run_rag_evaluation.sh RUN_DIR DATASET_NAME SPLIT METHOD_NAME [extra evaluate_rag_run.py args]

Examples:
  scripts/run_rag_evaluation.sh

  # Smoke-test only; final reports should compute BERTScore.
  scripts/run_rag_evaluation.sh --disable-bert-score

  scripts/run_rag_evaluation.sh \
    raptor_runs/qasper/qasper_retrieval_ablation_raptor \
    qasper \
    test \
    raptor
USAGE
}

run_one() {
  local run_dir="$1"
  local dataset_name="$2"
  local split="$3"
  local method_name="$4"
  shift 4

  if [[ ! -f "$run_dir/rag/qa_predictions.jsonl" ]]; then
    echo "Skipping $run_dir: missing rag/qa_predictions.jsonl" >&2
    return 0
  fi

  echo "Evaluating dataset=$dataset_name split=${split:-auto} run=$run_dir" >&2
  command=(
    python scripts/evaluate_rag_run.py
    --run-dir "$run_dir"
    --dataset-name "$dataset_name"
    --method-name "$method_name"
    --output-dir raptor_evaluations
    --ks 5 10
    --generation-top-k 10
  )
  if [[ -n "${split//[[:space:]]/}" ]]; then
    command+=(--split "$split")
  fi
  command+=("$@")
  "${command[@]}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ge 4 && -d "$1" ]]; then
  RUN_DIR="$1"
  DATASET_NAME="$2"
  SPLIT="$3"
  METHOD_NAME="$4"
  shift 4
  run_one "$RUN_DIR" "$DATASET_NAME" "$SPLIT" "$METHOD_NAME" "$@"
  exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
  shift
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  usage
  exit 2
fi

METHOD_NAME="${METHOD_NAME:-raptor}"
SPLIT="${SPLIT:-}"
found=0

while IFS= read -r -d '' run_dir; do
  dataset_name="$(basename "$(dirname "$run_dir")")"
  run_one "$run_dir" "$dataset_name" "$SPLIT" "$METHOD_NAME" "$@"
  found=1
done < <(
  find raptor_runs -mindepth 2 -maxdepth 2 -type d \
    ! -path '*/_loader_reports' \
    -print0 | sort -z
)

if [[ "$found" -eq 0 ]]; then
  echo "No run directories found under raptor_runs/<dataset>/<run>." >&2
  exit 1
fi
