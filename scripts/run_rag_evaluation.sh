#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  # Evaluate every completed run under raptor_<top-k>_runs/<dataset>/<run>/
  scripts/run_rag_evaluation.sh [--top-k K] [extra evaluate_rag_run.py args]

  # Explicit all-runs mode, useful when extra args might look positional
  scripts/run_rag_evaluation.sh --top-k K --all [extra evaluate_rag_run.py args]

  # Evaluate one run
  scripts/run_rag_evaluation.sh [--top-k K] RUN_DIR DATASET_NAME SPLIT METHOD_NAME [extra evaluate_rag_run.py args]

Examples:
  scripts/run_rag_evaluation.sh --top-k 10
  scripts/run_rag_evaluation.sh --top-k 5 --disable-bert-score

  # Smoke-test only; final reports should compute BERTScore.
  scripts/run_rag_evaluation.sh --top-k 10 --disable-bert-score

  scripts/run_rag_evaluation.sh \
    --top-k 10 \
    raptor_10_runs/qasper/qasper_retrieval_ablation_raptor \
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

  echo "Evaluating dataset=$dataset_name split=${split:-auto} top_k=$TOP_K run=$run_dir" >&2
  command=(
    python scripts/evaluate_rag_run.py
    --run-dir "$run_dir"
    --dataset-name "$dataset_name"
    --method-name "$method_name"
    --output-dir "$EVALUATION_ROOT"
    --ks 5 10
    --generation-top-k "$TOP_K"
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

TOP_K="${TOP_K:-10}"
RUNS_ROOT="${RUNS_ROOT:-}"
EVALUATION_ROOT="${EVALUATION_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top-k)
      if [[ $# -lt 2 ]]; then
        echo "--top-k requires a value" >&2
        exit 2
      fi
      TOP_K="$2"
      shift 2
      ;;
    --runs-root)
      if [[ $# -lt 2 ]]; then
        echo "--runs-root requires a value" >&2
        exit 2
      fi
      RUNS_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "--output-dir requires a value" >&2
        exit 2
      fi
      EVALUATION_ROOT="$2"
      shift 2
      ;;
    --all)
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! "$TOP_K" =~ ^[0-9]+$ || "$TOP_K" -lt 1 ]]; then
  echo "--top-k must be a positive integer" >&2
  exit 2
fi

RUNS_ROOT="${RUNS_ROOT:-raptor_${TOP_K}_runs}"
EVALUATION_ROOT="${EVALUATION_ROOT:-raptor_${TOP_K}_evaluations}"

if [[ $# -ge 4 && -d "$1" ]]; then
  RUN_DIR="$1"
  DATASET_NAME="$2"
  SPLIT="$3"
  METHOD_NAME="$4"
  shift 4
  run_one "$RUN_DIR" "$DATASET_NAME" "$SPLIT" "$METHOD_NAME" "$@"
  exit 0
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  usage
  exit 2
fi

METHOD_NAME="${METHOD_NAME:-raptor}"
SPLIT="${SPLIT:-}"
found=0

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "Run root does not exist: $RUNS_ROOT" >&2
  exit 1
fi

while IFS= read -r -d '' run_dir; do
  dataset_name="$(basename "$(dirname "$run_dir")")"
  run_one "$run_dir" "$dataset_name" "$SPLIT" "$METHOD_NAME" "$@"
  found=1
done < <(
  find "$RUNS_ROOT" -mindepth 2 -maxdepth 2 -type d \
    ! -path '*/_loader_reports' \
    -print0 | sort -z
)

if [[ "$found" -eq 0 ]]; then
  echo "No run directories found under ${RUNS_ROOT}/<dataset>/<run>." >&2
  exit 1
fi
