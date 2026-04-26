#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raptor.experiment_runner import run_experiment


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a standalone RAPTOR dataset experiment and save raw artifacts."
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name used in the output path.")
    parser.add_argument(
        "--default-yaml",
        required=True,
        help="Reference YAML file whose defaults should be mapped into the RAPTOR run.",
    )
    parser.add_argument("--run-name", help="Optional run name. Defaults to the YAML run_name or a UTC timestamp.")
    parser.add_argument(
        "--output-root",
        help="Optional output root. Defaults to the YAML output_root, usually raptor_10_runs.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        help="Override retrieval.top_k for the run, e.g. 5 or 10.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing trees and skip completed QA predictions when possible.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    result = run_experiment(
        dataset_name=args.dataset_name,
        default_yaml_path=args.default_yaml,
        run_name=args.run_name,
        output_root=args.output_root,
        retrieval_top_k=args.retrieval_top_k,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
