# RAPTOR Evaluation Outputs

This folder is the default root for compact evaluation artifacts produced by
`scripts/evaluate_rag_run.py`.

The evaluator writes reports as:

```text
raptor_evaluations/
  <dataset-name>/
    <run-name>/
      metrics_summary.json
      metrics_per_query.jsonl
      leaderboard_row.json
      evaluation_manifest.json
```

Use `--output-dir raptor_evaluations` to keep final reports here. The metric
files intentionally exclude full contexts and raw retrieval traces; those remain
in the original `raptor_runs/<dataset>/<run>/` directory and are referenced by
`evaluation_manifest.json`.
