#!/usr/bin/env python3
"""Compact evaluator for existing RAPTOR retrieval and QA artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "raptor_evaluations"
RETRIEVAL_METRIC_PREFIXES = ("recall", "mrr", "ndcg", "hit_rate")
RAG_METRIC_KEYS = (
    "exact_match",
    "token_f1",
    "rouge_l",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)
EFFICIENCY_FIELDS = (
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "context_tokens",
    "answer_tokens",
    "peak_gpu_memory_mb",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing RAPTOR retrieval payloads and QA predictions into compact "
            "metrics artifacts. This script never regenerates retrieval or answers."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Existing RAPTOR run directory.")
    parser.add_argument(
        "--labels-file",
        help=(
            "Optional labels/reference file. Defaults to <run-dir>/selection/qa_entries.json "
            "when present."
        ),
    )
    parser.add_argument(
        "--answers-file",
        help=(
            "Optional generated-answer/prediction file. Defaults to "
            "<run-dir>/rag/qa_predictions.jsonl."
        ),
    )
    parser.add_argument(
        "--predictions-file",
        dest="answers_file",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Evaluation output root. Results are written under "
            "<output-dir>/<dataset-name>/<run-name>. Defaults to 'raptor_evaluations'."
        ),
    )
    parser.add_argument("--method-name", required=True, help="Comparable method name, e.g. raptor.")
    parser.add_argument("--dataset-name", help="Dataset name. Defaults to run manifest when available.")
    parser.add_argument("--split", help="Dataset split. Defaults to labels metadata when available.")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10], help="Retrieval cutoffs.")
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=10,
        help="Top-k retrieval context used by the already-generated answers.",
    )
    parser.add_argument(
        "--disable-bert-score",
        action="store_true",
        help="Write null BERTScore metrics without loading the BERTScore model.",
    )
    parser.add_argument("--bert-score-model", default="roberta-large")
    parser.add_argument("--bert-score-lang", default="en")
    parser.add_argument("--bert-score-batch-size", type=int, default=16)
    parser.add_argument(
        "--bert-score-device",
        default="cpu",
        help="Device passed to bert_score.score, e.g. cuda:0 or cpu.",
    )
    return parser


def package_version(package_name: str) -> Optional[str]:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=False, separators=(",", ":")))
            handle.write("\n")


def infer_records_from_json(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("qa_entries", "records", "data", "predictions", "results", "queries"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

        if payload and all(isinstance(value, dict) for value in payload.values()):
            rows = []
            for key, value in payload.items():
                row = dict(value)
                row.setdefault("query_id", key)
                rows.append(row)
            return rows

        return [payload]

    return []


def read_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
        return rows

    if suffix == ".json":
        return infer_records_from_json(read_json(path))

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported file format for {path}. Use JSON, JSONL, or CSV.")


def get_nested(record: Dict[str, Any], key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_present(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = get_nested(record, key)
        if value is not None:
            return value

    metadata_value = record.get("metadata")
    if isinstance(metadata_value, dict):
        for key in keys:
            value = get_nested(metadata_value, key)
            if value is not None:
                return value

    return None


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] in "[{\"" or stripped in {"null", "true", "false"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def ordered_unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def maybe_split_delimited(text: str) -> List[str]:
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text and "::chunk::" not in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text.strip()] if text.strip() else []


def coerce_text_list(value: Any) -> List[str]:
    value = parse_jsonish(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("text", "answer", "reference_answer"):
            if key in value:
                return coerce_text_list(value[key])
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            items.extend(coerce_text_list(item))
        return [item for item in items if item]
    return [str(value)]


def coerce_id_list(value: Any) -> List[str]:
    value = parse_jsonish(value)
    if value is None:
        return []
    if isinstance(value, str):
        return ordered_unique(maybe_split_delimited(value))
    if isinstance(value, dict):
        for key in ("chunk_id", "id", "chunk_ids", "ids", "gold_chunk_ids", "silver_chunk_ids"):
            if key in value:
                return coerce_id_list(value[key])
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            items.extend(coerce_id_list(item))
        return ordered_unique(items)
    return ordered_unique([value])


def coerce_group_list(value: Any) -> List[List[str]]:
    value = parse_jsonish(value)
    if value is None:
        return []

    groups: List[List[str]] = []
    if isinstance(value, dict):
        for key in ("chunk_ids", "ids", "silver_chunk_ids", "gold_chunk_ids"):
            if key in value:
                group = coerce_id_list(value[key])
                return [group] if group else []
        return []

    if isinstance(value, str):
        parsed = parse_jsonish(value)
        if parsed is not value:
            return coerce_group_list(parsed)
        group = coerce_id_list(value)
        return [group] if group else []

    if isinstance(value, (list, tuple, set)):
        parsed_items = [parse_jsonish(item) for item in value]
        if parsed_items and all(not isinstance(item, (dict, list, tuple, set)) for item in parsed_items):
            group = coerce_id_list(parsed_items)
            return [group] if group else []

        for parsed_item in parsed_items:
            if isinstance(parsed_item, dict):
                groups.extend(coerce_group_list(parsed_item))
            elif isinstance(parsed_item, (list, tuple, set)):
                group = coerce_id_list(parsed_item)
                if group:
                    groups.append(group)
            elif parsed_item is not None:
                group = coerce_id_list(parsed_item)
                if group:
                    groups.append(group)

    return [ordered_unique(group) for group in groups if group]


def record_query_id(record: Dict[str, Any]) -> Optional[str]:
    value = first_present(record, ("query_id", "id", "question_id", "qid"))
    return str(value) if value is not None else None


def record_doc_id(record: Dict[str, Any]) -> Optional[str]:
    value = first_present(record, ("doc_id", "document_id", "paper_id", "book_id"))
    return str(value) if value is not None else None


def record_question(record: Dict[str, Any]) -> Optional[str]:
    value = first_present(record, ("question", "query", "prompt"))
    return str(value) if value is not None else None


def record_prediction(record: Dict[str, Any]) -> Optional[str]:
    value = first_present(
        record,
        ("prediction", "generated_answer", "answer", "model_answer", "response", "output"),
    )
    return str(value) if value is not None else None


def record_references(record: Dict[str, Any]) -> List[str]:
    value = first_present(
        record,
        (
            "reference_answers",
            "references",
            "answers",
            "gold_answers",
            "reference_answer",
            "ground_truth_answers",
        ),
    )
    return coerce_text_list(value)


def labels_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gold_chunk_ids": coerce_id_list(first_present(record, ("gold_chunk_ids",))),
        "silver_chunk_ids": coerce_id_list(first_present(record, ("silver_chunk_ids",))),
        "silver_chunk_groups": coerce_group_list(first_present(record, ("silver_chunk_groups",))),
    }


def label_fields_present(records: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    presence = {
        "gold_chunk_ids": False,
        "silver_chunk_ids": False,
        "silver_chunk_groups": False,
    }
    for record in records:
        for key in list(presence):
            if get_nested(record, key) is not None:
                presence[key] = True
            metadata_value = record.get("metadata")
            if isinstance(metadata_value, dict) and get_nested(metadata_value, key) is not None:
                presence[key] = True
    return presence


def index_records(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    by_query_id: Dict[str, Dict[str, Any]] = {}
    by_doc_question: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        query_id = record_query_id(record)
        if query_id is not None and query_id not in by_query_id:
            by_query_id[query_id] = record

        doc_id = record_doc_id(record)
        question = record_question(record)
        if doc_id is not None and question is not None:
            by_doc_question.setdefault((doc_id, question), record)
    return by_query_id, by_doc_question


def find_matching_record(
    query_id: str,
    doc_id: Optional[str],
    question: Optional[str],
    by_query_id: Dict[str, Dict[str, Any]],
    by_doc_question: Dict[Tuple[str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if query_id in by_query_id:
        return by_query_id[query_id]
    if doc_id is not None and question is not None:
        return by_doc_question.get((doc_id, question))
    return None


def load_node_descendant_map(run_dir: Path, doc_id: str) -> Dict[int, List[str]]:
    node_index_path = run_dir / "trees" / doc_id / "node_index.jsonl"
    if not node_index_path.exists():
        return {}

    mapping: Dict[int, List[str]] = {}
    with node_index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "node_index" not in row:
                continue
            mapping[int(row["node_index"])] = coerce_id_list(row.get("descendant_leaf_chunk_ids"))
    return mapping


def extract_score(record: Dict[str, Any]) -> Optional[float]:
    for key in ("score", "similarity", "distance", "retrieval_score"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def extract_retrieved_leaf_ids(
    retrieval_record: Optional[Dict[str, Any]],
    prediction_record: Optional[Dict[str, Any]],
    run_dir: Path,
    node_cache: Dict[str, Dict[int, List[str]]],
    expansion_notes: List[str],
) -> Tuple[List[str], List[Optional[float]]]:
    raw_ids: List[str] = []
    raw_scores: List[Optional[float]] = []

    if retrieval_record:
        expanded_chunks = retrieval_record.get("expanded_retrieved_chunks")
        if isinstance(expanded_chunks, list):
            for chunk_row in expanded_chunks:
                if not isinstance(chunk_row, dict):
                    continue
                chunk_id = chunk_row.get("chunk_id") or chunk_row.get("id")
                if chunk_id is None:
                    continue
                raw_ids.append(str(chunk_id))
                raw_scores.append(extract_score(chunk_row))

        if not raw_ids:
            retrieved_nodes = retrieval_record.get("retrieved_nodes")
            doc_id = record_doc_id(retrieval_record)
            if isinstance(retrieved_nodes, list) and doc_id:
                if doc_id not in node_cache:
                    node_cache[doc_id] = load_node_descendant_map(run_dir, doc_id)
                node_map = node_cache.get(doc_id, {})
                if node_map:
                    for node in retrieved_nodes:
                        if not isinstance(node, dict) or "node_index" not in node:
                            continue
                        try:
                            node_index = int(node["node_index"])
                        except (TypeError, ValueError):
                            continue
                        for chunk_id in node_map.get(node_index, []):
                            raw_ids.append(chunk_id)
                            raw_scores.append(extract_score(node))
                else:
                    expansion_notes.append(
                        f"No node_index.jsonl descendant map found for doc_id={doc_id}; "
                        "could not expand retrieved nodes for that document."
                    )

    if not raw_ids and prediction_record:
        for key in ("retrieved_chunk_ids", "retrieved_ids", "retrieved_ids_top10"):
            value = prediction_record.get(key)
            ids = coerce_id_list(value)
            if ids:
                raw_ids.extend(ids)
                raw_scores.extend([None] * len(ids))
                break

    deduped_ids: List[str] = []
    deduped_scores: List[Optional[float]] = []
    seen = set()
    for chunk_id, score in zip(raw_ids, raw_scores):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped_ids.append(chunk_id)
        deduped_scores.append(score)

    return deduped_ids, deduped_scores


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def answer_tokens(text: str) -> List[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def token_f1_pair(prediction: str, reference: str) -> float:
    prediction_tokens = answer_tokens(prediction)
    reference_tokens = answer_tokens(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l_pair(prediction: str, reference: str) -> float:
    prediction_tokens = answer_tokens(prediction)
    reference_tokens = answer_tokens(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def generation_metrics_without_bertscore(
    prediction: Optional[str],
    references: Sequence[str],
) -> Dict[str, Optional[float]]:
    metrics = {key: None for key in RAG_METRIC_KEYS}
    if prediction is None or not references:
        return metrics

    metrics["exact_match"] = exact_match(prediction, references)
    metrics["token_f1"] = max(token_f1_pair(prediction, reference) for reference in references)
    metrics["rouge_l"] = max(rouge_l_pair(prediction, reference) for reference in references)
    return metrics


def dcg(binary_relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(binary_relevances, start=1))


def retrieval_metric_keys(ks: Sequence[int]) -> List[str]:
    keys = []
    for prefix in RETRIEVAL_METRIC_PREFIXES:
        for k in ks:
            keys.append(f"{prefix}@{k}")
    return keys


def null_retrieval_metrics(ks: Sequence[int]) -> Dict[str, Optional[float]]:
    return {key: None for key in retrieval_metric_keys(ks)}


def flat_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    ks: Sequence[int],
) -> Tuple[bool, Dict[str, Optional[float]]]:
    metrics = null_retrieval_metrics(ks)
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return False, metrics

    for k in ks:
        top_k = list(retrieved_ids[:k])
        hits_by_rank = [1 if retrieved_id in relevant_set else 0 for retrieved_id in top_k]
        hit_count = sum(hits_by_rank)
        metrics[f"recall@{k}"] = hit_count / len(relevant_set)
        metrics[f"hit_rate@{k}"] = 1.0 if hit_count > 0 else 0.0
        first_hit_rank = next((index for index, value in enumerate(hits_by_rank, start=1) if value), None)
        metrics[f"mrr@{k}"] = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        ideal_len = min(len(relevant_set), k)
        ideal = [1] * ideal_len
        ideal_dcg = dcg(ideal)
        metrics[f"ndcg@{k}"] = dcg(hits_by_rank) / ideal_dcg if ideal_dcg else 0.0

    return True, metrics


def strict_group_metrics(
    retrieved_ids: Sequence[str],
    silver_groups: Sequence[Sequence[str]],
    ks: Sequence[int],
) -> Tuple[bool, Dict[str, Optional[float]]]:
    metrics = null_retrieval_metrics(ks)
    normalized_groups = [set(group) for group in silver_groups if group]
    if not normalized_groups:
        return False, metrics

    for k in ks:
        top_k = set(retrieved_ids[:k])
        hit_groups = [group for group in normalized_groups if group.issubset(top_k)]
        metrics[f"recall@{k}"] = len(hit_groups) / len(normalized_groups)
        metrics[f"hit_rate@{k}"] = 1.0 if hit_groups else 0.0

    return True, metrics


def union_metrics(
    retrieved_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    silver_groups: Sequence[Sequence[str]],
    ks: Sequence[int],
) -> Tuple[bool, Dict[str, Optional[float]]]:
    metrics = null_retrieval_metrics(ks)
    gold_set = set(gold_chunk_ids)
    normalized_groups = [set(group) for group in silver_groups if group]
    if not gold_set and not normalized_groups:
        return False, metrics

    for k in ks:
        top_k = set(retrieved_ids[:k])
        gold_hit = bool(gold_set.intersection(top_k)) if gold_set else False
        silver_strict_hit = any(group.issubset(top_k) for group in normalized_groups)
        metrics[f"hit_rate@{k}"] = 1.0 if gold_hit or silver_strict_hit else 0.0

    return True, metrics


def aggregate_metric_rows(
    rows: Sequence[Dict[str, Any]],
    view_name: str,
    ks: Sequence[int],
) -> Dict[str, Any]:
    metrics = null_retrieval_metrics(ks)
    eligible_rows = [row for row in rows if row["retrieval_by_view"][view_name]["eligible"]]
    if not eligible_rows:
        return {
            **metrics,
            "eligible_queries": 0,
        }

    for key in metrics:
        values = [
            row["retrieval_by_view"][view_name]["metrics"][key]
            for row in eligible_rows
            if row["retrieval_by_view"][view_name]["metrics"][key] is not None
        ]
        metrics[key] = mean(values) if values else None
    return {
        **metrics,
        "eligible_queries": len(eligible_rows),
    }


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary_stats(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": mean(numeric_values),
        "median": percentile(numeric_values, 0.5),
        "p95": percentile(numeric_values, 0.95),
    }


def numeric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def field_numeric(record: Optional[Dict[str, Any]], names: Sequence[str]) -> Optional[float]:
    if not record:
        return None
    for name in names:
        value = first_present(record, (name,))
        number = numeric_value(value)
        if number is not None:
            return number
    return None


def first_not_none(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def load_query_times(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    query_times_path = run_dir / "profiling" / "query_times.jsonl"
    rows = read_records(query_times_path)
    return {record_query_id(row): row for row in rows if record_query_id(row)}


def load_query_resource_usage(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    resource_path = run_dir / "profiling" / "resource_usage.jsonl"
    rows = read_records(resource_path)
    usage: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("phase") != "query":
            continue
        query_id = record_query_id(row)
        if query_id:
            usage[query_id] = row
    return usage


def sanitize_component(value: str) -> str:
    cleaned = str(value).strip().replace("/", "__").replace("\\", "__")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "unknown"


def infer_split(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    for record in records:
        split = first_present(record, ("split", "metadata.split"))
        if split is not None:
            return str(split)
    return None


def maybe_run_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            return manifest if isinstance(manifest, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def collect_query_order(
    retrieval_records: Sequence[Dict[str, Any]],
    prediction_records: Sequence[Dict[str, Any]],
) -> List[str]:
    ordered = []
    seen = set()
    for record in list(retrieval_records) + list(prediction_records):
        query_id = record_query_id(record)
        if query_id and query_id not in seen:
            seen.add(query_id)
            ordered.append(query_id)
    return ordered


def evaluate_bertscore(
    query_rows: List[Dict[str, Any]],
    bertscore_config: Dict[str, Any],
    disabled: bool,
) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    versions = {
        "bert-score": package_version("bert-score"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
    }

    if disabled:
        return "BERTScore disabled by --disable-bert-score.", versions

    pairs: List[Tuple[int, str, str]] = []
    for row_index, row in enumerate(query_rows):
        prediction = row.get("_prediction")
        references = row.get("_references") or []
        if prediction is None or not references:
            continue
        for reference in references:
            pairs.append((row_index, prediction, reference))

    if not pairs:
        return "No prediction/reference pairs were available for BERTScore.", versions

    try:
        from bert_score import score as bert_score_score  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional package.
        return f"BERTScore package could not be imported: {exc}", versions

    try:
        candidates = [candidate for _, candidate, _ in pairs]
        references = [reference for _, _, reference in pairs]
        precision, recall, f1 = bert_score_score(
            candidates,
            references,
            model_type=bertscore_config["model"],
            lang=bertscore_config["lang"],
            batch_size=bertscore_config["batch_size"],
            device=bertscore_config["device"],
            verbose=False,
            rescale_with_baseline=False,
        )
    except Exception as exc:  # pragma: no cover - model availability is environment dependent.
        return f"BERTScore could not be computed: {exc}", versions

    best_by_row: Dict[int, Tuple[float, float, float]] = {}
    for pair_index, (row_index, _, _) in enumerate(pairs):
        p_value = float(precision[pair_index].item())
        r_value = float(recall[pair_index].item())
        f_value = float(f1[pair_index].item())
        previous = best_by_row.get(row_index)
        if previous is None or f_value > previous[2]:
            best_by_row[row_index] = (p_value, r_value, f_value)

    for row_index, (p_value, r_value, f_value) in best_by_row.items():
        query_rows[row_index]["bertscore_precision"] = p_value
        query_rows[row_index]["bertscore_recall"] = r_value
        query_rows[row_index]["bertscore_f1"] = f_value

    return None, versions


def round_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_float(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_float(item) for item in value]
    return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")

    run_manifest = maybe_run_manifest(run_dir)
    run_name = str(run_manifest.get("run_name") or run_dir.name)
    dataset_name = str(args.dataset_name or run_manifest.get("dataset_name") or "unknown_dataset")

    retrieval_file = run_dir / "retrieval" / "retrieval_payloads.jsonl"
    predictions_file = Path(args.answers_file).expanduser().resolve() if args.answers_file else run_dir / "rag" / "qa_predictions.jsonl"
    default_labels_file = run_dir / "selection" / "qa_entries.json"
    labels_file = Path(args.labels_file).expanduser().resolve() if args.labels_file else default_labels_file
    query_times_file = run_dir / "profiling" / "query_times.jsonl"
    resource_usage_file = run_dir / "profiling" / "resource_usage.jsonl"

    retrieval_records = read_records(retrieval_file)
    prediction_records = read_records(predictions_file)
    label_records = read_records(labels_file)
    query_times = load_query_times(run_dir)
    resource_usage = load_query_resource_usage(run_dir)

    retrieval_by_query, retrieval_by_doc_question = index_records(retrieval_records)
    prediction_by_query, prediction_by_doc_question = index_records(prediction_records)
    labels_by_query, labels_by_doc_question = index_records(label_records)

    split = str(args.split or infer_split(label_records) or infer_split(prediction_records) or "unknown")
    ks = sorted(dict.fromkeys(args.ks))
    if not ks:
        raise SystemExit("--ks must contain at least one positive integer.")
    if any(k <= 0 for k in ks):
        raise SystemExit("--ks values must be positive integers.")

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_dir = output_root / sanitize_component(dataset_name) / sanitize_component(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_presence = label_fields_present(label_records)
    explicit_chunk_relevance_found = any(label_presence.values())
    query_order = collect_query_order(retrieval_records, prediction_records)
    if not query_order:
        raise SystemExit(
            "No query records found. Expected retrieval/retrieval_payloads.jsonl or rag/qa_predictions.jsonl."
        )

    expansion_notes: List[str] = []
    node_cache: Dict[str, Dict[int, List[str]]] = {}
    query_rows: List[Dict[str, Any]] = []

    for query_id in query_order:
        retrieval_record = retrieval_by_query.get(query_id)
        prediction_record = prediction_by_query.get(query_id)

        doc_id = (
            record_doc_id(prediction_record or {})
            or record_doc_id(retrieval_record or {})
        )
        question = (
            record_question(prediction_record or {})
            or record_question(retrieval_record or {})
        )
        if prediction_record is None and retrieval_record is not None:
            prediction_record = find_matching_record(
                query_id,
                doc_id,
                question,
                prediction_by_query,
                prediction_by_doc_question,
            )
        if retrieval_record is None and prediction_record is not None:
            retrieval_record = find_matching_record(
                query_id,
                doc_id,
                question,
                retrieval_by_query,
                retrieval_by_doc_question,
            )

        label_record = find_matching_record(query_id, doc_id, question, labels_by_query, labels_by_doc_question)
        labels = labels_from_record(label_record or {})

        retrieved_ids, retrieved_scores = extract_retrieved_leaf_ids(
            retrieval_record,
            prediction_record,
            run_dir,
            node_cache,
            expansion_notes,
        )

        references = record_references(prediction_record or {}) or record_references(label_record or {})
        prediction = record_prediction(prediction_record or {})

        generation_metrics = generation_metrics_without_bertscore(prediction, references)
        query_time_record = query_times.get(query_id, {})
        resource_record = resource_usage.get(query_id, {})

        retrieval_latency = first_not_none(
            field_numeric(prediction_record, ("retrieval_latency_ms",)),
            field_numeric(retrieval_record, ("retrieval_latency_ms",)),
            field_numeric(query_time_record, ("retrieval_latency_ms",)),
        )
        generation_latency = first_not_none(
            field_numeric(prediction_record, ("generation_latency_ms",)),
            field_numeric(query_time_record, ("generation_latency_ms",)),
        )
        total_latency = first_not_none(
            field_numeric(prediction_record, ("total_latency_ms",)),
            field_numeric(query_time_record, ("total_latency_ms",)),
        )
        context_tokens = first_not_none(
            field_numeric(prediction_record, ("context_tokens", "context_token_count")),
            field_numeric(retrieval_record, ("context_tokens", "context_token_count")),
        )
        answer_tokens_value = field_numeric(
            prediction_record,
            ("answer_tokens", "answer_token_count", "prediction_token_count"),
        )
        peak_gpu_memory = first_not_none(
            field_numeric(prediction_record, ("peak_gpu_memory_mb", "max_gpu_memory_mb", "gpu_peak_memory_mb")),
            field_numeric(resource_record, ("peak_gpu_memory_mb", "max_gpu_memory_mb", "gpu_peak_memory_mb")),
        )

        gold_eligible, gold_metrics = flat_retrieval_metrics(retrieved_ids, labels["gold_chunk_ids"], ks)
        silver_loose_eligible, silver_loose_metrics = flat_retrieval_metrics(
            retrieved_ids,
            labels["silver_chunk_ids"],
            ks,
        )
        silver_strict_eligible, silver_strict_metrics = strict_group_metrics(
            retrieved_ids,
            labels["silver_chunk_groups"],
            ks,
        )
        union_eligible, union_view_metrics = union_metrics(
            retrieved_ids,
            labels["gold_chunk_ids"],
            labels["silver_chunk_groups"],
            ks,
        )

        query_rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "question": question,
                "_prediction": prediction,
                "_references": references,
                "_retrieved_scores_top10": retrieved_scores[:10],
                "_labels": labels,
                "retrieval_by_view": {
                    "gold": {"eligible": gold_eligible, "metrics": gold_metrics},
                    "silver_loose": {
                        "eligible": silver_loose_eligible,
                        "metrics": silver_loose_metrics,
                    },
                    "silver_strict": {
                        "eligible": silver_strict_eligible,
                        "metrics": silver_strict_metrics,
                    },
                    "union": {"eligible": union_eligible, "metrics": union_view_metrics},
                },
                **generation_metrics,
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": generation_latency,
                "total_latency_ms": total_latency,
                "context_tokens": context_tokens,
                "answer_tokens": answer_tokens_value,
                "peak_gpu_memory_mb": peak_gpu_memory,
                "retrieved_ids_top10": retrieved_ids[:10],
            }
        )

    by_view = {
        view: aggregate_metric_rows(query_rows, view, ks)
        for view in ("gold", "silver_loose", "silver_strict", "union")
    }

    if by_view["gold"]["eligible_queries"] > 0:
        primary_relevance = "gold"
    elif by_view["silver_loose"]["eligible_queries"] > 0:
        primary_relevance = "silver_loose"
    elif by_view["silver_strict"]["eligible_queries"] > 0:
        primary_relevance = "silver_strict"
    else:
        primary_relevance = None

    if primary_relevance:
        retrieval_metrics = {
            key: by_view[primary_relevance].get(key)
            for key in retrieval_metric_keys(ks)
        }
    else:
        retrieval_metrics = null_retrieval_metrics(ks)

    bertscore_config = {
        "model": args.bert_score_model,
        "lang": args.bert_score_lang,
        "batch_size": args.bert_score_batch_size,
        "device": args.bert_score_device,
    }
    bertscore_reason, bertscore_versions = evaluate_bertscore(
        query_rows=query_rows,
        bertscore_config=bertscore_config,
        disabled=args.disable_bert_score,
    )

    rag_metrics = {
        key: mean([
            float(row[key])
            for row in query_rows
            if row.get(key) is not None
        ])
        for key in RAG_METRIC_KEYS
    }

    efficiency = {
        field: summary_stats([row.get(field) for row in query_rows])
        for field in EFFICIENCY_FIELDS
    }

    missing_metrics_reasons: Dict[str, str] = {}
    if not explicit_chunk_relevance_found:
        missing_metrics_reasons["retrieval_metrics"] = (
            "No explicit chunk-level gold_chunk_ids, silver_chunk_ids, or "
            "silver_chunk_groups fields were found in the labels file. doc_id was not "
            "used as a fallback relevance id."
        )
    elif primary_relevance is None:
        missing_metrics_reasons["retrieval_metrics"] = (
            "Chunk-level relevance fields were present but no evaluated query had "
            "non-empty relevance labels."
        )

    if bertscore_reason:
        missing_metrics_reasons["bertscore"] = bertscore_reason
    if by_view["silver_strict"]["eligible_queries"] > 0:
        missing_metrics_reasons["silver_strict_mrr_ndcg"] = (
            "silver_strict MRR and NDCG are null because strict relevance is defined "
            "as full evidence-group containment, not a flat relevant-id ranking."
        )
    if by_view["union"]["eligible_queries"] > 0:
        missing_metrics_reasons["union_recall_mrr_ndcg"] = (
            "union only reports HitRate@k as gold_hit@k OR silver_strict_hit@k; it "
            "does not flat-union gold and silver ids."
        )

    per_query_rows: List[Dict[str, Any]] = []
    for row in query_rows:
        if primary_relevance:
            primary_metrics = row["retrieval_by_view"][primary_relevance]["metrics"]
        else:
            primary_metrics = null_retrieval_metrics(ks)

        labels = row["_labels"]
        if primary_relevance == "gold":
            relevant_ids: Optional[List[str]] = labels["gold_chunk_ids"]
        elif primary_relevance == "silver_loose":
            relevant_ids = labels["silver_chunk_ids"]
        else:
            relevant_ids = None

        compact = {
            "query_id": row["query_id"],
            "doc_id": row["doc_id"],
            "question": row["question"],
            **{key: primary_metrics[key] for key in retrieval_metric_keys(ks)},
            "exact_match": row["exact_match"],
            "token_f1": row["token_f1"],
            "rouge_l": row["rouge_l"],
            "bertscore_precision": row["bertscore_precision"],
            "bertscore_recall": row["bertscore_recall"],
            "bertscore_f1": row["bertscore_f1"],
            "retrieval_latency_ms": row["retrieval_latency_ms"],
            "generation_latency_ms": row["generation_latency_ms"],
            "total_latency_ms": row["total_latency_ms"],
            "context_tokens": row["context_tokens"],
            "answer_tokens": row["answer_tokens"],
            "retrieved_ids_top10": row["retrieved_ids_top10"],
            "relevant_ids": relevant_ids,
        }
        if primary_relevance == "silver_strict":
            compact["relevant_groups"] = labels["silver_chunk_groups"]
        if any(score is not None for score in row["_retrieved_scores_top10"]):
            compact["retrieval_scores_top10"] = row["_retrieved_scores_top10"]
        per_query_rows.append(round_float(compact))

    summary = {
        "method_name": args.method_name,
        "dataset_name": dataset_name,
        "split": split,
        "run_name": run_name,
        "n_queries": len(query_rows),
        "k_values": ks,
        "primary_relevance": primary_relevance,
        "generation_top_k": args.generation_top_k,
        "retrieval_metrics": retrieval_metrics,
        "retrieval_metrics_by_relevance": by_view,
        "rag_metrics": rag_metrics,
        "efficiency": efficiency,
    }
    summary = round_float(summary)

    leaderboard_row = {
        "method_name": args.method_name,
        "dataset_name": dataset_name,
        "split": split,
        "run_name": run_name,
        "generation_top_k": args.generation_top_k,
        **{key: summary["retrieval_metrics"].get(key) for key in retrieval_metric_keys(ks)},
        "exact_match": summary["rag_metrics"]["exact_match"],
        "token_f1": summary["rag_metrics"]["token_f1"],
        "rouge_l": summary["rag_metrics"]["rouge_l"],
        "bertscore_precision": summary["rag_metrics"]["bertscore_precision"],
        "bertscore_recall": summary["rag_metrics"]["bertscore_recall"],
        "bertscore_f1": summary["rag_metrics"]["bertscore_f1"],
        "retrieval_latency_mean_ms": summary["efficiency"]["retrieval_latency_ms"]["mean"],
        "generation_latency_mean_ms": summary["efficiency"]["generation_latency_ms"]["mean"],
        "total_latency_mean_ms": summary["efficiency"]["total_latency_ms"]["mean"],
        "context_tokens_mean": summary["efficiency"]["context_tokens"]["mean"],
        "peak_gpu_memory_mean_mb": summary["efficiency"]["peak_gpu_memory_mb"]["mean"],
    }
    leaderboard_row = round_float(leaderboard_row)

    metrics_summary_path = output_dir / "metrics_summary.json"
    metrics_per_query_path = output_dir / "metrics_per_query.jsonl"
    leaderboard_row_path = output_dir / "leaderboard_row.json"
    evaluation_manifest_path = output_dir / "evaluation_manifest.json"

    input_files = {
        "run_dir": str(run_dir),
        "retrieval_file": str(retrieval_file) if retrieval_file.exists() else None,
        "answers_file": str(predictions_file) if predictions_file.exists() else None,
        "labels_file": str(labels_file) if labels_file.exists() else None,
        "query_times_file": str(query_times_file) if query_times_file.exists() else None,
        "resource_usage_file": str(resource_usage_file) if resource_usage_file.exists() else None,
        "run_manifest_file": str(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else None,
    }
    output_files = {
        "metrics_summary": str(metrics_summary_path),
        "metrics_per_query": str(metrics_per_query_path),
        "leaderboard_row": str(leaderboard_row_path),
        "evaluation_manifest": str(evaluation_manifest_path),
    }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files_used": input_files,
        "output_files_written": output_files,
        "relevance_source_used": str(labels_file) if labels_file.exists() else None,
        "explicit_chunk_level_relevance_fields_found": explicit_chunk_relevance_found,
        "chunk_level_relevance_field_presence": label_presence,
        "primary_relevance_choice": primary_relevance,
        "primary_relevance_choice_policy": (
            "Prefer gold when any gold_chunk_ids are eligible; else silver_loose; "
            "else silver_strict; else null."
        ),
        "generation_evaluated_from_top_k": args.generation_top_k,
        "bertscore_configuration": {
            **bertscore_config,
            "disabled": bool(args.disable_bert_score),
        },
        "raptor_node_to_leaf_expansion_method": (
            "Use retrieval_payloads.expanded_retrieved_chunks when available. "
            "Otherwise expand retrieved_nodes.node_index through "
            "trees/<doc_id>/node_index.jsonl descendant_leaf_chunk_ids. "
            "After expansion, deduplicate leaf chunk ids while preserving first occurrence."
        ),
        "assumptions": [
            "Existing QA predictions are evaluated as-is; no retrieval, context construction, or answer generation is rerun.",
            "Ranking metrics use expanded and deduplicated leaf chunk ids in the order emitted by the run.",
            "doc_id is never used as a fallback relevance id for chunk-level retrieval metrics.",
            "metrics_per_query.jsonl intentionally excludes full contexts and raw node traces.",
            "Node-level raw traces remain in the original retrieval payloads and are referenced by the manifest instead of duplicated.",
        ],
        "missing_metrics_and_reasons": missing_metrics_reasons,
        "node_expansion_notes": sorted(set(expansion_notes)),
        "command_used": " ".join(sys.argv),
        "package_versions": {
            "python": platform.python_version(),
            "bert-score": bertscore_versions.get("bert-score"),
            "torch": bertscore_versions.get("torch"),
            "transformers": bertscore_versions.get("transformers"),
        },
        "source_run_manifest": run_manifest,
    }
    manifest = round_float(manifest)

    write_json(metrics_summary_path, summary)
    write_jsonl(metrics_per_query_path, per_query_rows)
    write_json(leaderboard_row_path, leaderboard_row)
    write_json(evaluation_manifest_path, manifest)

    print(json.dumps(output_files, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
