"""Native NarrativeQA loader for RAPTOR experiments."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for NarrativeQA loading. Install with `pip install datasets`."
        ) from exc
    return load_dataset, get_dataset_config_names


def _coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_coerce_to_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [_coerce_to_text(item) for item in value.values()]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def _resolve_config(
    requested: str | None, dataset_name: str = "deepmind/narrativeqa"
) -> str | None:
    _, get_dataset_config_names = _datasets()
    try:
        configs = get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_narrativeqa_dataset_handle(*, config_name: str | None):
    load_dataset, _ = _datasets()
    try:
        return (
            load_dataset("deepmind/narrativeqa", name=config_name)
            if config_name
            else load_dataset("deepmind/narrativeqa")
        )
    except ValueError as exc:
        message = str(exc)
        if "Feature type 'List' not found" in message:
            raise RuntimeError(
                "Loading 'deepmind/narrativeqa' failed because the installed `datasets` "
                "package is too old for this dataset metadata. Upgrade datasets, for example "
                "with `python -m pip install -U \"datasets>=4\"`, and retry."
            ) from exc
        raise


def load_narrativeqa_dataset(
    *,
    split: str = "test",
    config_name: str | None = "default",
) -> dict[str, list[dict[str, Any]]]:
    """
    Load NarrativeQA into the internal experiment-runner schema.
    """

    resolved_config = _resolve_config(config_name)
    logger.info(
        "Loading NarrativeQA split=%s config=%s",
        split,
        resolved_config,
    )
    dataset_handle = _load_narrativeqa_dataset_handle(config_name=resolved_config)

    documents: list[dict[str, Any]] = []
    qa_entries: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    question_counts_by_doc_id: dict[str, int] = {}

    for row in dataset_handle[split]:
        if not isinstance(row, dict):
            continue

        document_payload = row.get("document") or {}
        doc_id = (
            document_payload.get("id")
            if isinstance(document_payload, dict)
            else None
        ) or row.get("document_id") or row.get("doc_id")
        if not doc_id:
            continue

        doc_id = str(doc_id)
        document_text = _coerce_to_text(
            document_payload.get("text")
            if isinstance(document_payload, dict)
            else document_payload
        )

        if document_text and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            documents.append(
                {
                    "doc_id": doc_id,
                    "text": document_text,
                    "metadata": {
                        "dataset_name": "deepmind/narrativeqa",
                        "config_name": resolved_config,
                        "split": split,
                        "kind": (
                            document_payload.get("kind")
                            if isinstance(document_payload, dict)
                            else None
                        ),
                        "title": (
                            document_payload.get("title")
                            if isinstance(document_payload, dict)
                            else None
                        ),
                    },
                }
            )

        question = row.get("question")
        if isinstance(question, dict):
            question = question.get("text")
        if not isinstance(question, str) or not question.strip():
            continue

        answer_text = _coerce_to_text(row.get("answers"))
        if not answer_text:
            continue

        question_index = question_counts_by_doc_id.get(doc_id, 0)
        question_counts_by_doc_id[doc_id] = question_index + 1

        qa_entries.append(
            {
                "query_id": f"{doc_id}::qa::{question_index}",
                "doc_id": doc_id,
                "question": question.strip(),
                "reference_answers": [answer_text],
                "metadata": {
                    "retrieval_spans": [],
                    "dataset_name": "deepmind/narrativeqa",
                    "config_name": resolved_config,
                    "split": split,
                },
            }
        )

    logger.info(
        "Loaded NarrativeQA documents=%d qa_entries=%d",
        len(documents),
        len(qa_entries),
    )
    return {"documents": documents, "qa_entries": qa_entries}
