"""Native QuALITY loader for RAPTOR experiments."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DATASET_ID = "tasksource/QuALITY"
_VALID_SPLITS = {"train", "validation"}
_SPLIT_ALIASES = {
    "default": "validation",
    "dev": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "validation",
    "train": "train",
}


def _datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for QuALITY loading. Install with `pip install datasets`."
        ) from exc
    return load_dataset


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


def _normalize_split(split: str | None) -> str:
    raw = str(split or "validation").strip().lower()
    normalized = _SPLIT_ALIASES.get(raw, raw)
    if normalized not in _VALID_SPLITS:
        raise ValueError(
            "Unsupported QuALITY split. Use one of: train, validation "
            f"(got: {split!r})."
        )
    return normalized


def _load_quality_split(split_name: str):
    load_dataset = _datasets()
    try:
        return load_dataset(_DATASET_ID, split=split_name)
    except TypeError:
        return load_dataset(_DATASET_ID)[split_name]


def _row_doc_id(row: dict[str, Any], *, fallback_index: int) -> str:
    for key in ("article_id", "document_id", "doc_id"):
        value = row.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return f"article:{str(value).strip()}"
    value = row.get("title")
    if isinstance(value, str) and value.strip():
        safe_title = "_".join(value.strip().split())
        return f"article:{safe_title}"
    return f"article:{fallback_index}"


def _options_list(row: dict[str, Any]) -> list[str]:
    raw_options = row.get("options")
    if isinstance(raw_options, list):
        return [
            text
            for text in (_coerce_to_text(item) for item in raw_options)
            if text
        ]
    return []


def _gold_option_index(row: dict[str, Any], options: list[str]) -> int | None:
    for key in ("gold_label", "writer_label"):
        value = row.get(key)
        try:
            index = int(value)
        except Exception:
            continue
        if 1 <= index <= len(options):
            return index - 1
        if 0 <= index < len(options):
            return index
    return None


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("title", "source", "author", "topic", "year"):
        value = _coerce_to_text(row.get(key))
        if value:
            payload[key] = value
    difficult = row.get("difficult")
    try:
        if difficult is not None:
            payload["difficult"] = int(difficult)
    except Exception:
        pass
    return payload


def load_quality_dataset(
    *,
    split: str = "validation",
    config_name: str | None = "default",
) -> dict[str, list[dict[str, Any]]]:
    """
    Load QuALITY into the internal experiment-runner schema.
    """

    _ = config_name
    split_name = _normalize_split(split)
    logger.info("Loading QuALITY split=%s", split_name)
    rows = _load_quality_split(split_name)

    documents: list[dict[str, Any]] = []
    qa_entries: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        doc_id = _row_doc_id(row, fallback_index=index)
        article_text = _coerce_to_text(row.get("article"))
        if article_text and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            documents.append(
                {
                    "doc_id": doc_id,
                    "text": article_text,
                    "metadata": {
                        "dataset_name": _DATASET_ID,
                        "config_name": "default",
                        "split": split_name,
                        **_metadata(row),
                    },
                }
            )

        question = _coerce_to_text(row.get("question"))
        if not question:
            continue

        options = _options_list(row)
        gold_index = _gold_option_index(row, options)
        if gold_index is None or gold_index >= len(options):
            continue

        answer_text = options[gold_index]
        if not answer_text:
            continue

        qa_metadata = {
            "retrieval_spans": [],
            "choices": options,
            "gold_option_index": gold_index,
            "gold_option_label": gold_index + 1,
            "dataset_name": _DATASET_ID,
            "config_name": "default",
            "split": split_name,
        }
        row_metadata = _metadata(row)
        if row_metadata:
            qa_metadata.update(row_metadata)

        qa_entries.append(
            {
                "query_id": str(row.get("question_unique_id") or f"{doc_id}:q{index}"),
                "doc_id": doc_id,
                "question": question,
                "reference_answers": [answer_text],
                "metadata": qa_metadata,
            }
        )

    logger.info(
        "Loaded QuALITY documents=%d qa_entries=%d",
        len(documents),
        len(qa_entries),
    )
    return {"documents": documents, "qa_entries": qa_entries}
