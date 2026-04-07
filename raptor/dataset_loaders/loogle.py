"""Native LooGLE loader for RAPTOR experiments."""

from __future__ import annotations

import ast
import json
import logging
import os
from importlib.metadata import version as pkg_version
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_DATASET_IDS = ("bigai-nlco/LooGLE", "bigainlco/LooGLE")
_LEGACY_CONFIG_ALIASES = {
    "longdep_summarization": "summarization",
}


def _datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for LooGLE loading. Install with `pip install datasets`."
        ) from exc
    return load_dataset, get_dataset_config_names


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        if not raw:
            return None
        return int(raw.split(".", 1)[0])
    except Exception:
        return None


def _coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_coerce_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [_coerce_to_text(item) for item in value.values()]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def _normalize_requested_config(requested: str | None) -> str:
    raw = str(requested or "").strip()
    if not raw:
        return "shortdep_qa"
    return _LEGACY_CONFIG_ALIASES.get(raw, raw)


def _config_candidates(requested: str | None) -> list[str]:
    normalized = _normalize_requested_config(requested)
    candidates = [normalized]
    for old_name, new_name in _LEGACY_CONFIG_ALIASES.items():
        if normalized == new_name:
            candidates.append(old_name)
        elif normalized == old_name:
            candidates.append(new_name)
    return candidates


def _get_config_names(dataset_name: str) -> list[str]:
    _, get_dataset_config_names = _datasets()
    major = _datasets_version_major()
    kwargs: dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            return list(get_dataset_config_names(dataset_name, **kwargs) or [])
        except TypeError:
            return list(get_dataset_config_names(dataset_name) or [])
    except Exception:
        return []


def _resolve_dataset_and_config(
    requested_config_name: str | None,
) -> tuple[str, str | None]:
    desired_candidates = _config_candidates(requested_config_name)
    for dataset_name in _DATASET_IDS:
        configs = _get_config_names(dataset_name)
        if not configs:
            continue
        for config_name in desired_candidates:
            if config_name in configs:
                return dataset_name, config_name
        if "shortdep_qa" in configs:
            return dataset_name, "shortdep_qa"
        return dataset_name, configs[0]
    return _DATASET_IDS[0], desired_candidates[0] if desired_candidates else None


def _load_loogle_dataset_handle(*, config_name: str | None):
    load_dataset, _ = _datasets()
    major = _datasets_version_major()
    preferred_name, preferred_config = _resolve_dataset_and_config(config_name)
    dataset_order = [preferred_name] + [
        dataset_name for dataset_name in _DATASET_IDS if dataset_name != preferred_name
    ]

    config_candidates: list[str | None] = [preferred_config]
    for candidate in _config_candidates(config_name):
        if candidate not in config_candidates:
            config_candidates.append(candidate)

    kwargs_candidates: list[dict[str, Any]] = []
    if major is not None and major >= 4:
        kwargs_candidates.append({"revision": "refs/convert/parquet"})
    kwargs_candidates.append({})
    if major is None or major < 4:
        kwargs_candidates.append({"trust_remote_code": True})

    def attempt_load(*, force_offline: bool):
        previous_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
        offline_download_config = None
        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                from datasets import DownloadConfig  # type: ignore

                offline_download_config = DownloadConfig(local_files_only=True)
            except Exception:
                offline_download_config = None

        last_exception: Exception | None = None
        try:
            for dataset_name in dataset_order:
                for candidate_config in config_candidates:
                    for kwargs in kwargs_candidates:
                        call_kwargs = dict(kwargs)
                        if (
                            force_offline
                            and offline_download_config is not None
                            and "download_config" not in call_kwargs
                        ):
                            call_kwargs["download_config"] = offline_download_config
                        try:
                            return (
                                load_dataset(
                                    dataset_name,
                                    name=candidate_config,
                                    **call_kwargs,
                                )
                                if candidate_config
                                else load_dataset(dataset_name, **call_kwargs)
                            )
                        except TypeError:
                            try:
                                return (
                                    load_dataset(dataset_name, name=candidate_config)
                                    if candidate_config
                                    else load_dataset(dataset_name)
                                )
                            except Exception as exc:
                                last_exception = exc
                        except Exception as exc:
                            last_exception = exc
            return last_exception
        finally:
            if force_offline:
                if previous_hub_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = previous_hub_offline
                if previous_datasets_offline is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = previous_datasets_offline

    first_try = attempt_load(force_offline=False)
    if not isinstance(first_try, Exception):
        return first_try

    second_try = attempt_load(force_offline=True)
    if not isinstance(second_try, Exception):
        return second_try

    requested = config_name if config_name is not None else "default"
    raise RuntimeError(
        f"Failed to load LooGLE dataset (requested config={requested!r}) "
        f"from any known dataset id: {_DATASET_IDS}."
    ) from second_try


def _extract_document_text(row: dict[str, Any]) -> str | None:
    for key in ("context", "input", "document", "text", "article", "passage"):
        text = _coerce_to_text(row.get(key))
        if text:
            return text
    return None


def _extract_doc_id(row: dict[str, Any], *, fallback_index: int) -> str:
    for key in ("doc_id", "document_id", "docid", "title"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"doc_{fallback_index}"


def _to_text_list(value: Any) -> list[str]:
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        for item in value:
            items.extend(_to_text_list(item))
        return items
    text = _coerce_to_text(value)
    if text:
        items.append(text)
    return items


def _parse_qa_pairs(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
    return []


def _iter_rows_for_split(dataset_handle: Any, *, split: str) -> Iterable[dict[str, Any]]:
    rows = dataset_handle[split]
    for row in rows:
        if isinstance(row, dict):
            yield row


def load_loogle_dataset(
    *,
    split: str = "test",
    config_name: str | None = "shortdep_qa",
) -> dict[str, list[dict[str, Any]]]:
    """
    Load LooGLE into the internal experiment-runner schema.
    """

    logger.info("Loading LooGLE split=%s config=%s", split, config_name)
    dataset_handle = _load_loogle_dataset_handle(config_name=config_name)
    resolved_dataset_name, resolved_config = _resolve_dataset_and_config(config_name)

    documents: list[dict[str, Any]] = []
    qa_entries: list[dict[str, Any]] = []

    for index, row in enumerate(_iter_rows_for_split(dataset_handle, split=split)):
        doc_id = _extract_doc_id(row, fallback_index=index)
        text = _extract_document_text(row) or ""
        if text:
            documents.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": {
                        "dataset_name": resolved_dataset_name,
                        "config_name": resolved_config,
                        "split": split,
                        "title": row.get("title"),
                    },
                }
            )

        qa_pairs = _parse_qa_pairs(row.get("qa_pairs"))
        if qa_pairs:
            for question_index, pair in enumerate(qa_pairs):
                question = _coerce_to_text(pair.get("Q") or pair.get("question"))
                if not question:
                    continue
                answer_texts = _to_text_list(pair.get("A") or pair.get("answer"))
                evidence_spans = _to_text_list(pair.get("S") or pair.get("evidence"))
                if answer_texts or evidence_spans:
                    qa_entries.append(
                        {
                            "query_id": f"{doc_id}::qa::{question_index}",
                            "doc_id": doc_id,
                            "question": question,
                            "reference_answers": answer_texts,
                            "metadata": {
                                "retrieval_spans": evidence_spans,
                                "dataset_name": resolved_dataset_name,
                                "config_name": resolved_config,
                                "split": split,
                            },
                        }
                    )
            continue

        question = _coerce_to_text(
            row.get("question") or row.get("Q") or row.get("query")
        )
        if not question:
            continue
        answer_texts = _to_text_list(
            row.get("answer") or row.get("A") or row.get("answers")
        )
        evidence_spans = _to_text_list(
            row.get("evidence") or row.get("S") or row.get("span")
        )
        if answer_texts or evidence_spans:
            qa_entries.append(
                {
                    "query_id": f"{doc_id}::qa::0",
                    "doc_id": doc_id,
                    "question": question,
                    "reference_answers": answer_texts,
                    "metadata": {
                        "retrieval_spans": evidence_spans,
                        "dataset_name": resolved_dataset_name,
                        "config_name": resolved_config,
                        "split": split,
                    },
                }
            )

    logger.info(
        "Loaded LooGLE documents=%d qa_entries=%d",
        len(documents),
        len(qa_entries),
    )
    return {"documents": documents, "qa_entries": qa_entries}
