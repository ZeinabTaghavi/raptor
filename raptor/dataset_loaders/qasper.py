"""Native QASPER loader for RAPTOR experiments."""

from __future__ import annotations

import logging
from importlib.metadata import version as pkg_version
from typing import Any

logger = logging.getLogger(__name__)


def _datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for QASPER loading. Install with `pip install datasets`."
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


def _resolve_config(
    requested: str | None, dataset_name: str = "allenai/qasper"
) -> str | None:
    _, get_dataset_config_names = _datasets()
    major = _datasets_version_major()
    kwargs = {}
    if major is None or major < 4:
        kwargs["trust_remote_code"] = True
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            configs = get_dataset_config_names(dataset_name, **kwargs) or []
        except TypeError:
            configs = get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_qasper_dataset_handle(*, config_name: str | None):
    load_dataset, _ = _datasets()
    major = _datasets_version_major()

    if major is not None and major >= 4:
        modern_kwargs = {"revision": "refs/convert/parquet"}
        try:
            return (
                load_dataset("allenai/qasper", name=config_name, **modern_kwargs)
                if config_name
                else load_dataset("allenai/qasper", **modern_kwargs)
            )
        except TypeError:
            return (
                load_dataset("allenai/qasper", name=config_name)
                if config_name
                else load_dataset("allenai/qasper")
            )
        except Exception as exc:
            message = str(exc)
            if (
                "Dataset scripts are no longer supported" in message
                and "qasper.py" in message
            ):
                raise RuntimeError(
                    "Loading 'allenai/qasper' failed with datasets>=4 because script "
                    "execution is disabled and the parquet fallback could not be resolved."
                ) from exc
            raise

    legacy_kwargs = {"trust_remote_code": True}
    try:
        try:
            return (
                load_dataset("allenai/qasper", name=config_name, **legacy_kwargs)
                if config_name
                else load_dataset("allenai/qasper", **legacy_kwargs)
            )
        except TypeError:
            return (
                load_dataset("allenai/qasper", name=config_name)
                if config_name
                else load_dataset("allenai/qasper")
            )
    except RuntimeError as exc:
        message = str(exc)
        if "trust_remote_code" in message:
            raise RuntimeError(
                "Loading 'allenai/qasper' requires script loading. Use `datasets<4` "
                "or a parquet-converted branch."
            ) from exc
        raise


def _normalize_document_text(full_text: Any) -> str:
    paragraphs = (full_text or {}).get("paragraphs", "") if isinstance(full_text, dict) else full_text
    if isinstance(paragraphs, list):
        if paragraphs and isinstance(paragraphs[0], list):
            return "\n".join(
                paragraph
                for section in paragraphs
                for paragraph in section
                if isinstance(paragraph, str) and paragraph.strip()
            )
        return "\n".join(
            paragraph for paragraph in paragraphs if isinstance(paragraph, str) and paragraph.strip()
        )
    return _coerce_to_text(paragraphs)


def load_qasper_dataset(
    *,
    split: str = "test",
    config_name: str | None = "default",
) -> dict[str, list[dict[str, Any]]]:
    """
    Load QASPER into the internal experiment-runner schema.

    Returns:
        {
            "documents": [{"doc_id": ..., "text": ..., "metadata": ...}, ...],
            "qa_entries": [{"query_id": ..., "doc_id": ..., "question": ..., "reference_answers": ..., "metadata": ...}, ...],
        }
    """

    resolved_config = _resolve_config(config_name)
    logger.info("Loading QASPER split=%s config=%s", split, resolved_config)
    dataset = _load_qasper_dataset_handle(config_name=resolved_config)

    documents: list[dict[str, Any]] = []
    qa_entries: list[dict[str, Any]] = []

    for row in dataset[split]:
        doc_id = str(row.get("id"))
        text = _normalize_document_text(row.get("full_text"))
        documents.append(
            {
                "doc_id": doc_id,
                "text": text,
                "metadata": {
                    "dataset_name": "allenai/qasper",
                    "config_name": resolved_config,
                    "split": split,
                    "title": row.get("title"),
                    "abstract": row.get("abstract"),
                },
            }
        )

        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        answers_all = qas.get("answers") or []
        for question_index, (question, answer_dict) in enumerate(
            zip(questions, answers_all)
        ):
            answer_texts: list[str] = []
            retrieval_spans: list[str] = []
            for answer in (answer_dict or {}).get("answer", []) or []:
                if not isinstance(answer, dict) or answer.get("unanswerable"):
                    continue
                if answer.get("extractive_spans"):
                    text_value = " ".join(
                        span
                        for span in answer.get("extractive_spans", [])
                        if isinstance(span, str)
                    ).strip()
                elif isinstance(answer.get("free_form_answer"), str) and answer.get(
                    "free_form_answer", ""
                ).strip():
                    text_value = answer["free_form_answer"].strip()
                elif answer.get("yes_no") is not None:
                    text_value = "Yes" if bool(answer["yes_no"]) else "No"
                else:
                    text_value = ""

                evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
                if text_value:
                    answer_texts.append(text_value)
                if isinstance(evidence, str) and evidence.strip():
                    retrieval_spans.append(evidence.strip())
                elif isinstance(evidence, list):
                    retrieval_spans.extend(
                        span.strip()
                        for span in evidence
                        if isinstance(span, str) and span.strip()
                    )

            if not answer_texts and not retrieval_spans:
                continue

            qa_entries.append(
                {
                    "query_id": f"{doc_id}::qa::{question_index}",
                    "doc_id": doc_id,
                    "question": str(question).strip(),
                    "reference_answers": answer_texts,
                    "metadata": {
                        "retrieval_spans": retrieval_spans,
                        "dataset_name": "allenai/qasper",
                        "config_name": resolved_config,
                        "split": split,
                    },
                }
            )

    logger.info(
        "Loaded QASPER documents=%d qa_entries=%d",
        len(documents),
        len(qa_entries),
    )
    return {"documents": documents, "qa_entries": qa_entries}
