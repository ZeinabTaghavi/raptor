"""Native NovelHopQA whole-book loader for RAPTOR experiments."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET_ID = "abhaygupta1266/novelhopqa"
_VALID_SPLITS = ("hop_1", "hop_2", "hop_3", "hop_4")
_CONFIG_ALIASES = {
    "default": "all",
    "full": "all",
    "all_hops": "all",
}


def _datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for NovelHopQA loading. Install with `pip install datasets`."
        ) from exc
    return load_dataset


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
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [_coerce_to_text(item) for item in value.values()]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def _normalize_config(config_name: str | None) -> str:
    raw = str(config_name or "all").strip().lower()
    normalized = _CONFIG_ALIASES.get(raw, raw)
    if normalized == "all" or normalized in _VALID_SPLITS:
        return normalized
    raise ValueError(
        "Unsupported NovelHopQA config_name. Use one of: all/default, hop_1, hop_2, hop_3, hop_4."
    )


def _selected_splits(mode: str) -> list[str]:
    if mode == "all":
        return list(_VALID_SPLITS)
    return [mode]


def _load_novelhopqa_split(split_name: str):
    load_dataset = _datasets()
    major = _datasets_version_major()
    kwargs: dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        return load_dataset(_DATASET_ID, "default", split=split_name, **kwargs)
    except TypeError:
        try:
            return load_dataset(_DATASET_ID, split=split_name, **kwargs)
        except TypeError:
            return load_dataset(_DATASET_ID, split=split_name)
    except Exception:
        return load_dataset(_DATASET_ID, "default", split=split_name)


def _safe_component(value: str, *, default: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return out or default


def _normalize_book_key(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _query_id(row: dict[str, Any], *, split_name: str, index: int) -> str:
    base = row.get("qid") or row.get("question_id") or row.get("id") or index
    return f"{split_name}:{_safe_component(str(base), default=str(index))}"


def _find_top_file_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_file() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _find_child_dir_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _looks_like_books_root(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_file():
        return root.suffix.lower() == ".txt"
    if _find_top_file_ci(root, "bookmeta.json") is not None:
        return True
    if _find_child_dir_ci(root, "Books") is not None:
        return True
    try:
        return any(child.is_file() and child.suffix.lower() == ".txt" for child in root.iterdir())
    except Exception:
        return False


def _coerce_books_root(raw_root: Path) -> Path:
    root = raw_root.expanduser().resolve()
    if _looks_like_books_root(root):
        return root
    try:
        for child in root.iterdir():
            if child.is_dir() and _looks_like_books_root(child):
                return child.resolve()
    except Exception:
        pass
    return root


def _resolve_books_root(raw_root: str | os.PathLike[str] | None) -> Path:
    configured_raw = str(raw_root).strip() if raw_root is not None else ""
    env_books_root = str(os.environ.get("NOVELHOPQA_BOOKS_ROOT") or "").strip()
    env_novelqa_root = str(os.environ.get("NOVELQA_DATASET_DIR") or "").strip()
    candidates = [
        ("NOVELHOPQA_BOOKS_ROOT", env_books_root),
        ("NOVELQA_DATASET_DIR", env_novelqa_root),
        ("dataset.books_root", configured_raw),
    ]
    unusable_attempts: list[str] = []
    saw_candidate = False
    for label, configured in candidates:
        if not configured:
            continue
        saw_candidate = True
        root = _coerce_books_root(Path(configured))
        if _looks_like_books_root(root):
            if unusable_attempts:
                logger.warning(
                    "Ignoring unusable NovelHopQA corpus root candidate(s): %s. Using %s=%s",
                    ", ".join(unusable_attempts),
                    label,
                    root,
                )
            return root
        unusable_attempts.append(f"{label}={root}")

    if not saw_candidate:
        raise RuntimeError(
            "NovelHopQA whole-book loading requires a book corpus root. "
            "Set dataset.loader.books_root, dataset.books_root, or NOVELHOPQA_BOOKS_ROOT "
            "to a directory containing bookmeta.json and book text files."
        )
    raise RuntimeError(
        "NovelHopQA whole-book loading could not find a usable corpus root at "
        f"{'; '.join(unusable_attempts)}. Expected bookmeta.json, a Books/ directory, or .txt files."
    )


def _iter_bookmeta_entries(payload: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                out.append(item)
        return out
    if isinstance(payload, dict):
        books = payload.get("books")
        if isinstance(books, list):
            for item in books:
                if isinstance(item, dict):
                    out.append(item)
            return out
        for key, value in payload.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("BID", key)
                out.append(row)
    return out


def _candidate_text_paths(root: Path, file_name: str) -> list[Path]:
    file_path = Path(str(file_name))
    if file_path.is_absolute():
        return [file_path]
    if len(file_path.parts) > 1:
        return [(root / file_path).resolve()]
    return [
        (root / "Books" / "PublicDomain" / file_path).resolve(),
        (root / "Books" / "publicdomain" / file_path).resolve(),
        (root / "Books" / "CopyrightProtected" / file_path).resolve(),
        (root / "Books" / "copyrightprotected" / file_path).resolve(),
        (root / "Books" / file_path).resolve(),
        (root / file_path).resolve(),
    ]


def _book_doc_id(title: str, *, fallback: str) -> str:
    return f"book:{_safe_component(title, default=fallback)}"


def _book_title(row: dict[str, Any]) -> str | None:
    return _coerce_to_text(row.get("book") or row.get("title") or row.get("book_title"))


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _iter_title_variants(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants: list[str] = [raw]
    cleanup_suffixes = (
        " complete",
        " unabridged",
        " illustrated",
        " with illustrations",
    )
    for separator in (":", ";", ",", " - ", " — ", " – "):
        if separator in raw:
            head = raw.split(separator, 1)[0].strip()
            if head:
                variants.append(head)
    for candidate in list(variants):
        lowered = candidate.casefold()
        for prefix in ("the ", "a ", "an "):
            if lowered.startswith(prefix):
                trimmed = candidate[len(prefix):].strip()
                if trimmed:
                    variants.append(trimmed)
        for suffix in cleanup_suffixes:
            if lowered.endswith(suffix):
                trimmed = candidate[: -len(suffix)].strip(" ,;-:")
                if trimmed:
                    variants.append(trimmed)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in variants:
        key = _normalize_book_key(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _title_like_lines(text: str, *, limit: int = 5) -> list[str]:
    raw_candidates: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.replace("\ufeff", "").replace("\xa0", " ").strip()
        if not line:
            continue
        lowered = line.casefold()
        if "project gutenberg ebook of" in lowered:
            match = re.search(
                r"project gutenberg ebook of\s+(.+?)(?:,\s+by\b|$)",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                title = match.group(1).strip(" .,:;!-")
                if title:
                    raw_candidates.append(title)
                    if len(raw_candidates) >= limit:
                        break
            continue
        if (
            "project gutenberg" in lowered
            or "www.gutenberg.org" in lowered
            or lowered.startswith("***")
            or lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
            or lowered.startswith("release date")
            or lowered.startswith("language:")
            or lowered.startswith("contents")
            or lowered.startswith("chapter ")
            or lowered.startswith("book ")
        ):
            continue
        if len(line) > 160 or sum(ch.isalpha() for ch in line) < 3:
            continue
        raw_candidates.append(line)
        if len(raw_candidates) >= limit:
            break

    out: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        normalized = candidate.strip(" .,:;!-")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)

    stitched_parts: list[str] = []
    total_words = 0
    for candidate in raw_candidates[:3]:
        lowered = candidate.casefold()
        if stitched_parts and (
            lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
        ):
            break
        if len(candidate) > 80:
            break
        stitched_parts.append(candidate.strip(" .,:;!-"))
        total_words += len(candidate.split())
        if len(stitched_parts) < 2 or total_words > 12:
            continue
        stitched = " ".join(part for part in stitched_parts if part).strip(" .,:;!-")
        key = stitched.casefold()
        if stitched and key not in seen:
            seen.add(key)
            out.append(stitched)
    return out


def _register_book_aliases(
    books: dict[str, tuple[str, str]],
    *,
    doc_id: str,
    text: str,
    candidates: list[str],
) -> None:
    for raw in candidates:
        for variant in _iter_title_variants(raw):
            key = _normalize_book_key(variant)
            if key:
                books.setdefault(key, (doc_id, text))


def _resolve_report_dir(raw_report_dir: str | os.PathLike[str] | None) -> Path:
    env_report_dir = str(os.environ.get("NOVELHOPQA_REPORT_DIR") or "").strip()
    configured = env_report_dir or (str(raw_report_dir).strip() if raw_report_dir is not None else "")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "raptor_runs" / "novelhopqa" / "_loader_reports"


def _write_title_report(path: Path, titles: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        {str(title).strip() for title in titles if str(title).strip()},
        key=str.casefold,
    )
    content = "\n".join(ordered)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _load_books_from_root(root: Path) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    books: dict[str, tuple[str, str]] = {}
    titles_by_doc_id: dict[str, str] = {}
    bookmeta_path = _find_top_file_ci(root, "bookmeta.json")
    if bookmeta_path is not None and bookmeta_path.exists():
        payload = _read_json(bookmeta_path)
        for index, row in enumerate(_iter_bookmeta_entries(payload)):
            title = _coerce_to_text(row.get("title") or row.get("book") or row.get("name"))
            doc_id = str(
                row.get("BID") or row.get("bid") or _book_doc_id(title, fallback=f"book_{index}")
            ).strip()
            txt_name = row.get("txtfile") or row.get("txt_file") or row.get("book_file") or row.get("text_file")
            candidate_names: list[str] = []
            if isinstance(txt_name, str) and txt_name.strip():
                candidate_names.append(txt_name.strip())
            if doc_id:
                candidate_names.extend([f"{doc_id}.txt", f"{doc_id.upper()}.txt", f"{doc_id.lower()}.txt"])
            if not title or not candidate_names:
                continue
            seen_candidates: set[Path] = set()
            for raw_name in candidate_names:
                for candidate in _candidate_text_paths(root, raw_name):
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    if not candidate.exists():
                        continue
                    text = _load_text(candidate)
                    if not text:
                        continue
                    aliases = [title, doc_id, Path(raw_name).stem, candidate.stem, *_title_like_lines(text)]
                    canonical_title = str(title or candidate.stem or doc_id).strip()
                    if canonical_title:
                        titles_by_doc_id.setdefault(doc_id, canonical_title)
                    _register_book_aliases(books, doc_id=doc_id, text=text, candidates=aliases)
                    break
                else:
                    continue
                break
        if books:
            return books, titles_by_doc_id

    if root.is_file() and root.suffix.lower() == ".txt":
        title = root.stem
        text = _load_text(root)
        if not text:
            return {}, {}
        doc_id = _book_doc_id(title, fallback="book")
        titles_by_doc_id[doc_id] = title
        _register_book_aliases(books, doc_id=doc_id, text=text, candidates=[title, *_title_like_lines(text)])
        return books, titles_by_doc_id

    try:
        txt_files = sorted(path for path in root.rglob("*.txt") if path.is_file())
    except Exception:
        txt_files = []
    for path in txt_files:
        text = _load_text(path)
        if not text:
            continue
        stem = path.stem
        doc_id = _book_doc_id(stem, fallback=stem)
        title_candidates = _title_like_lines(text)
        canonical_title = str(title_candidates[0] if title_candidates else stem).strip()
        if canonical_title:
            titles_by_doc_id.setdefault(doc_id, canonical_title)
        _register_book_aliases(books, doc_id=doc_id, text=text, candidates=[stem, *title_candidates])
    return books, titles_by_doc_id


def _load_novelhopqa_all(
    *,
    mode: str,
    split: str,
    books_root: str | os.PathLike[str] | None,
    subset_mode: bool,
    report_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, str], Path]:
    _ = split
    root = _resolve_books_root(books_root)
    books_by_key, titles_by_doc_id = _load_books_from_root(root)
    if not books_by_key:
        raise RuntimeError(
            "NovelHopQA whole-book loading found zero books under "
            f"{root}. Expected title-mapped .txt files or bookmeta.json entries."
        )

    documents: dict[str, str] = {}
    qa_entries: list[dict[str, Any]] = []
    missing_books: set[str] = set()
    used_doc_ids: set[str] = set()

    for split_name in _selected_splits(mode):
        rows = _load_novelhopqa_split(split_name)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            book_title = _book_title(row)
            if not book_title:
                continue
            book_record = books_by_key.get(_normalize_book_key(book_title))
            if book_record is None:
                missing_books.add(book_title)
                continue
            doc_id, document_text = book_record
            used_doc_ids.add(doc_id)
            if document_text and doc_id not in documents:
                documents[doc_id] = document_text

            context = _coerce_to_text(row.get("context") or row.get("passage") or row.get("document"))
            question = _coerce_to_text(row.get("question") or row.get("query"))
            answer = _coerce_to_text(row.get("answer") or row.get("gold_answer"))
            if not context or not question or not answer:
                continue
            qa_entries.append(
                {
                    "query_id": _query_id(row, split_name=split_name, index=index),
                    "doc_id": doc_id,
                    "question": question,
                    "reference_answers": [answer],
                    "metadata": {
                        "book_title": book_title,
                        "gold_context_window": context,
                        "retrieval_span_mode": "window",
                        "retrieval_spans": [context],
                        "dataset_hop_split": split_name,
                    },
                }
            )

    _write_title_report(report_dir / "novelhop_query_remained.txt", missing_books)
    unused_books = {
        title for doc_id, title in titles_by_doc_id.items() if doc_id not in used_doc_ids
    }
    _write_title_report(report_dir / "novelhop_book_remained.txt", unused_books)

    if missing_books:
        sample = ", ".join(sorted(missing_books)[:5])
        if not subset_mode:
            raise RuntimeError(
                "NovelHopQA whole-book loading could not resolve book texts for "
                f"{len(missing_books)} title(s) under {root}. Example title(s): {sample}. "
                "Provide dataset.loader.books_root or NOVELHOPQA_BOOKS_ROOT pointing to a corpus with matching titles, "
                "or enable subset mode."
            )
        logger.warning(
            "NovelHopQA subset mode enabled; skipped %d unresolved title(s) under %s. Example title(s): %s. Reports written to %s",
            len(missing_books),
            root,
            sample,
            report_dir,
        )

    return documents, qa_entries, titles_by_doc_id, root


def load_novelhopqa_dataset(
    *,
    split: str = "test",
    config_name: str | None = "all",
    books_root: str | os.PathLike[str] | None = None,
    subset_mode: bool | None = None,
    report_dir: str | os.PathLike[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    mode = _normalize_config(config_name)
    resolved_subset_mode = _is_truthy(
        os.environ.get("NOVELHOPQA_SUBSET_MODE") if subset_mode is None else subset_mode
    )
    resolved_report_dir = _resolve_report_dir(report_dir)

    logger.info(
        "Loading NovelHopQA split=%s config=%s books_root=%s subset_mode=%s",
        split,
        mode,
        books_root or os.environ.get("NOVELHOPQA_BOOKS_ROOT") or "",
        resolved_subset_mode,
    )
    docs, qa_rows, titles_by_doc_id, resolved_books_root = _load_novelhopqa_all(
        mode=mode,
        split=split,
        books_root=books_root,
        subset_mode=resolved_subset_mode,
        report_dir=resolved_report_dir,
    )

    documents = [
        {
            "doc_id": doc_id,
            "text": text,
            "metadata": {
                "dataset_name": _DATASET_ID,
                "config_name": mode,
                "split": split,
                "book_title": titles_by_doc_id.get(doc_id, ""),
                "books_root": str(resolved_books_root),
            },
        }
        for doc_id, text in docs.items()
    ]

    qa_entries = []
    for row in qa_rows:
        metadata = {
            "dataset_name": _DATASET_ID,
            "config_name": mode,
            "split": split,
            "books_root": str(resolved_books_root),
            **dict(row.get("metadata") or {}),
        }
        qa_entries.append(
            {
                "query_id": str(row["query_id"]),
                "doc_id": str(row["doc_id"]),
                "question": str(row["question"]),
                "reference_answers": list(row.get("reference_answers") or []),
                "metadata": metadata,
            }
        )

    logger.info(
        "Loaded NovelHopQA documents=%d qa_entries=%d report_dir=%s",
        len(documents),
        len(qa_entries),
        resolved_report_dir,
    )
    return {"documents": documents, "qa_entries": qa_entries}
