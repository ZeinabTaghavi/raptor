import csv
import inspect
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .EmbeddingModels import (HashEmbeddingModel, OpenAIEmbeddingModel,
                              SBertEmbeddingModel, TransformersEmbeddingModel)
from .QAModels import (ExtractiveQAModel, GPT3TurboQAModel, GPT4QAModel,
                       TransformersQAModel, UnifiedQAModel, VLLMQAModel)
from .SummarizationModels import (ExtractiveSummarizationModel,
                                  GPT3SummarizationModel,
                                  GPT3TurboSummarizationModel,
                                  TransformersSummarizationModel,
                                  VLLMSummarizationModel)
from ._compat import get_default_tokenizer
from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .tree_retriever import FIXED_RETRIEVAL_TOP_K, TreeRetriever, TreeRetrieverConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DocumentRecord:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QARecord:
    query_id: str
    doc_id: str
    question: str
    reference_answers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDataset:
    documents: List[DocumentRecord]
    qa_entries: List[QARecord]


def _deep_get(data: Dict[str, Any], dotted_path: str, default=None):
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _first_present(data: Dict[str, Any], paths: Sequence[str], default=None):
    for path in paths:
        value = _deep_get(data, path)
        if value is not None:
            return value, path
    return default, None


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _infer_format(path: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.lower()

    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "text"
    raise ValueError(f"Could not infer file format for {path!r}.")


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_reference_answers(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _read_structured_records(path: str, file_format: str, records_path: Optional[str]) -> List[Dict[str, Any]]:
    file_format = _infer_format(path, file_format)

    if file_format == "jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if file_format == "json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = _deep_get(payload, records_path) if records_path else payload
        if not isinstance(records, list):
            raise ValueError(f"Expected list-like JSON records in {path}.")
        return records

    if file_format == "yaml":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        records = _deep_get(payload, records_path) if records_path else payload
        if not isinstance(records, list):
            raise ValueError(f"Expected list-like YAML records in {path}.")
        return records

    if file_format == "csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported structured file format: {file_format}")


def _load_documents(dataset_config: Dict[str, Any]) -> List[DocumentRecord]:
    documents_config = dataset_config["documents"]
    documents_path = documents_config["path"]
    documents_format = _infer_format(documents_path, documents_config.get("format"))
    split_value = dataset_config.get("split")

    if documents_format == "text":
        with open(documents_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        doc_id = documents_config.get("doc_id") or Path(documents_path).stem
        metadata = {"source_path": documents_path}
        if split_value is not None:
            metadata["split"] = split_value
        return [DocumentRecord(doc_id=str(doc_id), text=text, metadata=metadata)]

    raw_documents = _read_structured_records(
        documents_path,
        documents_format,
        documents_config.get("records_path"),
    )
    split_field = documents_config.get("split_field")
    id_field = documents_config.get("id_field", "doc_id")
    text_field = documents_config.get("text_field", "text")

    documents = []
    for index, row in enumerate(raw_documents):
        if split_value is not None and split_field and row.get(split_field) != split_value:
            continue
        doc_id = row.get(id_field, f"doc_{index}")
        text = row.get(text_field)
        if text is None:
            raise ValueError(f"Document record {doc_id!r} is missing text field {text_field!r}.")
        metadata = {key: value for key, value in row.items() if key not in {id_field, text_field}}
        documents.append(DocumentRecord(doc_id=str(doc_id), text=str(text), metadata=metadata))

    return documents


def _load_qa_entries(dataset_config: Dict[str, Any]) -> List[QARecord]:
    qa_config = dataset_config["qa"]
    qa_path = qa_config["path"]
    qa_format = _infer_format(qa_path, qa_config.get("format"))
    split_value = dataset_config.get("split")
    split_field = qa_config.get("split_field")

    raw_entries = _read_structured_records(
        qa_path,
        qa_format,
        qa_config.get("records_path"),
    )

    query_id_field = qa_config.get("query_id_field", "query_id")
    doc_id_field = qa_config.get("doc_id_field", "doc_id")
    question_field = qa_config.get("question_field", "question")
    reference_answers_field = qa_config.get("reference_answers_field", "reference_answers")

    qa_entries = []
    for index, row in enumerate(raw_entries):
        if split_value is not None and split_field and row.get(split_field) != split_value:
            continue

        question = row.get(question_field)
        if question is None:
            raise ValueError(f"QA record at index {index} is missing question field {question_field!r}.")

        query_id = row.get(query_id_field) or f"query_{index}"
        doc_id = row.get(doc_id_field)
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {query_id_field, doc_id_field, question_field, reference_answers_field}
        }
        qa_entries.append(
            QARecord(
                query_id=str(query_id),
                doc_id=str(doc_id) if doc_id is not None else "",
                question=str(question),
                reference_answers=_normalize_reference_answers(row.get(reference_answers_field)),
                metadata=metadata,
            )
        )

    return qa_entries


def _load_combined_dataset(dataset_config: Dict[str, Any]) -> LoadedDataset:
    combined_config = dataset_config["combined"]
    combined_path = combined_config["path"]
    combined_format = _infer_format(combined_path, combined_config.get("format"))
    split_value = dataset_config.get("split")

    records = _read_structured_records(
        combined_path,
        combined_format,
        combined_config.get("records_path"),
    )

    doc_id_field = combined_config.get("doc_id_field", "doc_id")
    text_field = combined_config.get("text_field", "text")
    qa_field = combined_config.get("qa_field", "qa_entries")
    split_field = combined_config.get("split_field")
    qa_query_id_field = combined_config.get("qa_query_id_field", "query_id")
    qa_question_field = combined_config.get("qa_question_field", "question")
    qa_reference_answers_field = combined_config.get("qa_reference_answers_field", "reference_answers")

    documents = []
    qa_entries = []

    for doc_index, row in enumerate(records):
        if split_value is not None and split_field and row.get(split_field) != split_value:
            continue

        doc_id = str(row.get(doc_id_field, f"doc_{doc_index}"))
        text = row.get(text_field)
        if text is None:
            raise ValueError(f"Combined dataset record {doc_id!r} is missing text field {text_field!r}.")

        metadata = {key: value for key, value in row.items() if key not in {doc_id_field, text_field, qa_field}}
        documents.append(DocumentRecord(doc_id=doc_id, text=str(text), metadata=metadata))

        raw_qa_entries = row.get(qa_field, [])
        if not isinstance(raw_qa_entries, list):
            raise ValueError(f"Combined dataset record {doc_id!r} has a non-list {qa_field!r} field.")

        for qa_index, qa_entry in enumerate(raw_qa_entries):
            question = qa_entry.get(qa_question_field)
            if question is None:
                raise ValueError(f"QA entry {qa_index} for document {doc_id!r} is missing question text.")
            query_id = qa_entry.get(qa_query_id_field) or f"{doc_id}::query::{qa_index}"
            qa_metadata = {
                key: value
                for key, value in qa_entry.items()
                if key not in {qa_query_id_field, qa_question_field, qa_reference_answers_field}
            }
            qa_entries.append(
                QARecord(
                    query_id=str(query_id),
                    doc_id=doc_id,
                    question=str(question),
                    reference_answers=_normalize_reference_answers(
                        qa_entry.get(qa_reference_answers_field)
                    ),
                    metadata=qa_metadata,
                )
            )

    return LoadedDataset(documents=documents, qa_entries=qa_entries)


def _load_named_dataset(dataset_config: Dict[str, Any]) -> LoadedDataset:
    loader_config = dataset_config.get("loader") or {}
    loader_name = loader_config.get("name")
    if not loader_name:
        raise ValueError("dataset.loader.name is required for named dataset loading.")

    from .dataset_loaders import SUPPORTED_DATASET_LOADERS

    if loader_name not in SUPPORTED_DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset loader {loader_name!r}. Supported loaders: {sorted(SUPPORTED_DATASET_LOADERS)}"
        )

    load_function = SUPPORTED_DATASET_LOADERS[loader_name]
    payload = load_function(
        split=dataset_config.get("split", "test"),
        **{
            key: value
            for key, value in loader_config.items()
            if key != "name" and value is not None
        },
    )

    documents = [
        DocumentRecord(
            doc_id=str(document["doc_id"]),
            text=str(document["text"]),
            metadata=dict(document.get("metadata") or {}),
        )
        for document in payload.get("documents", [])
    ]
    qa_entries = [
        QARecord(
            query_id=str(entry["query_id"]),
            doc_id=str(entry["doc_id"]),
            question=str(entry["question"]),
            reference_answers=_normalize_reference_answers(
                entry.get("reference_answers")
            ),
            metadata=dict(entry.get("metadata") or {}),
        )
        for entry in payload.get("qa_entries", [])
    ]
    return LoadedDataset(documents=documents, qa_entries=qa_entries)


def load_dataset(dataset_config: Dict[str, Any]) -> LoadedDataset:
    if dataset_config.get("loader", {}).get("name"):
        dataset = _load_named_dataset(dataset_config)
    elif dataset_config.get("combined", {}).get("path"):
        dataset = _load_combined_dataset(dataset_config)
    else:
        documents = _load_documents(dataset_config)
        qa_entries = _load_qa_entries(dataset_config)
        dataset = LoadedDataset(documents=documents, qa_entries=qa_entries)

    if len(dataset.documents) == 1:
        sole_doc_id = dataset.documents[0].doc_id
        for qa_entry in dataset.qa_entries:
            if not qa_entry.doc_id:
                qa_entry.doc_id = sole_doc_id

    missing_doc_ids = sorted(
        {qa_entry.doc_id for qa_entry in dataset.qa_entries if qa_entry.doc_id}
        - {document.doc_id for document in dataset.documents}
    )
    if missing_doc_ids:
        raise ValueError(
            "QA entries reference unknown doc_ids: " + ", ".join(missing_doc_ids)
        )

    if any(not qa_entry.doc_id for qa_entry in dataset.qa_entries):
        raise ValueError(
            "Each QA entry must have a doc_id unless the dataset contains exactly one document."
        )

    return dataset


def _apply_selection(dataset: LoadedDataset, dataset_config: Dict[str, Any]) -> LoadedDataset:
    selection = dataset_config.get("selection", {})

    selected_documents = list(dataset.documents)
    selected_qa_entries = list(dataset.qa_entries)

    requested_doc_ids = selection.get("doc_ids") or []
    if requested_doc_ids:
        document_by_id = {document.doc_id: document for document in selected_documents}
        selected_documents = [
            document_by_id[doc_id]
            for doc_id in requested_doc_ids
            if doc_id in document_by_id
        ]
    max_docs = selection.get("max_docs")
    if max_docs is not None:
        selected_documents = selected_documents[: int(max_docs)]

    selected_doc_ids = {document.doc_id for document in selected_documents}
    selected_qa_entries = [
        qa_entry for qa_entry in selected_qa_entries if qa_entry.doc_id in selected_doc_ids
    ]

    requested_query_ids = selection.get("query_ids") or []
    if requested_query_ids:
        query_by_id = {qa_entry.query_id: qa_entry for qa_entry in selected_qa_entries}
        selected_qa_entries = [
            query_by_id[query_id]
            for query_id in requested_query_ids
            if query_id in query_by_id
        ]

    max_questions_per_doc = selection.get("max_questions_per_doc")
    if max_questions_per_doc is not None:
        counts = defaultdict(int)
        filtered_entries = []
        for qa_entry in selected_qa_entries:
            if counts[qa_entry.doc_id] >= int(max_questions_per_doc):
                continue
            filtered_entries.append(qa_entry)
            counts[qa_entry.doc_id] += 1
        selected_qa_entries = filtered_entries

    max_questions = selection.get("max_questions")
    if max_questions is not None:
        selected_qa_entries = selected_qa_entries[: int(max_questions)]

    return LoadedDataset(documents=selected_documents, qa_entries=selected_qa_entries)


def _package_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_dump_yaml(payload: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _write_json(path: Path, payload: Any):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_jsonl(path: Path, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _git_commit(workdir: Path):
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=workdir,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def _cpu_memory_mb():
    try:
        import resource
    except ImportError:
        return None

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(value / (1024 * 1024), 3)
    return round(value / 1024, 3)


def _gpu_memory_snapshot():
    try:
        import torch
    except ImportError:
        return {"gpu_memory_mb": None, "gpu_peak_memory_mb": None}

    if not torch.cuda.is_available():
        return {"gpu_memory_mb": None, "gpu_peak_memory_mb": None}

    device = torch.cuda.current_device()
    return {
        "gpu_memory_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3),
        "gpu_peak_memory_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3
        ),
    }


def _reset_gpu_peak_memory():
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _resource_usage_record(phase: str, doc_id: Optional[str] = None, query_id: Optional[str] = None):
    snapshot = _gpu_memory_snapshot()
    return {
        "phase": phase,
        "doc_id": doc_id,
        "query_id": query_id,
        "cpu_memory_mb": _cpu_memory_mb(),
        **snapshot,
    }


def _hardware_summary():
    summary = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch
    except ImportError:
        summary["cuda_available"] = False
        return summary

    summary["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        summary["cuda_device_count"] = torch.cuda.device_count()
        summary["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return summary


def _node_layer_map(tree) -> Dict[int, int]:
    layer_map = {}
    for layer_number, nodes in tree.layer_to_nodes.items():
        for node in nodes:
            layer_map[node.index] = layer_number
    return layer_map


def _leaf_chunk_rows(tree, doc_id: str, tokenizer) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    leaf_nodes = sorted(tree.leaf_nodes.values(), key=lambda node: node.index)
    rows = []
    lookup = {}
    for chunk_index, node in enumerate(leaf_nodes):
        chunk_id = f"{doc_id}::chunk::{chunk_index}"
        row = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "raw_text": node.text,
            "token_count": len(tokenizer.encode(node.text)),
            "node_index": node.index,
        }
        rows.append(row)
        lookup[node.index] = row
    return rows, lookup


def _descendant_leaf_lookup(tree, leaf_lookup: Dict[int, Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    cache: Dict[int, List[Dict[str, Any]]] = {}

    def resolve(node_index: int):
        if node_index in cache:
            return cache[node_index]

        node = tree.all_nodes[node_index]
        if not node.children:
            cache[node_index] = [leaf_lookup[node_index]]
            return cache[node_index]

        descendant_rows = []
        for child_index in sorted(node.children):
            descendant_rows.extend(resolve(child_index))
        unique_rows = []
        seen_chunk_ids = set()
        for row in descendant_rows:
            if row["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(row["chunk_id"])
            unique_rows.append(row)
        cache[node_index] = unique_rows
        return cache[node_index]

    for node_index in tree.all_nodes:
        resolve(node_index)

    return cache


def _tree_artifact_rows(tree, doc_id: str, tokenizer, build_time_ms: float):
    layer_map = _node_layer_map(tree)
    leaf_rows, leaf_lookup = _leaf_chunk_rows(tree, doc_id, tokenizer)
    descendant_lookup = _descendant_leaf_lookup(tree, leaf_lookup)

    node_rows = []
    token_count_by_layer = defaultdict(int)
    total_tree_token_count = 0

    for node in sorted(tree.all_nodes.values(), key=lambda current: (layer_map[current.index], current.index)):
        token_count = len(tokenizer.encode(node.text))
        token_count_by_layer[layer_map[node.index]] += token_count
        total_tree_token_count += token_count
        node_rows.append(
            {
                "doc_id": doc_id,
                "node_index": node.index,
                "layer_number": layer_map[node.index],
                "is_leaf": not bool(node.children),
                "children": sorted(node.children),
                "descendant_leaf_chunk_ids": [
                    chunk_row["chunk_id"] for chunk_row in descendant_lookup[node.index]
                ],
                "text": node.text,
                "token_count": token_count,
            }
        )

    tree_stats = {
        "doc_id": doc_id,
        "num_nodes": len(tree.all_nodes),
        "num_leaf_nodes": len(tree.leaf_nodes),
        "num_layers": tree.num_layers,
        "total_tree_token_count": total_tree_token_count,
        "token_count_by_layer": {str(key): value for key, value in sorted(token_count_by_layer.items())},
        "build_time_ms": round(build_time_ms, 3),
        "peak_cpu_memory_mb": _cpu_memory_mb(),
        "peak_gpu_memory_mb": _gpu_memory_snapshot()["gpu_peak_memory_mb"],
    }

    return tree_stats, leaf_rows, node_rows, descendant_lookup


def _retrieved_chunk_rows(retrieved_nodes: List[Dict[str, Any]], descendant_lookup: Dict[int, List[Dict[str, Any]]]):
    chunk_rows = []
    for retrieved_node in retrieved_nodes:
        for chunk_row in descendant_lookup[retrieved_node["node_index"]]:
            chunk_rows.append(
                {
                    "chunk_id": chunk_row["chunk_id"],
                    "chunk_index": chunk_row["chunk_index"],
                    "via_node_index": retrieved_node["node_index"],
                    "via_node_rank": retrieved_node["rank"],
                }
            )
    return chunk_rows


def _dedup_chunk_ids(chunk_rows: List[Dict[str, Any]]) -> List[str]:
    return _ordered_unique(row["chunk_id"] for row in chunk_rows)


def _normalize_model_config(model_config: Dict[str, Any], default_provider: str, default_model: Optional[str] = None):
    provider = str(model_config.get("provider", default_provider)).lower()
    normalized = dict(model_config)
    normalized["provider"] = provider
    if default_model and "model" not in normalized:
        normalized["model"] = default_model
    return normalized


def _build_vllm_engine_kwargs(
    model_config: Dict[str, Any], *, excluded_keys: Sequence[str]
) -> Dict[str, Any]:
    engine_kwargs = {
        key: value
        for key, value in model_config.items()
        if key not in set(excluded_keys) | {"vllm_kwargs"}
    }

    extra_kwargs = model_config.get("vllm_kwargs", {}) or {}
    if isinstance(extra_kwargs, dict):
        engine_kwargs.update(extra_kwargs)

    return engine_kwargs


def _validate_runtime_dependencies(model_config: Dict[str, Any], label: str) -> None:
    provider = str(model_config.get("provider", "")).lower()
    if provider not in {"transformers", "hf"}:
        return

    uses_accelerate_features = any(
        model_config.get(key) is not None for key in ("device_map", "tp_plan")
    )
    if not uses_accelerate_features:
        return

    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"The {label} model uses provider='transformers' with device_map/tp_plan, which "
            "requires the 'accelerate' package for multi-GPU loading. Install it in the RAPTOR "
            "environment with `pip install accelerate`, or remove device_map to force single-device loading."
        ) from exc


def _build_embedding_model(model_config: Dict[str, Any]):
    provider = model_config["provider"]
    if provider == "openai":
        return OpenAIEmbeddingModel(model=model_config.get("model", "text-embedding-ada-002"))
    if provider in {"sbert", "sentence_transformers"}:
        return SBertEmbeddingModel(
            model_name=model_config.get(
                "model", "sentence-transformers/multi-qa-mpnet-base-cos-v1"
            )
        )
    if provider in {"transformers", "hf", "contriever"}:
        default_model_name = "facebook/contriever" if provider == "contriever" else "facebook/contriever"
        return TransformersEmbeddingModel(
            model_name=model_config.get("model", default_model_name),
            pooling=model_config.get("pooling", "mean"),
            normalize=bool(model_config.get("normalize", True)),
            max_length=int(model_config.get("max_length", 512)),
            trust_remote_code=bool(model_config.get("trust_remote_code", True)),
            device=model_config.get("device"),
        )
    if provider in {"hash", "local_hash"}:
        return HashEmbeddingModel(dimension=int(model_config.get("dimension", 256)))
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _build_summarization_model(model_config: Dict[str, Any]):
    provider = model_config["provider"]
    if provider in {"openai_chat", "gpt", "openai"}:
        model_name = model_config.get("model", "gpt-3.5-turbo")
        if model_name == "text-davinci-003":
            return GPT3SummarizationModel(model=model_name)
        return GPT3TurboSummarizationModel(model=model_name)
    if provider in {"transformers", "hf"}:
        return TransformersSummarizationModel(
            model_name=model_config["model"],
            pipeline_kwargs={
                key: value
                for key, value in model_config.items()
                if key
                not in {"provider", "model", "temperature", "top_p", "stop"}
            },
            temperature=float(model_config.get("temperature", 0.0)),
            top_p=float(model_config.get("top_p", 1.0)),
            stop=model_config.get("stop"),
        )
    if provider == "vllm":
        return VLLMSummarizationModel(
            model_name=model_config["model"],
            engine_kwargs=_build_vllm_engine_kwargs(
                model_config,
                excluded_keys={"provider", "model", "temperature", "top_p", "stop", "device"},
            ),
            temperature=float(model_config.get("temperature", 0.0)),
            top_p=float(model_config.get("top_p", 1.0)),
            stop=model_config.get("stop"),
        )
    if provider in {"extractive", "local"}:
        return ExtractiveSummarizationModel()
    raise ValueError(f"Unsupported summarization provider: {provider}")


def _build_qa_model(model_config: Dict[str, Any]):
    provider = model_config["provider"]
    if provider in {"openai_chat", "gpt", "openai"}:
        model_name = model_config.get("model", "gpt-3.5-turbo")
        if model_name == "gpt-4":
            return GPT4QAModel(model=model_name)
        return GPT3TurboQAModel(model=model_name)
    if provider in {"transformers", "hf"}:
        return TransformersQAModel(
            model_name=model_config["model"],
            pipeline_kwargs={
                key: value
                for key, value in model_config.items()
                if key
                not in {
                    "provider",
                    "model",
                    "default_max_tokens",
                    "temperature",
                    "top_p",
                    "stop",
                }
            },
            default_max_tokens=int(model_config.get("default_max_tokens", 256)),
            temperature=float(model_config.get("temperature", 0.0)),
            top_p=float(model_config.get("top_p", 1.0)),
            stop=model_config.get("stop"),
        )
    if provider == "vllm":
        return VLLMQAModel(
            model_name=model_config["model"],
            engine_kwargs=_build_vllm_engine_kwargs(
                model_config,
                excluded_keys={
                    "provider",
                    "model",
                    "default_max_tokens",
                    "temperature",
                    "top_p",
                    "stop",
                    "device",
                },
            ),
            default_max_tokens=int(model_config.get("default_max_tokens", 256)),
            temperature=float(model_config.get("temperature", 0.0)),
            top_p=float(model_config.get("top_p", 1.0)),
            stop=model_config.get("stop"),
        )
    if provider in {"unifiedqa"}:
        return UnifiedQAModel(model_name=model_config.get("model", "allenai/unifiedqa-v2-t5-3b-1363200"))
    if provider in {"extractive", "local"}:
        return ExtractiveQAModel()
    raise ValueError(f"Unsupported QA provider: {provider}")


def resolve_run_config(
    dataset_name: str,
    default_yaml_path: str,
    run_name: Optional[str] = None,
    output_root: Optional[str] = None,
    resume: bool = False,
):
    yaml_path = Path(default_yaml_path).resolve()
    with open(yaml_path, "r", encoding="utf-8") as handle:
        raw_reference = yaml.safe_load(handle) or {}

    explicit_config = raw_reference.get("raptor_run", raw_reference.get("raptor", raw_reference))

    notes = ["Per-document retrieval scope is enabled for all QA queries."]
    mapped_fields = []

    dataset_source_name, dataset_source_name_path = _first_present(
        raw_reference,
        ["dataset.loader.name", "dataset.name"],
    )
    if dataset_source_name_path:
        mapped_fields.append(f"Mapped dataset source from {dataset_source_name_path}.")

    dataset_split, split_path = _first_present(
        raw_reference,
        ["dataset.split", "data.split", "split"],
    )
    if split_path:
        mapped_fields.append(f"Mapped dataset split from {split_path}.")

    max_docs, max_docs_path = _first_present(
        raw_reference,
        [
            "dataset.max_docs",
            "dataset.sample_size",
            "sample.max_documents",
            "sample_size",
            "max_docs",
        ],
    )
    if max_docs_path:
        mapped_fields.append(f"Mapped max_docs from {max_docs_path}.")

    max_questions, max_questions_path = _first_present(
        raw_reference,
        ["dataset.max_questions", "questions.max_questions", "max_questions"],
    )
    if max_questions_path:
        mapped_fields.append(f"Mapped max_questions from {max_questions_path}.")

    max_questions_per_doc, max_questions_per_doc_path = _first_present(
        raw_reference,
        [
            "dataset.max_questions_per_doc",
            "questions.max_questions_per_doc",
            "max_questions_per_doc",
        ],
    )
    if max_questions_per_doc_path:
        mapped_fields.append(f"Mapped max_questions_per_doc from {max_questions_per_doc_path}.")

    retrieval_top_k, retrieval_top_k_path = _first_present(
        raw_reference,
        ["retrieval.top_k", "retrieval.retrieve_k", "top_k"],
    )
    if retrieval_top_k_path:
        mapped_fields.append(f"Mapped retrieval top_k from {retrieval_top_k_path}.")

    chunk_size, chunk_size_path = _first_present(
        raw_reference,
        ["tree_builder.max_tokens", "ingest.chunk_size", "chunk.max_tokens", "chunk_size"],
    )
    if chunk_size_path:
        mapped_fields.append(f"Mapped tree max_tokens from {chunk_size_path}.")

    dataset_loader_config_name, dataset_loader_config_path = _first_present(
        raw_reference,
        ["dataset.loader.config_name", "dataset.config_name"],
    )
    if dataset_loader_config_path:
        mapped_fields.append(
            f"Mapped dataset loader config_name from {dataset_loader_config_path}."
        )

    dataset_books_root, dataset_books_root_path = _first_present(
        raw_reference,
        ["dataset.loader.books_root", "dataset.books_root"],
    )
    if dataset_books_root_path:
        mapped_fields.append(
            f"Mapped dataset loader books_root from {dataset_books_root_path}."
        )

    generation_max_tokens, generation_max_tokens_path = _first_present(
        raw_reference,
        ["generation.max_tokens", "model.generate.max_tokens"],
    )
    if generation_max_tokens_path:
        mapped_fields.append(
            f"Mapped generation max_tokens from {generation_max_tokens_path}."
        )

    generation_temperature, generation_temperature_path = _first_present(
        raw_reference,
        ["generation.temperature", "model.generate.temperature"],
    )
    if generation_temperature_path:
        mapped_fields.append(
            f"Mapped generation temperature from {generation_temperature_path}."
        )

    generation_stop, generation_stop_path = _first_present(
        raw_reference,
        ["generation.stop", "model.generate.stop"],
    )
    if generation_stop_path:
        mapped_fields.append(f"Mapped generation stop from {generation_stop_path}.")

    generation_top_p, generation_top_p_path = _first_present(
        raw_reference,
        ["generation.top_p", "model.sampling.top_p"],
    )
    if generation_top_p_path:
        mapped_fields.append(f"Mapped generation top_p from {generation_top_p_path}.")

    model_backend, model_backend_path = _first_present(
        raw_reference,
        ["models.qa.provider", "model.backend"],
    )
    if model_backend_path:
        mapped_fields.append(f"Mapped model backend from {model_backend_path}.")

    model_id, model_id_path = _first_present(
        raw_reference,
        ["models.qa.model", "model.model_id"],
    )
    if model_id_path:
        mapped_fields.append(f"Mapped model id from {model_id_path}.")

    model_tensor_parallel_size, model_tensor_parallel_size_path = _first_present(
        raw_reference,
        ["models.qa.tensor_parallel_size", "model.tensor_parallel_size"],
    )
    if model_tensor_parallel_size_path:
        mapped_fields.append(
            f"Mapped tensor_parallel_size from {model_tensor_parallel_size_path}."
        )

    model_dtype, model_dtype_path = _first_present(
        raw_reference,
        ["models.qa.dtype", "model.dtype"],
    )
    if model_dtype_path:
        mapped_fields.append(f"Mapped dtype from {model_dtype_path}.")

    model_gpu_memory_utilization, model_gpu_memory_utilization_path = _first_present(
        raw_reference,
        ["models.qa.gpu_memory_utilization", "model.gpu_memory_utilization"],
    )
    if model_gpu_memory_utilization_path:
        mapped_fields.append(
            f"Mapped gpu_memory_utilization from {model_gpu_memory_utilization_path}."
        )

    model_device, model_device_path = _first_present(
        raw_reference,
        ["models.qa.device", "model.device"],
    )
    if model_device_path:
        mapped_fields.append(f"Mapped device from {model_device_path}.")

    model_trust_remote_code, model_trust_remote_code_path = _first_present(
        raw_reference,
        ["models.qa.trust_remote_code", "model.trust_remote_code"],
    )
    if model_trust_remote_code_path:
        mapped_fields.append(
            f"Mapped trust_remote_code from {model_trust_remote_code_path}."
        )

    model_max_model_len, model_max_model_len_path = _first_present(
        raw_reference,
        ["models.qa.max_model_len", "model.max_model_len"],
    )
    if model_max_model_len_path:
        mapped_fields.append(
            f"Mapped max_model_len from {model_max_model_len_path}."
        )

    model_enforce_eager, model_enforce_eager_path = _first_present(
        raw_reference,
        ["models.qa.enforce_eager", "model.enforce_eager"],
    )
    if model_enforce_eager_path:
        mapped_fields.append(
            f"Mapped enforce_eager from {model_enforce_eager_path}."
        )

    model_vllm_max_num_seqs, model_vllm_max_num_seqs_path = _first_present(
        raw_reference,
        ["models.qa.vllm_kwargs.max_num_seqs", "model.vllm_kwargs.max_num_seqs"],
    )
    if model_vllm_max_num_seqs_path:
        mapped_fields.append(
            f"Mapped vllm_kwargs.max_num_seqs from {model_vllm_max_num_seqs_path}."
        )

    documents_path, documents_path_source = _first_present(
        raw_reference,
        [
            "dataset.documents.path",
            "documents.path",
            "dataset.corpus.path",
        ],
    )
    qa_path, qa_path_source = _first_present(
        raw_reference,
        [
            "dataset.qa.path",
            "qa.path",
            "questions.path",
        ],
    )
    combined_path, combined_path_source = _first_present(
        raw_reference,
        [
            "dataset.combined.path",
            "combined.path",
            "dataset.path",
        ],
    )
    if documents_path_source:
        mapped_fields.append(f"Mapped documents path from {documents_path_source}.")
    if qa_path_source:
        mapped_fields.append(f"Mapped QA path from {qa_path_source}.")
    if combined_path_source:
        mapped_fields.append(f"Mapped combined dataset path from {combined_path_source}.")

    now_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved_run_name = run_name or explicit_config.get("run_name") or raw_reference.get("run_name") or now_name
    resolved_output_root = (
        output_root
        or explicit_config.get("output_root")
        or "raptor_runs"
    )

    normalized_loader_name = None
    if isinstance(dataset_source_name, str):
        lowered_dataset_source = dataset_source_name.lower()
        if lowered_dataset_source in {"qasper", "allenai/qasper"}:
            normalized_loader_name = "qasper"
        elif lowered_dataset_source in {
            "loogle",
            "bigai-nlco/loogle",
            "bigainlco/loogle",
        }:
            normalized_loader_name = "loogle"
        elif lowered_dataset_source in {
            "narrativeqa",
            "deepmind/narrativeqa",
            "nqa",
        }:
            normalized_loader_name = "narrativeqa"
        elif lowered_dataset_source in {
            "quality",
            "tasksource/quality",
        }:
            normalized_loader_name = "quality"
        elif lowered_dataset_source in {
            "novelhopqa",
            "abhaygupta1266/novelhopqa",
        }:
            normalized_loader_name = "novelhopqa"
        elif lowered_dataset_source not in {"", "files"}:
            normalized_loader_name = lowered_dataset_source

    explicit_loader_config = dict(explicit_config.get("dataset", {}).get("loader", {}) or {})
    loader_settings = {
        key: value
        for key, value in explicit_loader_config.items()
        if key not in {"name", "config_name", "books_root", "report_dir"} and value is not None
    }
    explicit_or_mapped_books_root = explicit_loader_config.get("books_root", dataset_books_root)
    if explicit_or_mapped_books_root is not None:
        loader_settings["books_root"] = _resolve_path(
            yaml_path.parent, str(explicit_or_mapped_books_root)
        )
    explicit_report_dir = explicit_loader_config.get("report_dir")
    if explicit_report_dir is not None:
        loader_settings["report_dir"] = _resolve_path(
            yaml_path.parent, str(explicit_report_dir)
        )

    dataset_config = {
        "name": dataset_name,
        "split": explicit_config.get("dataset", {}).get("split", dataset_split),
        "loader": {
            "name": explicit_loader_config.get("name", normalized_loader_name),
            "config_name": explicit_loader_config.get("config_name", dataset_loader_config_name),
            **loader_settings,
        },
        "documents": {
            "path": _resolve_path(
                yaml_path.parent,
                explicit_config.get("dataset", {}).get("documents", {}).get("path", documents_path),
            ),
            "format": explicit_config.get("dataset", {}).get("documents", {}).get("format"),
            "records_path": explicit_config.get("dataset", {}).get("documents", {}).get("records_path"),
            "id_field": explicit_config.get("dataset", {}).get("documents", {}).get("id_field", "doc_id"),
            "text_field": explicit_config.get("dataset", {}).get("documents", {}).get("text_field", "text"),
            "split_field": explicit_config.get("dataset", {}).get("documents", {}).get("split_field"),
            "doc_id": explicit_config.get("dataset", {}).get("documents", {}).get("doc_id"),
        },
        "qa": {
            "path": _resolve_path(
                yaml_path.parent,
                explicit_config.get("dataset", {}).get("qa", {}).get("path", qa_path),
            ),
            "format": explicit_config.get("dataset", {}).get("qa", {}).get("format"),
            "records_path": explicit_config.get("dataset", {}).get("qa", {}).get("records_path"),
            "query_id_field": explicit_config.get("dataset", {}).get("qa", {}).get("query_id_field", "query_id"),
            "doc_id_field": explicit_config.get("dataset", {}).get("qa", {}).get("doc_id_field", "doc_id"),
            "question_field": explicit_config.get("dataset", {}).get("qa", {}).get("question_field", "question"),
            "reference_answers_field": explicit_config.get("dataset", {}).get("qa", {}).get("reference_answers_field", "reference_answers"),
            "split_field": explicit_config.get("dataset", {}).get("qa", {}).get("split_field"),
        },
        "combined": {
            "path": _resolve_path(
                yaml_path.parent,
                explicit_config.get("dataset", {}).get("combined", {}).get("path", combined_path),
            ),
            "format": explicit_config.get("dataset", {}).get("combined", {}).get("format"),
            "records_path": explicit_config.get("dataset", {}).get("combined", {}).get("records_path"),
            "doc_id_field": explicit_config.get("dataset", {}).get("combined", {}).get("doc_id_field", "doc_id"),
            "text_field": explicit_config.get("dataset", {}).get("combined", {}).get("text_field", "text"),
            "qa_field": explicit_config.get("dataset", {}).get("combined", {}).get("qa_field", "qa_entries"),
            "qa_query_id_field": explicit_config.get("dataset", {}).get("combined", {}).get("qa_query_id_field", "query_id"),
            "qa_question_field": explicit_config.get("dataset", {}).get("combined", {}).get("qa_question_field", "question"),
            "qa_reference_answers_field": explicit_config.get("dataset", {}).get("combined", {}).get("qa_reference_answers_field", "reference_answers"),
            "split_field": explicit_config.get("dataset", {}).get("combined", {}).get("split_field"),
        },
        "selection": {
            "doc_ids": explicit_config.get("dataset", {}).get("selection", {}).get("doc_ids", []),
            "query_ids": explicit_config.get("dataset", {}).get("selection", {}).get("query_ids", []),
            "max_docs": explicit_config.get("dataset", {}).get("selection", {}).get("max_docs", max_docs),
            "max_questions": explicit_config.get("dataset", {}).get("selection", {}).get("max_questions", max_questions),
            "max_questions_per_doc": explicit_config.get("dataset", {}).get("selection", {}).get(
                "max_questions_per_doc", max_questions_per_doc
            ),
        },
        "per_document_retrieval": True,
    }

    if (
        not dataset_config["loader"]["name"]
        and not dataset_config["combined"]["path"]
        and not dataset_config["documents"]["path"]
    ):
        raise ValueError(
            "Could not resolve a dataset source from the YAML. Add dataset.loader.name, dataset.documents.path, or dataset.combined.path."
        )
    if (
        not dataset_config["loader"]["name"]
        and not dataset_config["combined"]["path"]
        and not dataset_config["qa"]["path"]
    ):
        raise ValueError(
            "Could not resolve a QA path from the YAML. Add dataset.qa.path or use dataset.combined.path."
        )

    inferred_backend = str(model_backend).lower() if model_backend is not None else None
    if inferred_backend == "vllm":
        default_embedding_provider = "contriever"
        default_embedding_model = "facebook/contriever"
        default_generation_provider = "vllm"
        default_generation_model = model_id or "Qwen/Qwen2-0.5B-Instruct"
    elif inferred_backend in {"transformers", "hf"}:
        default_embedding_provider = "contriever"
        default_embedding_model = "facebook/contriever"
        default_generation_provider = "transformers"
        default_generation_model = model_id or "Qwen/Qwen2.5-7B-Instruct"
    else:
        default_embedding_provider = "openai"
        default_embedding_model = "text-embedding-ada-002"
        default_generation_provider = "openai"
        default_generation_model = "gpt-3.5-turbo"

    base_embedding_config = explicit_config.get("models", {}).get("embedding", {})
    if not base_embedding_config and default_embedding_provider in {"contriever", "transformers"}:
        base_embedding_config = {
            "provider": default_embedding_provider,
            "model": default_embedding_model,
            "trust_remote_code": True,
        }
        if model_device is not None:
            base_embedding_config["device"] = model_device

    base_summarization_config = explicit_config.get("models", {}).get("summarization", {})
    base_qa_config = explicit_config.get("models", {}).get("qa", {})
    if not base_summarization_config and inferred_backend in {"vllm", "transformers", "hf"}:
        base_summarization_config = {
            "provider": default_generation_provider,
            "model": default_generation_model,
            "trust_remote_code": True if model_trust_remote_code is None else model_trust_remote_code,
        }
    if not base_qa_config and inferred_backend in {"vllm", "transformers", "hf"}:
        base_qa_config = {
            "provider": default_generation_provider,
            "model": default_generation_model,
            "trust_remote_code": True if model_trust_remote_code is None else model_trust_remote_code,
        }

    for config_block in (base_summarization_config, base_qa_config):
        if not config_block:
            continue
        if model_tensor_parallel_size is not None:
            config_block.setdefault("tensor_parallel_size", model_tensor_parallel_size)
        if model_dtype is not None:
            config_block.setdefault("dtype", model_dtype)
        if model_gpu_memory_utilization is not None:
            config_block.setdefault(
                "gpu_memory_utilization", model_gpu_memory_utilization
            )
        if model_max_model_len is not None:
            config_block.setdefault("max_model_len", model_max_model_len)
        if model_enforce_eager is not None:
            config_block.setdefault("enforce_eager", model_enforce_eager)
        if model_vllm_max_num_seqs is not None:
            vllm_kwargs = config_block.setdefault("vllm_kwargs", {})
            if not isinstance(vllm_kwargs, dict):
                vllm_kwargs = {}
                config_block["vllm_kwargs"] = vllm_kwargs
            vllm_kwargs.setdefault("max_num_seqs", model_vllm_max_num_seqs)
        if model_device is not None and config_block.get("provider") != "vllm":
            config_block.setdefault("device", model_device)

    if base_summarization_config:
        base_summarization_config.setdefault("temperature", 0.0)
        base_summarization_config.setdefault("top_p", 1.0)

    if base_qa_config:
        if generation_temperature is not None:
            base_qa_config.setdefault("temperature", generation_temperature)
        if generation_top_p is not None:
            base_qa_config.setdefault("top_p", generation_top_p)
        if generation_stop is not None:
            base_qa_config.setdefault("stop", generation_stop)

    embedding_config = _normalize_model_config(
        base_embedding_config,
        default_provider=default_embedding_provider,
        default_model=default_embedding_model,
    )
    summarization_config = _normalize_model_config(
        base_summarization_config,
        default_provider=default_generation_provider,
        default_model=default_generation_model,
    )
    qa_config = _normalize_model_config(
        base_qa_config,
        default_provider=default_generation_provider,
        default_model=default_generation_model,
    )

    resolved_config = {
        "dataset_loader_settings": dataset_config,
        "llm_model": qa_config,
        "embedding_model": embedding_config,
        "tree_builder_settings": {
            "builder_type": "cluster",
            "max_tokens": explicit_config.get("tree_builder", {}).get(
                "max_tokens",
                chunk_size,
            )
            or 100,
            "num_layers": explicit_config.get("tree_builder", {}).get(
                "num_layers",
                _first_present(raw_reference, ["tree_builder.num_layers", "num_layers"], 5)[0],
            )
            or 5,
            "threshold": explicit_config.get("tree_builder", {}).get("threshold", 0.5),
            "top_k": explicit_config.get("tree_builder", {}).get("top_k", 5),
            "selection_mode": explicit_config.get("tree_builder", {}).get("selection_mode", "top_k"),
            "summarization_length": explicit_config.get("tree_builder", {}).get("summarization_length", 100),
            "reduction_dimension": explicit_config.get("tree_builder", {}).get("reduction_dimension", 10),
            "clustering_params": explicit_config.get("tree_builder", {}).get("clustering_params", {}),
            "summarization_model": summarization_config,
        },
        "retrieval_settings": {
            "start_layer": explicit_config.get("retrieval", {}).get("start_layer"),
            "num_layers": explicit_config.get("retrieval", {}).get("num_layers"),
            "top_k": explicit_config.get("retrieval", {}).get(
                "top_k", retrieval_top_k
            )
            or FIXED_RETRIEVAL_TOP_K,
            "max_tokens": explicit_config.get("retrieval", {}).get("max_tokens", 3500),
            "collapse_tree": explicit_config.get("retrieval", {}).get("collapse_tree", True),
            "threshold": explicit_config.get("retrieval", {}).get("threshold", 0.5),
            "selection_mode": explicit_config.get("retrieval", {}).get("selection_mode", "top_k"),
        },
        "generation_settings": {
            "qa_model": qa_config,
            "prompt_template_name": explicit_config.get("generation", {}).get("prompt_template_name"),
            "max_tokens": explicit_config.get("generation", {}).get(
                "max_tokens", generation_max_tokens
            ),
            "temperature": explicit_config.get("generation", {}).get(
                "temperature", generation_temperature
            ),
            "top_p": explicit_config.get("generation", {}).get("top_p", generation_top_p),
            "stop": explicit_config.get("generation", {}).get("stop", generation_stop),
        },
        "profiling_settings": {
            "record_resource_usage": explicit_config.get("profiling", {}).get("record_resource_usage", True),
        },
        "run_settings": {
            "dataset_name": dataset_name,
            "run_name": resolved_run_name,
            "output_root": str((PROJECT_ROOT / resolved_output_root).resolve())
            if not Path(resolved_output_root).is_absolute()
            else resolved_output_root,
            "resume": bool(resume or explicit_config.get("resume", False)),
            "default_experiment_yaml": str(yaml_path),
        },
        "mapping_notes": mapped_fields,
    }

    top_level_keys = sorted(raw_reference.keys())
    recognized_keys = {"raptor", "raptor_run", "dataset", "documents", "qa", "questions", "retrieval", "tree_builder", "models", "generation", "profiling", "split", "max_docs", "sample_size", "max_questions", "run_name", "output_dir", "sample", "ingest", "model", "evaluation"}
    ignored_top_level = [key for key in top_level_keys if key not in recognized_keys]
    if ignored_top_level:
        notes.append("Unmapped top-level YAML keys: " + ", ".join(ignored_top_level))

    if _deep_get(raw_reference, "ingest.strategy") is not None:
        notes.append(
            "Ignored ingest.strategy because RAPTOR always builds its own tree-based ingestion pipeline."
        )
    if _deep_get(raw_reference, "ingest.chunk_overlap") is not None:
        notes.append(
            "Ignored ingest.chunk_overlap because the current RAPTOR runner does not expose overlap control."
        )
    if _deep_get(raw_reference, "retrieval.retriever") is not None:
        notes.append(
            "Ignored retrieval.retriever because RAPTOR uses tree retrieval rather than BM25."
        )
    if _deep_get(raw_reference, "retrieval.mode") is not None:
        notes.append(
            "Ignored retrieval.mode because the RAPTOR runner always saves raw retrieval outputs and generated answers."
        )
    if _deep_get(raw_reference, "model.backend") is not None and not explicit_config.get("models"):
        notes.append(
            "Mapped the source YAML model backend into RAPTOR-native model settings because no explicit RAPTOR models block was provided."
        )
    if raw_reference.get("output_dir") is not None and not explicit_config.get("output_root"):
        notes.append(
            "Ignored output_dir from the source YAML and used the RAPTOR default output root structure under raptor_runs/<dataset>/<run_name>."
        )
    if not Path(resolved_output_root).is_absolute():
        notes.append(
            "Resolved relative output_root under the RAPTOR project root so artifacts remain inside this repository."
        )

    notes.extend(mapped_fields)

    return resolved_config, notes


def _model_name(model) -> str:
    return getattr(model, "model", getattr(model, "__class__", type(model)).__name__)


def _invoke_qa_model(qa_model, context: str, question: str, generation_settings: Dict[str, Any]):
    answer_question = qa_model.answer_question
    kwargs = {}
    parameters = inspect.signature(answer_question).parameters

    if "max_tokens" in parameters and generation_settings.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(generation_settings["max_tokens"])

    stop_value = generation_settings.get("stop")
    if "stop_sequence" in parameters and stop_value:
        if isinstance(stop_value, list):
            kwargs["stop_sequence"] = stop_value[0]
        else:
            kwargs["stop_sequence"] = stop_value

    if "temperature" in parameters and generation_settings.get("temperature") is not None:
        kwargs["temperature"] = float(generation_settings["temperature"])

    if "top_p" in parameters and generation_settings.get("top_p") is not None:
        kwargs["top_p"] = float(generation_settings["top_p"])

    return answer_question(context, question, **kwargs)


def run_experiment(
    dataset_name: str,
    default_yaml_path: str,
    run_name: Optional[str] = None,
    output_root: Optional[str] = None,
    resume: bool = False,
):
    resolved_config, notes = resolve_run_config(
        dataset_name=dataset_name,
        default_yaml_path=default_yaml_path,
        run_name=run_name,
        output_root=output_root,
        resume=resume,
    )

    _validate_runtime_dependencies(
        resolved_config["tree_builder_settings"]["summarization_model"],
        label="summarization",
    )
    _validate_runtime_dependencies(
        resolved_config["generation_settings"]["qa_model"],
        label="qa",
    )

    tokenizer = get_default_tokenizer()
    dataset = _apply_selection(
        load_dataset(resolved_config["dataset_loader_settings"]),
        resolved_config["dataset_loader_settings"],
    )

    run_settings = resolved_config["run_settings"]
    run_root = Path(run_settings["output_root"]) / dataset_name / run_settings["run_name"]
    config_dir = run_root / "config"
    selection_dir = run_root / "selection"
    corpus_dir = run_root / "corpus"
    trees_dir = run_root / "trees"
    retrieval_dir = run_root / "retrieval"
    rag_dir = run_root / "rag"
    profiling_dir = run_root / "profiling"

    for directory in [
        config_dir,
        selection_dir,
        corpus_dir,
        trees_dir,
        retrieval_dir,
        rag_dir,
        profiling_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    default_yaml_copy = config_dir / "default_experiment.yaml"
    if not default_yaml_copy.exists():
        shutil.copyfile(run_settings["default_experiment_yaml"], default_yaml_copy)
    _safe_dump_yaml(resolved_config, config_dir / "raptor_run.yaml")

    selected_doc_ids = [document.doc_id for document in dataset.documents]
    _write_json(selection_dir / "selected_doc_ids.json", selected_doc_ids)
    _write_json(selection_dir / "qa_entries.json", [asdict(entry) for entry in dataset.qa_entries])

    document_rows = [
        {
            "doc_id": document.doc_id,
            "text": document.text,
            "character_count": len(document.text),
            "token_count": len(tokenizer.encode(document.text)),
            **({"metadata": document.metadata} if document.metadata else {}),
        }
        for document in dataset.documents
    ]
    _write_jsonl(corpus_dir / "documents.jsonl", document_rows)

    tree_builder_settings = resolved_config["tree_builder_settings"]
    retrieval_settings = resolved_config["retrieval_settings"]
    generation_settings = resolved_config["generation_settings"]

    embedding_model = _build_embedding_model(resolved_config["embedding_model"])
    summarization_model = _build_summarization_model(tree_builder_settings["summarization_model"])
    qa_model = _build_qa_model(generation_settings["qa_model"])

    if hasattr(summarization_model, "warm_up"):
        summarization_model.warm_up()

    builder_config = ClusterTreeConfig(
        tokenizer=tokenizer,
        max_tokens=int(tree_builder_settings["max_tokens"]),
        num_layers=int(tree_builder_settings["num_layers"]),
        threshold=float(tree_builder_settings["threshold"]),
        top_k=int(tree_builder_settings["top_k"]),
        selection_mode=tree_builder_settings["selection_mode"],
        summarization_length=int(tree_builder_settings["summarization_length"]),
        summarization_model=summarization_model,
        embedding_models={"primary": embedding_model},
        cluster_embedding_model="primary",
        reduction_dimension=int(tree_builder_settings["reduction_dimension"]),
        clustering_params=tree_builder_settings["clustering_params"],
    )
    build_times_path = profiling_dir / "build_times.jsonl"
    query_times_path = profiling_dir / "query_times.jsonl"
    resource_usage_path = profiling_dir / "resource_usage.jsonl"
    retrieval_payloads_path = retrieval_dir / "retrieval_payloads.jsonl"
    qa_predictions_path = rag_dir / "qa_predictions.jsonl"

    resume_enabled = run_settings["resume"]
    existing_query_ids = set()
    if resume_enabled and qa_predictions_path.exists():
        with open(qa_predictions_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_query_ids.add(json.loads(line)["query_id"])
    elif qa_predictions_path.exists():
        qa_predictions_path.unlink()

    if retrieval_payloads_path.exists() and not resume_enabled:
        retrieval_payloads_path.unlink()
    if build_times_path.exists() and not resume_enabled:
        build_times_path.unlink()
    if query_times_path.exists() and not resume_enabled:
        query_times_path.unlink()
    if resource_usage_path.exists() and not resume_enabled:
        resource_usage_path.unlink()

    trees_by_doc_id = {}
    descendant_lookup_by_doc_id = {}

    for document in dataset.documents:
        doc_tree_dir = trees_dir / document.doc_id
        doc_tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = doc_tree_dir / "tree.pkl"
        tree_stats_path = doc_tree_dir / "tree_stats.json"
        leaf_chunks_path = doc_tree_dir / "leaf_chunks.jsonl"
        node_index_path = doc_tree_dir / "node_index.jsonl"

        loaded_existing_tree = resume_enabled and tree_path.exists()
        if loaded_existing_tree:
            with open(tree_path, "rb") as handle:
                tree = pickle.load(handle)
            build_time_ms = 0.0
            if tree_stats_path.exists():
                with open(tree_stats_path, "r", encoding="utf-8") as handle:
                    existing_tree_stats = json.load(handle)
                build_time_ms = float(existing_tree_stats.get("build_time_ms", 0.0))
        else:
            tree_builder = ClusterTreeBuilder(builder_config)
            _reset_gpu_peak_memory()
            build_started_at = time.perf_counter()
            tree = tree_builder.build_from_text(document.text)
            build_time_ms = (time.perf_counter() - build_started_at) * 1000
            with open(tree_path, "wb") as handle:
                pickle.dump(tree, handle)

        tree_stats, leaf_rows, node_rows, descendant_lookup = _tree_artifact_rows(
            tree=tree,
            doc_id=document.doc_id,
            tokenizer=tokenizer,
            build_time_ms=build_time_ms,
        )
        _write_json(tree_stats_path, tree_stats)
        _write_jsonl(leaf_chunks_path, leaf_rows)
        _write_jsonl(node_index_path, node_rows)
        if not loaded_existing_tree:
            _append_jsonl(
                build_times_path,
                {"doc_id": document.doc_id, "build_time_ms": round(build_time_ms, 3)},
            )
            if resolved_config["profiling_settings"]["record_resource_usage"]:
                _append_jsonl(
                    resource_usage_path,
                    _resource_usage_record(phase="build", doc_id=document.doc_id),
                )

        trees_by_doc_id[document.doc_id] = tree
        descendant_lookup_by_doc_id[document.doc_id] = descendant_lookup

    if hasattr(qa_model, "warm_up"):
        qa_model.warm_up()

    questions_by_doc_id = defaultdict(list)
    for qa_entry in dataset.qa_entries:
        questions_by_doc_id[qa_entry.doc_id].append(qa_entry)

    for doc_id, qa_entries in questions_by_doc_id.items():
        tree = trees_by_doc_id[doc_id]
        descendant_lookup = descendant_lookup_by_doc_id[doc_id]
        retriever = TreeRetriever(
            TreeRetrieverConfig(
                tokenizer=tokenizer,
                threshold=float(retrieval_settings["threshold"]),
                top_k=FIXED_RETRIEVAL_TOP_K,
                selection_mode=retrieval_settings["selection_mode"],
                context_embedding_model="primary",
                embedding_model=embedding_model,
                num_layers=retrieval_settings["num_layers"],
                start_layer=retrieval_settings["start_layer"],
            ),
            tree,
        )

        for qa_entry in qa_entries:
            if qa_entry.query_id in existing_query_ids:
                continue

            _reset_gpu_peak_memory()
            retrieval_started_at = time.perf_counter()
            retrieval_payload = retriever.retrieve_with_metadata(
                query=qa_entry.question,
                start_layer=retrieval_settings["start_layer"],
                num_layers=retrieval_settings["num_layers"],
                top_k=FIXED_RETRIEVAL_TOP_K,
                max_tokens=int(retrieval_settings["max_tokens"]),
                collapse_tree=bool(retrieval_settings["collapse_tree"]),
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_started_at) * 1000

            expanded_chunks = _retrieved_chunk_rows(
                retrieval_payload["retrieved_nodes"],
                descendant_lookup,
            )
            retrieved_chunk_ids = _dedup_chunk_ids(expanded_chunks)
            context = retrieval_payload["context"]
            context_token_count = len(tokenizer.encode(context))

            generation_started_at = time.perf_counter()
            prediction = _invoke_qa_model(
                qa_model,
                context,
                qa_entry.question,
                generation_settings,
            )
            generation_latency_ms = (time.perf_counter() - generation_started_at) * 1000
            total_latency_ms = retrieval_latency_ms + generation_latency_ms

            retrieval_artifact = {
                "query_id": qa_entry.query_id,
                "doc_id": qa_entry.doc_id,
                "question": qa_entry.question,
                "retrieved_nodes": retrieval_payload["retrieved_nodes"],
                "layer_information": retrieval_payload["layer_information"],
                "expanded_retrieved_chunks": expanded_chunks,
                "context": context,
                "context_token_count": context_token_count,
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "retrieval_config": {
                    **dict(retrieval_settings),
                    "top_k": FIXED_RETRIEVAL_TOP_K,
                },
            }
            _append_jsonl(retrieval_payloads_path, retrieval_artifact)

            qa_artifact = {
                "query_id": qa_entry.query_id,
                "doc_id": qa_entry.doc_id,
                "question": qa_entry.question,
                "prediction": prediction,
                "reference_answers": qa_entry.reference_answers,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "context": context,
                "context_token_count": context_token_count,
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "generation_latency_ms": round(generation_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "answer_token_count": len(tokenizer.encode(str(prediction))),
                "model_name": _model_name(qa_model),
                "prompt_template_name": generation_settings.get("prompt_template_name"),
            }
            _append_jsonl(qa_predictions_path, qa_artifact)

            _append_jsonl(
                query_times_path,
                {
                    "query_id": qa_entry.query_id,
                    "doc_id": qa_entry.doc_id,
                    "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                    "generation_latency_ms": round(generation_latency_ms, 3),
                    "total_latency_ms": round(total_latency_ms, 3),
                },
            )
            if resolved_config["profiling_settings"]["record_resource_usage"]:
                _append_jsonl(
                    resource_usage_path,
                    _resource_usage_record(
                        phase="query",
                        doc_id=qa_entry.doc_id,
                        query_id=qa_entry.query_id,
                    ),
                )

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "run_name": run_settings["run_name"],
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(Path(default_yaml_path).resolve().parent),
        "python_version": sys.version,
        "package_versions": {
            "PyYAML": _package_version("PyYAML"),
            "numpy": _package_version("numpy"),
            "openai": _package_version("openai"),
            "tiktoken": _package_version("tiktoken"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "scikit-learn": _package_version("scikit-learn"),
            "umap-learn": _package_version("umap-learn"),
        },
        "hardware_summary": _hardware_summary(),
        "selected_documents_count": len(dataset.documents),
        "selected_questions_count": len(dataset.qa_entries),
        "artifact_paths": {
            "selection": str(selection_dir),
            "corpus": str(corpus_dir),
            "trees": str(trees_dir),
            "retrieval": str(retrieval_dir),
            "rag": str(rag_dir),
            "profiling": str(profiling_dir),
        },
        "config_paths": {
            "default_experiment_yaml": str(default_yaml_copy),
            "raptor_run_yaml": str(config_dir / "raptor_run.yaml"),
        },
        "notes_about_assumptions_or_ignored_yaml_fields": notes,
    }
    _write_json(run_root / "run_manifest.json", run_manifest)

    return {
        "run_root": str(run_root),
        "selected_documents_count": len(dataset.documents),
        "selected_questions_count": len(dataset.qa_entries),
    }
