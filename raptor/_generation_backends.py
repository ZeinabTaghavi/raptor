from __future__ import annotations

import os
from typing import Any


_VLLM_CACHE: dict[tuple[Any, ...], Any] = {}
_TRANSFORMERS_PIPELINE_CACHE: dict[tuple[Any, ...], Any] = {}


def _freeze(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def trim_at_stop(text: str, stop):
    if not stop:
        return text.strip()

    stop_values = stop if isinstance(stop, list) else [stop]
    output = text
    for stop_value in stop_values:
        if not stop_value:
            continue
        marker_index = output.find(stop_value)
        if marker_index != -1:
            output = output[:marker_index]
    return output.strip()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _infer_tensor_parallel_size() -> int | None:
    for env_name in (
        "RAPTOR_TENSOR_PARALLEL_SIZE",
        "SAADI_TENSOR_PARALLEL_SIZE",
        "TENSOR_PARALLEL_SIZE",
    ):
        raw = str(os.getenv(env_name, "")).strip()
        if raw:
            return int(raw)

    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        return None

    raw = str(cuda_visible).strip()
    if not raw or raw == "-1":
        return None

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None

    return max(1, len(parts))


def _normalize_vllm_engine_kwargs(engine_kwargs: dict[str, Any] | None = None):
    normalized = dict(engine_kwargs or {})
    extra_kwargs = normalized.pop("vllm_kwargs", None)
    if isinstance(extra_kwargs, dict):
        normalized.update(extra_kwargs)

    normalized.setdefault("trust_remote_code", True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if normalized.get("tensor_parallel_size") is None:
        inferred = _infer_tensor_parallel_size()
        if inferred is not None:
            normalized["tensor_parallel_size"] = inferred
    elif normalized.get("tensor_parallel_size") is not None:
        normalized["tensor_parallel_size"] = int(normalized["tensor_parallel_size"])

    if normalized.get("gpu_memory_utilization") is not None:
        normalized["gpu_memory_utilization"] = float(
            normalized["gpu_memory_utilization"]
        )

    if normalized.get("max_model_len") is not None:
        normalized["max_model_len"] = int(normalized["max_model_len"])

    if normalized.get("enforce_eager") is not None:
        normalized["enforce_eager"] = _coerce_bool(normalized["enforce_eager"])

    return normalized


def _get_or_create_vllm_llm(
    *, model_name: str, engine_kwargs: dict[str, Any] | None = None
):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "vllm is required for provider='vllm'. Install it in the RAPTOR environment."
        ) from exc

    normalized_kwargs = _normalize_vllm_engine_kwargs(engine_kwargs)
    cache_key = (model_name, _freeze(normalized_kwargs))
    llm = _VLLM_CACHE.get(cache_key)
    if llm is not None:
        return llm

    try:
        llm = LLM(model=model_name, **normalized_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"vLLM failed to initialize for model '{model_name}'. "
            f"engine_kwargs={normalized_kwargs}. "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}. "
            "Check tensor_parallel_size, gpu_memory_utilization, dtype, and whether other models already occupy GPU memory."
        ) from exc

    _VLLM_CACHE[cache_key] = llm
    return llm


def warm_vllm_engine(*, model_name: str, engine_kwargs: dict[str, Any] | None = None):
    _get_or_create_vllm_llm(model_name=model_name, engine_kwargs=engine_kwargs)


def generate_with_vllm(
    *,
    model_name: str,
    prompt: str,
    engine_kwargs: dict[str, Any] | None = None,
    max_tokens: int = 150,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop=None,
):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vllm is required for provider='vllm'. Install it in the RAPTOR environment."
        ) from exc

    llm = _get_or_create_vllm_llm(model_name=model_name, engine_kwargs=engine_kwargs)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    return trim_at_stop(outputs[0].outputs[0].text, stop)


def generate_with_transformers(
    *,
    model_name: str,
    prompt: str,
    pipeline_kwargs: dict[str, Any] | None = None,
    max_tokens: int = 150,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop=None,
):
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers is required for provider='transformers'. Install project requirements first."
        ) from exc

    pipeline_kwargs = dict(pipeline_kwargs or {})
    pipeline_kwargs.setdefault("device_map", "auto")
    pipeline_kwargs.setdefault("torch_dtype", "auto")
    pipeline_kwargs.setdefault("trust_remote_code", True)
    cache_key = (model_name, _freeze(pipeline_kwargs))

    generator = _TRANSFORMERS_PIPELINE_CACHE.get(cache_key)
    if generator is None:
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            **pipeline_kwargs,
        )
        _TRANSFORMERS_PIPELINE_CACHE[cache_key] = generator

    do_sample = temperature is not None and float(temperature) > 0
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=float(temperature) if temperature is not None else 0.0,
        top_p=float(top_p) if top_p is not None else 1.0,
        return_full_text=False,
    )
    return trim_at_stop(outputs[0]["generated_text"], stop)
