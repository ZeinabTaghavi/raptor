from __future__ import annotations

import logging
from typing import Any


_VLLM_CACHE: dict[tuple[Any, ...], Any] = {}
_TRANSFORMERS_PIPELINE_CACHE: dict[tuple[Any, ...], Any] = {}
LOGGER = logging.getLogger(__name__)


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
    engine_kwargs = dict(engine_kwargs or {})

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        LOGGER.warning(
            "vLLM is unavailable for %s; falling back to transformers generation.",
            model_name,
        )
        return generate_with_transformers(
            model_name=model_name,
            prompt=prompt,
            pipeline_kwargs=_vllm_engine_kwargs_to_transformers_kwargs(engine_kwargs),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    engine_kwargs.setdefault("trust_remote_code", True)
    cache_key = (model_name, _freeze(engine_kwargs))

    llm = _VLLM_CACHE.get(cache_key)
    if llm is None:
        try:
            llm = LLM(model=model_name, **engine_kwargs)
        except Exception as exc:
            LOGGER.warning(
                "vLLM failed to initialize for %s; falling back to transformers. Root error: %s",
                model_name,
                exc,
            )
            return generate_with_transformers(
                model_name=model_name,
                prompt=prompt,
                pipeline_kwargs=_vllm_engine_kwargs_to_transformers_kwargs(engine_kwargs),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        _VLLM_CACHE[cache_key] = llm

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    try:
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    except Exception as exc:
        LOGGER.warning(
            "vLLM generation failed for %s; falling back to transformers. Root error: %s",
            model_name,
            exc,
        )
        return generate_with_transformers(
            model_name=model_name,
            prompt=prompt,
            pipeline_kwargs=_vllm_engine_kwargs_to_transformers_kwargs(engine_kwargs),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
    return trim_at_stop(outputs[0].outputs[0].text, stop)


def _vllm_engine_kwargs_to_transformers_kwargs(engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    pipeline_kwargs: dict[str, Any] = {
        "device_map": "auto",
    }

    for key in ("trust_remote_code", "revision"):
        if key in engine_kwargs:
            pipeline_kwargs[key] = engine_kwargs[key]

    dtype = engine_kwargs.get("dtype")
    if dtype is not None:
        pipeline_kwargs["torch_dtype"] = dtype

    return pipeline_kwargs


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
    torch_dtype = pipeline_kwargs.get("torch_dtype", "auto")
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype, torch_dtype)
    pipeline_kwargs["torch_dtype"] = torch_dtype
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
