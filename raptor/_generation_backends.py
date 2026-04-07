from __future__ import annotations

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
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vllm is required for provider='vllm'. Install it in the RAPTOR environment."
        ) from exc

    engine_kwargs = dict(engine_kwargs or {})
    engine_kwargs.setdefault("trust_remote_code", True)
    cache_key = (model_name, _freeze(engine_kwargs))

    llm = _VLLM_CACHE.get(cache_key)
    if llm is None:
        llm = LLM(model=model_name, **engine_kwargs)
        _VLLM_CACHE[cache_key] = llm

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
