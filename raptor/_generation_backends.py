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


def _get_or_create_vllm_llm(*, model_name: str, engine_kwargs: dict[str, Any] | None = None):
    engine_kwargs = dict(engine_kwargs or {})
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            f"vLLM is required for model {model_name!r}, but it is not installed in this environment. "
            "Install and run with vLLM instead of falling back to transformers."
        ) from exc

    engine_kwargs.setdefault("trust_remote_code", True)
    cache_key = (model_name, _freeze(engine_kwargs))
    llm = _VLLM_CACHE.get(cache_key)
    if llm is not None:
        return llm

    LOGGER.info(
        "Initializing vLLM engine for %s with engine kwargs: %s",
        model_name,
        engine_kwargs,
    )
    try:
        llm = LLM(model=model_name, **engine_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"vLLM failed to initialize for model {model_name!r}. "
            "The run is configured to use vLLM only, so it will stop instead of falling back. "
            f"Root error: {exc}"
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
    engine_kwargs = dict(engine_kwargs or {})
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            f"vLLM is required for model {model_name!r}, but it is not installed in this environment."
        ) from exc

    llm = _get_or_create_vllm_llm(model_name=model_name, engine_kwargs=engine_kwargs)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    try:
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    except Exception as exc:
        raise RuntimeError(
            f"vLLM generation failed for model {model_name!r}. "
            "The run is configured to use vLLM only, so it will stop instead of falling back. "
            f"Root error: {exc}"
        ) from exc
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
        pipeline_kwargs["dtype"] = dtype

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
    dtype = pipeline_kwargs.pop("torch_dtype", pipeline_kwargs.pop("dtype", "auto"))
    if isinstance(dtype, str) and dtype != "auto":
        dtype = getattr(torch, dtype, dtype)
    pipeline_kwargs["dtype"] = dtype

    if "device_map" in pipeline_kwargs:
        try:
            import accelerate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Transformers multi-GPU loading requires the 'accelerate' package when "
                "device_map is set. Install it with `pip install accelerate`, or use "
                "`device: 0` for a single-GPU run."
            ) from exc
    elif "device" not in pipeline_kwargs:
        pipeline_kwargs["device"] = 0 if torch.cuda.is_available() else -1
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
