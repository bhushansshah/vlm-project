"""Backend registry.

A model is addressed by a spec string ``<provider>:<model-name>``::

    openai:gpt-5-mini
    anthropic:claude-sonnet-5-5
    gemini:gemini-2.5-flash
    openrouter:qwen/qwen2.5-vl-72b-instruct
    local:Qwen/Qwen2.5-VL-7B-Instruct          # in-process HF transformers
    vllm:Qwen/Qwen2.5-VL-7B-Instruct           # OpenAI-compatible server (vLLM, SGLang, ...)
    ollama:qwen2.5vl:7b                        # Ollama's OpenAI-compatible endpoint
    dummy:echo                                 # offline smoke testing

Backends are imported lazily so that only the SDK you use has to be installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import VLM, NonRetryableError

# provider -> (module, class name, default constructor options)
_REGISTRY: dict[str, tuple[str, str, dict[str, Any]]] = {
    "openai": ("openai_api", "OpenAIResponsesVLM", {}),
    "anthropic": ("anthropic_api", "AnthropicVLM", {}),
    "gemini": ("gemini_api", "GeminiVLM", {}),
    "openrouter": ("openai_compat", "OpenAICompatVLM", {"preset": "openrouter"}),
    "openai-compat": ("openai_compat", "OpenAICompatVLM", {}),
    "vllm": ("openai_compat", "OpenAICompatVLM", {"preset": "vllm"}),
    "ollama": ("openai_compat", "OpenAICompatVLM", {"preset": "ollama"}),
    "local": ("hf_local", "HFLocalVLM", {}),
    "dummy": ("dummy", "DummyVLM", {}),
}


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def parse_spec(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"Model spec must look like '<provider>:<model>', got {spec!r}")
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider {provider!r}. Available: {available_providers()}")
    return provider, model


def load_model(spec: str, **options: Any) -> VLM:
    provider, model = parse_spec(spec)
    module_name, cls_name, defaults = _REGISTRY[provider]
    module = importlib.import_module(f".{module_name}", __name__)
    cls = getattr(module, cls_name)
    obj = cls(model, **{**defaults, **options})
    obj.provider = provider
    return obj


__all__ = ["VLM", "NonRetryableError", "load_model", "parse_spec", "available_providers"]
