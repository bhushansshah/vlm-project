"""Backend for any server speaking the OpenAI Chat Completions API.

Covers OpenRouter (as the VisualOverload authors used it) and locally served models:
vLLM (``vllm serve <model>``), SGLang, Ollama, LM Studio, llama.cpp server, TGI, ...
Serving local models this way is usually much faster than in-process HF inference,
because the server batches the concurrent requests the runner sends.
"""

from __future__ import annotations

import os
from typing import Any

from ..images import ImagePolicy, encode_image
from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage
from .base import VLM, NonRetryableError

PRESETS: dict[str, dict[str, str | None]] = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "vllm": {"base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
}


class OpenAICompatVLM(VLM):
    """Options:
        preset: "openrouter" | "vllm" | "ollama" (sets base_url and API-key env var)
        base_url: server URL, overrides the preset (e.g. http://gpu-node-12:8000/v1)
        api_key: explicit key; otherwise read from the preset's env var
        image_detail: optional OpenAI-style "detail" field on image parts
        max_tokens_param: "max_tokens" (default, widest support) or "max_completion_tokens"
        max_side, max_pixels: optional client-side downscaling (see ImagePolicy)
    """

    provider = "openai-compat"

    def __init__(
        self,
        model: str,
        preset: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        image_detail: str | None = None,
        max_tokens_param: str = "max_tokens",
        timeout: float = 600,
        max_retries: int = 2,
        max_side: int | None = None,
        max_pixels: int | None = None,
        **options: Any,
    ):
        p = PRESETS.get(preset or "", {})
        base_url = base_url or p.get("base_url")
        if base_url is None:
            raise ValueError("openai-compat backend needs base_url=... (or a preset)")
        key_env = p.get("api_key_env")
        api_key = api_key or (os.environ.get(key_env) if key_env else None) or "EMPTY"
        super().__init__(model, preset=preset, base_url=base_url, image_detail=image_detail,
                         max_side=max_side, max_pixels=max_pixels, **options)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.image_detail = image_detail
        self.max_tokens_param = max_tokens_param
        self.image_policy = ImagePolicy(max_side=max_side, max_pixels=max_pixels)

    def _content(self, msg: Message) -> list[dict[str, Any]] | str:
        if msg.role != "user":
            return "".join(p.text for p in msg.content if isinstance(p, TextPart))
        parts: list[dict[str, Any]] = []
        for p in msg.content:
            if isinstance(p, TextPart):
                parts.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                url: dict[str, Any] = {"url": encode_image(p, self.image_policy).data_url}
                if self.image_detail:
                    url["detail"] = self.image_detail
                parts.append({"type": "image_url", "image_url": url})
        return parts

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": self._content(m)} for m in messages],
        }
        if config.max_tokens is not None:
            kwargs[self.max_tokens_param] = config.max_tokens
        for name in ("temperature", "top_p", "seed"):
            if getattr(config, name) is not None:
                kwargs[name] = getattr(config, name)
        # Unknown keys go in the request body, e.g. OpenRouter's {"provider": {...}} or
        # vLLM's {"top_k": 1, "mm_processor_kwargs": {...}}.
        if config.extra:
            kwargs["extra_body"] = dict(config.extra)

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError,
                openai.NotFoundError) as e:
            raise NonRetryableError(str(e)) from e

        if not resp.choices:
            # OpenRouter reports upstream provider failures in-band
            err = getattr(resp, "error", None) or getattr(resp, "model_extra", {}).get("error")
            raise RuntimeError(f"empty choices from server: {err}")
        choice = resp.choices[0]
        usage = Usage()
        if resp.usage is not None:
            details = getattr(resp.usage, "completion_tokens_details", None)
            usage = Usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                reasoning_tokens=getattr(details, "reasoning_tokens", None),
            )
        return Completion(text=choice.message.content, finish_reason=choice.finish_reason, usage=usage, raw=resp)
