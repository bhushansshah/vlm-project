"""OpenAI backend via the Responses API (same API the VisualOverload authors used for o3/o4-mini)."""

from __future__ import annotations

import os
from typing import Any

from ..images import ImagePolicy, encode_image
from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage
from .base import VLM, NonRetryableError


class OpenAIResponsesVLM(VLM):
    """Options:
        image_detail: "auto" (default, as in the paper) | "low" | "high" | "original"
        api_key, base_url, timeout, max_retries: client settings
        max_side, max_pixels: optional client-side downscaling (see ImagePolicy)
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        image_detail: str = "auto",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600,
        max_retries: int = 2,
        max_side: int | None = None,
        max_pixels: int | None = None,
        **options: Any,
    ):
        super().__init__(model, image_detail=image_detail, max_side=max_side, max_pixels=max_pixels, **options)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.image_detail = image_detail
        # Responses API allows up to 50 MB per request; this guards against pathological files.
        self.image_policy = ImagePolicy(max_side=max_side, max_pixels=max_pixels, max_b64_bytes=20_000_000)

    def _content(self, msg: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for p in msg.content:
            if isinstance(p, TextPart):
                kind = "output_text" if msg.role == "assistant" else "input_text"
                parts.append({"type": kind, "text": p.text})
            elif isinstance(p, ImagePart):
                img = encode_image(p, self.image_policy)
                parts.append({"type": "input_image", "image_url": img.data_url, "detail": self.image_detail})
        return parts

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        import openai

        instructions = "\n\n".join(
            p.text for m in messages if m.role == "system" for p in m.content if isinstance(p, TextPart)
        )
        inputs = [{"role": m.role, "content": self._content(m)} for m in messages if m.role != "system"]

        kwargs: dict[str, Any] = {"model": self.model, "input": inputs}
        if instructions:
            kwargs["instructions"] = instructions
        if config.max_tokens is not None:
            kwargs["max_output_tokens"] = config.max_tokens
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        kwargs.update(config.extra)  # e.g. reasoning={"effort": "low"}

        try:
            resp = self.client.responses.create(**kwargs)
        except (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError,
                openai.NotFoundError) as e:
            raise NonRetryableError(str(e)) from e

        usage = Usage()
        if resp.usage is not None:
            details = getattr(resp.usage, "output_tokens_details", None)
            usage = Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                reasoning_tokens=getattr(details, "reasoning_tokens", None),
            )
        finish = resp.status
        if resp.incomplete_details is not None:
            finish = f"incomplete:{resp.incomplete_details.reason}"
        return Completion(text=resp.output_text, finish_reason=finish, usage=usage, raw=resp)
