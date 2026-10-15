"""Anthropic (Claude) backend via the Messages API."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..images import ImagePolicy, encode_image
from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage
from .base import VLM, NonRetryableError

log = logging.getLogger(__name__)

# The API rejects images above 5 MB; some VisualOverload JPEGs exceed this once base64-encoded,
# so those are re-encoded (quality first, then size) just enough to fit. The API also
# downscales large images server-side, so this rarely changes what the model sees.
_MAX_IMAGE_B64_BYTES = 5 * 1024 * 1024


class AnthropicVLM(VLM):
    """Options:
        default_max_tokens: used when GenerationConfig.max_tokens is None (the API requires it)
        api_key, base_url, timeout, max_retries: client settings
        max_side, max_pixels: optional client-side downscaling (see ImagePolicy)
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        default_max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600,
        max_retries: int = 2,
        max_side: int | None = None,
        max_pixels: int | None = None,
        **options: Any,
    ):
        super().__init__(model, default_max_tokens=default_max_tokens, max_side=max_side,
                         max_pixels=max_pixels, **options)
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.default_max_tokens = default_max_tokens
        self.image_policy = ImagePolicy(max_side=max_side, max_pixels=max_pixels,
                                        max_b64_bytes=_MAX_IMAGE_B64_BYTES)

    def _content(self, msg: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for p in msg.content:
            if isinstance(p, TextPart):
                parts.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                img = encode_image(p, self.image_policy)
                parts.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img.mime_type, "data": img.b64},
                })
        return parts

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        import anthropic

        system = "\n\n".join(
            p.text for m in messages if m.role == "system" for p in m.content if isinstance(p, TextPart)
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.max_tokens or self.default_max_tokens,
            "messages": [{"role": m.role, "content": self._content(m)} for m in messages if m.role != "system"],
        }
        if system:
            kwargs["system"] = system
        # Current Claude models do not take sampling parameters; older ones still do, so
        # forward them untyped only when explicitly requested.
        sampling = {k: v for k, v in (("temperature", config.temperature), ("top_p", config.top_p)) if v is not None}
        if sampling:
            kwargs["extra_body"] = sampling
        kwargs.update(config.extra)  # e.g. thinking={...}, output_config={"effort": "low"}

        try:
            resp = self.client.messages.create(**kwargs)
        except (anthropic.BadRequestError, anthropic.AuthenticationError,
                anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise NonRetryableError(str(e)) from e

        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
        return Completion(text=text, finish_reason=resp.stop_reason, usage=usage, raw=resp)
