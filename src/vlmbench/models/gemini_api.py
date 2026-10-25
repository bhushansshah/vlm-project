"""Google Gemini backend via the google-genai SDK.

Mirrors the authors' setup: image part first, then the prompt, with safety filters
disabled (several paintings show violence or nudity that would otherwise be blocked).
"""

from __future__ import annotations

import os
from typing import Any

from ..images import ImagePolicy, encode_image
from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage
from .base import VLM, NonRetryableError

_SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]


class GeminiVLM(VLM):
    """Options:
        media_resolution: per-image token budget, e.g. "MEDIA_RESOLUTION_HIGH"
            or "MEDIA_RESOLUTION_ULTRA_HIGH" (default: model default, as in the paper)
        disable_safety: set all safety thresholds to BLOCK_NONE (default True, as in the paper)
        api_key: defaults to $GEMINI_API_KEY, then $GOOGLE_API_KEY
        vertexai, project, location: use Vertex AI instead of the Gemini Developer API
        timeout: request timeout in seconds
        max_side, max_pixels: optional client-side downscaling (see ImagePolicy)
    """

    provider = "gemini"

    def __init__(
        self,
        model: str,
        media_resolution: str | None = None,
        disable_safety: bool = True,
        api_key: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        timeout: float = 600,
        max_side: int | None = None,
        max_pixels: int | None = None,
        **options: Any,
    ):
        super().__init__(model, media_resolution=media_resolution, disable_safety=disable_safety,
                         max_side=max_side, max_pixels=max_pixels, **options)
        from google import genai
        from google.genai import types

        self.types = types
        if vertexai:
            self.client = genai.Client(vertexai=True, project=project, location=location,
                                       http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        else:
            self.client = genai.Client(
                api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )
        self.media_resolution = media_resolution
        self.disable_safety = disable_safety
        # Inline requests are limited to 20 MB in total.
        self.image_policy = ImagePolicy(max_side=max_side, max_pixels=max_pixels, max_b64_bytes=15_000_000)

    def _parts(self, msg: Message) -> list[Any]:
        t = self.types
        parts = []
        for p in msg.content:
            if isinstance(p, TextPart):
                parts.append(t.Part.from_text(text=p.text))
            elif isinstance(p, ImagePart):
                img = encode_image(p, self.image_policy)
                kw = {"media_resolution": {"level": self.media_resolution}} if self.media_resolution else {}
                parts.append(t.Part.from_bytes(data=img.data, mime_type=img.mime_type, **kw))
        return parts

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        from google.genai import errors

        t = self.types
        system = "\n\n".join(
            p.text for m in messages if m.role == "system" for p in m.content if isinstance(p, TextPart)
        )
        contents = [
            t.Content(role="model" if m.role == "assistant" else "user", parts=self._parts(m))
            for m in messages if m.role != "system"
        ]
        cfg: dict[str, Any] = {
            "system_instruction": system or None,
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "stop_sequences": config.stop,
        }
        if self.disable_safety:
            cfg["safety_settings"] = [
                t.SafetySetting(category=c, threshold="BLOCK_NONE") for c in _SAFETY_CATEGORIES
            ]
        cfg.update(config.extra)  # e.g. thinking_config={"thinking_budget": 0}

        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=t.GenerateContentConfig(**cfg)
            )
        except errors.ClientError as e:
            if e.code == 429:
                raise  # quota / rate limit: let the runner back off and retry
            raise NonRetryableError(str(e)) from e

        feedback = resp.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            raise NonRetryableError(f"prompt blocked: {feedback.block_reason}")

        finish = None
        if resp.candidates:
            fr = resp.candidates[0].finish_reason
            finish = getattr(fr, "value", fr)
        um = resp.usage_metadata
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", None),
            output_tokens=getattr(um, "candidates_token_count", None),
            reasoning_tokens=getattr(um, "thoughts_token_count", None),
            cached_tokens=getattr(um, "cached_content_token_count", None),
        )
        return Completion(text=resp.text, finish_reason=finish, usage=usage, raw=resp)
