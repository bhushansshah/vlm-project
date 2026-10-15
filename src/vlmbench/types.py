"""Provider-neutral message, generation, and result types.

Every backend consumes a list of ``Message`` objects and returns a ``Completion``.
Messages can hold several text and image parts and the list can span several turns,
so the same backends serve the single-shot baseline and the agentic pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from PIL import Image

Role = Literal["system", "user", "assistant"]


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    """An image given either as a file path or an in-memory PIL image."""

    image: Union[str, Path, Image.Image]

    def load(self) -> Image.Image:
        if isinstance(self.image, Image.Image):
            return self.image
        return Image.open(self.image)


ContentPart = Union[TextPart, ImagePart]


@dataclass
class Message:
    role: Role
    content: list[ContentPart]

    @classmethod
    def user(cls, *parts: ContentPart | str) -> "Message":
        return cls("user", [TextPart(p) if isinstance(p, str) else p for p in parts])

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls("system", [TextPart(text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls("assistant", [TextPart(text)])


@dataclass
class GenerationConfig:
    """Sampling settings shared by all backends.

    ``None`` means "use the provider / model default". The paper used greedy decoding
    for open-weight models and provider defaults for proprietary APIs.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # Provider-specific passthrough, e.g. {"reasoning": {"effort": "low"}} for OpenAI,
    # {"thinking": {...}} for Anthropic, {"thinking_config": {...}} for Gemini.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class Completion:
    text: str | None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # provider response object, not serialized

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "usage": asdict(self.usage),
        }
