"""Provider-neutral message, generation, and result types.

Every backend consumes a list of ``Message`` objects and returns a ``Completion``.
Messages can hold several text and image parts and the list can span several turns,
so the same backends serve the single-shot baseline and the agentic pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Literal, Union

from PIL import Image

Role = Literal["system", "user", "assistant", "tool"]


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
    """A turn. Agents using native tool calling also set ``tool_calls`` on an assistant turn
    ([{"id", "name", "arguments"}]) and send each result as a "tool" turn with ``tool_call_id``.
    Backends without native tool messages never receive these (the agent falls back to text)."""
    role: Role
    content: list[ContentPart]
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

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
    stop: list[str] | None = None  # stop sequences (where the backend supports them)
    # Tool specs ({"name", "description", "parameters"}) to declare natively where the backend supports
    # it; native tool calls come back as <action>{"name": ..., "arguments": ...}</action> text.
    tools: list[dict[str, Any]] | None = None
    # Provider-specific passthrough, e.g. {"reasoning": {"effort": "low"}} for OpenAI,
    # {"thinking": {...}} for Anthropic, {"thinking_config": {...}} for Gemini.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    """Token counts and cost of one call (or a sum of calls). None = the provider did not report it.

    input_tokens counts every prompt token, cached ones included; cached_tokens is how many of them
    were read from the provider's prompt cache (billed cheaper). cost is the charge in USD as the
    provider reports it (OpenRouter does; the OpenAI, Anthropic and Gemini APIs do not)."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cost: float | None = None

    def add(self, other: Usage) -> None:
        """Add another call's usage in place. A field stays None only if neither side reported it."""
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if b is not None:
                setattr(self, f.name, b if a is None else a + b)


def usage_totals(usages: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    """Sum saved usage dicts (records from older runs may lack fields). cost_usd / cached are None
    when no call reported them; cost_reported_calls says how many calls the cost covers."""
    tot = {"input": 0, "output": 0, "reasoning": 0, "cached": 0}
    n_calls = n_cost = n_cached = 0
    cost = 0.0
    for u in usages:
        if not u:
            continue
        n_calls += 1
        for key, f in (("input", "input_tokens"), ("output", "output_tokens"),
                       ("reasoning", "reasoning_tokens"), ("cached", "cached_tokens")):
            tot[key] += u.get(f) or 0
        n_cached += u.get("cached_tokens") is not None
        if u.get("cost") is not None:
            n_cost += 1
            cost += u["cost"]
    if not n_cached:
        tot["cached"] = None
    return {**tot, "calls": n_calls, "cost_usd": round(cost, 6) if n_cost else None, "cost_reported_calls": n_cost}


@dataclass
class Completion:
    text: str | None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # provider response object, not serialized
    trace: list[dict[str, Any]] | None = None  # agent steps (tool calls and results), when an agent produced it

    def to_dict(self) -> dict[str, Any]:
        d = {"text": self.text, "finish_reason": self.finish_reason, "usage": asdict(self.usage)}
        if self.trace is not None:
            d["trace"] = self.trace
        return d
