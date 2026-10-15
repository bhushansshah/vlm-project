from __future__ import annotations

import abc
from typing import Any, ClassVar

from ..types import Completion, GenerationConfig, Message


class VLM(abc.ABC):
    """A vision-language model backend.

    Subclasses implement ``generate`` for one conversation. ``generate_batch`` has a
    sequential default; local backends may override it with true batching.
    """

    #: provider key used in model specs, e.g. "openai" in "openai:gpt-5-mini"
    provider: ClassVar[str]
    #: whether concurrent ``generate`` calls from multiple threads are safe & useful
    supports_concurrency: ClassVar[bool] = True

    def __init__(self, model: str, **options: Any):
        self.model = model
        self.options = options

    @abc.abstractmethod
    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion: ...

    def generate_batch(
        self, conversations: list[list[Message]], config: GenerationConfig
    ) -> list[Completion]:
        return [self.generate(m, config) for m in conversations]

    def describe(self) -> dict[str, Any]:
        """Serializable description recorded in each run's config."""
        return {"provider": self.provider, "model": self.model, "options": self.options}


class NonRetryableError(Exception):
    """Raised by backends for errors a retry cannot fix (bad request, auth, content block)."""
