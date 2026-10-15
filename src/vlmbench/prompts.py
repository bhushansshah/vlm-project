"""Prompt builders: turn a ``Sample`` into the conversation sent to the model.

``default`` reproduces the paper exactly: the dataset's ``default_prompt``, no system
prompt, image first. Further strategies can be added with ``@register_prompt``.
"""

from __future__ import annotations

from typing import Callable

from .data import Sample
from .types import ImagePart, Message

PromptBuilder = Callable[[Sample], list[Message]]
PROMPTS: dict[str, PromptBuilder] = {}


def register_prompt(name: str):
    def deco(fn: PromptBuilder) -> PromptBuilder:
        PROMPTS[name] = fn
        return fn
    return deco


def build_messages(name: str, sample: Sample) -> list[Message]:
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt {name!r}. Available: {sorted(PROMPTS)}")
    return PROMPTS[name](sample)


def format_options(options: tuple[str, ...]) -> str:
    return "\n".join(f"{chr(65 + i)}. {o.strip()}" for i, o in enumerate(options))


@register_prompt("default")
def default_prompt(s: Sample) -> list[Message]:
    """The authors' prompt, e.g. "<question> Options:\\nA. yes\\nB. no\\nAnswer with the
    option's letter from the given choices directly." """
    return [Message.user(ImagePart(s.image_path), s.default_prompt)]


_FORMAT = {
    "choice": "the option's letter from the given choices",
    "counting": "a single number",
    "ocr": "the exact text as written in the image",
}


@register_prompt("think_then_answer")
def think_then_answer(s: Sample) -> list[Message]:
    """Let the model reason first, then give the final answer in <answer></answer> tags
    (the official judge extracts answers from these tags)."""
    body = s.question
    if s.question_type == "choice":
        body += "\nOptions:\n" + format_options(s.options)
    body += (
        "\n\nLook carefully at the relevant parts of the image and think step by step. "
        f"Then give your final answer as {_FORMAT[s.question_type]} inside <answer></answer> tags."
    )
    return [Message.user(ImagePart(s.image_path), body)]
