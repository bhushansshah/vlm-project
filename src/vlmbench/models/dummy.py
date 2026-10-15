"""Offline backend for testing the pipeline end to end without any API key or GPU."""

from __future__ import annotations

import random
import re
from typing import Any

from ..types import Completion, GenerationConfig, ImagePart, Message, TextPart, Usage
from .base import VLM


class DummyVLM(VLM):
    """Model names:
        dummy:random  - a random valid answer for each question format (a chance-level baseline)
        dummy:echo    - returns the prompt text (useful for checking prompts)
    """

    provider = "dummy"

    def __init__(self, model: str, seed: int = 0, **options: Any):
        super().__init__(model, seed=seed, **options)
        self.seed = seed

    def generate(self, messages: list[Message], config: GenerationConfig) -> Completion:
        prompt = "\n".join(p.text for m in messages for p in m.content if isinstance(p, TextPart))
        n_images = sum(isinstance(p, ImagePart) for m in messages for p in m.content)
        if self.model == "echo":
            text = f"[{n_images} image(s)] {prompt}"
        else:
            rng = random.Random(f"{self.seed}:{prompt}")
            letters = re.findall(r"^([A-D])\. ", prompt, flags=re.M)
            if letters:
                text = rng.choice(letters)
            elif "number" in prompt.lower():
                text = str(rng.randint(0, 20))
            else:
                text = "lorem"
        return Completion(text=text, finish_reason="stop", usage=Usage(len(prompt.split()), 1))
