"""Agent registry: an agent wraps a backend ``VLM`` and is itself a ``VLM``.

    tools   tool-augmented ReAct agent (pipeline 2)
"""

from __future__ import annotations

from typing import Any

from ..models.base import VLM
from .tool_agent import ToolAgent, summarize_traces

AGENTS: dict[str, type[VLM]] = {"tools": ToolAgent}


def build_agent(name: str, model: VLM, **options: Any) -> VLM:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent {name!r}. Available: {sorted(AGENTS)}")
    return AGENTS[name](model, **options)


__all__ = ["AGENTS", "ToolAgent", "build_agent", "summarize_traces"]
