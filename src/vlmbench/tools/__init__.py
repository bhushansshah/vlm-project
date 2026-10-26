"""Tool registry. Add a tool by subclassing ``Tool`` (or ``ModelTool``) and listing it here."""

from __future__ import annotations

from typing import Any

from .base import ModelTool, Tool, ToolContext, ToolError, ToolResult
from .colour import Colour
from .count import Count, Mark
from .depth import Depth, Distance
from .detect import Detect
from .enhance import Enhance
from .image_ops import Grid, Zoom
from .ocr import OCR
from .order import Order
from .pose import Orient
from .viewpoint import Viewpoint

TOOLS: dict[str, type[Tool]] = {t.name: t for t in (Zoom, Grid, Detect, OCR, Depth, Count, Mark, Distance,
                                                    Enhance, Order, Orient, Viewpoint, Colour)}
DEFAULT_TOOLS = ("zoom", "grid", "detect", "ocr", "depth")


def build_tools(names: list[str] | tuple[str, ...] | None = None,
                options: dict[str, dict[str, Any]] | None = None) -> list[Tool]:
    """Instantiate tools by name; ``options`` maps a tool name to its constructor options,
    e.g. {"detect": {"box_threshold": 0.35}, "zoom": {"max_side": 768}}."""
    names = list(names or DEFAULT_TOOLS)
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        raise ValueError(f"Unknown tool(s) {unknown}. Available: {sorted(TOOLS)}")
    return [TOOLS[n](**(options or {}).get(n, {})) for n in names]


__all__ = ["DEFAULT_TOOLS", "TOOLS", "ModelTool", "Tool", "ToolContext", "ToolError", "ToolResult", "build_tools"]
