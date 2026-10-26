"""Tool interface for tool-using agents.

Every tool works on the question's ORIGINAL image and speaks one coordinate system: a box is
``{"left": .., "top": .., "right": .., "bottom": ..}`` on a 0-1000 scale (left/right horizontal,
top/bottom vertical), whatever resolution the model itself sees. Named fields, not a bare list:
models trained on different box conventions ([x1, y1, x2, y2] vs [y1, x1, y2, x2]) silently
swapped axes with lists. Lists are still accepted, read as [left, top, right, bottom].
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

from PIL import Image

SCALE = 1000
_LOAD_LOCK = threading.Lock()  # tools load one at a time: concurrent `from transformers import ...` can fail


class ToolError(Exception):
    """A problem with the call (bad arguments, nothing found...): shown to the model, not fatal."""


@dataclass
class ToolResult:
    text: str
    images: list[Image.Image] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # machine-readable result, stored in the trace


@dataclass
class ToolContext:
    """Per-question state shared by the tools."""

    image: Image.Image  # original image, RGB
    cache: dict[str, Any] = field(default_factory=dict)  # per-question results tools may reuse (e.g. a depth map)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


BOX_KEYS = ("left", "top", "right", "bottom")


def fmt_box(box) -> str:
    """A 0-1000 box as the named-field JSON the tools accept, for tool result text."""
    return "{" + ", ".join(f'"{k}": {round(v)}' for k, v in zip(BOX_KEYS, box)) + "}"


def parse_box(box: Any, name: str = "box") -> tuple[float, float, float, float]:
    """Validate a 0-1000 box (named fields, or a [left, top, right, bottom] list); clips to the
    image and orders the corners. Returns (left, top, right, bottom)."""
    if isinstance(box, dict):
        if not all(k in box for k in BOX_KEYS):
            raise ToolError(f'{name} needs "left", "top", "right" and "bottom" (0-1000), got {box!r}')
        box = [box[k] for k in BOX_KEYS]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ToolError(f'{name} must be {{"left": .., "top": .., "right": .., "bottom": ..}} on a 0-1000 scale, '
                        f"got {box!r}")
    try:
        x1, y1, x2, y2 = (min(max(float(v), 0.0), SCALE) for v in box)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must contain four numbers, got {box!r}") from None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ToolError(f"{name} {box!r} is empty")
    return x1, y1, x2, y2


def to_pixels(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = box
    return (int(x1 * w / SCALE), int(y1 * h / SCALE), max(int(x1 * w / SCALE) + 1, round(x2 * w / SCALE)),
            max(int(y1 * h / SCALE) + 1, round(y2 * h / SCALE)))


def to_norm(px_box, size: tuple[int, int]) -> list[int]:
    w, h = size
    x1, y1, x2, y2 = px_box
    return [round(x1 * SCALE / w), round(y1 * SCALE / h), round(x2 * SCALE / w), round(y2 * SCALE / h)]


def region_of(ctx: ToolContext, region: Any, name: str = "region") -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Crop for an optional box argument called ``name`` (whole image when None)."""
    box = parse_box(region, name) if region is not None else (0.0, 0.0, float(SCALE), float(SCALE))
    return ctx.image.crop(to_pixels(box, ctx.size)), box


def sub_to_orig(px_box, crop_size: tuple[int, int], region: tuple[float, float, float, float]) -> list[int]:
    """Pixel box inside a crop -> 0-1000 box on the original image."""
    cw, ch = crop_size
    rx1, ry1, rx2, ry2 = region
    x1, y1, x2, y2 = px_box
    return [round(rx1 + x1 / cw * (rx2 - rx1)), round(ry1 + y1 / ch * (ry2 - ry1)),
            round(rx1 + x2 / cw * (rx2 - rx1)), round(ry1 + y2 / ch * (ry2 - ry1))]


class Tool(abc.ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    #: JSON-schema "properties" of the arguments
    parameters: ClassVar[dict[str, Any]]
    required: ClassVar[tuple[str, ...]] = ()
    #: example arguments, shown to the model by instructions that list examples
    example: ClassVar[dict[str, Any]] = {}

    def __init__(self, **options: Any):
        self.options = options

    @abc.abstractmethod
    def run(self, ctx: ToolContext, **args: Any) -> ToolResult: ...

    def spec(self) -> dict[str, Any]:
        """Description shown to the model."""
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.parameters, "required": list(self.required)}}

    def __call__(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        if "box" in args and "box" not in self.parameters and "region" in self.parameters and "region" not in args:
            args = {**args, "region": args["box"]}  # a common slip: zoom calls it "box"
            del args["box"]
        missing = [k for k in self.required if k not in args]
        if missing:
            raise ToolError(f"missing argument(s): {', '.join(missing)}")
        unknown = set(args) - set(self.parameters)
        if unknown:
            raise ToolError(f"unknown argument(s): {', '.join(sorted(unknown))}; allowed: {', '.join(self.parameters)}")
        return self.run(ctx, **args)


class ModelTool(Tool):
    """A tool backed by a neural network: loaded lazily, once, and used under a lock
    (several questions may run concurrently in threads)."""

    _lock: ClassVar[threading.Lock]

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._lock = threading.Lock()
        cls._loaded = None

    def device(self) -> str:
        dev = self.options.get("device")
        if dev:
            return dev
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def model(self):
        if type(self)._loaded is None:
            with _LOAD_LOCK:
                if type(self)._loaded is None:
                    type(self)._loaded = self.load()
        return type(self)._loaded

    @abc.abstractmethod
    def load(self): ...
