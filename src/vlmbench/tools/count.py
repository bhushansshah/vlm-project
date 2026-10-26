"""Mark-and-verify counting: numbered marks the model checks one by one, instead of trusting a number.

``count`` proposes candidates (Grounding DINO at a low threshold, so fewer misses) and numbers them
on the image; the model accepts or rejects each number. ``mark`` puts numbered dots on points the
model chooses, so it can check its own tally (every object marked once, nothing marked twice).
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .base import SCALE, Tool, ToolContext, ToolError, ToolResult, fmt_box, region_of
from .detect import Detect
from .image_ops import BOX_SCHEMA, resize_for_view

POINT_SCHEMA = {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"], "description": "a point on the 0-1000 scale (x horizontal, y vertical)"}


def draw_marks(img: Image.Image, region, points: list[tuple[float, float]], boxes: list | None = None) -> Image.Image:
    """Numbered dots (and optional thin boxes) on a view of ``region``; points/boxes in 0-1000 original coords."""
    view = resize_for_view(img, 768 * 768, 1280 * 1280).convert("RGB")
    rx1, ry1, rx2, ry2 = region
    w, h = view.size

    def px(x, y):
        return (x - rx1) / (rx2 - rx1) * w, (y - ry1) / (ry2 - ry1) * h

    d = ImageDraw.Draw(view)
    r = max(4, min(w, h) // 120)
    font = ImageFont.load_default(size=max(11, min(w, h) // 50))
    for b in boxes or []:
        (x1, y1), (x2, y2) = px(b[0], b[1]), px(b[2], b[3])
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=1)
    for i, (x, y) in enumerate(points, 1):
        cx, cy = px(x, y)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0), outline=(255, 255, 255), width=1)
        d.text((cx + r + 1, cy - r - 1), str(i), fill=(255, 255, 255), font=font, stroke_width=2,
               stroke_fill=(255, 0, 0))
    return view


class Count(Tool):
    name = "count"
    description = ("Count objects by checking candidates one by one. Finds possible instances of the object "
                   "(generously, so it rarely misses one, but some candidates are wrong or duplicates) and "
                   "shows them as numbered marks. Look at every number and decide which ones really are the "
                   "object; your count = the numbers you accept plus any real objects without a mark. Give a "
                   "region to count only part of the image.")
    parameters = {"query": {"type": "string", "description": "what to count, a short noun phrase, e.g. 'cup'"},
                  "region": {**BOX_SCHEMA, "description": "optional region; default is the whole image"}}
    required = ("query",)
    example = {"query": "cup"}

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.detector = Detect(**options)

    def run(self, ctx: ToolContext, query: str = "", region: Any = None) -> ToolResult:
        label = str(query).strip().lower()
        if not label:
            raise ToolError("query must name the object to count")
        crop, reg = region_of(ctx, region)
        dets = self.detector.find(crop, reg, [label], self.options.get("box_threshold", 0.2))
        dets = dets[: self.options.get("max_candidates", 80)]
        r = [round(v) for v in reg]
        where = "the whole image" if r == [0, 0, 1000, 1000] else f"region {fmt_box(r)}"
        if not dets:
            return ToolResult(text=f"No candidate '{label}' found in {where}. Count them yourself (zoom in; "
                                   "use mark to number the ones you see).", data={"candidates": [], "region": r})
        centers = [((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2) for d in dets]
        lines = [f"{i}. center ({cx:.0f}, {cy:.0f}) box {fmt_box(d['box'])} (score {d['score']:.2f})"
                 for i, (d, (cx, cy)) in enumerate(zip(dets, centers), 1)]
        return ToolResult(
            text=(f"{len(dets)} candidate '{label}' marks in {where} (NOT a final count: check each number).\n"
                  + "\n".join(lines)),
            images=[draw_marks(crop, reg, centers, [d["box"] for d in dets])],
            data={"region": r, "candidates": [{"box": d["box"], "score": round(d["score"], 3)} for d in dets]})


class Mark(Tool):
    name = "mark"
    description = ("Put numbered dots on points you choose, to check your own count or list: mark each object "
                   "you counted once, then look at the result to see that every object has exactly one dot "
                   "(none missed, none counted twice).")
    parameters = {"points": {"type": "array", "minItems": 1, "maxItems": 200, "items": POINT_SCHEMA,
                             "description": "e.g. [{\"x\": 120, \"y\": 340}, {\"x\": 410, \"y\": 360}]"},
                  "region": {**BOX_SCHEMA, "description": "optional region to show; default is the whole image"}}
    required = ("points",)
    example = {"points": [{"x": 120, "y": 340}, {"x": 410, "y": 360}]}

    def run(self, ctx: ToolContext, points: Any = None, region: Any = None) -> ToolResult:
        if not isinstance(points, list) or not points:
            raise ToolError('points must be a non-empty list of {"x": .., "y": ..}')
        pts = []
        for i, p in enumerate(points, 1):
            try:
                x, y = (float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1]))
            except (KeyError, IndexError, TypeError, ValueError):
                raise ToolError(f'point {i} must be {{"x": .., "y": ..}} on the 0-1000 scale, got {p!r}') from None
            pts.append((min(max(x, 0.0), SCALE), min(max(y, 0.0), SCALE)))
        crop, reg = region_of(ctx, region)
        outside = [i for i, (x, y) in enumerate(pts, 1) if not (reg[0] <= x <= reg[2] and reg[1] <= y <= reg[3])]
        text = f"Marked {len(pts)} point(s) in region {fmt_box(reg)}."
        if outside:
            text += f" Points {outside} are outside the shown region."
        return ToolResult(text=text, images=[draw_marks(crop, reg, pts)], data={"points": pts})
