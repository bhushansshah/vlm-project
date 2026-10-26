"""Name the main colours of objects (k-means on their pixels), e.g. a shirt or a cart's umbrella."""

from __future__ import annotations

import colorsys
from typing import Any

from .base import Tool, ToolContext, ToolError, ToolResult, fmt_box, parse_box, to_pixels
from .image_ops import BOX_SCHEMA


def colour_name(rgb) -> str:
    """Basic colour word for an RGB triple (0-255), from hue, saturation and brightness."""
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.18:
        return "black"
    if s < 0.15:
        return "white" if v > 0.85 else "light gray" if v > 0.6 else "gray" if v > 0.35 else "dark gray"
    if s < 0.35 and v < 0.5:
        return "dark gray"
    if h < 15 or h >= 345:
        base = "red"
    elif h < 40:
        base = "orange"
    elif h < 65:
        base = "yellow"
    elif h < 160:
        base = "green"
    elif h < 195:
        base = "cyan"
    elif h < 255:
        base = "blue"
    elif h < 290:
        base = "purple"
    else:
        base = "pink"
    if base in ("orange", "red", "yellow") and v < 0.55:
        return "brown" if base != "red" or s < 0.7 else "dark red"
    if base == "red" and s < 0.5 and v > 0.7:
        return "pink"
    if v < 0.4:
        return f"dark {base}"
    if s < 0.45 and v > 0.75:
        return f"light {base}"
    return base


def dominant_colours(pixels, k: int = 3, iters: int = 12, seed: int = 0) -> list[tuple[tuple[int, int, int], float]]:
    """k-means on an (N, 3) array: [(rgb, share)], biggest first."""
    import numpy as np

    px = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
    k = min(k, len(px))
    rng = np.random.default_rng(seed)
    centres = px[rng.choice(len(px), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((px[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        centres = np.stack([px[labels == j].mean(0) if np.any(labels == j) else centres[j] for j in range(k)])
    shares = np.bincount(labels, minlength=k) / len(px)
    order = np.argsort(-shares)
    return [(tuple(int(round(c)) for c in centres[j]), float(shares[j])) for j in order]


class Colour(Tool):
    name = "colour"
    description = ("Name the main colours of up to 4 objects (or parts: trousers, a flag, an umbrella's stripes), "
                   "with each colour's share of the box. Box tightly around the part you mean; the centre of the "
                   "box counts most. Shadows, lighting and background shift the result: use it to settle close "
                   "calls (blue vs purple, white vs light gray) together with zoom, not instead of looking.")
    parameters = {"objects": {"type": "array", "minItems": 1, "maxItems": 4,
                              "items": {"type": "object", "properties": {"label": {"type": "string"},
                                                                         "box": BOX_SCHEMA}}}}
    required = ("objects",)
    example = {"objects": [{"label": "jacket", "box": {"left": 410, "top": 380, "right": 470, "bottom": 520}}]}

    def run(self, ctx: ToolContext, objects: Any = None) -> ToolResult:
        import numpy as np

        if not isinstance(objects, list) or not objects:
            raise ToolError('objects must be a non-empty list of {"label": ..., "box": {...}}')
        lines, data = [], []
        for i, obj in enumerate(objects[:4], 1):
            if not isinstance(obj, dict) or "box" not in obj:
                raise ToolError(f"object {i} needs a box")
            b = parse_box(obj["box"], f"object {i} box")
            x1, y1, x2, y2 = to_pixels(b, ctx.size)
            mx, my = (x2 - x1) // 6, (y2 - y1) // 6  # central part: less background
            crop = ctx.image.crop((x1 + mx, y1 + my, max(x2 - mx, x1 + mx + 1), max(y2 - my, y1 + my + 1)))
            crop.thumbnail((48, 48))
            cols = dominant_colours(np.asarray(crop.convert("RGB")))
            named = [(colour_name(c), share, c) for c, share in cols]
            label = str(obj.get("label", f"object {i}"))
            lines.append(f"{i}. {label} {fmt_box([round(v) for v in b])}: "
                         + ", ".join(f"{n} {round(100 * sh)}% (rgb {c[0]},{c[1]},{c[2]})" for n, sh, c in named))
            data.append({"label": label, "colours": [{"name": n, "share": round(sh, 3), "rgb": list(c)}
                                                     for n, sh, c in named]})
        return ToolResult(text="Main colours (largest share first):\n" + "\n".join(lines), data={"objects": data})
