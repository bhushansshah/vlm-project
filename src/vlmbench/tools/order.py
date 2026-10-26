"""Number every instance of an object in a stated order ("the third bicycle from the right")."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .base import Tool, ToolContext, ToolError, ToolResult, fmt_box, region_of
from .depth import Depth
from .detect import Detect
from .image_ops import BOX_SCHEMA, resize_for_view

ORDERS = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "near_to_far", "far_to_near")
ORDINALS = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 61)]


def sort_key(order: str, box, nearness: float | None = None) -> float:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return {"left_to_right": cx, "right_to_left": -cx, "top_to_bottom": cy, "bottom_to_top": -cy,
            "near_to_far": -(nearness or 0.0), "far_to_near": nearness or 0.0}[order]


def draw_ordered(img: Image.Image, dets: list[dict]) -> Image.Image:
    """Boxes labeled with their position in the order (1st, 2nd, ...)."""
    view = resize_for_view(img, 640 * 640, 1280 * 1280).convert("RGB")
    s = view.width / img.width
    d = ImageDraw.Draw(view)
    font = ImageFont.load_default(size=max(14, view.width // 40))
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = (v * s for v in det["px"])
        d.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=max(2, view.width // 350))
        d.text((x1 + 2, max(0, y1 - font.size - 2)), ORDINALS[i], fill=(255, 255, 255), font=font,
               stroke_width=3, stroke_fill=(200, 80, 0))
    return view


class Order(Tool):
    name = "order"
    description = ("Find every instance of an object and number them in a stated order: left_to_right, "
                   "right_to_left, top_to_bottom, bottom_to_top, near_to_far or far_to_near (by estimated depth). "
                   "Returns the ordered list with boxes and an image labeling each instance 1st, 2nd, 3rd, ... "
                   "Use it for 'the third X from the right' and 'which X, counting from the left' questions. Give "
                   "a region to order only part of the scene (e.g. 'the trees behind the sign'). The detector can "
                   "miss, merge or add instances: check the labeled image and adjust the count yourself.")
    parameters = {"query": {"type": "string", "description": "what to find, e.g. 'person' or 'streetlight'"},
                  "order": {"type": "string", "enum": list(ORDERS)},
                  "region": {**BOX_SCHEMA, "description": "optional region; default is the whole image"}}
    required = ("query", "order")
    example = {"query": "bicycle", "order": "right_to_left"}

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.detect = Detect(**options.get("detect", {}))
        self.depth = Depth(**options.get("depth", {}))

    def run(self, ctx: ToolContext, query: str = "", order: str = "", region: Any = None) -> ToolResult:
        if order not in ORDERS:
            raise ToolError(f"order must be one of {', '.join(ORDERS)}, got {order!r}")
        labels = [q.strip().lower() for q in str(query).split(",") if q.strip()]
        if not labels:
            raise ToolError("query must name the object to order")
        crop, reg = region_of(ctx, region)
        dets = self.detect.find(crop, reg, labels, self.options.get("box_threshold", 0.3))
        if not dets:  # fall back to weaker detections rather than "none"
            dets = self.detect.find(crop, reg, labels, self.options.get("low_threshold", 0.18))
        if not dets:
            raise ToolError(f"no '{query}' found here; zoom in and count them yourself, or try another name")
        if order in ("near_to_far", "far_to_near"):
            import numpy as np

            dmap = self.depth._depth_map(ctx)
            h, w = dmap.shape
            for d in dets:
                x1, y1, x2, y2 = (int(d["box"][0] * w / 1000), int(d["box"][1] * h / 1000),
                                  int(d["box"][2] * w / 1000), int(d["box"][3] * h / 1000))
                cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
                cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
                d["nearness"] = float(np.median(dmap[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]))
        dets.sort(key=lambda d: sort_key(order, d["box"], d.get("nearness")))
        lines = [f"{ORDINALS[i]}: {d['label']} box {fmt_box(d['box'])} (score {d['score']:.2f}"
                 + (f", nearness {d['nearness']:.2f}" if "nearness" in d else "") + ")"
                 for i, d in enumerate(dets[:60])]
        r = [round(v) for v in reg]
        where = "the whole image" if r == [0, 0, 1000, 1000] else f"region {fmt_box(r)}"
        return ToolResult(
            text=(f"{len(dets)} '{query}' in {where}, ordered {order.replace('_', ' ')}:\n" + "\n".join(lines)
                  + "\nCheck the labeled image: if an instance is missed or doubled, fix the order before answering."),
            images=[draw_ordered(crop, dets)],
            data={"order": order, "count": len(dets), "region": r,
                  "items": [{"label": d["label"], "box": d["box"], "score": round(d["score"], 3)} for d in dets]})
