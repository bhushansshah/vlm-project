"""Text reading (EasyOCR)."""

from __future__ import annotations

from typing import Any

from .base import ModelTool, ToolContext, ToolResult, fmt_box, region_of, sub_to_orig
from .image_ops import BOX_SCHEMA, resize_for_view


class OCR(ModelTool):
    name = "ocr"
    description = ("Read the text in the image or in a region of it (signs, licence plates, labels, captions). "
                   "Returns each piece of text with its box and confidence. For small text, give a region "
                   "around it.")
    parameters = {"region": {**BOX_SCHEMA, "description": "optional region to read; default is the whole image"}}
    example = {"region": {"left": 600, "top": 700, "right": 900, "bottom": 800}}

    def load(self):
        import easyocr

        return easyocr.Reader(self.options.get("languages", ["en"]), gpu=self.device().startswith("cuda"),
                              verbose=False)

    def run(self, ctx: ToolContext, region: Any = None) -> ToolResult:
        import numpy as np

        crop, reg = region_of(ctx, region)
        view = resize_for_view(crop, 1024 * 1024, 2560 * 1440, max_side=2560)  # enlarge small text
        reader = self.model()
        with self._lock:
            found = reader.readtext(np.array(view.convert("RGB")))
        s = crop.width / view.width
        items = []
        for pts, text, conf in found:
            xs, ys = [p[0] * s for p in pts], [p[1] * s for p in pts]
            items.append({"text": text, "conf": float(conf),
                          "box": sub_to_orig([min(xs), min(ys), max(xs), max(ys)], crop.size, reg)})
        items.sort(key=lambda i: (i["box"][1] // 20, i["box"][0]))  # rows top to bottom, then left to right
        r = [round(v) for v in reg]
        where = "the whole image" if r == [0, 0, 1000, 1000] else f"region {fmt_box(r)}"
        if not items:
            return ToolResult(text=f"No text found in {where}.", data={"region": r, "items": []})
        lines = [f'"{i["text"]}" box {fmt_box(i["box"])} (confidence {i["conf"]:.2f})' for i in items[:80]]
        return ToolResult(text=f"Text found in {where}, top to bottom:\n" + "\n".join(lines),
                          data={"region": r, "items": [{**i, "conf": round(i["conf"], 3)} for i in items]})
