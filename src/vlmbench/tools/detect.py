"""Open-vocabulary object detection (Grounding DINO): find, locate and count objects named in words."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .base import ModelTool, ToolContext, ToolError, ToolResult, fmt_box, region_of, sub_to_orig
from .image_ops import BOX_SCHEMA, resize_for_view


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[dict], iou: float = 0.5) -> list[dict]:
    kept: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(_iou(d["px"], k["px"]) < iou for k in kept):
            kept.append(d)
    return kept


def draw_boxes(img: Image.Image, dets: list[dict]) -> Image.Image:
    """Numbered boxes on a view of the searched region."""
    view = resize_for_view(img, 640 * 640, 1024 * 1024).convert("RGB")
    s = view.width / img.width
    d = ImageDraw.Draw(view)
    font = ImageFont.load_default(size=max(12, view.width // 45))
    for i, det in enumerate(dets, 1):
        x1, y1, x2, y2 = (v * s for v in det["px"])
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(2, view.width // 400))
        d.text((x1 + 2, y1 + 1), str(i), fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(255, 0, 0))
    return view


class Detect(ModelTool):
    name = "detect"
    description = ("Find every instance of an object described in words (e.g. 'person', 'red car', 'cushion') "
                   "and return how many were found, their boxes, and an image with numbered boxes. Use it to "
                   "count and to locate objects. Use short noun phrases; separate several with commas. Give a "
                   "region to search only part of the image (better for small objects). Detections can miss "
                   "or duplicate objects: check the numbered image.")
    parameters = {"query": {"type": "string", "description": "what to find, e.g. 'cushion' or 'red car, bus'"},
                  "region": {**BOX_SCHEMA, "description": "optional region to search; default is the whole image"}}
    required = ("query",)
    example = {"query": "chair"}

    def load(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = self.options.get("model", "IDEA-Research/grounding-dino-tiny")
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device()).eval()
        return proc, model

    def find(self, crop: Image.Image, reg, labels: list[str], threshold: float) -> list[dict]:
        """Detections in a crop, deduplicated, in reading order, with 0-1000 boxes on the original."""
        import torch

        proc, model = self.model()
        with self._lock, torch.inference_mode():
            inputs = proc(images=crop, text=[labels], return_tensors="pt").to(model.device)
            out = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=threshold,
                text_threshold=min(threshold, self.options.get("text_threshold", 0.25)),
                target_sizes=[crop.size[::-1]])[0]
        names = res.get("text_labels", res.get("labels"))
        dets = [{"px": [float(v) for v in b], "score": float(s), "label": str(n)}
                for b, s, n in zip(res["boxes"].tolist(), res["scores"].tolist(), names)]
        # drop boxes that are the whole search area (a common false positive) and duplicates
        area = crop.width * crop.height
        dets = [d for d in dets if (d["px"][2] - d["px"][0]) * (d["px"][3] - d["px"][1]) < 0.9 * area]
        dets = nms(dets)
        dets.sort(key=lambda d: (d["px"][1], d["px"][0]))  # reading order, so numbers are easy to follow
        for d in dets:
            d["box"] = sub_to_orig(d["px"], crop.size, reg)
        return dets

    def run(self, ctx: ToolContext, query: str = "", region: Any = None) -> ToolResult:
        labels = [q.strip().lower() for q in str(query).split(",") if q.strip()]
        if not labels:
            raise ToolError("query must name at least one object")
        crop, reg = region_of(ctx, region)
        dets = self.find(crop, reg, labels, self.options.get("box_threshold", 0.3))

        r = [round(v) for v in reg]
        where = "the whole image" if r == [0, 0, 1000, 1000] else f"region {fmt_box(r)}"
        if not dets and self.options.get("low_threshold", 0.15) is None:  # candidates off (old behaviour)
            return ToolResult(text=f"No '{query}' detected in {where}.", data={"count": 0, "region": r})
        if not dets:
            # "nothing found" is often a miss (small, unusual or partly hidden objects): show the
            # low-confidence candidates instead, clearly marked, and let the model look
            weak = self.find(crop, reg, labels, self.options.get("low_threshold", 0.15))[:20]
            note = ("The detector often misses small, unusual or partly hidden objects, so this does NOT mean "
                    "the object is absent: look at the image yourself (zoom in) before deciding.")
            if not weak:
                return ToolResult(text=f"No confident '{query}' detections in {where}. {note}",
                                  data={"count": 0, "region": r, "low_confidence": []})
            lines = [f"{i}. {d['label']}? box {fmt_box(d['box'])} (score {d['score']:.2f})"
                     for i, d in enumerate(weak, 1)]
            return ToolResult(
                text=(f"No confident '{query}' detections in {where}. Low-confidence candidates (may be wrong): "
                      f"\n" + "\n".join(lines) + f"\n{note}"),
                images=[draw_boxes(crop, weak)],
                data={"count": 0, "region": r, "low_confidence": [
                    {"label": d["label"], "box": d["box"], "score": round(d["score"], 3)} for d in weak]})
        counts = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        lines = [f"{i}. {d['label']} box {fmt_box(d['box'])} (score {d['score']:.2f})"
                 for i, d in enumerate(dets[:60], 1)]
        if len(dets) > 60:
            lines.append(f"... and {len(dets) - 60} more")
        summary = ", ".join(f"{n} x '{k}'" for k, n in counts.items())
        return ToolResult(
            text=f"Detected {len(dets)} object(s) in {where}: {summary}.\n" + "\n".join(lines),
            images=[draw_boxes(crop, dets)],
            data={"count": len(dets), "counts": counts, "region": r,
                  "detections": [{"label": d["label"], "box": d["box"], "score": round(d["score"], 3)} for d in dets]})
