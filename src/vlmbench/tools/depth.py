"""Relative depth (Depth Anything V2): which objects are nearer to or farther from the camera."""

from __future__ import annotations

from typing import Any

from .base import ModelTool, Tool, ToolContext, ToolError, ToolResult, parse_box, to_pixels
from .image_ops import BOX_SCHEMA


class Depth(ModelTool):
    name = "depth"
    description = ("Compare how far objects are from the camera. Give each object a label and a box; returns "
                   "them ordered from nearest to farthest, with a relative nearness score (1 = nearest point "
                   "in the image, 0 = farthest). Scores are relative, not metres.")
    parameters = {"objects": {"type": "array", "minItems": 1,
                              "items": {"type": "object", "properties": {"label": {"type": "string"},
                                                                         "box": BOX_SCHEMA}},
                              "description": "e.g. [{\"label\": \"bus\", \"box\": {\"left\": 100, \"top\": 400, "
                                             "\"right\": 300, \"bottom\": 700}}]"}}
    required = ("objects",)
    example = {"objects": [{"label": "bus", "box": {"left": 100, "top": 400, "right": 300, "bottom": 700}},
                           {"label": "tree", "box": {"left": 650, "top": 200, "right": 750, "bottom": 600}}]}

    def load(self):
        from transformers import pipeline

        dev = self.device()
        return pipeline("depth-estimation", model=self.options.get("model", "depth-anything/Depth-Anything-V2-Small-hf"),
                        device=0 if dev.startswith("cuda") else -1)

    def _depth_map(self, ctx: ToolContext):
        import numpy as np

        if "depth" not in ctx.cache:
            pipe = self.model()
            with self._lock:
                out = pipe(ctx.image)
            d = np.asarray(out["depth"], dtype=np.float32)  # resized to the image; brighter = nearer
            lo, hi = float(d.min()), float(d.max())
            ctx.cache["depth"] = (d - lo) / (hi - lo) if hi > lo else d * 0
        return ctx.cache["depth"]

    def run(self, ctx: ToolContext, objects: Any = None) -> ToolResult:
        import numpy as np

        if not isinstance(objects, list) or not objects:
            raise ToolError("objects must be a non-empty list of {\"label\": ..., \"box\": [...]}")
        dmap = self._depth_map(ctx)
        h, w = dmap.shape
        results = []
        for i, obj in enumerate(objects, 1):
            if not isinstance(obj, dict) or "box" not in obj:
                raise ToolError(f"object {i} needs a box")
            x1, y1, x2, y2 = to_pixels(parse_box(obj["box"], f"object {i} box"), (w, h))
            # central half of the box: less background bleeding in at the edges
            cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
            cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
            patch = dmap[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]
            results.append({"label": str(obj.get("label", f"object {i}")), "box": obj["box"],
                            "nearness": round(float(np.median(patch)), 3)})
        results.sort(key=lambda r: -r["nearness"])
        lines = [f"{k}. {r['label']} (nearness {r['nearness']:.2f})" for k, r in enumerate(results, 1)]
        return ToolResult(text="Nearest to farthest:\n" + "\n".join(lines), data={"objects": results})


class Distance(Tool):
    name = "distance"
    description = ("Estimate how far objects are from a reference object (e.g. 'which is closer to the robot "
                   "gripper?'), in 3D, not just on the picture. Give the reference and the objects with tight "
                   "boxes (box the part that matters, e.g. the gripper, not the whole arm). Combines image "
                   "position with estimated depth; rough, so only trust clear differences. For distance from "
                   "the camera, use depth.")
    parameters = {"reference": {"type": "object", "properties": {"label": {"type": "string"}, "box": BOX_SCHEMA},
                                "description": "the object distances are measured from"},
                  "objects": {"type": "array", "minItems": 1,
                              "items": {"type": "object", "properties": {"label": {"type": "string"},
                                                                         "box": BOX_SCHEMA}},
                              "description": "the objects to compare"}}
    required = ("reference", "objects")
    example = {"reference": {"label": "gripper", "box": {"left": 400, "top": 100, "right": 500, "bottom": 250}},
               "objects": [{"label": "cup", "box": {"left": 550, "top": 500, "right": 620, "bottom": 600}},
                           {"label": "pen", "box": {"left": 200, "top": 520, "right": 320, "bottom": 560}}]}

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.depth = Depth(**options)  # shares the per-question depth map with the depth tool

    def _point(self, dmap, obj, name: str):
        """Rough 3D point of a box: centre on the image, depth from the median nearness of its middle."""
        import numpy as np

        if not isinstance(obj, dict) or "box" not in obj:
            raise ToolError(f"{name} needs a label and a box")
        box = parse_box(obj["box"], f"{name} box")
        h, w = dmap.shape
        x1, y1, x2, y2 = to_pixels(box, (w, h))
        cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
        cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
        near = float(np.median(dmap[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)]))
        # nearness is relative inverse depth; turn it into a rough depth and back-project with an
        # assumed 60 degree horizontal field of view (only the ranking of distances is meaningful)
        z = 1.0 / (near + self.options.get("depth_offset", 0.1))
        f = w / (2 * np.tan(np.radians(self.options.get("fov_deg", 60.0)) / 2))
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        return np.array([(u - w / 2) / f * z, (v - h / 2) / f * z, z]), box

    def run(self, ctx: ToolContext, reference: Any = None, objects: Any = None) -> ToolResult:
        import numpy as np

        if not isinstance(objects, list) or not objects:
            raise ToolError('objects must be a non-empty list of {"label": ..., "box": {...}}')
        dmap = self.depth._depth_map(ctx)
        ref, _ = self._point(dmap, reference, "reference")
        rows = []
        for i, obj in enumerate(objects, 1):
            p, box = self._point(dmap, obj, f"object {i}")
            label = str(obj.get("label", f"object {i}"))
            rows.append({"label": label, "box": [round(v) for v in box], "distance": float(np.linalg.norm(p - ref))})
        rb = parse_box(reference["box"], "reference box")
        rc = ((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2)
        far = max(r["distance"] for r in rows) or 1.0
        for r in rows:
            b = r["box"]
            r["on_image"] = round(float(np.hypot((b[0] + b[2]) / 2 - rc[0], (b[1] + b[3]) / 2 - rc[1])))
            r["relative"] = round(r["distance"] / far, 2)
        rows.sort(key=lambda r: r["distance"])
        ref_label = str(reference.get("label", "reference"))
        lines = [f"{k}. {r['label']} (3D distance {r['relative']:.2f} relative to the farthest; "
                 f"{r['on_image']} units apart on the image)" for k, r in enumerate(rows, 1)]
        note = ""
        if len(rows) > 1 and rows[1]["relative"] - rows[0]["relative"] < 0.1:
            note = "\nThe two closest are within the estimate's error: look at the image to decide."
        return ToolResult(text=f"Closest to farthest from {ref_label}:\n" + "\n".join(lines) + note,
                          data={"reference": ref_label, "objects": [
                              {k: v for k, v in r.items() if k != "distance"} for r in rows]})
