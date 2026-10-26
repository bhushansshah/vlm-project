"""Move the camera to an object in the scene: where are other objects from ITS point of view?

Each object becomes a point on a top-down (bird's-eye) map: sideways position from where it is in
the image, distance from the camera from estimated depth (the same back-projection as the distance
tool). The map is then re-centred on the reference and turned so the reference looks "up", which
turns "from the woman's perspective, where is the bus?" into reading a map. Models answering these
questions on their own mostly describe left and right as seen from the camera; this tool does the
change of viewpoint explicitly.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .base import Tool, ToolContext, ToolError, ToolResult
from .depth import Distance
from .image_ops import BOX_SCHEMA
from .pose import describe_angle

FACING_WORDS = {"away_from_camera": 0.0, "image_right": 90.0, "toward_camera": 180.0, "image_left": 270.0}


def facing_angle(facing: Any) -> float:
    """Facing as degrees: 0 = away from the camera, 90 = toward image-right, 180 = toward the camera,
    270 = toward image-left. Accepts those numbers or the four words."""
    if isinstance(facing, (int, float)) and not isinstance(facing, bool):
        return float(facing) % 360
    key = str(facing).strip().lower().replace(" ", "_").replace("-", "_")
    if key in FACING_WORDS:
        return FACING_WORDS[key]
    try:
        return float(key) % 360
    except ValueError:
        raise ToolError(f"facing must be one of {', '.join(FACING_WORDS)} or an angle in degrees "
                        f"(0 = away from the camera, 90 = image-right, 180 = toward the camera, 270 = image-left); "
                        f"got {facing!r}") from None


def relative_position(ref_xz, facing: float, target_xz) -> dict[str, float]:
    """Target position in the reference's own frame. Map axes: x to the image's right, z away from
    the camera. Returns ``forward`` (> 0 in front of the reference), ``right`` (> 0 on the
    reference's right) and ``bearing`` (degrees: 0 ahead, 90 right, -90 left, +-180 behind)."""
    a = math.radians(facing)
    fx, fz = math.sin(a), math.cos(a)  # facing direction on the map
    rx, rz = fz, -fx  # the reference's right-hand side
    dx, dz = target_xz[0] - ref_xz[0], target_xz[1] - ref_xz[1]
    forward, right = dx * fx + dz * fz, dx * rx + dz * rz
    return {"forward": forward, "right": right, "bearing": math.degrees(math.atan2(right, forward))}


def direction_words(bearing: float, margin: float = 15.0) -> tuple[str, str]:
    """('front left', note): quadrant name, plus a warning when the target is near an axis."""
    fb = "front" if abs(bearing) < 90 else "rear"
    lr = "right" if bearing > 0 else "left"
    note = ""
    if abs(bearing) < margin or abs(bearing) > 180 - margin:
        note = f"nearly straight {'ahead' if abs(bearing) < 90 else 'behind'}, so left/right is uncertain"
    elif abs(abs(bearing) - 90) < margin:
        note = f"nearly level with the reference (to its {lr}), so front/rear is uncertain"
    return f"{fb} {lr}", note


def draw_map(ref_label: str, rows: list[dict], camera: dict[str, float], size: int = 560) -> Image.Image:
    """Top-down map in the reference's frame: reference at the centre looking up."""
    img = Image.new("RGB", (size, size), (250, 248, 240))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    c = size / 2
    pts = [(r["right"], r["forward"]) for r in rows] + [(camera["right"], camera["forward"])]
    reach = max(max(abs(x), abs(y)) for x, y in pts) or 1.0
    s = (size / 2 - 48) / reach

    def xy(right, forward):
        return c + right * s, c - forward * s

    d.line([(c, 24), (c, size - 24)], fill=(200, 196, 186), width=1)
    d.line([(24, c), (size - 24, c)], fill=(200, 196, 186), width=1)
    for text, (tx, ty) in {"FRONT LEFT": (30, 30), "FRONT RIGHT": (size - 125, 30),
                           "REAR LEFT": (30, size - 46), "REAR RIGHT": (size - 120, size - 46)}.items():
        d.text((tx, ty), text, fill=(150, 145, 135), font=font)
    # reference: a triangle pointing up = the direction it faces
    d.polygon([(c, c - 16), (c - 10, c + 10), (c + 10, c + 10)], fill=(200, 60, 40))
    d.text((c + 14, c - 8), f"{ref_label} (facing up)", fill=(120, 30, 20), font=font)
    cx, cy = xy(camera["right"], camera["forward"])
    d.rectangle([cx - 7, cy - 5, cx + 7, cy + 5], outline=(60, 60, 60), width=2)
    d.text((cx + 10, cy - 8), "camera", fill=(60, 60, 60), font=font)
    for i, r in enumerate(rows, 1):
        x, y = xy(r["right"], r["forward"])
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(40, 90, 170))
        d.text((x + 10, y - 8), f"{i}. {r['label']}", fill=(20, 50, 110), font=font)
    return img


class Viewpoint(Tool):
    name = "viewpoint"
    description = ("Answer 'from X's point of view, where is Y?' by moving the camera to X. Give the reference "
                   "(whose point of view counts) with a box and the direction it FACES, and the targets with "
                   "boxes. The tool puts everything on a top-down map (sideways position from the image, distance "
                   "from estimated depth), turns the map so the reference looks up, and reports each target as "
                   "front/rear and left/right OF THE REFERENCE, with the map image. facing: 'toward_camera', "
                   "'away_from_camera', 'image_left', 'image_right', or an angle (0 = away from the camera, 90 = "
                   "toward image-right, 180 = toward the camera, 270 = toward image-left); for a person, get it "
                   "from orient. The facing direction matters most: a person facing the camera has their left on "
                   "the image's right. Depth is rough, so near-axis results are flagged.")
    parameters = {
        "reference": {"type": "object", "description": "the object whose point of view counts",
                      "properties": {"label": {"type": "string"}, "box": BOX_SCHEMA,
                                     "facing": {"description": "direction it faces: a word or an angle"}}},
        "targets": {"type": "array", "minItems": 1, "description": "objects to locate from the reference's view",
                    "items": {"type": "object", "properties": {"label": {"type": "string"}, "box": BOX_SCHEMA}}},
    }
    required = ("reference", "targets")
    example = {"reference": {"label": "woman in red", "box": {"left": 300, "top": 400, "right": 360, "bottom": 800},
                             "facing": "toward_camera"},
               "targets": [{"label": "bus", "box": {"left": 600, "top": 350, "right": 900, "bottom": 650}}]}

    def __init__(self, **options: Any):
        super().__init__(**options)
        self.distance = Distance(**options)  # depth map + back-projection, shared with depth/distance

    def run(self, ctx: ToolContext, reference: Any = None, targets: Any = None) -> ToolResult:
        if not isinstance(reference, dict) or "box" not in reference:
            raise ToolError('reference needs "label", "box" and "facing"')
        if "facing" not in reference:
            raise ToolError("reference needs \"facing\": which way it faces ('toward_camera', 'away_from_camera', "
                            "'image_left', 'image_right' or an angle); use orient for a person")
        if not isinstance(targets, list) or not targets:
            raise ToolError('targets must be a non-empty list of {"label": ..., "box": {...}}')
        facing = facing_angle(reference["facing"])
        dmap = self.distance.depth._depth_map(ctx)
        ref_p, _ = self.distance._point(dmap, reference, "reference")
        ref_xz = (float(ref_p[0]), float(ref_p[2]))
        ref_label = str(reference.get("label", "reference"))
        rows = []
        for i, t in enumerate(targets, 1):
            p, box = self.distance._point(dmap, t, f"target {i}")
            rel = relative_position(ref_xz, facing, (float(p[0]), float(p[2])))
            words, note = direction_words(rel["bearing"])
            rows.append({"label": str(t.get("label", f"target {i}")), "box": [round(v) for v in box],
                         "direction": words, "note": note, **{k: round(v, 3) for k, v in rel.items()}})
        camera = relative_position(ref_xz, facing, (0.0, 0.0))
        lines = []
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['label']}: {r['direction'].upper()} of {ref_label} "
                         f"(bearing {r['bearing']:.0f} degrees: 0 = straight ahead, 90 = right, -90 = left, 180 = "
                         f"behind){'; ' + r['note'] if r['note'] else ''}")
        cam_words, _ = direction_words(camera["bearing"])
        text = (f"From {ref_label}'s point of view (facing {describe_angle(facing)}, angle {facing:.0f}):\n"
                + "\n".join(lines)
                + f"\n(The camera is to the {cam_words} of {ref_label}: a quick check that the facing direction is "
                  "right. If the facing direction is wrong, every answer here is wrong.)")
        return ToolResult(text=text, images=[draw_map(ref_label, rows, camera)],
                          data={"reference": ref_label, "facing": facing, "targets": rows,
                                "camera_bearing": round(camera["bearing"], 1)})
