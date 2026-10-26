"""Pure image tools (no neural network): zoom into a region, overlay a coordinate grid."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .base import SCALE, Tool, ToolContext, ToolResult, fmt_box, region_of

# Below this many pixels on its shorter side a crop holds little detail, however much it is enlarged.
SMALL_CROP_PX = 128

BOX_SCHEMA = {"type": "object",
              "properties": {k: {"type": "number"} for k in ("left", "top", "right", "bottom")},
              "required": ["left", "top", "right", "bottom"],
              "description": "0-1000 scale of the original image: left/right horizontal, top/bottom vertical"}


def resize_for_view(img: Image.Image, min_pixels: int, max_pixels: int, max_side: int = 2048,
                    max_upscale: float = 8.0) -> Image.Image:
    """Bring the image's AREA into [min_pixels, max_pixels]: small crops are enlarged so the
    model's vision encoder gives them more detail, big ones are shrunk to save tokens. Area (not
    the longer side) keeps thin strips readable: a 2000x200 crop stays 2000x200."""
    area = img.width * img.height
    s = 1.0
    if area < min_pixels:
        s = min((min_pixels / area) ** 0.5, max_upscale)
    elif area > max_pixels:
        s = (max_pixels / area) ** 0.5
    s = min(s, max_side / max(img.size))
    if abs(s - 1.0) < 0.01:
        return img
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.LANCZOS)


def draw_grid(img: Image.Image, region: tuple[float, float, float, float], step: int | None = None) -> Image.Image:
    """Grid lines labeled in ORIGINAL-image 0-1000 coordinates over a (possibly cropped) view."""
    img = img.convert("RGB").copy()
    rx1, ry1, rx2, ry2 = region
    span = max(rx2 - rx1, ry2 - ry1)
    if step is None:  # about 5-10 lines per side, on round numbers
        step = next(s for s in (5, 10, 20, 25, 50, 100, 200, 250) if span / s <= 10)
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(10, min(w, h) // 40))

    def label(xy, text):
        d.text(xy, text, fill=(255, 255, 0, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))

    first_x = (int(rx1) // step + 1) * step
    for gx in range(first_x, int(rx2), step):
        px = (gx - rx1) / (rx2 - rx1) * w
        d.line([(px, 0), (px, h)], fill=(255, 255, 0, 110), width=1)
        label((px + 2, 2), str(gx))
    first_y = (int(ry1) // step + 1) * step
    for gy in range(first_y, int(ry2), step):
        py = (gy - ry1) / (ry2 - ry1) * h
        d.line([(0, py), (w, py)], fill=(255, 255, 0, 110), width=1)
        label((2, py + 2), str(gy))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


class Zoom(Tool):
    name = "zoom"
    description = ("Crop a region of the original image and show it enlarged, to see small or distant details "
                   "(small objects, text, fine attributes). Optionally overlay a grid labeled in original "
                   "coordinates to plan the next zoom.")
    parameters = {"box": BOX_SCHEMA,
                  "grid": {"type": "boolean", "description": "overlay coordinate grid lines (default false)"}}
    required = ("box",)
    example = {"box": {"left": 400, "top": 300, "right": 600, "bottom": 500}}

    def run(self, ctx: ToolContext, box: Any = None, grid: bool = False) -> ToolResult:
        crop, region = region_of(ctx, box, "box")
        # enlarging past a few times only makes blur bigger, and blurry crops invite guessing
        view = resize_for_view(crop, self.options.get("min_pixels", 512 * 512),
                               self.options.get("max_pixels", 1024 * 1024),
                               max_upscale=self.options.get("max_upscale", 3.0))
        if grid:
            view = draw_grid(view, region)
        r = [round(v) for v in region]
        factor = view.width / crop.width
        text = (f"Zoomed view of region {fmt_box(r)} ({crop.width}x{crop.height} px in the original, shown at "
                f"{view.width}x{view.height}, {factor:.1f}x).")
        if min(crop.size) < SMALL_CROP_PX:
            text += (f" Note: this region is only {crop.width}x{crop.height} pixels in the original, so it holds "
                     "little detail; zooming in further cannot add any. Do not trust details you cannot see "
                     "clearly here; decide from what is clearly visible.")
        return ToolResult(
            text=text,
            images=[view], data={"region": r, "crop_px": list(crop.size), "view_px": list(view.size)})


class Grid(Tool):
    name = "grid"
    description = ("Show the image (or a region of it) with grid lines labeled in 0-1000 coordinates. Use it to "
                   "find the coordinates of objects before zooming or detecting, or to count region by region.")
    parameters = {"region": {**BOX_SCHEMA, "description": "optional region; default is the whole image"},
                  "step": {"type": "integer", "description": "grid spacing in 0-1000 units (default: automatic)"}}
    example: dict[str, Any] = {}

    def run(self, ctx: ToolContext, region: Any = None, step: int | None = None) -> ToolResult:
        crop, reg = region_of(ctx, region)
        if step is not None and not 5 <= int(step) <= SCALE // 2:
            step = None
        view = draw_grid(resize_for_view(crop, 768 * 768, self.options.get("max_pixels", 1280 * 1280)), reg,
                         int(step) if step else None)
        return ToolResult(text=f"Grid overlay on region {fmt_box(reg)}; vertical lines are labeled with "
                               "left/right values and horizontal lines with top/bottom values (0-1000).", images=[view], data={"region": [round(v) for v in reg]})
