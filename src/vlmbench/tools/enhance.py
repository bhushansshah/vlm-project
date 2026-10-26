"""Super-resolution of a small region (Swin2SR x4): more legible detail than plain enlargement."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageFilter

from .base import ModelTool, ToolContext, ToolError, ToolResult, fmt_box, region_of
from .image_ops import BOX_SCHEMA, resize_for_view

MAX_INPUT_SIDE = 320  # larger regions already hold enough pixels: zoom is enough (and SR on CPU is slow)


class Enhance(ModelTool):
    name = "enhance"
    description = ("Sharpen a SMALL region with a super-resolution network (4x), for details that stay blurry "
                   "with zoom: small text, a watch, a logo, a far-away person's clothing. Use it on regions up to "
                   f"about {MAX_INPUT_SIDE} pixels wide in the original (zoom says when a region is small); for "
                   "bigger regions use zoom. Slower than zoom. The network can invent texture: trust clear shapes "
                   "and colours, not tiny strokes.")
    parameters = {"box": BOX_SCHEMA}
    required = ("box",)
    example = {"box": {"left": 480, "top": 420, "right": 520, "bottom": 470}}

    def load(self):
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

        model_id = self.options.get("model", "caidas/swin2SR-classical-sr-x4-64")
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(self.device()).eval()
        return proc, model

    def upscale(self, crop: Image.Image) -> Image.Image:
        import numpy as np
        import torch

        proc, model = self.model()
        with self._lock, torch.inference_mode():
            inputs = proc(crop, return_tensors="pt").to(model.device)
            out = model(**inputs).reconstruction[0]
        arr = out.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
        img = Image.fromarray((arr * 255).round().astype(np.uint8))
        scale = getattr(model.config, "upscale", 4)
        return img.crop((0, 0, crop.width * scale, crop.height * scale))  # drop the processor's padding

    def run(self, ctx: ToolContext, box: Any = None) -> ToolResult:
        crop, region = region_of(ctx, box, "box")
        if max(crop.size) > self.options.get("max_input_side", MAX_INPUT_SIDE):
            raise ToolError(f"this region is {crop.width}x{crop.height} px in the original, big enough for zoom; "
                            f"enhance is for regions up to {MAX_INPUT_SIDE} px wide")
        if min(crop.size) < 8:
            raise ToolError(f"this region is only {crop.width}x{crop.height} px: nothing to enhance")
        try:
            sr = self.upscale(crop.convert("RGB"))
            how = "super-resolved 4x"
        except Exception as e:  # noqa: BLE001 - fall back rather than fail the step
            sr = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.LANCZOS).filter(
                ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
            how = f"enlarged 4x and sharpened (super-resolution unavailable: {type(e).__name__})"
        view = resize_for_view(sr, 512 * 512, 1024 * 1024, max_upscale=1.0)
        r = [round(v) for v in region]
        return ToolResult(
            text=(f"Enhanced view of region {fmt_box(r)} ({crop.width}x{crop.height} px in the original, {how}, "
                  f"shown at {view.width}x{view.height}). Fine texture may be invented by the network."),
            images=[view], data={"region": r, "crop_px": list(crop.size), "view_px": list(view.size)})
