"""Image preprocessing and encoding for API backends.

VisualOverload images are ~4K (~8.3 MP, 1.5-4 MB JPEGs). Resolution is the
variable under study, so we never resize silently: by default the original bytes are
sent as-is, and an image is only downscaled / re-encoded when it violates an explicit
``ImagePolicy`` (user-set ``max_side`` / ``max_pixels``, or a hard provider byte limit).
Every transformation is logged once per (image, policy) via ``EncodedImage.transform``.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .types import ImagePart

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePolicy:
    max_side: int | None = None  # longest edge in px
    max_pixels: int | None = None  # width * height
    max_b64_bytes: int | None = None  # hard limit on the base64 payload of one image
    jpeg_quality: int = 95


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    mime_type: str
    size: tuple[int, int]
    transform: str  # "original" or a description of what was changed

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.b64}"


_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}


def _b64_len(n_bytes: int) -> int:
    return 4 * math.ceil(n_bytes / 3)


def _target_size(w: int, h: int, policy: ImagePolicy) -> tuple[int, int]:
    scale = 1.0
    if policy.max_side and max(w, h) > policy.max_side:
        scale = min(scale, policy.max_side / max(w, h))
    if policy.max_pixels and w * h > policy.max_pixels:
        scale = min(scale, math.sqrt(policy.max_pixels / (w * h)))
    return max(1, int(w * scale)), max(1, int(h * scale))


def _to_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_image(part: ImagePart, policy: ImagePolicy = ImagePolicy()) -> EncodedImage:
    if isinstance(part.image, (str, Path)):
        return _encode_path(str(part.image), policy)
    return _encode_pil(part.image, policy)


@lru_cache(maxsize=32)
def _encode_path(path: str, policy: ImagePolicy) -> EncodedImage:
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        fmt, size = img.format, img.size
        target = _target_size(*size, policy)
        fits = policy.max_b64_bytes is None or _b64_len(len(raw)) <= policy.max_b64_bytes
        if target == size and fits and fmt in _MIME:
            return EncodedImage(raw, _MIME[fmt], size, "original")
        img.load()
        enc = _encode_pil(img, policy)
    log.info("Image %s: %s", Path(path).name, enc.transform)
    return enc


def _encode_pil(img: Image.Image, policy: ImagePolicy) -> EncodedImage:
    orig = img.size
    w, h = _target_size(*orig, policy)
    quality = policy.jpeg_quality
    while True:
        resized = img if (w, h) == img.size else img.resize((w, h), Image.Resampling.LANCZOS)
        data = _to_jpeg(resized, quality)
        if policy.max_b64_bytes is None or _b64_len(len(data)) <= policy.max_b64_bytes:
            break
        # Lower quality first (cheap, keeps resolution), then shrink.
        if quality > 85:
            quality = 85
        else:
            w, h = int(w * 0.9), int(h * 0.9)
    if (w, h) == orig:
        transform = f"re-encoded as JPEG q={quality}"
    else:
        transform = f"{orig[0]}x{orig[1]} -> {w}x{h}, JPEG q={quality}"
    return EncodedImage(data, "image/jpeg", (w, h), transform)
