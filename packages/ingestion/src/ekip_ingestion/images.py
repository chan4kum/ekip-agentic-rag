"""Safe image inspection: read metadata from the header without decoding pixels."""

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from ekip_ingestion.limits import ParseLimits

_ALLOWED = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}


@dataclass(frozen=True)
class ImageInfo:
    mime_type: str
    width: int
    height: int


def inspect_image(data: bytes, limits: ParseLimits) -> ImageInfo | None:
    """Return metadata for an allowed image type within limits, else None.

    SVG is deliberately NOT allowed (it is XML that can carry scripts and external references).
    Opening reads only the header; `size` never decodes the pixels, so a decompression bomb is
    rejected by its declared dimensions before any memory is spent on it.
    """
    if not data or len(data) > limits.max_image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            mime = _ALLOWED.get(img.format or "")
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None
    if mime is None or width < 1 or height < 1 or width * height > limits.max_image_pixels:
        return None
    return ImageInfo(mime, width, height)


def is_meaningful(info: ImageInfo, size_bytes: int, limits: ParseLimits) -> bool:
    """Cheap, LLM-free filter: drop icons, bullets and spacers before any vision call."""
    return (
        min(info.width, info.height) >= limits.min_image_side_px
        and size_bytes >= limits.min_image_bytes
    )


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
