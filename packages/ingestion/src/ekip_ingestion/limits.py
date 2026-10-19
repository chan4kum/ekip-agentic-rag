"""Hard limits applied to every untrusted document. Tunable, but never absent."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParseLimits:
    max_document_bytes: int = 100 * 1024 * 1024
    max_pages: int = 2_000
    max_blocks: int = 200_000
    max_images: int = 500
    max_image_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 100_000_000  # decompression-bomb guard (header read only, no decode)
    max_zip_uncompressed_bytes: int = 500 * 1024 * 1024  # DOCX is a zip: zip-bomb guard
    min_image_side_px: int = 48  # smaller images are icons/bullets/spacers
    min_image_bytes: int = 1_024
    max_table_cells: int = 20_000


DEFAULT_LIMITS = ParseLimits()
