"""Content-based file identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_content_identity(path: str | Path) -> tuple[int, str]:
    """Return the byte count and BLAKE2b digest of one file read."""

    digest = hashlib.blake2b(digest_size=32)
    size_bytes = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size_bytes += len(block)
            digest.update(block)
    return size_bytes, digest.hexdigest()


__all__ = ["file_content_identity"]
