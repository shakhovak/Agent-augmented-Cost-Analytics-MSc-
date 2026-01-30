"""File hash and dataset version ID utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 hash of a file. Used as dataset version for CSV snapshots.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """
    Useful for hashing config content or query specs if needed.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
