"""Order-independent deterministic random draws for dataset generators."""

from __future__ import annotations

import hashlib


def unit_interval_draw(*parts: object) -> float:
    """Return a deterministic uniform draw in the closed unit interval."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float((1 << 64) - 1)
