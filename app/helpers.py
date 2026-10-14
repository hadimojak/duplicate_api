from __future__ import annotations

import hashlib
from typing import Any


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalize_value(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_digest(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()

    return str(value)


def record_identity(
    item: dict[str, Any],
    index: int,
) -> str:

    explicit_id = (
        item.get("record_id")
        or item.get("id")
        or item.get("image_id")
    )

    if explicit_id:
        return str(explicit_id)

    source = "|".join(
        [
            str(item.get("claimid", "")),
            str(item.get("claim_id", "")),
            str(item.get("folder_id", "")),
            str(item.get("full_path", "")),
            str(item.get("path", "")),
            str(item.get("filename", "")),
            str(index),
        ]
    )

    return hashlib.sha1(
        source.encode("utf-8")
    ).hexdigest()
