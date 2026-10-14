from __future__ import annotations

from pathlib import Path
from typing import Any

from app.helpers import (
    normalize_value,
    record_identity,
)
from app.models import ImageRecord


# =============================================================================
# DATASET PARSER
# =============================================================================

def extract_items(
    payload: Any,
) -> list[dict[str, Any]]:

    if isinstance(payload, list):

        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if not isinstance(payload, dict):

        raise ValueError(
            "ساختار JSON باید آرایه یا شیء JSON باشد."
        )

    for key in (
        "records",
        "data",
        "items",
    ):

        value = payload.get(key)

        if isinstance(value, list):

            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    claims = payload.get("claims")

    if isinstance(claims, list):

        return [
            item
            for item in claims
            if isinstance(item, dict)
        ]

    flattened: list[
        dict[str, Any]
    ] = []

    for key, value in payload.items():

        if isinstance(value, dict):

            item = dict(value)

            if not any(
                item.get(k)
                for k in (
                    "claimid",
                    "claim_id",
                    "claimId",
                )
            ):
                item["claimid"] = key

            flattened.append(item)

        elif isinstance(value, list):

            for child in value:

                if not isinstance(
                    child,
                    dict,
                ):
                    continue

                item = dict(child)

                if not any(
                    item.get(k)
                    for k in (
                        "claimid",
                        "claim_id",
                        "claimId",
                    )
                ):
                    item["claimid"] = key

                flattened.append(item)

    return flattened


def get_full_path(
    item: dict[str, Any],
) -> str | None:

    for key in (
        "full_path",
        "path",
        "file_path",
        "image_path",
        "filepath",
    ):

        value = normalize_value(
            item.get(key)
        )

        if value:
            return value

    return None


def make_record(
    item: dict[str, Any],
    index: int,
) -> ImageRecord | None:

    full_path = get_full_path(item)

    if not full_path:
        return None

    filename = normalize_value(
        item.get("filename")
        or item.get("file_name")
        or Path(full_path).name
    )

    if not filename:
        filename = Path(full_path).name

    folder_id = normalize_value(
        item.get("folder_id")
        or item.get("folder")
        or item.get("case_id")
        or item.get("claimid")
        or item.get("claim_id")
        or item.get("claimId")
    )

    if not folder_id:
        folder_id = "UNKNOWN"

    claimid = normalize_value(
        item.get("claimid")
        or item.get("claim_id")
        or item.get("claimId")
    )

    attach_group = normalize_value(
        item.get("attach_group")
        or item.get("attachment_group")
        or item.get("attachGroup")
    )

    return ImageRecord(
        record_id=record_identity(
            item,
            index,
        ),
        folder_id=folder_id,
        claimid=claimid,
        full_path=full_path,
        filename=filename,
        attach_group=attach_group,
    )
