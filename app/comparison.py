from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from app.models import ImageRecord


# =============================================================================
# PUBLIC RECORD
# =============================================================================

def public_record(
    record: ImageRecord,
) -> dict[str, Any]:

    return {
        "record_id": record.record_id,
        "folder_id": record.folder_id,
        "claimid": record.claimid,
        "filename": record.filename,
        "full_path": record.full_path,
        "attach_group": record.attach_group,
        "file_size": record.file_size,
        "image_size": record.image_size,
        "file_sha256": record.file_sha256,
        "pixel_sha256": record.pixel_sha256,
        "is_blank": record.is_blank,
        "is_xray": record.is_xray,
        "excluded": record.excluded,
        "exclusion_reason": record.exclusion_reason,
    }


# =============================================================================
# COMPARISON
# =============================================================================

def compare_record(
    query: ImageRecord,
    candidates: Iterable[ImageRecord],
) -> list[dict[str, Any]]:

    matches: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for candidate in candidates:

        if candidate.record_id == query.record_id:
            continue

        # فقط بین folder_idهای متفاوت
        if candidate.folder_id == query.folder_id:
            continue

        if candidate.excluded:
            continue

        duplicate_type: str | None = None
        risk_level: str | None = None

        if (
            query.file_sha256
            and candidate.file_sha256
            == query.file_sha256
        ):

            duplicate_type = (
                "exact_file"
            )

            risk_level = (
                "VERY_HIGH"
            )

        elif (
            query.pixel_sha256
            and candidate.pixel_sha256
            == query.pixel_sha256
        ):

            duplicate_type = (
                "exact_pixel"
            )

            risk_level = "HIGH"

        if (
            duplicate_type is None
            or risk_level is None
        ):
            continue

        identity = (
            candidate.record_id,
            duplicate_type,
        )

        if identity in seen:
            continue

        seen.add(identity)

        matches.append(
            {
                "risk_level": risk_level,
                "duplicate_type": duplicate_type,
                "query": public_record(
                    query
                ),
                "matched_record": public_record(
                    candidate
                ),
            }
        )

    matches.sort(
        key=lambda item: (
            0
            if item["risk_level"]
            == "VERY_HIGH"
            else 1,
            str(
                item["matched_record"][
                    "folder_id"
                ]
            ),
            str(
                item["matched_record"][
                    "filename"
                ]
            ),
        )
    )

    return matches


def create_summary(
    query: ImageRecord,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:

    by_risk: dict[
        str,
        int,
    ] = defaultdict(int)

    by_type: dict[
        str,
        int,
    ] = defaultdict(int)

    folders: set[str] = set()

    for match in matches:

        by_risk[
            match["risk_level"]
        ] += 1

        by_type[
            match["duplicate_type"]
        ] += 1

        folders.add(
            str(
                match["matched_record"][
                    "folder_id"
                ]
            )
        )

    return {
        "query_folder_id": query.folder_id,
        "query_claimid": query.claimid,
        "total_matches": len(matches),
        "very_high_count": by_risk.get(
            "VERY_HIGH",
            0,
        ),
        "high_count": by_risk.get(
            "HIGH",
            0,
        ),
        "exact_file_count": by_type.get(
            "exact_file",
            0,
        ),
        "exact_pixel_count": by_type.get(
            "exact_pixel",
            0,
        ),
        "matched_folder_count": len(
            folders
        ),
        "matched_folders": sorted(
            folders
        ),
    }
