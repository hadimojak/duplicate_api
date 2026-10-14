from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.builder import rebuild_repository_sync
from app.config import CACHE_LOCK
from app.errors import raise_http_error
from app.models import DatasetLoadResponse
from app.repository import get_repository

router = APIRouter()


# =============================================================================
# DATASET LOAD
# =============================================================================

@router.post(
    "/v1/dataset/load",
    response_model=DatasetLoadResponse,
)
async def load_dataset() -> DatasetLoadResponse:

    try:

        (
            loaded,
            skipped,
            total,
        ) = await run_in_threadpool(
            rebuild_repository_sync
        )

        return DatasetLoadResponse(
            loaded=loaded,
            skipped=skipped,
            total_repository_records=total,
        )

    except Exception as exc:

        raise_http_error(
            exc,
            "خطا در بارگذاری Dataset",
        )

        raise


# =============================================================================
# DATASET STATS
# =============================================================================

@router.get(
    "/v1/dataset/stats"
)
def dataset_stats() -> dict[str, Any]:

    with CACHE_LOCK:

        repository = get_repository()

        records = list(
            repository.records.values()
        )

        file_hash_groups = len(
            repository.by_file_hash
        )

        pixel_hash_groups = len(
            repository.by_pixel_hash
        )

    excluded = [
        record
        for record in records
        if record.excluded
    ]

    return {
        "total": len(records),
        "active": (
            len(records)
            - len(excluded)
        ),
        "excluded": len(excluded),
        "blank": sum(
            record.is_blank
            for record in records
        ),
        "xray": sum(
            record.is_xray
            for record in records
        ),
        "processing_errors": sum(
            (
                record.exclusion_reason
                or ""
            ).startswith(
                "PROCESSING_ERROR"
            )
            for record in records
        ),
        "file_hash_groups": (
            file_hash_groups
        ),
        "pixel_hash_groups": (
            pixel_hash_groups
        ),
    }
