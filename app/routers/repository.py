from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool

from app.builder import rebuild_repository_sync
from app.comparison import (
    create_summary,
    public_record,
)
from app.errors import raise_http_error
from app.hashing import process_record
from app.models import ImageRecord
from app.repository_service import (
    add_record_to_repository_sync,
    delete_repository_record_sync,
)
from app.uploads import save_upload
from app.validators import require_folder_id

router = APIRouter()


# =============================================================================
# ADD TO REPOSITORY
# =============================================================================

@router.post(
    "/v1/repository/add"
)
async def add_image_to_repository(
    file: UploadFile = File(...),
    folder_id: str = Query(...),
    claimid: str | None = Query(
        default=None
    ),
    attach_group: str | None = Query(
        default=None
    ),
) -> dict[str, Any]:

    folder_id = require_folder_id(
        folder_id
    )

    temp_path: Path | None = None

    try:

        temp_path, size = (
            await save_upload(
                file
            )
        )

        record = ImageRecord(
            record_id=(
                "record-"
                f"{uuid.uuid4().hex}"
            ),
            folder_id=folder_id,
            claimid=claimid,
            full_path=str(
                temp_path
            ),
            filename=(
                file.filename
                or temp_path.name
            ),
            attach_group=attach_group,
            file_size=size,
        )

        record = await run_in_threadpool(
            process_record,
            record,
        )

        if record.excluded:

            return {
                "ok": True,
                "record": public_record(
                    record
                ),
                "matches_before_insert": [],
                "summary": {
                    "total_matches": 0,
                    "excluded": True,
                    "exclusion_reason": (
                        record.exclusion_reason
                    ),
                },
            }

        matches = await run_in_threadpool(
            add_record_to_repository_sync,
            record,
        )

        return {
            "ok": True,
            "record": public_record(
                record
            ),
            "matches_before_insert": matches,
            "summary": create_summary(
                record,
                matches,
            ),
        }

    finally:

        if temp_path is not None:

            temp_path.unlink(
                missing_ok=True
            )


# =============================================================================
# DELETE
# =============================================================================

@router.delete(
    "/v1/repository/{record_id}"
)
async def delete_repository_record(
    record_id: str,
) -> dict[str, Any]:

    try:

        await run_in_threadpool(
            delete_repository_record_sync,
            record_id,
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail="رکورد یافت نشد.",
        ) from exc

    return {
        "ok": True,
        "deleted_record_id": record_id,
    }


# =============================================================================
# REBUILD
# =============================================================================

@router.post(
    "/v1/repository/rebuild"
)
async def rebuild_repository() -> dict[str, Any]:

    try:

        (
            loaded,
            skipped,
            total,
        ) = await run_in_threadpool(
            rebuild_repository_sync
        )

        return {
            "ok": True,
            "loaded": loaded,
            "skipped": skipped,
            "total": total,
        }

    except Exception as exc:

        raise_http_error(
            exc,
            "خطا در بازسازی Repository",
        )

        raise
