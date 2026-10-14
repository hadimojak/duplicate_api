from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from app.comparison import (
    create_summary,
    public_record,
)
from app.config import MAX_BATCH_SIZE
from app.models import CompareResponse
from app.uploads import process_uploaded_image
from app.validators import require_folder_id

router = APIRouter()


# =============================================================================
# COMPARE ONE IMAGE
# =============================================================================

@router.post(
    "/v1/image/compare",
    response_model=CompareResponse,
)
async def compare_new_image(
    file: UploadFile = File(...),
    folder_id: str = Query(...),
    claimid: str | None = Query(
        default=None
    ),
    attach_group: str | None = Query(
        default=None
    ),
) -> CompareResponse:

    folder_id = require_folder_id(
        folder_id
    )

    temp_path: Path | None = None

    try:

        (
            query,
            matches,
            temp_path,
        ) = await process_uploaded_image(
            file=file,
            folder_id=folder_id,
            claimid=claimid,
            attach_group=attach_group,
        )

        if query.excluded:

            return CompareResponse(
                ok=True,
                query=public_record(
                    query
                ),
                matches=[],
                summary={
                    "total_matches": 0,
                    "excluded": True,
                    "exclusion_reason": (
                        query.exclusion_reason
                    ),
                },
            )

        return CompareResponse(
            ok=True,
            query=public_record(
                query
            ),
            matches=matches,
            summary=create_summary(
                query,
                matches,
            ),
        )

    finally:

        if temp_path is not None:

            temp_path.unlink(
                missing_ok=True
            )


# =============================================================================
# COMPARE BATCH
# =============================================================================

@router.post(
    "/v1/images/compare"
)
async def compare_batch_images(
    files: list[UploadFile] = File(...),
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

    if not files:

        raise HTTPException(
            status_code=400,
            detail=(
                "حداقل یک فایل "
                "ارسال کنید."
            ),
        )

    if (
        len(files)
        > MAX_BATCH_SIZE
    ):

        raise HTTPException(
            status_code=413,
            detail=(
                f"حداکثر "
                f"{MAX_BATCH_SIZE} "
                "فایل مجاز است."
            ),
        )

    results: list[
        dict[str, Any]
    ] = []

    for file in files:

        filename = (
            file.filename
        )

        try:

            (
                query,
                matches,
                temp_path,
            ) = await process_uploaded_image(
                file=file,
                folder_id=folder_id,
                claimid=claimid,
                attach_group=attach_group,
            )

            try:

                if query.excluded:

                    results.append(
                        {
                            "ok": True,
                            "filename": filename,
                            "query": public_record(
                                query
                            ),
                            "matches": [],
                            "summary": {
                                "total_matches": 0,
                                "excluded": True,
                                "exclusion_reason": (
                                    query.exclusion_reason
                                ),
                            },
                        }
                    )

                else:

                    results.append(
                        {
                            "ok": True,
                            "filename": filename,
                            "query": public_record(
                                query
                            ),
                            "matches": matches,
                            "summary": create_summary(
                                query,
                                matches,
                            ),
                        }
                    )

            finally:

                temp_path.unlink(
                    missing_ok=True
                )

        except HTTPException as exc:

            results.append(
                {
                    "ok": False,
                    "filename": filename,
                    "status_code": (
                        exc.status_code
                    ),
                    "error": exc.detail,
                }
            )

        except Exception as exc:

            results.append(
                {
                    "ok": False,
                    "filename": filename,
                    "error": str(exc),
                }
            )

    return {
        "ok": True,
        "folder_id": folder_id,
        "total_files": len(
            results
        ),
        "successful_files": sum(
            bool(
                item.get("ok")
            )
            for item in results
        ),
        "failed_files": sum(
            not bool(
                item.get("ok")
            )
            for item in results
        ),
        "results": results,
    }
