from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from PIL import (
    Image,
    UnidentifiedImageError,
)

from app.comparison import compare_record
from app.config import (
    ALLOWED_EXTENSIONS,
    CACHE_LOCK,
    MAX_IMAGE_SIZE,
    UPLOAD_DIR,
)
from app.hashing import process_record
from app.models import ImageRecord
from app.repository import get_repository


# =============================================================================
# UPLOAD
# =============================================================================

async def save_upload(
    upload: UploadFile,
) -> tuple[Path, int]:

    original_name = (
        upload.filename
        or ""
    )

    suffix = Path(
        original_name
    ).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".img"

    path = (
        UPLOAD_DIR
        / f"{uuid.uuid4().hex}{suffix}"
    )

    total_size = 0

    try:

        with path.open(
            "wb"
        ) as output:

            while True:

                chunk = await upload.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if (
                    total_size
                    > MAX_IMAGE_SIZE
                ):

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "حجم فایل نباید "
                            "بیشتر از "
                            f"{MAX_IMAGE_SIZE} "
                            "بایت باشد."
                        ),
                    )

                output.write(
                    chunk
                )

        if total_size <= 0:

            raise HTTPException(
                status_code=400,
                detail="فایل خالی است.",
            )

        try:

            with Image.open(
                path
            ) as image:

                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
        ) as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    "فایل ارسال‌شده "
                    "تصویر معتبر نیست."
                ),
            ) from exc

        return (
            path,
            total_size,
        )

    except BaseException:

        path.unlink(
            missing_ok=True
        )

        raise


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

async def process_uploaded_image(
    file: UploadFile,
    folder_id: str,
    claimid: str | None,
    attach_group: str | None,
) -> tuple[
    ImageRecord,
    list[dict[str, Any]],
    Path,
]:

    temp_path, size = (
        await save_upload(
            file
        )
    )

    query = ImageRecord(
        record_id=(
            "query-"
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

    query = await run_in_threadpool(
        process_record,
        query,
    )

    if query.excluded:

        return (
            query,
            [],
            temp_path,
        )

    # Lock فقط برای گرفتن snapshot بسیار کوتاه است.
    # هیچ پردازش سنگینی داخل Lock انجام نمی‌شود.
    with CACHE_LOCK:

        repository = get_repository()

        candidates = (
            repository.candidates_for(
                query
            )
        )

    matches = await run_in_threadpool(
        compare_record,
        query,
        candidates,
    )

    return (
        query,
        matches,
        temp_path,
    )
