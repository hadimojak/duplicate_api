from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.comparison import compare_record
from app.config import (
    ALLOWED_EXTENSIONS,
    CACHE_LOCK,
    UPLOAD_DIR,
)
from app.models import ImageRecord
from app.repository import get_repository


# =============================================================================
# ADD TO REPOSITORY
# =============================================================================

def add_record_to_repository_sync(
    record: ImageRecord,
) -> list[dict[str, Any]]:

    # تمام عملیات hash قبلاً خارج از lock انجام شده است.
    with CACHE_LOCK:

        repository = get_repository()

        candidates = (
            repository.candidates_for(
                record
            )
        )

        matches = compare_record(
            record,
            candidates,
        )

        safe_name = Path(
            record.filename
        ).name

        extension = Path(
            safe_name
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            extension = ".img"

        permanent_path = (
            UPLOAD_DIR
            / (
                f"{record.record_id}"
                f"{extension}"
            )
        )

        shutil.copy2(
            record.full_path,
            permanent_path,
        )

        record.full_path = str(
            permanent_path
        )

        repository.add(
            record
        )

        repository.save()

    return matches


# =============================================================================
# DELETE
# =============================================================================

def delete_repository_record_sync(
    record_id: str,
) -> None:

    with CACHE_LOCK:

        repository = get_repository()

        record = repository.get(
            record_id
        )

        if record is None:

            raise KeyError(
                record_id
            )

        stored_path = Path(
            record.full_path
        )

        repository.remove(
            record_id
        )

        repository.save()

        try:

            stored_path.resolve().relative_to(
                UPLOAD_DIR.resolve()
            )

            stored_path.unlink(
                missing_ok=True
            )

        except ValueError:

            pass
