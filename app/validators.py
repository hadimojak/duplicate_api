from __future__ import annotations

from fastapi import HTTPException


def require_folder_id(
    folder_id: str | None,
) -> str:

    value = (
        folder_id or ""
    ).strip()

    if not value:

        raise HTTPException(
            status_code=422,
            detail=(
                "folder_id الزامی است."
            ),
        )

    return value
