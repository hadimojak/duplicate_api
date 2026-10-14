from __future__ import annotations

import json

from fastapi import HTTPException


def raise_http_error(
    exc: Exception,
    context: str,
) -> None:

    if isinstance(
        exc,
        FileNotFoundError,
    ):

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    if isinstance(
        exc,
        json.JSONDecodeError,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "فایل Dataset "
                "دارای JSON معتبر نیست."
            ),
        ) from exc

    raise HTTPException(
        status_code=500,
        detail=(
            f"{context}: "
            f"{exc}"
        ),
    ) from exc
