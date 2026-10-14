from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.config import (
    CACHE_LOCK,
    ensure_directories,
)
from app.repository import get_repository
from app.routers import (
    compare,
    dataset,
    repository,
    system,
)


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(
    _: FastAPI,
):

    ensure_directories()

    try:

        # Load cache در startup سنگین است،
        # بنابراین خارج از event loop اجرا می‌شود.
        loaded = await run_in_threadpool(
            get_repository().load
        )

        if not loaded:

            # Cache خراب/قدیمی فقط باعث خالی‌ماندن Repository می‌شود.
            # Dataset به‌صورت خودکار در startup rebuild نمی‌شود.
            with CACHE_LOCK:
                get_repository().clear()

    except Exception:

        with CACHE_LOCK:
            get_repository().clear()

    yield


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title=(
        "Claim Image Duplicate "
        "Detection API"
    ),
    version="4.0.0",
    description=(
        "تشخیص تصاویر تکراری بر اساس "
        "File SHA256 و Pixel SHA256 "
        "با حذف Blank و X-Ray و "
        "مقایسه فقط بین folder_idهای متفاوت."
    ),
    lifespan=lifespan,
)

app.include_router(
    system.router
)

app.include_router(
    dataset.router
)

app.include_router(
    compare.router
)

app.include_router(
    repository.router
)
