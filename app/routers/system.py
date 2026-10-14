from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.config import (
    CACHE_LOCK,
    CACHE_PATH,
    DATASET_JSON,
    UPLOAD_DIR,
    USE_GPU,
)
from app.hashing import (
    cpu_function_available,
    gpu_function_available,
)
from app.repository import get_repository

router = APIRouter()


# =============================================================================
# ROOT
# =============================================================================

@router.get("/")
def root() -> dict[str, Any]:

    with CACHE_LOCK:

        repository_count = len(
            get_repository().records
        )

    return {
        "service": (
            "Claim Image Duplicate "
            "Detection API"
        ),
        "version": "4.0.0",
        "ok": True,
        "repository_records": repository_count,
    }


# =============================================================================
# HEALTH
# =============================================================================

@router.get("/health")
def health() -> dict[str, Any]:

    with CACHE_LOCK:

        repository = get_repository()

        repository_records = len(
            repository.records
        )

        active_records = sum(
            not record.excluded
            for record
            in repository.records.values()
        )

    return {
        "ok": True,
        "repository_records": repository_records,
        "active_records": active_records,
        "dataset_json": str(
            DATASET_JSON
        ),
        "cache_path": str(
            CACHE_PATH
        ),
        "upload_dir": str(
            UPLOAD_DIR
        ),
        "use_gpu": USE_GPU,
        "gpu_function_available": (
            gpu_function_available()
        ),
        "cpu_function_available": (
            cpu_function_available()
        ),
    }
