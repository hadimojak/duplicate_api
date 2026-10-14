from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class ImageRecord:
    record_id: str
    folder_id: str
    claimid: str | None
    full_path: str
    filename: str
    attach_group: str | None = None

    file_size: int | None = None
    file_sha256: str | None = None
    pixel_sha256: str | None = None
    image_size: str | None = None

    is_blank: bool = False
    is_xray: bool = False
    excluded: bool = False
    exclusion_reason: str | None = None

    risk_level: str | None = None
    duplicate_type: str | None = None
    duplicate_with: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# API MODELS
# =============================================================================

class DatasetLoadResponse(BaseModel):
    loaded: int
    skipped: int
    total_repository_records: int


class CompareResponse(BaseModel):
    ok: bool
    query: dict[str, Any]
    matches: list[dict[str, Any]]
    summary: dict[str, Any]
