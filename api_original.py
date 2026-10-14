from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

import all_data


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(
    os.getenv("BASE_DIR", ".")
).resolve()

DATASET_JSON = Path(
    os.getenv(
        "DATASET_JSON",
        str(BASE_DIR / "all_claims.json"),
    )
).resolve()

CACHE_PATH = Path(
    os.getenv(
        "CACHE_PATH",
        str(BASE_DIR / "api_cache.json"),
    )
).resolve()

UPLOAD_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        str(BASE_DIR / "api_uploads"),
    )
).resolve()

MAX_IMAGE_SIZE = int(
    os.getenv(
        "MAX_IMAGE_SIZE",
        str(50 * 1024 * 1024),
    )
)

MAX_BATCH_SIZE = int(
    os.getenv(
        "MAX_BATCH_SIZE",
        "500",
    )
)

USE_GPU = os.getenv(
    "USE_GPU",
    "true",
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}

CACHE_VERSION = "api-v4"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# مهم:
# این Lock فقط در بخش‌های کوتاه و synchronous مربوط به Repository گرفته می‌شود.
# هیچ await سنگینی داخل این Lock انجام نمی‌شود.
CACHE_LOCK = threading.RLock()

# برای جلوگیری از اجرای هم‌زمان pixel hashing روی GPU
# در چند request، در صورت استفاده از GPU.
GPU_PROCESS_LOCK = threading.Lock()


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


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_digest(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()

    return str(value)


def record_identity(
    item: dict[str, Any],
    index: int,
) -> str:

    explicit_id = (
        item.get("record_id")
        or item.get("id")
        or item.get("image_id")
    )

    if explicit_id:
        return str(explicit_id)

    source = "|".join(
        [
            str(item.get("claimid", "")),
            str(item.get("claim_id", "")),
            str(item.get("folder_id", "")),
            str(item.get("full_path", "")),
            str(item.get("path", "")),
            str(item.get("filename", "")),
            str(index),
        ]
    )

    return hashlib.sha1(
        source.encode("utf-8")
    ).hexdigest()


# =============================================================================
# DATASET PARSER
# =============================================================================

def extract_items(
    payload: Any,
) -> list[dict[str, Any]]:

    if isinstance(payload, list):

        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if not isinstance(payload, dict):

        raise ValueError(
            "ساختار JSON باید آرایه یا شیء JSON باشد."
        )

    for key in (
        "records",
        "data",
        "items",
    ):

        value = payload.get(key)

        if isinstance(value, list):

            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    claims = payload.get("claims")

    if isinstance(claims, list):

        return [
            item
            for item in claims
            if isinstance(item, dict)
        ]

    flattened: list[
        dict[str, Any]
    ] = []

    for key, value in payload.items():

        if isinstance(value, dict):

            item = dict(value)

            if not any(
                item.get(k)
                for k in (
                    "claimid",
                    "claim_id",
                    "claimId",
                )
            ):
                item["claimid"] = key

            flattened.append(item)

        elif isinstance(value, list):

            for child in value:

                if not isinstance(
                    child,
                    dict,
                ):
                    continue

                item = dict(child)

                if not any(
                    item.get(k)
                    for k in (
                        "claimid",
                        "claim_id",
                        "claimId",
                    )
                ):
                    item["claimid"] = key

                flattened.append(item)

    return flattened


def get_full_path(
    item: dict[str, Any],
) -> str | None:

    for key in (
        "full_path",
        "path",
        "file_path",
        "image_path",
        "filepath",
    ):

        value = normalize_value(
            item.get(key)
        )

        if value:
            return value

    return None


def make_record(
    item: dict[str, Any],
    index: int,
) -> ImageRecord | None:

    full_path = get_full_path(item)

    if not full_path:
        return None

    filename = normalize_value(
        item.get("filename")
        or item.get("file_name")
        or Path(full_path).name
    )

    if not filename:
        filename = Path(full_path).name

    folder_id = normalize_value(
        item.get("folder_id")
        or item.get("folder")
        or item.get("case_id")
        or item.get("claimid")
        or item.get("claim_id")
        or item.get("claimId")
    )

    if not folder_id:
        folder_id = "UNKNOWN"

    claimid = normalize_value(
        item.get("claimid")
        or item.get("claim_id")
        or item.get("claimId")
    )

    attach_group = normalize_value(
        item.get("attach_group")
        or item.get("attachment_group")
        or item.get("attachGroup")
    )

    return ImageRecord(
        record_id=record_identity(
            item,
            index,
        ),
        folder_id=folder_id,
        claimid=claimid,
        full_path=full_path,
        filename=filename,
        attach_group=attach_group,
    )


# =============================================================================
# HASH REPOSITORY
# =============================================================================

class HashRepository:

    def __init__(self) -> None:

        self.records: dict[
            str,
            ImageRecord,
        ] = {}

        self.by_file_hash: dict[
            str,
            list[str],
        ] = defaultdict(list)

        self.by_pixel_hash: dict[
            str,
            list[str],
        ] = defaultdict(list)

    def clear(self) -> None:

        self.records.clear()
        self.by_file_hash.clear()
        self.by_pixel_hash.clear()

    def _append_unique(
        self,
        mapping: dict[str, list[str]],
        key: str,
        record_id: str,
    ) -> None:

        values = mapping[key]

        if record_id not in values:
            values.append(record_id)

    def add(
        self,
        record: ImageRecord,
    ) -> None:

        old = self.records.get(
            record.record_id
        )

        if old is not None:
            self.remove(
                old.record_id
            )

        self.records[
            record.record_id
        ] = record

        if record.excluded:
            return

        if record.file_sha256:

            self._append_unique(
                self.by_file_hash,
                record.file_sha256,
                record.record_id,
            )

        if record.pixel_sha256:

            self._append_unique(
                self.by_pixel_hash,
                record.pixel_sha256,
                record.record_id,
            )

    def remove(
        self,
        record_id: str,
    ) -> None:

        record = self.records.pop(
            record_id,
            None,
        )

        if record is None:
            return

        if record.file_sha256:

            values = self.by_file_hash.get(
                record.file_sha256
            )

            if values is not None:

                if record_id in values:
                    values.remove(record_id)

                if not values:

                    del self.by_file_hash[
                        record.file_sha256
                    ]

        if record.pixel_sha256:

            values = self.by_pixel_hash.get(
                record.pixel_sha256
            )

            if values is not None:

                if record_id in values:
                    values.remove(record_id)

                if not values:

                    del self.by_pixel_hash[
                        record.pixel_sha256
                    ]

    def get(
        self,
        record_id: str,
    ) -> ImageRecord | None:

        return self.records.get(
            record_id
        )

    def all(self) -> list[ImageRecord]:

        return list(
            self.records.values()
        )

    def candidates_for(
        self,
        query: ImageRecord,
    ) -> list[ImageRecord]:

        ids: set[str] = set()

        if query.file_sha256:

            ids.update(
                self.by_file_hash.get(
                    query.file_sha256,
                    [],
                )
            )

        if query.pixel_sha256:

            ids.update(
                self.by_pixel_hash.get(
                    query.pixel_sha256,
                    [],
                )
            )

        return [
            self.records[record_id]
            for record_id in ids
            if record_id in self.records
        ]

    def save(self) -> None:

        payload = {
            "cache_version": CACHE_VERSION,
            "records": [
                record.to_dict()
                for record in self.records.values()
            ],
        }

        temporary = CACHE_PATH.with_name(
            CACHE_PATH.name
            + ".tmp."
            + uuid.uuid4().hex
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

        temporary.replace(
            CACHE_PATH
        )

    def load(self) -> bool:

        if not CACHE_PATH.exists():
            return False

        with CACHE_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(file)

        if not isinstance(
            payload,
            dict,
        ):
            return False

        if payload.get(
            "cache_version"
        ) != CACHE_VERSION:

            return False

        records = payload.get(
            "records",
            [],
        )

        if not isinstance(
            records,
            list,
        ):
            return False

        self.clear()

        for item in records:

            if not isinstance(
                item,
                dict,
            ):
                continue

            try:

                record = ImageRecord(
                    record_id=str(
                        item["record_id"]
                    ),
                    folder_id=str(
                        item["folder_id"]
                    ),
                    claimid=item.get(
                        "claimid"
                    ),
                    full_path=str(
                        item["full_path"]
                    ),
                    filename=str(
                        item["filename"]
                    ),
                    attach_group=item.get(
                        "attach_group"
                    ),
                    file_size=item.get(
                        "file_size"
                    ),
                    file_sha256=item.get(
                        "file_sha256"
                    ),
                    pixel_sha256=item.get(
                        "pixel_sha256"
                    ),
                    image_size=item.get(
                        "image_size"
                    ),
                    is_blank=bool(
                        item.get(
                            "is_blank",
                            False,
                        )
                    ),
                    is_xray=bool(
                        item.get(
                            "is_xray",
                            False,
                        )
                    ),
                    excluded=bool(
                        item.get(
                            "excluded",
                            False,
                        )
                    ),
                    exclusion_reason=item.get(
                        "exclusion_reason"
                    ),
                    risk_level=item.get(
                        "risk_level"
                    ),
                    duplicate_type=item.get(
                        "duplicate_type"
                    ),
                    duplicate_with=item.get(
                        "duplicate_with"
                    ),
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):

                continue

            self.add(
                record
            )

        return True


REPOSITORY = HashRepository()


# =============================================================================
# ALL_DATA INTEGRATION
# =============================================================================

def calculate_pixel_hash(
    path: Path,
) -> tuple[str, str]:

    gpu_function = getattr(
        all_data,
        "_pixel_sha256_gpu",
        None,
    )

    if (
        USE_GPU
        and callable(gpu_function)
    ):

        device = getattr(
            all_data,
            "DEVICE",
            None,
        )

        if device is not None:

            try:

                with GPU_PROCESS_LOCK:

                    result = gpu_function(
                        path,
                        device,
                    )

                if (
                    isinstance(
                        result,
                        tuple,
                    )
                    and len(result) == 2
                ):

                    digest, image_size = result

                    return (
                        normalize_digest(
                            digest
                        ),
                        str(
                            image_size
                        ),
                    )

            except Exception:
                pass

    cpu_function = getattr(
        all_data,
        "_pixel_sha256_cpu",
        None,
    )

    if not callable(
        cpu_function
    ):

        raise RuntimeError(
            "_pixel_sha256_cpu "
            "در all_data.py یافت نشد."
        )

    result = cpu_function(
        path
    )

    if (
        not isinstance(
            result,
            tuple,
        )
        or len(result) != 2
    ):

        raise RuntimeError(
            "خروجی _pixel_sha256_cpu "
            "باید (digest, image_size) باشد."
        )

    digest, image_size = result

    return (
        normalize_digest(
            digest
        ),
        str(
            image_size
        ),
    )


def process_record(
    record: ImageRecord,
) -> ImageRecord:

    path = Path(
        record.full_path
    )

    if not path.exists():

        record.excluded = True
        record.exclusion_reason = (
            "FILE_NOT_FOUND"
        )

        return record

    try:

        record.file_size = (
            path.stat().st_size
        )

        blank_function = getattr(
            all_data,
            "is_blank_image",
            None,
        )

        if not callable(
            blank_function
        ):

            raise RuntimeError(
                "is_blank_image "
                "در all_data.py یافت نشد."
            )

        try:

            blank_result = (
                blank_function(
                    path,
                    record.file_size,
                )
            )

        except TypeError:

            blank_result = (
                blank_function(
                    str(path),
                    record.file_size,
                )
            )

        record.is_blank = bool(
            blank_result
        )

        if record.is_blank:

            record.excluded = True
            record.exclusion_reason = (
                "BLANK_IMAGE"
            )

            return record

        xray_function = getattr(
            all_data,
            "is_xray_image",
            None,
        )

        if callable(
            xray_function
        ):

            try:

                xray_result = (
                    xray_function(
                        path
                    )
                )

            except TypeError:

                xray_result = (
                    xray_function(
                        str(path)
                    )
                )

            record.is_xray = bool(
                xray_result
            )

        if record.is_xray:

            record.excluded = True
            record.exclusion_reason = (
                "XRAY_IMAGE"
            )

            return record

        file_hash_function = getattr(
            all_data,
            "_file_sha256",
            None,
        )

        if not callable(
            file_hash_function
        ):

            raise RuntimeError(
                "_file_sha256 "
                "در all_data.py یافت نشد."
            )

        try:

            file_hash = (
                file_hash_function(
                    path
                )
            )

        except TypeError:

            file_hash = (
                file_hash_function(
                    str(path)
                )
            )

        record.file_sha256 = (
            normalize_digest(
                file_hash
            )
        )

        record.pixel_sha256, record.image_size = (
            calculate_pixel_hash(
                path
            )
        )

        record.excluded = False
        record.exclusion_reason = None

        return record

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        RuntimeError,
        TypeError,
    ) as exc:

        record.excluded = True

        record.exclusion_reason = (
            "PROCESSING_ERROR:"
            f"{type(exc).__name__}:"
            f"{str(exc)[:300]}"
        )

        return record


# =============================================================================
# PUBLIC RECORD
# =============================================================================

def public_record(
    record: ImageRecord,
) -> dict[str, Any]:

    return {
        "record_id": record.record_id,
        "folder_id": record.folder_id,
        "claimid": record.claimid,
        "filename": record.filename,
        "full_path": record.full_path,
        "attach_group": record.attach_group,
        "file_size": record.file_size,
        "image_size": record.image_size,
        "file_sha256": record.file_sha256,
        "pixel_sha256": record.pixel_sha256,
        "is_blank": record.is_blank,
        "is_xray": record.is_xray,
        "excluded": record.excluded,
        "exclusion_reason": record.exclusion_reason,
    }


# =============================================================================
# COMPARISON
# =============================================================================

def compare_record(
    query: ImageRecord,
    candidates: Iterable[ImageRecord],
) -> list[dict[str, Any]]:

    matches: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for candidate in candidates:

        if candidate.record_id == query.record_id:
            continue

        # فقط بین folder_idهای متفاوت
        if candidate.folder_id == query.folder_id:
            continue

        if candidate.excluded:
            continue

        duplicate_type: str | None = None
        risk_level: str | None = None

        if (
            query.file_sha256
            and candidate.file_sha256
            == query.file_sha256
        ):

            duplicate_type = (
                "exact_file"
            )

            risk_level = (
                "VERY_HIGH"
            )

        elif (
            query.pixel_sha256
            and candidate.pixel_sha256
            == query.pixel_sha256
        ):

            duplicate_type = (
                "exact_pixel"
            )

            risk_level = "HIGH"

        if (
            duplicate_type is None
            or risk_level is None
        ):
            continue

        identity = (
            candidate.record_id,
            duplicate_type,
        )

        if identity in seen:
            continue

        seen.add(identity)

        matches.append(
            {
                "risk_level": risk_level,
                "duplicate_type": duplicate_type,
                "query": public_record(
                    query
                ),
                "matched_record": public_record(
                    candidate
                ),
            }
        )

    matches.sort(
        key=lambda item: (
            0
            if item["risk_level"]
            == "VERY_HIGH"
            else 1,
            str(
                item["matched_record"][
                    "folder_id"
                ]
            ),
            str(
                item["matched_record"][
                    "filename"
                ]
            ),
        )
    )

    return matches


def create_summary(
    query: ImageRecord,
    matches: list[dict[str, Any]],
) -> dict[str, Any]:

    by_risk: dict[
        str,
        int,
    ] = defaultdict(int)

    by_type: dict[
        str,
        int,
    ] = defaultdict(int)

    folders: set[str] = set()

    for match in matches:

        by_risk[
            match["risk_level"]
        ] += 1

        by_type[
            match["duplicate_type"]
        ] += 1

        folders.add(
            str(
                match["matched_record"][
                    "folder_id"
                ]
            )
        )

    return {
        "query_folder_id": query.folder_id,
        "query_claimid": query.claimid,
        "total_matches": len(matches),
        "very_high_count": by_risk.get(
            "VERY_HIGH",
            0,
        ),
        "high_count": by_risk.get(
            "HIGH",
            0,
        ),
        "exact_file_count": by_type.get(
            "exact_file",
            0,
        ),
        "exact_pixel_count": by_type.get(
            "exact_pixel",
            0,
        ),
        "matched_folder_count": len(
            folders
        ),
        "matched_folders": sorted(
            folders
        ),
    }


# =============================================================================
# DATASET BUILD
#
# Heavy processing is done completely outside CACHE_LOCK.
# Only the final repository replacement + cache write uses CACHE_LOCK.
# =============================================================================

def build_repository_from_json() -> tuple[
    HashRepository,
    int,
    int,
]:

    if not DATASET_JSON.exists():

        raise FileNotFoundError(
            f"Dataset یافت نشد: "
            f"{DATASET_JSON}"
        )

    with DATASET_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(file)

    items = extract_items(
        payload
    )

    new_repository = (
        HashRepository()
    )

    loaded = 0
    skipped = 0

    for index, item in enumerate(
        items
    ):

        record = make_record(
            item,
            index,
        )

        if record is None:

            skipped += 1
            continue

        processed = process_record(
            record
        )

        new_repository.add(
            processed
        )

        loaded += 1

    return (
        new_repository,
        loaded,
        skipped,
    )


def install_repository(
    new_repository: HashRepository,
) -> None:

    global REPOSITORY

    with CACHE_LOCK:

        new_repository.save()

        REPOSITORY = (
            new_repository
        )


def rebuild_repository_sync() -> tuple[
    int,
    int,
    int,
]:

    (
        new_repository,
        loaded,
        skipped,
    ) = build_repository_from_json()

    install_repository(
        new_repository
    )

    total = len(
        new_repository.records
    )

    return (
        loaded,
        skipped,
        total,
    )


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

        repository = REPOSITORY

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
            REPOSITORY.load
        )

        if not loaded:

            # Cache خراب/قدیمی فقط باعث خالی‌ماندن Repository می‌شود.
            # Dataset به‌صورت خودکار در startup rebuild نمی‌شود.
            with CACHE_LOCK:
                REPOSITORY.clear()

    except Exception:

        with CACHE_LOCK:
            REPOSITORY.clear()

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


# =============================================================================
# ROOT
# =============================================================================

@app.get("/")
def root() -> dict[str, Any]:

    with CACHE_LOCK:

        repository_count = len(
            REPOSITORY.records
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

@app.get("/health")
def health() -> dict[str, Any]:

    with CACHE_LOCK:

        repository_records = len(
            REPOSITORY.records
        )

        active_records = sum(
            not record.excluded
            for record
            in REPOSITORY.records.values()
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
        "gpu_function_available": callable(
            getattr(
                all_data,
                "_pixel_sha256_gpu",
                None,
            )
        ),
        "cpu_function_available": callable(
            getattr(
                all_data,
                "_pixel_sha256_cpu",
                None,
            )
        ),
    }


# =============================================================================
# DATASET LOAD
# =============================================================================

@app.post(
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

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except json.JSONDecodeError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "فایل Dataset "
                "دارای JSON معتبر نیست."
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "خطا در بارگذاری Dataset: "
                f"{exc}"
            ),
        ) from exc


# =============================================================================
# DATASET STATS
# =============================================================================

@app.get(
    "/v1/dataset/stats"
)
def dataset_stats() -> dict[str, Any]:

    with CACHE_LOCK:

        repository = REPOSITORY

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


# =============================================================================
# COMPARE ONE IMAGE
# =============================================================================

@app.post(
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

    folder_id = (
        folder_id or ""
    ).strip()

    if not folder_id:

        raise HTTPException(
            status_code=422,
            detail=(
                "folder_id الزامی است."
            ),
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

@app.post(
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

    folder_id = (
        folder_id or ""
    ).strip()

    if not folder_id:

        raise HTTPException(
            status_code=422,
            detail=(
                "folder_id الزامی است."
            ),
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


# =============================================================================
# ADD TO REPOSITORY
# =============================================================================

def add_record_to_repository_sync(
    record: ImageRecord,
) -> list[dict[str, Any]]:

    # تمام عملیات hash قبلاً خارج از lock انجام شده است.
    with CACHE_LOCK:

        candidates = (
            REPOSITORY.candidates_for(
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

        REPOSITORY.add(
            record
        )

        REPOSITORY.save()

    return matches


@app.post(
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

    folder_id = (
        folder_id or ""
    ).strip()

    if not folder_id:

        raise HTTPException(
            status_code=422,
            detail=(
                "folder_id الزامی است."
            ),
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

def delete_repository_record_sync(
    record_id: str,
) -> None:

    with CACHE_LOCK:

        record = REPOSITORY.get(
            record_id
        )

        if record is None:

            raise KeyError(
                record_id
            )

        stored_path = Path(
            record.full_path
        )

        REPOSITORY.remove(
            record_id
        )

        REPOSITORY.save()

        try:

            stored_path.resolve().relative_to(
                UPLOAD_DIR.resolve()
            )

            stored_path.unlink(
                missing_ok=True
            )

        except ValueError:

            pass


@app.delete(
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

@app.post(
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

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except json.JSONDecodeError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "فایل Dataset "
                "دارای JSON معتبر نیست."
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "خطا در بازسازی Repository: "
                f"{exc}"
            ),
        ) from exc


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=False,
        workers=1,
    )
