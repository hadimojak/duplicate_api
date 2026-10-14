from __future__ import annotations

import os
import threading
from pathlib import Path


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


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
