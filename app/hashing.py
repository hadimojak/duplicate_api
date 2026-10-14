from __future__ import annotations

from pathlib import Path

from PIL import UnidentifiedImageError

import all_data

from app.config import (
    GPU_PROCESS_LOCK,
    USE_GPU,
)
from app.helpers import normalize_digest
from app.models import ImageRecord


# =============================================================================
# ALL_DATA INTEGRATION
# =============================================================================

def gpu_function_available() -> bool:
    return callable(
        getattr(
            all_data,
            "_pixel_sha256_gpu",
            None,
        )
    )


def cpu_function_available() -> bool:
    return callable(
        getattr(
            all_data,
            "_pixel_sha256_cpu",
            None,
        )
    )


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
