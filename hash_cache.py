from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat, UnidentifiedImageError


BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
DATASET_JSON = Path(
    os.getenv("DATASET_JSON", str(BASE_DIR / "all_claims_new.json"))
).resolve()
OUTPUT_PKL = Path(
    os.getenv("OUTPUT_PKL", str(BASE_DIR / "hash_cache_all.pkl"))
).resolve()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PIXEL_HASH_SIZE = (64, 64)
BLANK_THRESHOLD = 250
BLANK_PIXEL_RATIO = 0.95
XRAY_ASPECT_RATIO_MIN = 0.5
XRAY_ASPECT_RATIO_MAX = 2.0
XRAY_MEAN_BRIGHTNESS_MAX = 100
XRAY_STD_MAX = 60


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def extract_items(payload: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if isinstance(payload, list):
        for claim in payload:
            if not isinstance(claim, dict):
                continue
            claimid = normalize_value(
                claim.get("claimid")
                or claim.get("claim_id")
                or claim.get("claimId")
            )
            attachments = claim.get("attachments")
            if not isinstance(attachments, dict):
                continue
            for attach_group, files in attachments.items():
                if not isinstance(files, list):
                    continue
                for file_info in files:
                    if not isinstance(file_info, dict):
                        continue
                    path = normalize_value(
                        file_info.get("path")
                        or file_info.get("full_path")
                        or file_info.get("file_path")
                    )
                    if not path:
                        continue
                    results.append(
                        {
                            "claimid": claimid,
                            "folder_id": Path(path).parent.name,
                            "full_path": str(BASE_DIR / path),
                            "filename": normalize_value(
                                file_info.get("name")
                                or file_info.get("filename")
                            ) or Path(path).name,
                            "attach_group": normalize_value(attach_group),
                        }
                    )
        return results

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        results.append(item)
            elif isinstance(value, dict):
                results.append(value)

    return results


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixel_sha256_cpu(path: Path) -> tuple[str, str]:
    with Image.open(path) as img:
        img = img.convert("L").resize(PIXEL_HASH_SIZE)
        pixels = img.tobytes()
        digest = hashlib.sha256(pixels).hexdigest()
        image_size = f"{img.width}x{img.height}"
    return digest, image_size


def is_blank_image(path: Path, file_size: int) -> bool:
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            stat = ImageStat.Stat(img)
            mean_brightness = stat.mean[0]
            if mean_brightness < BLANK_THRESHOLD:
                return False
            histogram = img.histogram()
            total_pixels = img.width * img.height
            white_pixels = sum(histogram[BLANK_THRESHOLD:])
            return (white_pixels / total_pixels) >= BLANK_PIXEL_RATIO
    except Exception:
        return False


def is_xray_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 1.0
            if not (XRAY_ASPECT_RATIO_MIN <= aspect_ratio <= XRAY_ASPECT_RATIO_MAX):
                return False
            stat = ImageStat.Stat(img)
            mean_brightness = stat.mean[0]
            std_dev = stat.stddev[0]
            return (
                mean_brightness <= XRAY_MEAN_BRIGHTNESS_MAX
                and std_dev <= XRAY_STD_MAX
            )
    except Exception:
        return False


def process_single_record(item: dict[str, Any], index: int) -> dict[str, Any]:
    full_path = item.get("full_path", "")
    path = Path(full_path)

    record_id = f"{item.get('folder_id', 'UNKNOWN')}/{item.get('filename', index)}"

    result: dict[str, Any] = {
        "record_id": record_id,
        "folder_id": item.get("folder_id", "UNKNOWN"),
        "claimid": item.get("claimid"),
        "filename": item.get("filename"),
        "full_path": full_path,
        "attach_group": item.get("attach_group"),
        "file_size": None,
        "image_size": None,
        "file_sha256": None,
        "pixel_sha256": None,
        "is_blank": False,
        "is_xray": False,
        "excluded": False,
        "exclusion_reason": None,
    }

    if not path.exists():
        result["excluded"] = True
        result["exclusion_reason"] = "FILE_NOT_FOUND"
        return result

    try:
        result["file_size"] = path.stat().st_size

        if is_blank_image(path, result["file_size"]):
            result["is_blank"] = True
            result["excluded"] = True
            result["exclusion_reason"] = "BLANK_IMAGE"
            return result

        if is_xray_image(path):
            result["is_xray"] = True
            result["excluded"] = True
            result["exclusion_reason"] = "XRAY_IMAGE"
            return result

        result["file_sha256"] = _file_sha256(path)
        pixel_hash, image_size = _pixel_sha256_cpu(path)
        result["pixel_sha256"] = pixel_hash
        result["image_size"] = image_size

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        RuntimeError,
        TypeError,
    ) as exc:
        result["excluded"] = True
        result["exclusion_reason"] = (
            f"PROCESSING_ERROR:{type(exc).__name__}:{str(exc)[:300]}"
        )

    return result


def build_hash_cache() -> list[dict[str, Any]]:
    if not DATASET_JSON.exists():
        print(f"ERROR: Dataset not found: {DATASET_JSON}", file=sys.stderr)
        sys.exit(1)

    with DATASET_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = extract_items(payload)
    print(f"Found {len(items)} images to process")

    results: list[dict[str, Any]] = [None] * len(items)  # type: ignore[assignment]

    with ProcessPoolExecutor() as executor:
        future_to_index = {
            executor.submit(process_single_record, item, i): i
            for i, item in enumerate(items)
        }

        completed = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                results[i] = {
                    "record_id": f"error-{i}",
                    "folder_id": "UNKNOWN",
                    "filename": items[i].get("filename", f"error-{i}"),
                    "full_path": items[i].get("full_path", ""),
                    "excluded": True,
                    "exclusion_reason": f"PROCESSING_ERROR:{type(exc).__name__}:{str(exc)[:300]}",
                }
            completed += 1
            if completed % 100 == 0 or completed == len(items):
                print(f"  Progress: {completed}/{len(items)}")

    return results


def save_cache(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(results)} records to {output_path}")


def main() -> None:
    print(f"Reading dataset: {DATASET_JSON}")
    results = build_hash_cache()

    ok_count = sum(1 for r in results if not r["excluded"])
    excluded_count = sum(1 for r in results if r["excluded"])
    blank_count = sum(1 for r in results if r.get("is_blank"))
    xray_count = sum(1 for r in results if r.get("is_xray"))

    print(f"\nResults:")
    print(f"  Total: {len(results)}")
    print(f"  OK: {ok_count}")
    print(f"  Excluded: {excluded_count}")
    print(f"    Blank: {blank_count}")
    print(f"    X-ray: {xray_count}")

    save_cache(results, OUTPUT_PKL)


if __name__ == "__main__":
    main()
