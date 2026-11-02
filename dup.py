from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import os
import json
import hashlib
import logging
import pickle
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps

# ── GPU (اختیاری) ─────────────────────────────────────────
try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    from torchvision.io import read_image, ImageReadMode
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
 
# ─── تنظیمات ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "TD", "download")
JSON_PATH    = os.path.join(BASE_DIR, "all_claims.json")
EXCEL_DIR    = os.path.join(BASE_DIR, "excel_output")
CACHE_FILE   = os.path.join(BASE_DIR, "hash_cache_all.pkl")
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_WORKERS = 8
CHUNK_SIZE  = 1 << 20
PIXEL_THUMB = 512

# ── فیلتر برگه سفید ───────────────────────────────────────
MAX_FILE_BYTES_FOR_BLANK_CHECK = 10_000
BLANK_VAR_THRESH               = 8.0
BLANK_ENTROPY_THRESH           = 1.5

# ── فیلتر رادیولوژی ───────────────────────────────────────
XRAY_MAX_MEAN       = 80
XRAY_MIN_STD        = 40
XRAY_MAX_COLOR_DIFF = 15

_CACHE_VERSION = "v6-download-scan"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─── ساختار رکورد ─────────────────────────────────────────
@dataclass
class ImageRecord:
    folder_id:    str
    claimid:      str
    attach_group: str
    filename:     str
    rel_path:     str
    full_path:    str
    file_size:    int = 0
    file_sha256:  str = ""
    pixel_sha256: str = ""
    img_size:     str = ""
    error:        str = ""
    skipped:      str = ""


# ─── Cache ────────────────────────────────────────────────
def load_cache(cache_path: str, fingerprint: str | None = None) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict) or not isinstance(cache.get("ok"), list):
            log.warning("Cache file has no hashed records")
            return None
        log.info(
            f"Loaded cache — {len(cache['ok']):,} hashed, "
            f"{len(cache.get('err') or []):,} errors, "
            f"{len(cache.get('skipped') or []):,} skipped"
        )
        if fingerprint is not None and cache.get("json_fingerprint") != fingerprint:
            log.info("Cache file list changed; new images will be added")
        return cache
    except Exception as e:
        log.warning(f"Could not read cache: {e}")
        return None


def _cache_filename(rec: dict) -> str:
    raw = rec.get("rel_path") or rec.get("filename") or ""
    return os.path.basename(str(raw).replace("\\", "/"))


def append_new_images(
    records: list[dict],
    ok: list[dict],
    err: list[dict],
    skipped: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    known = {
        name
        for bucket in (ok, err, skipped)
        for rec in bucket
        if (name := _cache_filename(rec))
    }
    already = len(known)
    new_records = []
    for rec in records:
        name = _cache_filename(rec)
        if not name or name in known:
            continue
        known.add(name)
        new_records.append(rec)
    log.info(f"New images to hash: {len(new_records):,} (already cached filenames: {already:,})")
    if not new_records:
        return ok, err, skipped

    gpu_ok = _TORCH_OK and torch.cuda.is_available()
    if gpu_ok:
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        new_ok, new_err, new_skipped = hash_all_gpu(new_records, torch.device("cuda"))
    else:
        log.info(f"CPU: {MAX_WORKERS} workers")
        new_ok, new_err, new_skipped = hash_all_cpu(new_records)

    ok.extend(new_ok)
    err.extend(new_err)
    skipped.extend(new_skipped)
    stored = ok + err + skipped
    save_cache(CACHE_FILE, _fingerprint_dicts(stored), ok, err, skipped)
    log.info(f"Added {len(new_records):,} images to {CACHE_FILE}")
    return ok, err, skipped


def save_cache(cache_path: str, fingerprint: str,
               ok: list[dict], err: list[dict], skipped: list[dict]):
    with open(cache_path, "wb") as f:
        pickle.dump({
            "cache_version":    _CACHE_VERSION,
            "json_fingerprint": fingerprint,
            "blank_var_thresh": BLANK_VAR_THRESH,
            "blank_ent_thresh": BLANK_ENTROPY_THRESH,
            "xray_max_mean":    XRAY_MAX_MEAN,
            "xray_min_std":     XRAY_MIN_STD,
            "xray_max_color":   XRAY_MAX_COLOR_DIFF,
            "created_at":       datetime.now().isoformat(),
            "ok":               ok,
            "err":              err,
            "skipped":          skipped,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Cache saved → {cache_path}")


# ─── بارگذاری JSON ────────────────────────────────────────
def extract_folder_id(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return parts[-2]
    return parts[0] if len(parts) == 2 else "unknown"


def _norm_rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def scan_download_images(download_dir: str) -> list[ImageRecord]:
    if not os.path.isdir(download_dir):
        log.error(f"Image folder not found: {download_dir}")
        return []

    records = []
    for root, _, files in os.walk(download_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            full = os.path.join(root, name)
            rel = _norm_rel(os.path.relpath(full, BASE_DIR))
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            records.append(ImageRecord(
                folder_id    = extract_folder_id(rel),
                claimid      = "",
                attach_group = "",
                filename     = name,
                rel_path     = rel,
                full_path    = full,
                file_size    = size,
            ))
    records.sort(key=lambda r: r.rel_path)
    log.info(f"TD/download: {len(records):,} images")
    return records


def claim_index_from_claims(claims: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in claims:
        claimid = str(entry.get("claimid", ""))
        attachments_data = entry.get("attachments")
        if not isinstance(attachments_data, dict):
            continue
        for group, files in attachments_data.items():
            if not isinstance(files, list):
                continue
            for item in files:
                if isinstance(item, dict):
                    rel = item.get("path", "")
                    display_name = item.get("name", "")
                elif isinstance(item, str):
                    rel, display_name = item, ""
                else:
                    continue
                key = _norm_rel(rel)
                if not key or key in index:
                    continue
                index[key] = {
                    "claimid": claimid,
                    "attach_group": str(group),
                    "name": display_name,
                }
    log.info(f"Claims index: {len(index):,} paths")
    return index


def load_claim_index(json_path: str) -> dict[str, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return claim_index_from_claims(data)


def attach_claim_details(records: list[dict], index: dict[str, dict]) -> int:
    matched = 0
    for rec in records:
        info = index.get(_norm_rel(rec.get("rel_path", "")))
        if not info:
            continue
        rec["claimid"] = info["claimid"]
        rec["attach_group"] = info["attach_group"]
        if info.get("name"):
            rec["filename"] = info["name"]
        matched += 1
    return matched


# ─── تشخیص برگه سفید ──────────────────────────────────────
def _image_entropy(arr: np.ndarray) -> float:
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def is_blank_image_bytes(data: bytes, file_size: int) -> str:
    real_size = file_size if file_size > 0 else len(data)
    if real_size > MAX_FILE_BYTES_FOR_BLANK_CHECK:
        return ""
    try:
        with Image.open(BytesIO(data)) as img:
            gray = ImageOps.exif_transpose(img).convert("L")
            gray.thumbnail((64, 64), Image.LANCZOS)
            arr = np.array(gray, dtype=np.float32)
        var = float(np.var(arr))
        ent = _image_entropy(arr.astype(np.uint8))
        if var < BLANK_VAR_THRESH and ent < BLANK_ENTROPY_THRESH:
            return f"blank_image(var={var:.1f},ent={ent:.2f})"
    except Exception:
        pass
    return ""


def is_blank_image(path: str, file_size: int) -> str:
    real_size = file_size if file_size > 0 else 0
    try:
        real_size = real_size or os.path.getsize(path)
    except OSError:
        return ""
    if real_size > MAX_FILE_BYTES_FOR_BLANK_CHECK:
        return ""
    try:
        with open(path, "rb") as f:
            return is_blank_image_bytes(f.read(), real_size)
    except OSError:
        return ""


# ─── تشخیص رادیولوژی ──────────────────────────────────────
def is_xray_image_bytes(data: bytes) -> str:
    try:
        with Image.open(BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((128, 128), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)

        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        mean_brightness = float(np.mean(gray))
        std_brightness  = float(np.std(gray))

        rg_diff = float(np.mean(np.abs(arr[:,:,0] - arr[:,:,1])))
        rb_diff = float(np.mean(np.abs(arr[:,:,0] - arr[:,:,2])))
        color_diff = (rg_diff + rb_diff) / 2.0

        if (mean_brightness < XRAY_MAX_MEAN
                and std_brightness  > XRAY_MIN_STD
                and color_diff      < XRAY_MAX_COLOR_DIFF):
            return (f"xray_image("
                    f"mean={mean_brightness:.1f},"
                    f"std={std_brightness:.1f},"
                    f"color_diff={color_diff:.1f})")
    except Exception:
        pass
    return ""


def is_xray_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return is_xray_image_bytes(f.read())
    except OSError:
        return ""


# ─── توابع هش ─────────────────────────────────────────────
def _file_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixel_sha256_bytes(data: bytes) -> tuple[str, str]:
    with Image.open(BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        thumb = img.copy()
        thumb.thumbnail((PIXEL_THUMB, PIXEL_THUMB), Image.LANCZOS)
        digest = hashlib.sha256(
            thumb.tobytes() + f"{w}x{h}".encode()
        ).hexdigest()
    return digest, f"{w}x{h}"


def _pixel_sha256_cpu(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        return _pixel_sha256_bytes(f.read())


def _pixel_sha256_gpu(path: str, device) -> tuple[str, str]:
    try:
        t = read_image(path, mode=ImageReadMode.RGB).to(device)
        _, h, w = t.shape

        max_dim = max(h, w)
        if max_dim > PIXEL_THUMB:
            scale = PIXEL_THUMB / max_dim
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            resized = TF.resize(
                t, 
                [new_h, new_w], 
                interpolation=InterpolationMode.LANCZOS, 
                antialias=True
            )
        else:
            resized = t

        arr_bytes = resized.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()

        digest = hashlib.sha256(
            arr_bytes + f"{w}x{h}".encode()
        ).hexdigest()
        
        return digest, f"{w}x{h}"
    except Exception:
        return _pixel_sha256_cpu(path)


# ─── پردازش یک رکورد ──────────────────────────────────────
def process_one(rec: dict) -> dict:
    path = rec["full_path"]

    if not os.path.exists(path):
        rec["error"] = "file_not_found"
        return rec

    skip = is_blank_image(path, rec.get("file_size", 0))
    if skip:
        rec["skipped"] = skip
        return rec

    skip = is_xray_image(path)
    if skip:
        rec["skipped"] = skip
        return rec

    try:
        rec["file_sha256"]                   = _file_sha256(path)
        rec["pixel_sha256"], rec["img_size"] = _pixel_sha256_cpu(path)
    except Exception as e:
        rec["error"] = str(e)

    return rec


# ─── Hash همه فایل‌ها — GPU ────────────────────────────────
def hash_all_gpu(records: list[dict], device) -> tuple[list, list, list]:
    ok, err, skipped = [], [], []
    for r in tqdm(records, desc="GPU hashing", unit="img"):
        path = r["full_path"]
        if not os.path.exists(path):
            r["error"] = "file_not_found"
            err.append(r)
            continue

        skip = is_blank_image(path, r.get("file_size", 0))
        if not skip:
            skip = is_xray_image(path)
        if skip:
            r["skipped"] = skip
            skipped.append(r)
            continue

        try:
            r["file_sha256"]                   = _file_sha256(path)
            r["pixel_sha256"], r["img_size"]   = _pixel_sha256_gpu(path, device)
            ok.append(r)
        except Exception as e:
            r["error"] = str(e)
            err.append(r)

    return ok, err, skipped


# ─── Hash همه فایل‌ها — CPU ────────────────────────────────
def hash_all_cpu(records: list[dict]) -> tuple[list, list, list]:
    ok, err, skipped = [], [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one, r): r for r in records}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="CPU hashing", unit="img"):
            res = f.result()
            if   res.get("error"):   err.append(res)
            elif res.get("skipped"): skipped.append(res)
            else:                    ok.append(res)
    return ok, err, skipped


# ─── تشخیص تکراری‌ها ──────────────────────────────────────
def _image_id(rec: dict) -> str:
    name = os.path.basename(rec.get("rel_path") or rec.get("filename") or "")
    stem, _ext = os.path.splitext(name)
    return stem or name


def find_duplicate_pairs(records: list[dict]) -> pd.DataFrame:
    file_groups:  dict[str, list] = defaultdict(list)
    pixel_groups: dict[str, list] = defaultdict(list)

    for r in records:
        if r.get("error") or not r.get("file_sha256"):
            continue
        file_groups[r["file_sha256"]].append(r)
        pixel_groups[r["pixel_sha256"]].append(r)

    folder_pairs:   dict[tuple, dict] = {}
    seen_img_pairs: set[tuple]        = set()
    DUP_PRIORITY = {"exact_file": 2, "exact_pixel": 1}

    def register_pair(a: dict, b: dict, dup_type: str):
        if a["folder_id"] == b["folder_id"]:
            return
        img_key = tuple(sorted([a["full_path"], b["full_path"]]))
        if img_key in seen_img_pairs:
            return
        seen_img_pairs.add(img_key)

        fid1, fid2 = sorted([a["folder_id"], b["folder_id"]])
        r1 = a if a["folder_id"] == fid1 else b
        r2 = b if a["folder_id"] == fid1 else a
        folder_key = (fid1, fid2)

        if folder_key not in folder_pairs:
            folder_pairs[folder_key] = {
                "folder_id_1":       fid1,
                "claimid_1":         r1["claimid"],
                "folder_id_2":       fid2,
                "claimid_2":         r2["claimid"],
                "n_shared_images":   0,
                "best_dup_priority": 0,
                "best_dup_type":     "",
                "_files_1":          [],
                "_files_2":          [],
                "_dup_ids_1":        [],
                "_dup_ids_2":        [],
                "_groups_1":         [],
                "_groups_2":         [],
            }

        entry = folder_pairs[folder_key]
        entry["n_shared_images"] += 1
        entry["_files_1"].append(r1["filename"])
        entry["_files_2"].append(r2["filename"])
        id1, id2 = _image_id(r1), _image_id(r2)
        if id1 not in entry["_dup_ids_1"]:
            entry["_dup_ids_1"].append(id1)
        if id2 not in entry["_dup_ids_2"]:
            entry["_dup_ids_2"].append(id2)
        entry["_groups_1"].append(r1["attach_group"])
        entry["_groups_2"].append(r2["attach_group"])
        if DUP_PRIORITY[dup_type] > entry["best_dup_priority"]:
            entry["best_dup_priority"] = DUP_PRIORITY[dup_type]
            entry["best_dup_type"]     = dup_type

    for items in file_groups.values():
        if len(items) >= 2:
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    register_pair(items[i], items[j], "exact_file")

    for items in pixel_groups.values():
        if len(items) < 2:
            continue
        if len({x["file_sha256"] for x in items}) <= 1:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                register_pair(items[i], items[j], "exact_pixel")

    if not folder_pairs:
        return pd.DataFrame()

    RISK_MAP = {
        2: ("VERY_HIGH", "Identical file bytes across different folders"),
        1: ("HIGH",      "Identical pixel content with a different format"),
    }
    rows = []
    for entry in folder_pairs.values():
        risk_level, risk_reason = RISK_MAP.get(
            entry["best_dup_priority"], ("MEDIUM", "Similar content")
        )
        rows.append({
            "risk_level":      risk_level,
            "risk_reason":     risk_reason,
            "dup_type":        entry["best_dup_type"],
            "folder_id_1":     entry["folder_id_1"],
            "claimid_1":       entry["claimid_1"],
            "folder_id_2":     entry["folder_id_2"],
            "claimid_2":       entry["claimid_2"],
            "n_shared_images": entry["n_shared_images"],
            "filenames_1":         " | ".join(dict.fromkeys(entry["_files_1"])),
            "duplicateFilename1":  " | ".join(entry["_dup_ids_1"]),
            "filenames_2":         " | ".join(dict.fromkeys(entry["_files_2"])),
            "duplicateFilename2":  " | ".join(entry["_dup_ids_2"]),
            "attach_groups_1": " | ".join(dict.fromkeys(entry["_groups_1"])),
            "attach_groups_2": " | ".join(dict.fromkeys(entry["_groups_2"])),
        })

    df = pd.DataFrame(rows)
    risk_order = {"VERY_HIGH": 0, "HIGH": 1, "MEDIUM": 2}
    df["_ro"] = df["risk_level"].map(risk_order)
    df.sort_values(["_ro", "n_shared_images"], ascending=[True, False], inplace=True)
    df.drop(columns=["_ro"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── خلاصه پرونده‌های مشکوک ───────────────────────────────
def build_folder_summary(df: pd.DataFrame) -> pd.DataFrame:
    risk_order = {"VERY_HIGH": 0, "HIGH": 1, "MEDIUM": 2}
    rows: dict[str, dict] = {}

    for _, row in df.iterrows():
        for my_side, other_side in [("1", "2"), ("2", "1")]:
            fid = row[f"folder_id_{my_side}"]
            oth = row[f"folder_id_{other_side}"]
            if fid not in rows:
                rows[fid] = {
                    "folder_id":             fid,
                    "claimid":               row[f"claimid_{my_side}"],
                    "risk_level":            row["risk_level"],
                    "risk_reason":           row["risk_reason"],
                    "total_shared_images":   0,
                    "n_shared_with_folders": 0,
                    "shared_with_folders":   set(),
                }
            e = rows[fid]
            e["total_shared_images"] += row["n_shared_images"]
            e["shared_with_folders"].add(oth)
            if risk_order.get(row["risk_level"], 9) < risk_order.get(e["risk_level"], 9):
                e["risk_level"]  = row["risk_level"]
                e["risk_reason"] = row["risk_reason"]

    result = []
    for info in rows.values():
        info["n_shared_with_folders"] = len(info["shared_with_folders"])
        info["shared_with_folders"]   = " | ".join(sorted(info["shared_with_folders"]))
        result.append(info)

    out = pd.DataFrame(result)
    out["_ro"] = out["risk_level"].map(risk_order)
    out.sort_values(["_ro", "total_shared_images"], ascending=[True, False], inplace=True)
    out.drop(columns=["_ro"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ─── ذخیره Excel ──────────────────────────────────────────
PAIR_COLS = [
    "risk_level", "risk_reason", "dup_type",
    "folder_id_1", "claimid_1", "attach_groups_1", "filenames_1", "duplicateFilename1",
    "folder_id_2", "claimid_2", "attach_groups_2", "filenames_2", "duplicateFilename2",
    "n_shared_images",
]
MAX_EXCEL_BYTES = 5 * 1024 * 1024  # 5 MB per Excel file


def excel_output_path() -> str:
    os.makedirs(EXCEL_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(EXCEL_DIR, f"exact_duplicate_pairs_{stamp}.xlsx")


def _skipped_frame(skipped: list[dict]) -> pd.DataFrame:
    if not skipped:
        return pd.DataFrame()
    df_skip = pd.DataFrame(skipped)[[
        "folder_id", "claimid", "filename",
        "rel_path", "file_size", "skipped"
    ]].copy()
    df_skip["skip_type"] = df_skip["skipped"].apply(
        lambda s: "xray" if "xray_image" in str(s) else "blank"
    )
    return df_skip


def _write_excel_part(
    path: str,
    df_pairs: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_stats: pd.DataFrame,
    errors: list[dict] | None,
    df_skip: pd.DataFrame | None,
    part_no: int,
    part_total: int,
) -> str:
    stats = df_stats.copy()
    if part_total > 1:
        stats = pd.concat([
            stats,
            pd.DataFrame([
                {"metric": "Excel part", "value": f"{part_no}/{part_total}"},
            ]),
        ], ignore_index=True)

    def _write(p: str) -> None:
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            pairs = df_pairs[PAIR_COLS] if not df_pairs.empty else pd.DataFrame(columns=PAIR_COLS)
            pairs.to_excel(w, sheet_name="duplicate_pairs", index=False)
            df_summary.to_excel(w, sheet_name="suspicious_folders", index=False)
            very_high = pairs[pairs["risk_level"] == "VERY_HIGH"] if not pairs.empty else pairs
            very_high.to_excel(w, sheet_name="VERY_HIGH_only", index=False)
            stats.to_excel(w, sheet_name="statistics", index=False)
            if errors:
                pd.DataFrame(errors).to_excel(w, sheet_name="errors", index=False)
            if df_skip is not None and not df_skip.empty:
                df_skip.to_excel(w, sheet_name="skipped_images", index=False)
                df_xray = df_skip[df_skip["skip_type"] == "xray"]
                if not df_xray.empty:
                    df_xray.to_excel(w, sheet_name="skipped_xray", index=False)

    try:
        _write(path)
    except PermissionError:
        alt = f"{os.path.splitext(path)[0]}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        _write(alt)
        log.warning(f"File was locked → {alt}")
        path = alt
    return path


def _excel_part_path(base_path: str, part_no: int, part_total: int) -> str:
    if part_total <= 1:
        return base_path
    root, ext = os.path.splitext(base_path)
    return f"{root}_part{part_no:03d}of{part_total:03d}{ext}"


def _workbook_size_bytes(
    df_pairs: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_stats: pd.DataFrame,
    errors: list[dict] | None,
    df_skip: pd.DataFrame | None,
) -> int:
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        _write_excel_part(tmp, df_pairs, df_summary, df_stats, errors, df_skip, 1, 1)
        return os.path.getsize(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _max_pair_rows_for_limit(
    df_pairs: pd.DataFrame,
    df_stats: pd.DataFrame,
    errors: list[dict] | None,
    df_skip: pd.DataFrame | None,
    max_bytes: int,
) -> int:
    """Largest number of pair rows that keep the workbook under max_bytes."""
    n = len(df_pairs)
    if n == 0:
        return 0
    lo, hi, best = 1, n, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = df_pairs.iloc[:mid]
        summary = build_folder_summary(chunk) if not chunk.empty else pd.DataFrame()
        size = _workbook_size_bytes(chunk, summary, df_stats, errors, df_skip)
        if size <= max_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def save_excel(df_pairs, df_summary, df_stats, errors, skipped, path) -> str:
    """Write Excel output, splitting into multiple files each <= MAX_EXCEL_BYTES."""
    df_skip_all = _skipped_frame(skipped)
    full_size = _workbook_size_bytes(df_pairs, df_summary, df_stats, errors, df_skip_all)
    if full_size <= MAX_EXCEL_BYTES:
        saved = _write_excel_part(path, df_pairs, df_summary, df_stats, errors, df_skip_all, 1, 1)
        log.info(f"Excel saved → {saved} ({full_size / (1024 * 1024):.2f} MB)")
        return saved

    # First part may include errors/skipped; later parts are pairs-only.
    parts: list[tuple[pd.DataFrame, list[dict] | None, pd.DataFrame | None]] = []
    remaining = df_pairs.reset_index(drop=True)
    aux_errors, aux_skip = errors, df_skip_all

    empty_pairs = df_pairs.iloc[0:0].copy()
    while True:
        has_aux = bool(aux_errors) or (aux_skip is not None and not aux_skip.empty)
        if remaining.empty and not has_aux:
            break

        if remaining.empty:
            # Only aux sheets left — write them even if over the limit.
            parts.append((empty_pairs, aux_errors, aux_skip))
            break

        fit = _max_pair_rows_for_limit(
            remaining, df_stats, aux_errors, aux_skip, MAX_EXCEL_BYTES
        )
        if fit < 1 and has_aux:
            # Aux sheets alone push over the limit: write them separately, then pairs.
            parts.append((empty_pairs, aux_errors, aux_skip))
            aux_errors, aux_skip = None, None
            continue

        if fit < 1:
            fit = 1
            log.warning(
                "A single Excel chunk still exceeds %s MB; writing oversized part",
                MAX_EXCEL_BYTES // (1024 * 1024),
            )

        chunk = remaining.iloc[:fit].copy()
        parts.append((chunk, aux_errors, aux_skip))
        remaining = remaining.iloc[fit:].reset_index(drop=True)
        aux_errors, aux_skip = None, None

    total = len(parts)
    saved_paths: list[str] = []
    for i, (chunk, err_part, skip_part) in enumerate(parts, start=1):
        summary = build_folder_summary(chunk) if not chunk.empty else pd.DataFrame()
        part_path = _excel_part_path(path, i, total)
        saved = _write_excel_part(
            part_path, chunk, summary, df_stats, err_part, skip_part, i, total
        )
        size = os.path.getsize(saved)
        log.info(
            f"Excel part {i}/{total} saved → {saved} ({size / (1024 * 1024):.2f} MB)"
        )
        saved_paths.append(saved)

    return " | ".join(saved_paths)


def records_from_claims(claims: list[dict], image_root: str | None = None) -> list[ImageRecord]:
    root = image_root or BASE_DIR
    records = []
    seen: set[str] = set()
    for entry in claims:
        attachments_data = entry.get("attachments")
        if not isinstance(attachments_data, dict):
            continue
        for group, files in attachments_data.items():
            if not isinstance(files, list):
                continue
            for item in files:
                if isinstance(item, dict):
                    rel = item.get("path", "")
                    size = int(item.get("size") or 0)
                    display_name = item.get("name") or ""
                elif isinstance(item, str):
                    rel, size, display_name = item, 0, ""
                else:
                    continue
                rel = _norm_rel(rel)
                if not rel or rel in seen:
                    continue
                if os.path.splitext(rel)[1].lower() not in IMAGE_EXTS:
                    continue
                seen.add(rel)
                records.append(ImageRecord(
                    folder_id    = extract_folder_id(rel),
                    claimid      = str(entry.get("claimid", "")),
                    attach_group = str(group),
                    filename     = display_name or os.path.basename(rel),
                    rel_path     = rel,
                    full_path    = os.path.join(root, rel.replace("/", os.sep)),
                    file_size    = size,
                ))
    records.sort(key=lambda r: r.rel_path)
    return records


def records_from_claims_minio(claims: list[dict]) -> list[ImageRecord]:
    """Build image records keyed by MinIO object path (no local files)."""
    records = []
    seen: set[str] = set()
    for entry in claims:
        attachments_data = entry.get("attachments")
        if not isinstance(attachments_data, dict):
            continue
        for group, files in attachments_data.items():
            if not isinstance(files, list):
                continue
            for item in files:
                if not isinstance(item, dict):
                    continue
                rel = _norm_rel(str(item.get("path") or ""))
                if not rel or rel in seen:
                    continue
                if os.path.splitext(rel)[1].lower() not in IMAGE_EXTS:
                    continue
                seen.add(rel)
                records.append(ImageRecord(
                    folder_id    = extract_folder_id(rel),
                    claimid      = str(entry.get("claimid", "")),
                    attach_group = str(group),
                    filename     = item.get("name") or os.path.basename(rel),
                    rel_path     = rel,
                    full_path    = rel,
                    file_size    = int(item.get("size") or 0),
                ))
    records.sort(key=lambda r: r.rel_path)
    return records


def process_one_bytes(rec: dict, data: bytes) -> dict:
    if not data:
        rec["error"] = "empty_object"
        return rec
    if not rec.get("file_size"):
        rec["file_size"] = len(data)

    skip = is_blank_image_bytes(data, rec.get("file_size", 0))
    if skip:
        rec["skipped"] = skip
        return rec

    skip = is_xray_image_bytes(data)
    if skip:
        rec["skipped"] = skip
        return rec

    try:
        rec["file_sha256"] = _file_sha256_bytes(data)
        rec["pixel_sha256"], rec["img_size"] = _pixel_sha256_bytes(data)
    except Exception as e:
        rec["error"] = str(e)
    return rec


def hash_all_from_fetcher(
    records: list[dict],
    fetch_bytes: Callable[[str], bytes],
) -> tuple[list[dict], list[dict], list[dict]]:
    ok, err, skipped = [], [], []
    for r in tqdm(records, desc="MinIO hashing", unit="img"):
        key = _norm_rel(r.get("rel_path") or "")
        try:
            data = fetch_bytes(key)
        except Exception as e:
            r["error"] = f"minio_fetch_failed: {e}"
            err.append(r)
            continue
        res = process_one_bytes(r, data)
        if res.get("error"):
            err.append(res)
        elif res.get("skipped"):
            skipped.append(res)
        else:
            ok.append(res)
    return ok, err, skipped


def publish_results(
    ok: list[dict],
    err: list[dict],
    skipped: list[dict],
    json_path: str | None = None,
    claims: list[dict] | None = None,
) -> None:
    if claims is not None:
        claim_index = claim_index_from_claims(claims)
    elif json_path:
        claim_index = load_claim_index(json_path)
    else:
        claim_index = {}
    matched = attach_claim_details(ok, claim_index)
    matched += attach_claim_details(err, claim_index)
    matched += attach_claim_details(skipped, claim_index)
    log.info(f"Claim details found for {matched:,} images")

    n_blank = sum(1 for r in skipped if "blank_image" in str(r.get("skipped", "")))
    n_xray  = sum(1 for r in skipped if "xray_image" in str(r.get("skipped", "")))
    log.info(
        f"OK: {len(ok):,} | Errors: {len(err):,} | "
        f"Skipped: {len(skipped):,} (blank={n_blank:,}, xray={n_xray:,})"
    )

    log.info("Looking for duplicates...")
    df_pairs = find_duplicate_pairs(ok)
    if df_pairs.empty:
        log.info("No duplicate images found across folders.")
        return

    df_summary = build_folder_summary(df_pairs)
    vh = (df_pairs["risk_level"] == "VERY_HIGH").sum()
    hi = (df_pairs["risk_level"] == "HIGH").sum()
    folders = pd.concat([df_pairs["folder_id_1"], df_pairs["folder_id_2"]]).nunique()
    df_stats = pd.DataFrame([
        {"metric": "Images processed",                    "value": len(ok)},
        {"metric": "Filtered — blank page",               "value": n_blank},
        {"metric": "Filtered — X-ray",                    "value": n_xray},
        {"metric": "Errors / missing files",              "value": len(err)},
        {"metric": "Duplicate folder pairs",              "value": len(df_pairs)},
        {"metric": "VERY_HIGH — exact_file",              "value": int(vh)},
        {"metric": "HIGH — exact_pixel",                  "value": int(hi)},
        {"metric": "Folders involved",                    "value": int(folders)},
    ])
    saved = save_excel(df_pairs, df_summary, df_stats, err, skipped, excel_output_path())

    print("\n" + "=" * 60)
    print("  Duplicate image results")
    print("=" * 60)
    print(f"  Processed:                                 {len(ok):>8,}")
    print(f"  Filtered — blank page:                     {n_blank:>8,}")
    print(f"  Filtered — X-ray:                          {n_xray:>8,}")
    print(f"  Errors:                                    {len(err):>8,}")
    print(f"  -------------------------------------------------")
    print(f"  Duplicate folder pairs:                    {len(df_pairs):>8,}")
    print(f"  VERY_HIGH (exact_file):                    {vh:>8,}")
    print(f"  HIGH      (exact_pixel):                   {hi:>8,}")
    print(f"  Folders involved:                          {folders:>8,}")
    print(f"\n  Excel → {saved}")
    print("=" * 60)

    if not df_summary.empty:
        print("\n  Top 10 highest-risk folders:")
        print(f"  {'folder_id':<18} {'risk':<12} {'shared images':>14}  {'folders':>8}")
        print("  " + "─" * 56)
        for _, r in df_summary.head(10).iterrows():
            print(
                f"  {r['folder_id']:<18} {r['risk_level']:<12} "
                f"{r['total_shared_images']:>10}  {r['n_shared_with_folders']:>11}"
            )


def detect_from_json(json_path: str, image_root: str | None = None) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        claims = json.load(f)
    records = records_from_claims(claims, image_root)
    if not records:
        log.info("No images in the JSON to process.")
        return

    cache = load_cache(CACHE_FILE)
    ok = list(cache["ok"]) if cache else []
    err = list(cache["err"]) if cache else []
    skipped = list(cache["skipped"]) if cache else []
    ok, err, skipped = append_new_images([asdict(rec) for rec in records], ok, err, skipped)
    publish_results(ok, err, skipped, json_path=json_path, claims=claims)


def detect_from_claims(
    claims: list[dict],
    image_root: str,
) -> None:
    """Detect duplicates from in-memory claims using images already saved under image_root."""
    records = records_from_claims(claims, image_root)
    if not records:
        log.info("No images in claims to process.")
        return

    cache = load_cache(CACHE_FILE)
    ok = list(cache["ok"]) if cache else []
    err = list(cache["err"]) if cache else []
    skipped = list(cache["skipped"]) if cache else []
    ok, err, skipped = append_new_images([asdict(rec) for rec in records], ok, err, skipped)
    publish_results(ok, err, skipped, claims=claims)


def _fingerprint_dicts(records: list[dict]) -> str:
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda item: item.get("rel_path", "")):
        h.update(str(rec.get("rel_path", "")).encode())
        h.update(str(rec.get("file_size", 0)).encode())
        h.update(b"\n")
    return h.hexdigest()


# ─── main ──────────────────────────────────────────────────
def main():
    records = scan_download_images(DOWNLOAD_DIR)
    if not records:
        log.error("No images in TD/download to process.")
        return
    records_dicts = [asdict(r) for r in records]
    cache = load_cache(CACHE_FILE)
    ok = list(cache["ok"]) if cache else []
    err = list(cache["err"]) if cache else []
    skipped = list(cache["skipped"]) if cache else []
    ok, err, skipped = append_new_images(records_dicts, ok, err, skipped)
    publish_results(ok, err, skipped, json_path=JSON_PATH)


if __name__ == "__main__":
    main()