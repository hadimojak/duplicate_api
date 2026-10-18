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
BASE_DIR     = r"C:\Users\day\Documents\duplicate_fraud_api"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "TD", "download")
JSON_PATH    = r"C:\Users\day\Documents\duplicate_fraud_api\all_claims.json"
OUT_EXCEL    = r"C:\Users\day\Documents\duplicate_fraud_api\exact_all_duplicate_pairs.xlsx"
CACHE_FILE   = r"C:\Users\day\Documents\duplicate_fraud_api\hash_cache_all.pkl"
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
def load_cache(cache_path: str, fingerprint: str) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        checks = [
            cache.get("json_fingerprint")  == fingerprint,
            cache.get("blank_var_thresh")  == BLANK_VAR_THRESH,
            cache.get("blank_ent_thresh")  == BLANK_ENTROPY_THRESH,
            cache.get("xray_max_mean")     == XRAY_MAX_MEAN,
            cache.get("xray_min_std")      == XRAY_MIN_STD,
            cache.get("xray_max_color")    == XRAY_MAX_COLOR_DIFF,
            cache.get("cache_version")     == _CACHE_VERSION,
        ]
        if not all(checks):
            log.info("Cache بی‌اعتبار → هش‌گذاری مجدد")
            return None
        log.info(f"Cache معتبر — {len(cache['ok']):,} رکورد")
        return cache
    except Exception as e:
        log.warning(f"Cache خوانده نشد: {e}")
        return None


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
    log.info(f"Cache ذخیره → {cache_path}")


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
        log.error(f"پوشه تصاویر پیدا نشد: {download_dir}")
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
    log.info(f"TD/download: {len(records):,} تصویر")
    return records


def load_claim_index(json_path: str) -> dict[str, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: dict[str, dict] = {}
    for entry in data:
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
    log.info(f"JSON: {len(index):,} مسیر")
    return index


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


def _files_fingerprint(records: list[ImageRecord]) -> str:
    h = hashlib.sha256()
    for rec in records:
        h.update(rec.rel_path.encode())
        h.update(str(rec.file_size).encode())
        h.update(b"\n")
    return h.hexdigest()


# ─── تشخیص برگه سفید ──────────────────────────────────────
def _image_entropy(arr: np.ndarray) -> float:
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def is_blank_image(path: str, file_size: int) -> str:
    real_size = file_size if file_size > 0 else 0
    try:
        real_size = real_size or os.path.getsize(path)
    except OSError:
        return ""
    if real_size > MAX_FILE_BYTES_FOR_BLANK_CHECK:
        return ""
    try:
        with Image.open(path) as img:
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


# ─── تشخیص رادیولوژی ──────────────────────────────────────
def is_xray_image(path: str) -> str:
    try:
        with Image.open(path) as img:
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


# ─── توابع هش ─────────────────────────────────────────────
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixel_sha256_cpu(path: str) -> tuple[str, str]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        thumb = img.copy()
        thumb.thumbnail((PIXEL_THUMB, PIXEL_THUMB), Image.LANCZOS)
        digest = hashlib.sha256(
            thumb.tobytes() + f"{w}x{h}".encode()
        ).hexdigest()
    return digest, f"{w}x{h}"


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
        2: ("VERY_HIGH", "فایل خام یکسان بین پرونده‌های مختلف"),
        1: ("HIGH",      "محتوای پیکسلی یکسان — فرمت متفاوت"),
    }
    rows = []
    for entry in folder_pairs.values():
        risk_level, risk_reason = RISK_MAP.get(
            entry["best_dup_priority"], ("MEDIUM", "شباهت محتوا")
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


def save_excel(df_pairs, df_summary, df_stats, errors, skipped, path):
    def _write(p):
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            df_pairs[PAIR_COLS].to_excel(
                w, sheet_name="duplicate_pairs",    index=False)
            df_summary.to_excel(
                w, sheet_name="suspicious_folders", index=False)
            df_pairs[df_pairs["risk_level"] == "VERY_HIGH"][PAIR_COLS].to_excel(
                w, sheet_name="VERY_HIGH_only",     index=False)
            df_stats.to_excel(
                w, sheet_name="statistics",         index=False)
            if errors:
                pd.DataFrame(errors).to_excel(
                    w, sheet_name="errors",         index=False)
            if skipped:
                df_skip = pd.DataFrame(skipped)[[
                    "folder_id", "claimid", "filename",
                    "rel_path", "file_size", "skipped"
                ]].copy()
                df_skip["skip_type"] = df_skip["skipped"].apply(
                    lambda s: "xray" if "xray_image" in str(s) else "blank"
                )
                df_skip.to_excel(
                    w, sheet_name="skipped_images", index=False)

                df_xray = df_skip[df_skip["skip_type"] == "xray"]
                if not df_xray.empty:
                    df_xray.to_excel(
                        w, sheet_name="skipped_xray", index=False)

    try:
        _write(path)
    except PermissionError:
        alt = f"{os.path.splitext(path)[0]}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        _write(alt)
        log.warning(f"فایل قفل بود → {alt}")
        path = alt
    log.info(f"Excel ذخیره → {path}")


# ─── main ──────────────────────────────────────────────────
def main():
    records = scan_download_images(DOWNLOAD_DIR)
    if not records:
        log.error("تصویری در TD/download برای پردازش نیست.")
        return
    records_dicts = [asdict(r) for r in records]

    fingerprint = _files_fingerprint(records)
    cache       = load_cache(CACHE_FILE, fingerprint)

    if cache:
        ok, err, skipped = cache["ok"], cache["err"], cache["skipped"]
    else:
        gpu_ok = _TORCH_OK and torch.cuda.is_available()
        if gpu_ok:
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            ok, err, skipped = hash_all_gpu(records_dicts, torch.device("cuda"))
        else:
            log.info(f"CPU: {MAX_WORKERS} workers")
            ok, err, skipped = hash_all_cpu(records_dicts)
        save_cache(CACHE_FILE, fingerprint, ok, err, skipped)

    claim_index = load_claim_index(JSON_PATH)
    matched = attach_claim_details(ok, claim_index)
    matched += attach_claim_details(err, claim_index)
    matched += attach_claim_details(skipped, claim_index)
    log.info(f"جزئیات JSON برای {matched:,} تصویر از {len(records):,} پیدا شد")

    n_blank = sum(1 for r in skipped if "blank_image" in str(r.get("skipped","")))
    n_xray  = sum(1 for r in skipped if "xray_image"  in str(r.get("skipped","")))
    log.info(f"OK: {len(ok):,} | Errors: {len(err):,} | "
             f"Skipped: {len(skipped):,} (blank={n_blank:,}, xray={n_xray:,})")

    log.info("تشخیص تکراری‌ها...")
    df_pairs = find_duplicate_pairs(ok)

    if df_pairs.empty:
        log.info("هیچ تصویر تکراری بین پرونده‌ای پیدا نشد.")
        return

    df_summary = build_folder_summary(df_pairs)

    vh      = (df_pairs["risk_level"] == "VERY_HIGH").sum()
    hi      = (df_pairs["risk_level"] == "HIGH").sum()
    folders = pd.concat(
        [df_pairs["folder_id_1"], df_pairs["folder_id_2"]]
    ).nunique()

    df_stats = pd.DataFrame([
        {"metric": "کل تصاویر پردازش‌شده",               "value": len(ok)},
        {"metric": "فیلترشده — برگه سفید",                "value": n_blank},
        {"metric": "فیلترشده — رادیولوژی/اشعه ایکس",     "value": n_xray},
        {"metric": "خطا / فایل گم‌شده",                  "value": len(err)},
        {"metric": "جفت پرونده‌های تکراری (بدون تکرار)", "value": len(df_pairs)},
        {"metric": "VERY_HIGH — exact_file",               "value": int(vh)},
        {"metric": "HIGH — exact_pixel",                   "value": int(hi)},
        {"metric": "پرونده‌های منحصربه‌فرد درگیر",        "value": int(folders)},
    ])

    save_excel(df_pairs, df_summary, df_stats, err, skipped, OUT_EXCEL)

    print("\n" + "═" * 60)
    print("  نتایج تشخیص تصاویر تکراری")
    print("═" * 60)
    print(f"  کل پردازش‌شده:                            {len(ok):>8,}")
    print(f"  فیلتر — برگه سفید:                        {n_blank:>8,}")
    print(f"  فیلتر — رادیولوژی:                        {n_xray:>8,}")
    print(f"  خطا:                                       {len(err):>8,}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  جفت پرونده تکراری (هر جفت یک ردیف):      {len(df_pairs):>8,}")
    print(f"  VERY_HIGH (exact_file):                    {vh:>8,}")
    print(f"  HIGH      (exact_pixel):                   {hi:>8,}")
    print(f"  پرونده‌های درگیر:                          {folders:>8,}")
    print(f"\n  اکسل → {OUT_EXCEL}")
    print("═" * 60)

    if not df_summary.empty:
        print("\n  ۱۰ پرونده پرریسک‌ترین:")
        print(f"  {'folder_id':<18} {'risk':<12} {'عکس مشترک':>10}  {'با N پرونده':>11}")
        print("  " + "─" * 56)
        for _, r in df_summary.head(10).iterrows():
            print(f"  {r['folder_id']:<18} {r['risk_level']:<12} "
                  f"{r['total_shared_images']:>10}  {r['n_shared_with_folders']:>11}")


if __name__ == "__main__":
    main()