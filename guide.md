# Hash Cache Generation Guide

## What It Does

`hash_cache.py` reads `all_claims_new.json`, computes SHA256 hashes for every image, and saves results to `hash_cache_all.pkl` (same format as `hash_cache_all_old.pkl`).

## Run

If `python` is not in PATH, find it first:

```powershell
Get-ChildItem "C:\Users\day\AppData\Local\Programs\Python" -Recurse -Filter "python.exe" | Select-Object FullName
```

Then run with the full path:

```powershell
& "C:\Path\To\python.exe" hash_cache.py
```

## Output

- **File:** `hash_cache_all.pkl`
- **Format:** Pickled list of dicts, each containing:
  - `record_id`, `folder_id`, `claimid`, `filename`, `full_path`
  - `file_size`, `image_size`, `file_sha256`, `pixel_sha256`
  - `is_blank`, `is_xray`, `excluded`, `exclusion_reason`

## Requirements

Install Pillow first:

```powershell
& "C:\Path\To\python.exe" -m pip install Pillow
```

## Env Vars

| Var | Default | Description |
|---|---|---|
| `BASE_DIR` | `.` | Repo root |
| `DATASET_JSON` | `./all_claims_new.json` | Input claims list |
| `OUTPUT_PKL` | `./hash_cache_all.pkl` | Output pickle path |

---

# Duplicate Finder Guide

## What It Does

`find_duplicates.py` loads `hash_cache_all.pkl`, groups images by `file_sha256` (exact file duplicates) and `pixel_sha256` (same pixels, different file), and outputs all cross-folder duplicate pairs to `duplicates.json`.

## Run

```powershell
& "C:\Path\To\python.exe" find_duplicates.py
```

## Output

- **File:** `duplicates.json`
- **Format:**
  ```json
  {
    "summary": { "duplicate_groups": 0, "very_high_risk": 0, ... },
    "duplicates": [
      {
        "duplicate_type": "exact_file",
        "risk_level": "VERY_HIGH",
        "hash": "<sha256>",
        "records": [ { ... }, { ... } ]
      }
    ]
  }
  ```

## Duplicate Types

| Type | Risk | Meaning |
|---|---|---|
| `exact_file` | VERY_HIGH | Identical file bytes (same `file_sha256`) |
| `exact_pixel` | HIGH | Same pixel content (same `pixel_sha256`), different file encoding |

## Env Vars

| Var | Default | Description |
|---|---|---|
| `BASE_DIR` | `.` | Repo root |
| `CACHE_PKL` | `./hash_cache_all.pkl` | Input pickle path |
| `OUTPUT_JSON` | `./duplicates.json` | Output JSON path |

---

# Excel Export Guide

## What It Does

`export_excel.py` reads `duplicates.json` and writes `duplicates.csv` (Excel-compatible) with columns: `claimId1`, `claimId2`, `filename1`, `filename2`, `folder1`, `folder2`, `duplicate_type`, `risk_level`, `file_sha256`, `pixel_sha256`.

## Run

```powershell
& "C:\Path\To\python.exe" export_excel.py
```

## Output

- **File:** `duplicates.csv` (open directly in Excel)
- **Encoding:** UTF-8 with BOM (Excel-safe)

## Env Vars

| Var | Default | Description |
|---|---|---|
| `BASE_DIR` | `.` | Repo root |
| `INPUT_JSON` | `./duplicates.json` | Input JSON path |
| `OUTPUT_CSV` | `./duplicates.csv` | Output CSV path |
