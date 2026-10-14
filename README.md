# Claim Image Duplicate Detection API

Fraud duplicate detection for medical claim images, based on `داکیومنت کد تکراری ها.pdf`.

Detects reused images across different claim folders (`folder_id`) using dual hashing + pre-filters, exposed as FastAPI v4.0.0.

## 1. How It Works

### Detection logic

* `exact_file` — `SHA256` of whole file. 100% identical bytes. Risk `VERY_HIGH`.
* `exact_pixel` — resize + `SHA256` of normalized pixels (`_pixel_sha256_gpu` / `_pixel_sha256_cpu` in `all_data.py`, called from `app/hashing.py`). Same visual content despite format/size/metadata change. Risk `HIGH`.
* Pre-filters (run before hashing in `process_record()`):
  * `is_blank_image(path, file_size)` → `BLANK_IMAGE` → `excluded=true`
  * `is_xray_image(path)` → `XRAY_IMAGE` → `excluded=true`
  * Missing file → `FILE_NOT_FOUND`, decode/hash failure → `PROCESSING_ERROR:...`
* Comparison rule (`app/comparison.py:compare_record`):
  * Only across different `folder_id`s. Same folder is skipped.
  * Skips `excluded` candidates.
  * `file_sha256` match wins over `pixel_sha256` match.
  * Sorted `VERY_HIGH` first, then by `folder_id`, `filename`.

Original batch script: parallel `ProcessPoolExecutor` + optional `torch` GPU + Excel output (`exact pairs / suspicious / stats / errors`) + `hash_cache_all.pkl` for resume.

Current API: same hashing/filter logic, sequential build + in-memory `HashRepository` + JSON responses, no Excel.

### Pipeline

```
DATASET_JSON (all_claims.json)
  -> build_repository_from_json() [app/builder.py]
  -> make_record() [app/parser.py]
  -> process_record() [app/hashing.py: blank -> xray -> file_sha256 -> pixel_sha256]
  -> HashRepository.add() [by_file_hash / by_pixel_hash indexes]
  -> compare_record() / create_summary() [app/comparison.py]
```

## 2. Requirements

```
pandas==2.3.2
tqdm==4.67.1
Pillow==11.3.0
openpyxl==3.1.5
torch
torchvision
fastapi>=0.115,<1
uvicorn[standard]>=0.30,<1
python-multipart>=0.0.9,<1
```

Plus `all_data.py` (provides `_file_sha256`, `_pixel_sha256_cpu`, `_pixel_sha256_gpu`, `is_blank_image`, `is_xray_image`, `DEVICE`).

Stdlib: `os, json, hashlib, logging, pickle, datetime, collections, concurrent.futures, dataclasses, warnings`.

## 3. Install & Run

```powershell
pip install -r requirements.txt
python api.py
# uvicorn api:app --host 0.0.0.0 --port 8000
```

Docs: `http://localhost:8000/docs`

## 4. Configuration

Env vars (see `app/config.py`):

| Var | Default | Description |
|---|---|---|
| `BASE_DIR` | `.` | Repo root |
| `DATASET_JSON` | `./all_claims.json` | Input claims list `{folder_id, claimid, full_path, filename, attach_group}` |
| `CACHE_PATH` | `./api_cache.json` | Persisted repository (replaces old `hash_cache_all.pkl`) |
| `UPLOAD_DIR` | `./api_uploads` | Temp uploads |
| `MAX_IMAGE_SIZE` | `52428800` | 50 MB reject limit |
| `MAX_BATCH_SIZE` | `500` | Max files for `/v1/images/compare` |
| `USE_GPU` | `true` | Use `_pixel_sha256_gpu` if available, fallback CPU |
| `PORT` | `8000` | `api.py` listen port |

## 5. Usage Workflow

### 1) Health check

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/
```

Returns `repository_records, active_records, dataset_json, cache_path, use_gpu, gpu_function_available`.

### 2) Load dataset

Builds hashes for all images in `DATASET_JSON`:

```powershell
curl -X POST http://localhost:8000/v1/dataset/load
curl -X POST http://localhost:8000/v1/repository/rebuild
```

Response: `{loaded, skipped, total_repository_records}`.

### 3) Check stats

```powershell
curl http://localhost:8000/v1/dataset/stats
```

Returns `{total, active, excluded, blank, xray, processing_errors, file_hash_groups, pixel_hash_groups}`.

### 4) Compare new image(s) — no insert

```powershell
# single
curl -X POST "http://localhost:8000/v1/image/compare?folder_id=FOLDER123&claimid=CLM1" -F "file=@./test.jpg"

# batch (up to 500)
curl -X POST "http://localhost:8000/v1/images/compare?folder_id=FOLDER123" -F "files=@./a.jpg" -F "files=@./b.jpg"
```

Success response:

```json
{
  "ok": true,
  "query": {"folder_id": "...", "file_sha256": "...", "pixel_sha256": "...", "is_blank": false, "is_xray": false, "excluded": false},
  "matches": [{"risk_level": "VERY_HIGH", "duplicate_type": "exact_file", "matched_record": {...}}],
  "summary": {"total_matches": 1, "very_high_count": 1, "high_count": 0, "exact_file_count": 1, "exact_pixel_count": 0, "matched_folders": ["..."]}
}
```

If `query.excluded=true` (`BLANK_IMAGE` / `XRAY_IMAGE` / `PROCESSING_ERROR`), `matches=[]` by design.

### 5) Add to repository

```powershell
curl -X POST "http://localhost:8000/v1/repository/add?folder_id=FOLDER123&claimid=CLM1" -F "file=@./new.jpg"
curl -X DELETE http://localhost:8000/v1/repository/record-xxx
```

Returns `record + matches_before_insert + summary`.

## 6. Reference Performance (from doc)

| Metric | Total time (s) | Speed (imgs/s) |
|---|---|---|
| TOTAL processing | 141219.41 | 12.55 |
| hash_gpu | 141091.16 | 12.56 |
| hash.xray_filter | 102957.92 | 17.41 |
| hash.pixel_sha256_gpu_wall | 33998.20 | 52.14 |
| hash.file_sha256 | 1842.98 | 961.66 |

`file_sha256` is ~50x faster than pixel hash — blank/xray filters exist to skip expensive GPU work.

## 7. Project Structure

```
api.py                  # uvicorn entry
api_original.py         # legacy
all_claims.json         # DATASET_JSON input
hash_cache_all.pkl      # legacy pickle cache (ok/err/skipped)
app/main.py             # FastAPI app + lifespan cache load
app/config.py           # env config
app/builder.py          # build_repository_from_json / rebuild_repository_sync
app/hashing.py          # calculate_pixel_hash / process_record
app/comparison.py       # compare_record / create_summary / public_record
app/repository.py       # HashRepository + get_repository
app/repository_service.py
app/parser.py           # extract_items / make_record
app/uploads.py          # save_upload / process_uploaded_image
app/validators.py       # require_folder_id
app/models.py           # ImageRecord + CompareResponse
app/routers/system.py       # GET / , /health
app/routers/dataset.py      # POST /v1/dataset/load, GET /v1/dataset/stats
app/routers/compare.py      # POST /v1/image/compare, /v1/images/compare
app/routers/repository.py   # POST /v1/repository/add, DELETE, POST rebuild
```
