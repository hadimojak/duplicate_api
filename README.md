# Duplicate image detection

`dup.py` scans images under `TD/download`, hashes each file and its pixels, skips blank pages and X-ray images, then writes an Excel workbook of duplicate pairs, suspicious folders, stats, and skips. Each image id is written once in the filename fields. Claim id and attachment group come from `all_claims.json`, matched by the image path.

## Run

Use Python 3.14. PyTorch does not publish a build for Python 3.15, which is why `pip install` fails on the default interpreter.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python dup.py
```

Paths are set at the top of `dup.py`:

| Setting | Role |
|---|---|
| `BASE_DIR` | Project folder. Images are read from `TD/download` inside it |
| `JSON_PATH` | Claims JSON used to look up claim id and attachment group for each scanned file |
| `EXCEL_DIR` | Folder `excel output`. Each run writes `exact_duplicate_pairs_YYYYMMDD_HHMMSS.xlsx` |
| `CACHE_FILE` | Pickle cache so a rerun can skip hashing |

GPU hashing is used when CUDA is available. Otherwise the script hashes on CPU.

## Scheduled job

`job.py` checks Postgres every 5 minutes. It reads `TreatmentDamages` rows whose `Id` is greater than the last saved id, writes only those rows to `all_claims_new.json`, downloads their images from MinIO into `temp_images/TD/download`, runs duplicate detection, then deletes those temporary images. The checkpoint is `job_state.json`. A row is marked done only after that round finishes, so a failed round is tried again.

```powershell
.\.venv\Scripts\python.exe job.py
```

Leave that process running. Postgres is `127.0.0.1:5432`, database `duplicate_fraud`, user `postgres`, password `postgres`.
