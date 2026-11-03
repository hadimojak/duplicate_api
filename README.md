# Duplicate image detection

`dup.py` can still scan local images under `TD/download` for offline runs. The scheduled job path does not use claim JSON files or local image files.

## Run

Use Python 3.14. PyTorch does not publish a build for Python 3.15, which is why `pip install` fails on the default interpreter.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python dup.py
```

## Scheduled job

`job.py` checks Postgres every 5 minutes. It loads `TreatmentDamages` with `State = 10` and `Id > 1447005` that are not in `checked_ids`, builds claims in memory, downloads their images from MinIO into `temp_images/`, runs duplicate detection, then deletes those temp files and saves the ids in `job_state.json`. `min_id` stays fixed at `1447005`.

MinIO settings (env, with defaults):

```
MINIO_ENDPOINT=192.168.20.22
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=KKgWercFhZjtoIDnULTz
MINIO_SECRET_KEY=Flrr462el7E5cQFwo81HeA6x55zslRJnzboq7k2x
MINIO_BUCKET=policy-management
```

FTP upload (after Excel is created; panel backend can import from here):

```
FTP_ENABLED=true
FTP_HOST=your.ftp.host
FTP_PORT=21
FTP_USER=...
FTP_PASSWORD=...
FTP_DIR=/suspicious-duplicates
FTP_USE_TLS=false
```

Files are uploaded atomically (`*.uploading` then renamed). Set `FTP_ENABLED=false` or leave `FTP_HOST` empty to skip upload.

Local Excel copies are kept for now (`EXCEL_KEEP_LOCAL=true`). When you only want FTP, set:

```
EXCEL_KEEP_LOCAL=false
```

That deletes local Excel files after a successful FTP upload.

```powershell
.\.venv\Scripts\python.exe job.py
```
