# Duplicate image detection

`dup.py` scans images under `TD/download`, hashes each file and its pixels, skips blank pages and X-ray images, then writes an Excel workbook of pairs that appear in different folders. Claim id and attachment group come from `all_claims.json`, matched by the image path.

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
| `OUT_EXCEL` | Workbook of duplicate pairs, suspicious folders, stats, and skips |
| `CACHE_FILE` | Pickle cache so a rerun can skip hashing |

GPU hashing is used when CUDA is available. Otherwise the script hashes on CPU.
