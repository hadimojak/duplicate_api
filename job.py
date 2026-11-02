from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time

import psycopg
from minio import Minio
from minio.error import S3Error
from psycopg.rows import dict_row

from dup import BASE_DIR, detect_from_claims

STATE_PATH = os.path.join(BASE_DIR, "job_state.json")
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
INTERVAL_SECONDS = 5 * 60
CHUNK_SIZE = 500
# Fixed floor: everything up to this id is already checked. Never advance this value.
MIN_ID = 1447005

PG = dict(
    host="192.168.20.41",
    port=6002,
    dbname="postgres",
    user="td_user",
    password="td_user_pass",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("job")
stop = threading.Event()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def minio_from_env() -> tuple[Minio, str]:
    endpoint = os.getenv("MINIO_ENDPOINT", "192.168.20.22").strip()
    port = os.getenv("MINIO_PORT", "9000").strip()
    secure = _env_bool("MINIO_USE_SSL", "false")
    access_key = os.getenv("MINIO_ACCESS_KEY", "KKgWercFhZjtoIDnULTz")
    secret_key = os.getenv("MINIO_SECRET_KEY", "Flrr462el7E5cQFwo81HeA6x55zslRJnzboq7k2x")
    bucket = os.getenv("MINIO_BUCKET", "policy-management")
    client = Minio(
        f"{endpoint}:{port}",
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
    return client, bucket


MINIO, BUCKET = minio_from_env()


def request_stop(signum, _frame) -> None:
    if stop.is_set():
        log.warning("Forced exit")
        raise SystemExit(1)
    log.info("Shutdown requested; the current round will finish, then the job will stop")
    stop.set()


def load_checked_ids() -> set[int]:
    if not os.path.exists(STATE_PATH):
        return set()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get("checked_ids")
    if isinstance(ids, list):
        return {int(x) for x in ids}
    return set()


def save_checked_ids(checked_ids: set[int]) -> None:
    # MIN_ID is written for clarity but never changed; progress is only checked_ids.
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"min_id": MIN_ID, "checked_ids": sorted(checked_ids)}, f)
    log.info("Saved checked ids: %s total (fixed min_id=%s)", len(checked_ids), MIN_ID)


def fetch_new_rows(checked_ids: set[int], limit: int = CHUNK_SIZE) -> list[dict]:
    """Fetch State=10 rows with Id > MIN_ID that are not already checked."""
    with psycopg.connect(**PG, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if checked_ids:
                cur.execute(
                    """
                    SELECT "Id", "Properties"
                    FROM "TreatmentDamages"
                    WHERE "State" = 10
                      AND "Id" > %s
                      AND NOT ("Id" = ANY(%s))
                    ORDER BY "Id"
                    LIMIT %s
                    """,
                    (MIN_ID, list(checked_ids), limit),
                )
            else:
                cur.execute(
                    """
                    SELECT "Id", "Properties"
                    FROM "TreatmentDamages"
                    WHERE "State" = 10
                      AND "Id" > %s
                    ORDER BY "Id"
                    LIMIT %s
                    """,
                    (MIN_ID, limit),
                )
            return list(cur.fetchall())


def row_to_claim(row: dict) -> dict:
    props = row["Properties"] or {}
    if isinstance(props, str):
        props = json.loads(props)
    attachments_in = props.get("Attachments") or {}
    attachments: dict[str, list[dict]] = {}
    if isinstance(attachments_in, dict):
        for group, files in attachments_in.items():
            if not isinstance(files, list):
                continue
            cleaned = []
            for item in files:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("Path") or item.get("path") or "").replace("\\", "/")
                if not path:
                    continue
                cleaned.append({
                    "name": item.get("Name") or item.get("name") or os.path.basename(path),
                    "path": path,
                    "size": item.get("Size") or item.get("size") or 0,
                })
            if cleaned:
                attachments[str(group)] = cleaned
    return {
        "treatmentDamageId": int(row["Id"]),
        "claimid": str(props.get("ClaimId") or ""),
        "servicecaption": props.get("ServiceCaption") or "",
        "servicegroupcaption": props.get("ServiceGroupCaption") or "",
        "prescriptiondate": str(props.get("PrescriptionDate") or ""),
        "attachments": attachments,
    }


def download_images(claims: list[dict]) -> list[str]:
    """Download claim attachment images from MinIO into TEMP_DIR."""
    saved: list[str] = []
    for claim in claims:
        for files in claim["attachments"].values():
            for item in files:
                key = str(item.get("path") or "").replace("\\", "/").lstrip("/")
                if not key or os.path.splitext(key)[1].lower() not in IMAGE_EXTS:
                    continue
                local_path = os.path.join(TEMP_DIR, key.replace("/", os.sep))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    MINIO.fget_object(BUCKET, key, local_path)
                except S3Error as exc:
                    log.error("MinIO download failed for %s: %s", key, exc)
                    raise
                saved.append(local_path)
                log.info(
                    "Saved temp image claimId=%s treatmentDamageId=%s path=%s",
                    claim.get("claimid"),
                    claim.get("treatmentDamageId"),
                    local_path,
                )
    return saved


def remove_temp_images(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
            log.info("Removed temp image %s", path)
        except OSError as exc:
            log.warning("Could not remove %s: %s", path, exc)
    for root, _dirs, _files in os.walk(TEMP_DIR, topdown=False):
        if root == TEMP_DIR:
            continue
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass


def log_documents(claims: list[dict]) -> None:
    for claim in claims:
        docs = [
            item.get("path", "")
            for files in claim["attachments"].values()
            for item in files
        ]
        if docs:
            for path in docs:
                log.info(
                    "Document from db: treatmentDamageId=%s claimId=%s path=%s",
                    claim["treatmentDamageId"],
                    claim["claimid"],
                    path,
                )
        else:
            log.info(
                "Document from db: treatmentDamageId=%s claimId=%s path=(none)",
                claim["treatmentDamageId"],
                claim["claimid"],
            )


def process_chunk(rows: list[dict], chunk_no: int, checked_ids: set[int]) -> None:
    claims = [row_to_claim(row) for row in rows]
    log.info(
        "Chunk %s: %s rows, from id %s to %s",
        chunk_no,
        len(rows),
        rows[0]["Id"],
        rows[-1]["Id"],
    )
    log_documents(claims)
    saved_paths: list[str] = []
    try:
        saved_paths = download_images(claims)
        detect_from_claims(claims, TEMP_DIR)
        for row in rows:
            checked_ids.add(int(row["Id"]))
        save_checked_ids(checked_ids)
    finally:
        remove_temp_images(saved_paths)


def run_once() -> None:
    checked_ids = load_checked_ids()
    chunk_no = 0
    total_rows = 0

    log.info(
        "Starting round with %s already-checked ids; State=10 and Id > %s only",
        len(checked_ids),
        MIN_ID,
    )

    while not stop.is_set():
        rows = fetch_new_rows(checked_ids, CHUNK_SIZE)
        if not rows:
            if total_rows == 0:
                log.info(
                    "No new State=10 rows to check (%s already checked)",
                    len(checked_ids),
                )
            else:
                log.info(
                    "All new State=10 rows processed in %s chunk(s), %s rows total",
                    chunk_no,
                    total_rows,
                )
            return

        chunk_no += 1
        process_chunk(rows, chunk_no, checked_ids)
        total_rows += len(rows)
        if len(rows) < CHUNK_SIZE:
            log.info(
                "All new State=10 rows processed in %s chunk(s), %s rows total",
                chunk_no,
                total_rows,
            )
            return

    log.info(
        "Shutdown during chunking after %s chunk(s), %s rows; remaining rows stay for the next run",
        chunk_no,
        total_rows,
    )


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    log.info(
        "Job started; MinIO %s bucket=%s; interval %ss; chunks of %s",
        f"{os.getenv('MINIO_ENDPOINT', '192.168.20.22')}:{os.getenv('MINIO_PORT', '9000')}",
        BUCKET,
        INTERVAL_SECONDS,
        CHUNK_SIZE,
    )
    while not stop.is_set():
        started = time.monotonic()
        log.info("Round started")
        try:
            run_once()
        except Exception:
            log.exception("This round failed; checked_ids were not updated")
        elapsed = time.monotonic() - started
        log.info("Round finished in %.1f seconds", elapsed)
        if stop.is_set():
            break
        log.info("Interval paused during work; waiting %s seconds until the next round", INTERVAL_SECONDS)
        if stop.wait(INTERVAL_SECONDS):
            break
    log.info("Job stopped")


if __name__ == "__main__":
    main()
