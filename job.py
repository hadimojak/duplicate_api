from __future__ import annotations

import json
import logging
import os
import signal
import threading

import psycopg
from minio import Minio
from minio.error import S3Error
from psycopg.rows import dict_row

from dup import BASE_DIR, detect_from_json

JSON_PATH = os.path.join(BASE_DIR, "all_claims_new.json")
STATE_PATH = os.path.join(BASE_DIR, "job_state.json")
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
INTERVAL_SECONDS = 5 * 60

PG = dict(
    host="127.0.0.1",
    port=5432,
    dbname="duplicate_fraud",
    user="postgres",
    password="postgres",
)

MINIO = Minio(
    "192.168.20.22:9000",
    access_key="KKgWercFhZjtoIDnULTz",
    secret_key="Flrr462el7E5cQFwo81HeA6x55zslRJnzboq7k2x",
    secure=False,
)
BUCKET = "policy-management"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("job")
stop = threading.Event()


def request_stop(signum, _frame) -> None:
    if stop.is_set():
        log.warning("Forced exit")
        raise SystemExit(1)
    log.info("Shutdown requested; the current round will finish, then the job will stop")
    stop.set()


def load_last_id() -> int:
    if not os.path.exists(STATE_PATH):
        return 0
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return int(json.load(f).get("last_id", 0))


def save_last_id(last_id: int) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_id": last_id}, f)
    log.info("Saved last id: %s", last_id)


def fetch_new_rows(last_id: int) -> list[dict]:
    with psycopg.connect(**PG, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT "Id", "Properties"
                FROM "TreatmentDamages"
                WHERE "Id" > %s
                ORDER BY "Id"
                """,
                (last_id,),
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
                log.info("Saved temp image %s", local_path)
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


def write_new_claims(new_claims: list[dict]) -> None:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(new_claims, f, ensure_ascii=False)
    log.info("all_claims_new.json written from %s new TreatmentDamages rows", len(new_claims))


def run_once() -> None:
    last_id = load_last_id()
    rows = fetch_new_rows(last_id)
    if not rows:
        log.info("No new rows after id %s", last_id)
        return

    claims = [row_to_claim(row) for row in rows]
    log.info("Read %s rows, from id %s to %s", len(rows), rows[0]["Id"], rows[-1]["Id"])
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
    saved_paths: list[str] = []
    try:
        write_new_claims(claims)
        saved_paths = download_images(claims)
        detect_from_json(JSON_PATH, TEMP_DIR)
        save_last_id(int(rows[-1]["Id"]))
    finally:
        remove_temp_images(saved_paths)


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    log.info("Job started; running every %s seconds", INTERVAL_SECONDS)
    while not stop.is_set():
        try:
            run_once()
        except Exception:
            log.exception("This round failed; the last id was not advanced")
        if stop.is_set() or stop.wait(INTERVAL_SECONDS):
            break
    log.info("Job stopped")


if __name__ == "__main__":
    main()
