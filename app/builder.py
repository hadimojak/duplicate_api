from __future__ import annotations

import json

from app.config import DATASET_JSON
from app.hashing import process_record
from app.parser import (
    extract_items,
    make_record,
)
from app.repository import (
    HashRepository,
    install_repository,
)


# =============================================================================
# DATASET BUILD
#
# Heavy processing is done completely outside CACHE_LOCK.
# Only the final repository replacement + cache write uses CACHE_LOCK.
# =============================================================================

def build_repository_from_json() -> tuple[
    HashRepository,
    int,
    int,
]:

    if not DATASET_JSON.exists():

        raise FileNotFoundError(
            f"Dataset یافت نشد: "
            f"{DATASET_JSON}"
        )

    with DATASET_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(file)

    items = extract_items(
        payload
    )

    new_repository = (
        HashRepository()
    )

    loaded = 0
    skipped = 0

    for index, item in enumerate(
        items
    ):

        record = make_record(
            item,
            index,
        )

        if record is None:

            skipped += 1
            continue

        processed = process_record(
            record
        )

        new_repository.add(
            processed
        )

        loaded += 1

    return (
        new_repository,
        loaded,
        skipped,
    )


def rebuild_repository_sync() -> tuple[
    int,
    int,
    int,
]:

    (
        new_repository,
        loaded,
        skipped,
    ) = build_repository_from_json()

    install_repository(
        new_repository
    )

    total = len(
        new_repository.records
    )

    return (
        loaded,
        skipped,
        total,
    )
