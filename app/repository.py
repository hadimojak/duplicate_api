from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict

from app.config import (
    CACHE_LOCK,
    CACHE_PATH,
    CACHE_VERSION,
)
from app.models import ImageRecord


# =============================================================================
# HASH REPOSITORY
# =============================================================================

class HashRepository:

    def __init__(self) -> None:

        self.records: dict[
            str,
            ImageRecord,
        ] = {}

        self.by_file_hash: dict[
            str,
            list[str],
        ] = defaultdict(list)

        self.by_pixel_hash: dict[
            str,
            list[str],
        ] = defaultdict(list)

    def clear(self) -> None:

        self.records.clear()
        self.by_file_hash.clear()
        self.by_pixel_hash.clear()

    def _append_unique(
        self,
        mapping: dict[str, list[str]],
        key: str,
        record_id: str,
    ) -> None:

        values = mapping[key]

        if record_id not in values:
            values.append(record_id)

    def add(
        self,
        record: ImageRecord,
    ) -> None:

        old = self.records.get(
            record.record_id
        )

        if old is not None:
            self.remove(
                old.record_id
            )

        self.records[
            record.record_id
        ] = record

        if record.excluded:
            return

        if record.file_sha256:

            self._append_unique(
                self.by_file_hash,
                record.file_sha256,
                record.record_id,
            )

        if record.pixel_sha256:

            self._append_unique(
                self.by_pixel_hash,
                record.pixel_sha256,
                record.record_id,
            )

    def remove(
        self,
        record_id: str,
    ) -> None:

        record = self.records.pop(
            record_id,
            None,
        )

        if record is None:
            return

        if record.file_sha256:

            values = self.by_file_hash.get(
                record.file_sha256
            )

            if values is not None:

                if record_id in values:
                    values.remove(record_id)

                if not values:

                    del self.by_file_hash[
                        record.file_sha256
                    ]

        if record.pixel_sha256:

            values = self.by_pixel_hash.get(
                record.pixel_sha256
            )

            if values is not None:

                if record_id in values:
                    values.remove(record_id)

                if not values:

                    del self.by_pixel_hash[
                        record.pixel_sha256
                    ]

    def get(
        self,
        record_id: str,
    ) -> ImageRecord | None:

        return self.records.get(
            record_id
        )

    def all(self) -> list[ImageRecord]:

        return list(
            self.records.values()
        )

    def candidates_for(
        self,
        query: ImageRecord,
    ) -> list[ImageRecord]:

        ids: set[str] = set()

        if query.file_sha256:

            ids.update(
                self.by_file_hash.get(
                    query.file_sha256,
                    [],
                )
            )

        if query.pixel_sha256:

            ids.update(
                self.by_pixel_hash.get(
                    query.pixel_sha256,
                    [],
                )
            )

        return [
            self.records[record_id]
            for record_id in ids
            if record_id in self.records
        ]

    def save(self) -> None:

        payload = {
            "cache_version": CACHE_VERSION,
            "records": [
                record.to_dict()
                for record in self.records.values()
            ],
        }

        temporary = CACHE_PATH.with_name(
            CACHE_PATH.name
            + ".tmp."
            + uuid.uuid4().hex
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

        temporary.replace(
            CACHE_PATH
        )

    def load(self) -> bool:

        if not CACHE_PATH.exists():
            return False

        with CACHE_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(file)

        if not isinstance(
            payload,
            dict,
        ):
            return False

        if payload.get(
            "cache_version"
        ) != CACHE_VERSION:

            return False

        records = payload.get(
            "records",
            [],
        )

        if not isinstance(
            records,
            list,
        ):
            return False

        self.clear()

        for item in records:

            if not isinstance(
                item,
                dict,
            ):
                continue

            try:

                record = ImageRecord(
                    record_id=str(
                        item["record_id"]
                    ),
                    folder_id=str(
                        item["folder_id"]
                    ),
                    claimid=item.get(
                        "claimid"
                    ),
                    full_path=str(
                        item["full_path"]
                    ),
                    filename=str(
                        item["filename"]
                    ),
                    attach_group=item.get(
                        "attach_group"
                    ),
                    file_size=item.get(
                        "file_size"
                    ),
                    file_sha256=item.get(
                        "file_sha256"
                    ),
                    pixel_sha256=item.get(
                        "pixel_sha256"
                    ),
                    image_size=item.get(
                        "image_size"
                    ),
                    is_blank=bool(
                        item.get(
                            "is_blank",
                            False,
                        )
                    ),
                    is_xray=bool(
                        item.get(
                            "is_xray",
                            False,
                        )
                    ),
                    excluded=bool(
                        item.get(
                            "excluded",
                            False,
                        )
                    ),
                    exclusion_reason=item.get(
                        "exclusion_reason"
                    ),
                    risk_level=item.get(
                        "risk_level"
                    ),
                    duplicate_type=item.get(
                        "duplicate_type"
                    ),
                    duplicate_with=item.get(
                        "duplicate_with"
                    ),
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):

                continue

            self.add(
                record
            )

        return True


# =============================================================================
# ACTIVE REPOSITORY
# =============================================================================

REPOSITORY = HashRepository()


def get_repository() -> HashRepository:
    return REPOSITORY


def install_repository(
    new_repository: HashRepository,
) -> None:

    global REPOSITORY

    with CACHE_LOCK:

        new_repository.save()

        REPOSITORY = (
            new_repository
        )
