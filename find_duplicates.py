from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
CACHE_PKL = Path(
    os.getenv("CACHE_PKL", str(BASE_DIR / "hash_cache_all.pkl"))
).resolve()
OUTPUT_JSON = Path(
    os.getenv("OUTPUT_JSON", str(BASE_DIR / "duplicates.json"))
).resolve()


def load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"ERROR: Cache not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("rb") as f:
        return pickle.load(f)


def find_duplicate_groups(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    active = [r for r in records if not r.get("excluded")]

    by_file_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pixel_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in active:
        if r.get("file_sha256"):
            by_file_hash[r["file_sha256"]].append(r)
        if r.get("pixel_sha256"):
            by_pixel_hash[r["pixel_sha256"]].append(r)

    groups: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for hash_val, items in by_file_hash.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["folder_id"] == b["folder_id"]:
                    continue
                pair_key = tuple(sorted([a["record_id"], b["record_id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                groups.append({
                    "duplicate_type": "exact_file",
                    "risk_level": "VERY_HIGH",
                    "hash": hash_val,
                    "records": [a, b],
                })

    for hash_val, items in by_pixel_hash.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["folder_id"] == b["folder_id"]:
                    continue
                pair_key = tuple(sorted([a["record_id"], b["record_id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                groups.append({
                    "duplicate_type": "exact_pixel",
                    "risk_level": "HIGH",
                    "hash": hash_val,
                    "records": [a, b],
                })

    groups.sort(key=lambda g: (0 if g["risk_level"] == "VERY_HIGH" else 1, g["hash"]))

    return groups


def build_summary(
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(records)
    excluded = sum(1 for r in records if r.get("excluded"))
    active = total - excluded

    very_high = sum(1 for g in groups if g["risk_level"] == "VERY_HIGH")
    high = sum(1 for g in groups if g["risk_level"] == "HIGH")

    involved_folders: set[str] = set()
    involved_claims: set[str] = set()
    for g in groups:
        for r in g["records"]:
            involved_folders.add(r.get("folder_id", "UNKNOWN"))
            if r.get("claimid"):
                involved_claims.add(str(r["claimid"]))

    return {
        "total_records": total,
        "active_records": active,
        "excluded_records": excluded,
        "duplicate_groups": len(groups),
        "very_high_risk": very_high,
        "high_risk": high,
        "unique_folders_involved": len(involved_folders),
        "unique_claims_involved": len(involved_claims),
    }


def main() -> None:
    print(f"Loading cache: {CACHE_PKL}")
    records = load_cache(CACHE_PKL)
    print(f"Loaded {len(records)} records")

    groups = find_duplicate_groups(records)
    summary = build_summary(records, groups)

    print(f"\nResults:")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Active: {summary['active_records']}")
    print(f"  Excluded: {summary['excluded_records']}")
    print(f"  Duplicate groups: {summary['duplicate_groups']}")
    print(f"    VERY_HIGH (exact file): {summary['very_high_risk']}")
    print(f"    HIGH (exact pixel): {summary['high_risk']}")
    print(f"  Folders involved: {summary['unique_folders_involved']}")
    print(f"  Claims involved: {summary['unique_claims_involved']}")

    output = {
        "summary": summary,
        "duplicates": groups,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
