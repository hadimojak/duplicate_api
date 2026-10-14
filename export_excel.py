from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
INPUT_JSON = Path(
    os.getenv("INPUT_JSON", str(BASE_DIR / "duplicates.json"))
).resolve()
OUTPUT_CSV = Path(
    os.getenv("OUTPUT_CSV", str(BASE_DIR / "duplicates.csv"))
).resolve()


def load_duplicates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("duplicates", [])


def export_csv(groups: list[dict[str, Any]], output: Path) -> int:
    rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    paired: dict[tuple[str, str], dict[str, Any]] = {}

    for group in groups:
        recs = group.get("records", [])
        if len(recs) < 2:
            continue
        a, b = recs[0], recs[1]
        claim_a = str(a.get("claimid", ""))
        claim_b = str(b.get("claimid", ""))
        key = tuple(sorted([claim_a, claim_b]))

        if key not in paired:
            paired[key] = {
                "claimId1": key[0],
                "claimId2": key[1],
                "filenames1": [],
                "filenames2": [],
                "duplicate_types": set(),
                "risk_levels": set(),
            }

        entry = paired[key]
        fn_a = str(a.get("filename", ""))
        fn_b = str(b.get("filename", ""))
        if fn_a and fn_a not in entry["filenames1"]:
            entry["filenames1"].append(fn_a)
        if fn_b and fn_b not in entry["filenames2"]:
            entry["filenames2"].append(fn_b)
        entry["duplicate_types"].add(group.get("duplicate_type", ""))
        entry["risk_levels"].add(group.get("risk_level", ""))

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "claimId1",
            "claimId2",
            "filenames1",
            "filenames2",
            "duplicate_types",
            "risk_levels",
        ])

        for entry in paired.values():
            writer.writerow([
                entry["claimId1"],
                entry["claimId2"],
                ", ".join(entry["filenames1"]),
                ", ".join(entry["filenames2"]),
                ", ".join(sorted(entry["duplicate_types"])),
                ", ".join(sorted(entry["risk_levels"])),
            ])
            rows += 1

    return rows


def main() -> None:
    print(f"Loading: {INPUT_JSON}")
    groups = load_duplicates(INPUT_JSON)
    print(f"Found {len(groups)} duplicate groups")

    rows = export_csv(groups, OUTPUT_CSV)
    print(f"Exported {rows} rows to: {OUTPUT_CSV}")
    print("Open in Excel to view.")


if __name__ == "__main__":
    main()
