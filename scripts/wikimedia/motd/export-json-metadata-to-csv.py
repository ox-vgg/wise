#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

WANTED_FIELDS = [
    "ImageDescription",
    "DateTimeOriginal",
    "Artist",
    "UsageTerms",
    "LicenseUrl",
    "Credit",
    "Restrictions",
]

EXT_RANK = {
    ".webm": 0,
    ".ogv": 1,
    ".ogg": 2,
    ".oga": 3,
    ".mp4": 4,
    ".mid": 5,
}


def normalize_title(value: str) -> str:
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def normalize_loose(value: str) -> str:
    value = normalize_title(value)
    value = re.sub(r"[\"\',\.;:\-\(\)\[\]\{\}]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def pick_best_json(paths):
    def score(path):
        name = Path(path).name
        base = name[:-5] if name.lower().endswith(".json") else name
        ext = Path(base).suffix.lower()
        return (EXT_RANK.get(ext, 99), name.lower())

    return sorted(paths, key=score)[0]


def build_json_index(metadata_dir: Path):
    index = defaultdict(list)
    for json_path in metadata_dir.glob("*.json"):
        base = json_path.name[:-5]
        key = normalize_title(Path(base).stem)
        if key:
            index[key].append(str(json_path))

    resolved = {k: pick_best_json(v) for k, v in index.items()}
    duplicates = {k: v for k, v in index.items() if len(v) > 1}
    return resolved, duplicates


def extract_english(value: str) -> str:
    if not value:
        return ""
    pattern = re.compile(
        r"<[^>]*\blang=[\"']en[\"'][^>]*>(.*?)</[^>]+>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(value)
    if matches:
        joined = " ".join(m.strip() for m in matches if m.strip())
        return joined.strip()
    return value


def extract_field(extmetadata, key):
    if not extmetadata:
        return ""
    value = extmetadata.get(key)
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("value", "")
    if key == "ImageDescription":
        value = extract_english(value)
    return value


def resolve_media_rows(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id, path FROM media")
    rows = cursor.fetchall()
    conn.close()
    return rows


def load_overrides(overrides_csv: Path | None):
    if overrides_csv is None:
        return {}
    overrides = {}
    with open(overrides_csv, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            media_path = (row.get("media_path") or "").strip()
            json_title = (row.get("json_title") or "").strip()
            if not media_path or not json_title:
                continue
            overrides[normalize_title(Path(media_path).stem)] = normalize_title(Path(json_title).stem)
    return overrides


def export_metadata(
    project_dir: Path,
    metadata_dir: Path,
    out_csv: Path,
    allow_missing: bool,
    use_prefix_match: bool,
    overrides_csv: Path | None,
):
    db_path = project_dir / "metadata" / "internal.db"
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    json_index, duplicates = build_json_index(metadata_dir)
    overrides = load_overrides(overrides_csv)
    if duplicates:
        print("Found duplicate JSON titles; picking a deterministic file for each key:")
        for key, paths in sorted(duplicates.items()):
            chosen = pick_best_json(paths)
            print(f"  {key} -> {Path(chosen).name} (candidates: {len(paths)})")
    if overrides:
        print(f"Loaded {len(overrides)} manual overrides.")

    json_keys = list(json_index.keys())
    rows = resolve_media_rows(db_path)

    matched = 0
    missing = []
    ambiguous = []
    missing_imageinfo = []
    output_rows = []

    for media_id, media_path in rows:
        media_title = Path(media_path).stem
        key = normalize_title(media_title)
        override_key = overrides.get(key)
        json_path = json_index.get(override_key) if override_key else json_index.get(key)

        if json_path is None and use_prefix_match:
            media_loose = normalize_loose(media_title)
            candidates = [k for k in json_keys if normalize_loose(k).startswith(media_loose)]
            if len(candidates) == 1:
                json_path = json_index[candidates[0]]
            elif len(candidates) > 1:
                ambiguous.append((media_id, media_path, candidates))
                continue

        if json_path is None:
            missing.append((media_id, media_path))
            continue

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        imageinfo = data.get("imageinfo") or []
        if not imageinfo:
            missing_imageinfo.append((media_id, media_path, json_path))
            continue

        extmetadata = imageinfo[0].get("extmetadata", {})
        row = {"media_id": media_id}
        for field in WANTED_FIELDS:
            row[field] = extract_field(extmetadata, field)
        output_rows.append(row)
        matched += 1

    if ambiguous:
        print(f"Ambiguous prefix matches for {len(ambiguous)} media rows:")
        for media_id, media_path, candidates in ambiguous[:20]:
            preview = ", ".join(candidates[:3])
            print(f"  {media_id}: {media_path} -> {preview}")
        if len(ambiguous) > 20:
            print(f"  ... {len(ambiguous) - 20} more")

    if missing_imageinfo:
        print(f"JSON files missing imageinfo for {len(missing_imageinfo)} media rows.")

    if missing:
        print(f"Missing JSON metadata for {len(missing)} media rows.")
        for media_id, media_path in missing[:20]:
            print(f"  {media_id}: {media_path}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        if not allow_missing:
            raise SystemExit("Aborting because --allow-missing was not set.")

    if ambiguous:
        if not allow_missing:
            raise SystemExit("Aborting because ambiguous matches were found and --allow-missing was not set.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["media_id"] + WANTED_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    total = len(rows)
    print(f"Processed {total} media rows. Matched {matched}. Wrote {len(output_rows)} rows to {out_csv}.")


def main():
    parser = argparse.ArgumentParser(
        description="Export selected Wikimedia Commons JSON metadata to CSV for media-metadata.py",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="WISE project directory (expects metadata/internal.db)",
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        help="Directory containing Wikimedia Commons JSON metadata files",
    )
    parser.add_argument(
        "--out-csv-file",
        required=True,
        help="Output CSV filename",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing or ambiguous matches and still write output",
    )
    parser.add_argument(
        "--no-prefix-match",
        action="store_true",
        help="Disable prefix-based fallback matching",
    )
    parser.add_argument(
        "--overrides-csv",
        help="CSV with columns media_path,json_title to override matching",
    )

    args = parser.parse_args()
    export_metadata(
        project_dir=Path(args.project_dir),
        metadata_dir=Path(args.metadata_dir),
        out_csv=Path(args.out_csv_file),
        allow_missing=args.allow_missing,
        use_prefix_match=not args.no_prefix_match,
        overrides_csv=Path(args.overrides_csv) if args.overrides_csv else None,
    )


if __name__ == "__main__":
    main()
