#!/usr/bin/env python3

import os
import zipfile
import unicodedata
from collections import defaultdict

def normalize(name):
    """Normalize Unicode filenames to NFC to avoid mismatches."""
    return unicodedata.normalize('NFC', name)

def extract_matching_pairs(zip_path, extract_dir, dry_run=False):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_files = [normalize(name) for name in zf.namelist()]

        # Group files by stem (filename without extension)
        file_map = defaultdict(set)
        for name in all_files:
            if name.endswith('.mp4') or name.endswith('.json'):
                stem, ext = os.path.splitext(os.path.basename(name))
                file_map[stem].add(ext)

        # Only keep pairs where both .mp4 and .json exist
        matched_stems = [stem for stem, exts in file_map.items() if {'.mp4', '.json'} <= exts]

        if not dry_run and not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        for stem in matched_stems:
            for ext in ['.mp4', '.json']:
                filename = f"{stem}{ext}"
                normalized_filename = normalize(filename)

                try:
                    source_info = next(info for info in zf.infolist() if normalize(info.filename) == normalized_filename)
                    if dry_run:
                        print(f"[DRY RUN] Would extract: {source_info.filename}")
                    else:
                        zf.extract(source_info, path=extract_dir)
                        print(f"Extracted: {source_info.filename}")
                except StopIteration:
                    print(f"Warning: Could not find matching file for {normalized_filename}")

        # Warn about unmatched files
        unmatched = [stem for stem in file_map if stem not in matched_stems]
        if unmatched:
            print("\nUnmatched file stems (missing either .mp4 or .json):")
            for stem in unmatched:
                print(f" - {stem}: {file_map[stem]}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract matched .mp4/.json file pairs from a zip file.")
    parser.add_argument("zipfile", help="Path to the input .zip file")
    parser.add_argument("outdir", help="Directory to extract matched files to")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be extracted, without writing to disk")
    args = parser.parse_args()

    extract_matching_pairs(args.zipfile, args.outdir, args.dry_run)
