#!/usr/bin/env python

"""
This script prevents a list of media files from appearing in the search results.
This is achieved by removing all the vector_id associated with the specified media
files from the Faiss search index.

Usage :

python3 scripts/cinephile/remove-silent-videos-from-search-index.py \
  --project-dir /data/cinephile/wise-project/cinephile/ \
  --block-filename-list scripts/cinephile/silent-video-filenames.txt
"""

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


def read_filenames_from_txt(txt_path: Path) -> list[str]:
    """Reads a list of filenames from a plain text file."""
    if not txt_path.exists():
        logging.error(f"Blocklist file not found: {txt_path}")
        return []

    with open(txt_path, 'r', encoding='utf-8') as f:
        # Read lines, stripping leading/trailing whitespace and skipping empty lines
        return [line.strip() for line in f if line.strip()]


def get_vector_ids_for_files(db_path: Path, filenames: list[str], modality: str, feature_extractor_id: str) -> list[int]:
    """
    Connects to the SQLite database and retrieves the vector_ids for a given
    list of filenames, a modality, and a feature_extractor_id.
    """
    if not db_path.exists():
        logging.error(f"Database not found: {db_path}")
        return []

    vector_ids = []
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()

            # Create a temporary table to hold the filenames for an efficient query
            cursor.execute("CREATE TEMP TABLE blocklist (filename TEXT PRIMARY KEY)")
            cursor.executemany("INSERT OR IGNORE INTO blocklist (filename) VALUES (?)", [(name,) for name in filenames])

            # Check for any filenames in the blocklist that do not exist in the media table
            cursor.execute("SELECT filename FROM blocklist EXCEPT SELECT path FROM media")
            unmatched_files = [row[0] for row in cursor.fetchall()]

            if unmatched_files:
                logging.warning(f"The following {len(unmatched_files)} files were not found in the database and will be skipped:")
                for f in sorted(unmatched_files):
                    logging.warning(f"  - {f}")

            # Query to get vector_ids by joining media, vectors, and the blocklist
            query = """
                SELECT v.id
                FROM vectors v
                JOIN media m ON v.media_id = m.id
                JOIN blocklist b ON m.path = b.filename
                WHERE v.modality = ? AND v.feature_extractor_id = ?
            """
            cursor.execute(query, (modality, feature_extractor_id))
            vector_ids = [row[0] for row in cursor.fetchall()]

            cursor.execute("DROP TABLE blocklist")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")

    return vector_ids


def remove_from_faiss_index(index_path: Path, ids_to_remove: list[int]) -> bool:
    """Loads a Faiss index, removes vectors by ID, and saves it back."""
    if not index_path.exists():
        logging.error(f"Faiss index not found: {index_path}")
        return False

    try:
        index = faiss.read_index(str(index_path))

        # The `remove_ids` function requires a specific numpy array of int64
        ids_to_remove_np = np.array(ids_to_remove, dtype=np.int64)

        # The remove_ids function returns the number of elements removed.
        nb_removed = index.remove_ids(faiss.IDSelectorArray(ids_to_remove_np))
        logging.info(f"Removed {nb_removed} vectors from the Faiss index.")

        # Save the modified index
        faiss.write_index(index, str(index_path))
        logging.info(f"Successfully saved the updated index to {index_path}")
        return True

    except Exception as e:
        logging.error(f"Failed to modify Faiss index: {e}")
        return False


def main():
    """Main function to parse arguments and orchestrate the removal process."""
    parser = argparse.ArgumentParser(
        description="Remove files from a WISE project's search index."
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        required=True,
        help="Path to the WISE project directory."
    )
    parser.add_argument(
        '--block-filename-list',
        type=Path,
        required=True,
        help="Path to the TXT file containing filenames to remove."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Perform a dry run without modifying any files."
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    blocklist_path = args.block_filename_list.resolve()
    dry_run = args.dry_run

    if dry_run:
        logging.info("--- Starting DRY RUN mode. No files will be modified. ---")

    # Define paths based on WISE project structure
    db_path = project_dir / "metadata" / "internal.db"
    index_path = project_dir / "store/microsoft/clap/2023/four-datasets/index" / "audio-IndexFlatIP.faiss"

    # --- Step 1: Read filenames to block ---
    logging.info(f"Reading filenames from {blocklist_path}")
    filenames_to_remove = read_filenames_from_txt(blocklist_path)
    if not filenames_to_remove:
        logging.warning("No filenames found in the blocklist. Exiting.")
        return
    logging.info(f"Found {len(filenames_to_remove)} filenames to process.")

    # --- Step 2: Get vector IDs from the database ---
    logging.info("Fetching corresponding vector IDs from the database...")
    modality = 'AUDIO'
    feature_extractor_id = 'microsoft/clap/2023/four-datasets'
    vector_ids_to_remove = get_vector_ids_for_files(db_path, filenames_to_remove, modality, feature_extractor_id)
    if not vector_ids_to_remove:
        logging.warning("No matching vectors found in the database for the given files. Exiting.")
        return
    logging.info(f"Found {len(vector_ids_to_remove)} vectors to remove.")

    # --- Step 3: Perform removal or report actions for dry run ---
    if dry_run:
        logging.info(f"[DRY RUN] Would remove {len(vector_ids_to_remove)} vectors from the Faiss index at {index_path}")
        logging.info("--- DRY RUN complete. ---")
        return

    # --- Step 3: Remove vectors from the Faiss index (Live Run) ---
    logging.info(f"Removing vectors from the Faiss index at {index_path}")
    success = remove_from_faiss_index(index_path, vector_ids_to_remove)
    if not success:
        logging.error("Failed to remove vectors from Faiss index.")
        sys.exit(1)

    logging.info("Process completed successfully.")


if __name__ == "__main__":
    main()
