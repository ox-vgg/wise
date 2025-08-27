#!/bin/bash

# Notes:
# 1. CINEPHILE_DATA_DIR points to `/data/cinephile/` inside the container and is 
#    set inside the Dockerfile. This folder is volume mapped to a local folder
#    (e.g. /path/to/local/folder) where all data related to the Cinephile project
#    will be stored.
# 2. The video archives are downloaded from the Cinephile 2025 challenge website.
#    https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html
#    If you already have the DFF.zip and NIBG.zip files, you can place them in the
#    ${CINEPHILE_DATA_DIR}/temp/videos_zip directory to skip the download step.

set -euo pipefail

#------------------------------------------------------------------------------
# Sanity checks
#------------------------------------------------------------------------------
# Check if the environment is activated
if [[ -z "${CONDA_PREFIX}" || ! "${CONDA_PREFIX}" == *"/wise-env" ]]; then
    echo "Micromamba environment 'wise-env' could not be activated. Exiting."
    exit 1
fi

if [ -z "${CINEPHILE_DATA_DIR}" ]; then
    echo "CINEPHILE_DATA_DIR is not set. Exiting."
    exit 1
fi

#------------------------------------------------------------------------------
# Set cache directories
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Step 1 + 2: Download data
#------------------------------------------------------------------------------
bash scripts/cinephile/download-data.sh


#------------------------------------------------------------------------------
# Step 3: Extract audio and visual features from the video files.
#------------------------------------------------------------------------------
echo "--- Step 3: Extracting audio and visual features ---"
TEMP_DIR="${CINEPHILE_DATA_DIR}/temp"
PROJECT_DIR="${CINEPHILE_DATA_DIR}/wise-project/cinephile"
STATE_DIR="${CINEPHILE_DATA_DIR}/state/wise-project/cinephile"
FEATURE_SET1_EXTRACTION_SUCCESS_FILE="${STATE_DIR}/feature-set1-extraction.success"
VIDEO_FEATURE_ID1="mlfoundations/open_clip/ViT-L-16-SigLIP2-512/webli"
VIDEO_FEATURE_ID2="deepinsight/insightface/buffalo_l/_unknown"
AUDIO_FEATURE_ID="microsoft/clap/2023/four-datasets"
FAISS_INDEX_TYPE="IndexFlatIP"

# Create the state directory if it doesn't exist
mkdir -p "${STATE_DIR}"

# Check if the process has already completed successfully.
if [ -f "${FEATURE_SET1_EXTRACTION_SUCCESS_FILE}" ]; then
    echo "Feature extraction has already completed successfully. Skipping."
else
    # If the success file is missing, check if the project directory exists.
    # Its existence implies a previously failed or interrupted run.
    if [ -d "${PROJECT_DIR}" ]; then
        echo "An incomplete project directory was found at ${PROJECT_DIR}."
        echo "This typically happens if a previous feature extraction run was interrupted or failed."
        echo "To restart feature extraction, this directory needs to be deleted."
        read -p "Do you want to delete the existing project directory and restart feature extraction? (y/N): " -n 1 -r
        echo
        if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
            echo "Aborting feature extraction. Please manually clean up ${PROJECT_DIR} if you wish to restart."
            exit 1
        fi
        echo "Deleting existing project directory: ${PROJECT_DIR}"
        rm -rf "${PROJECT_DIR}" || { echo "Failed to delete ${PROJECT_DIR}. Please check permissions."; exit 1; }
    fi

    echo "Extracting features ... (takes ~14 hours)"
    python extract-features.py \
        "/data/cinephile/videos/" \
        --media-include "*.mp4" \
        --shard-maxcount 4096 \
        --shard-maxsize 20971520 \
        --num-workers 0 \
        --feature-store webdataset \
        --audio-feature-id "${AUDIO_FEATURE_ID}" \
        --video-feature-id "${VIDEO_FEATURE_ID1}" \
        --video-feature-id "${VIDEO_FEATURE_ID2}" \
        --project-dir "${PROJECT_DIR}"

    # If the python script completes successfully, create the success file.
    if [ $? -eq 0 ]; then
        echo "Feature extraction completed successfully."
        touch "${FEATURE_SET1_EXTRACTION_SUCCESS_FILE}"
    else
        echo "Feature extraction failed."
        # The script will exit here because of `set -e`
    fi
fi

echo "Step 3 complete."
echo

#------------------------------------------------------------------------------
# Step 4: Import media metadata
#------------------------------------------------------------------------------
echo "--- Step 4: Import media metadata ---"
METADATA_DIR="${CINEPHILE_DATA_DIR}/videos/" # mp4 videos and metadata json files are in same folder

METADATA_DB_FILE="${PROJECT_DIR}/metadata/internal.db"
METADATA_ID="cinephile"
METADATA_TABLE_NAME="metadata-${METADATA_ID}"

RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
    echo "Importing metadata from JSON files contained in ${METADATA_DIR}/ (takes few seconds) ..."
    if [ -f "${TEMP_DIR}/${METADATA_TABLE_NAME}.csv" ]; then
        echo "Deleting existing CSV file: ${TEMP_DIR}/${METADATA_TABLE_NAME}.csv"
        rm -f "${TEMP_DIR}/${METADATA_TABLE_NAME}.csv"
    fi
    python3 scripts/cinephile/export-cinephile-metadata-as-csv.py \
        --json-dir "${METADATA_DIR}" \
        --project-dir "${PROJECT_DIR}" \
        --out-csv-file "${TEMP_DIR}/${METADATA_TABLE_NAME}.csv"

    if [ ! -f "${TEMP_DIR}/${METADATA_TABLE_NAME}.csv" ]; then
        echo "Metadata export to CSV failed."
        exit 1
    fi

    python3 media-metadata.py \
      import \
      --metadata-id "${METADATA_ID}" \
      --from-csv "${TEMP_DIR}/${METADATA_TABLE_NAME}.csv" \
      --metadata-type "media" \
      --project-dir "${PROJECT_DIR}"

    # check if the sqlite table was created
    RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
    if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
        echo "Failed to create SQLite table: $METADATA_TABLE_NAME"
        exit 1
    fi
fi

echo "Step 4 complete."
echo

#------------------------------------------------------------------------------
# Step 5: Create search index
#------------------------------------------------------------------------------
echo "--- Step 5: Create search index ---"
FTS_CONFIG_FILENAME="${TEMP_DIR}/fts_config.json" # config for full text search on metadata
cat > "${FTS_CONFIG_FILENAME}" <<EOF
{
  "${METADATA_TABLE_NAME}": [
    "provider",
    "title",
    "type",
    "year",
    "country",
    "language",
    "data_provider",
    "dc_contributor",
    "dc_description",
    "edm_timespan_label",
    "edm_preview",
    "edm_place_latitude",
    "edm_place_longitude",
    "edm_place_label",
    "edm_place_alt_label",
    "edm_dataset_name",
    "edm_concept_label"
  ]
}
EOF

FAISS_INDEX_TYPE="IndexFlatIP"
VIDEO_INDEX_FILENAME1="${PROJECT_DIR}/store/${VIDEO_FEATURE_ID1}/index/video-${FAISS_INDEX_TYPE}.faiss"
VIDEO_INDEX_FILENAME2="${PROJECT_DIR}/store/${VIDEO_FEATURE_ID2}/index/video-${FAISS_INDEX_TYPE}.faiss"
AUDIO_INDEX_FILENAME="${PROJECT_DIR}/store/${AUDIO_FEATURE_ID}/index/audio-${FAISS_INDEX_TYPE}.faiss"
if [ ! -f "${VIDEO_INDEX_FILENAME1}" ] || [ ! -f "${VIDEO_INDEX_FILENAME2}" ] || [ ! -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Creating index (takes about 5 min.) ..."
    python create-index.py \
           --index-type "${FAISS_INDEX_TYPE}" \
           --fts-config "${FTS_CONFIG_FILENAME}" \
           --project-dir "$PROJECT_DIR"
    if [ $? -eq 0 ]; then
        echo "Search index creation completed successfully."
    else
        echo "Failed to create search index."
        exit 1
    fi
fi

echo "Step 5 complete."
echo

#------------------------------------------------------------------------------
# Step 6: Remove silent videos
#------------------------------------------------------------------------------
echo "--- Step 6: Remove silent videos ---"

if [ -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Removing silent videos from audio search index (takes about 1 min) ..."
    python3 scripts/cinephile/remove-silent-videos-from-search-index.py \
            --project-dir "$PROJECT_DIR" \
            --block-filename-list scripts/cinephile/silent-video-filenames.txt
    if [ $? -eq 0 ]; then
        echo "Silent videos removed successfully."
    else
        echo "Failed to remove silent videos."
        exit 1
    fi
fi

echo "Step 6 complete."
echo