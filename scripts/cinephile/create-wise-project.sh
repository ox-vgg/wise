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
# Step 0: Sanity checks
#------------------------------------------------------------------------------
if [ -z "${CINEPHILE_DATA_DIR}" ]; then
    echo "CINEPHILE_DATA_DIR is not set. Exiting."
    exit 1
fi

#------------------------------------------------------------------------------
# Step 1: Download the ZIP file containing videos and metadata
#------------------------------------------------------------------------------
echo "--- Step 1: Downloading video archives ---"

URL1="https://hessenbox.uni-marburg.de/dl/fiP3skhbYUzNcsnsQoS8ND/DFF.dir"
URL2="https://hessenbox.uni-marburg.de/dl/fiDW4TuWzSmqmeLPsNYgGf/NIBG.dir"
TARGET_DIR="${CINEPHILE_DATA_DIR}/temp/videos_zip"
FILE1="${TARGET_DIR}/DFF.zip"
FILE2="${TARGET_DIR}/NIBG.zip"

mkdir -p "${TARGET_DIR}"

if [ ! -f "${FILE1}" ]; then
  echo "Downloading DFF.zip (21GB, takes ~15 minutes) ..."
  curl -L -o "${FILE1}" "${URL1}"
else
  echo "DFF.zip already exists. Skipping download."
fi

if [ ! -f "${FILE2}" ]; then
  echo "Downloading NIBG.zip (18GB, takes ~12 minutes) ..."
  curl -L -o "${FILE2}" "${URL2}"
else
  echo "NIBG.zip already exists. Skipping download."
fi

echo "Step 1 complete."
echo

#------------------------------------------------------------------------------
# Step 2: Extract the archives using a custom script to handle unicode filenames.
#------------------------------------------------------------------------------
echo "--- Step 2: Extracting video archives ---"

UNZIP_DEST_DFF="${CINEPHILE_DATA_DIR}/videos/DFF"
UNZIP_DEST_NIBG="${CINEPHILE_DATA_DIR}/videos/NIBG"
EXTRACT_SCRIPT="/wise/scripts/cinephile/extract_matched_video_json_pairs.py"

if [ ! -d "${UNZIP_DEST_DFF}" ] || [ -z "$(ls -A "${UNZIP_DEST_DFF}")" ]; then
  echo "Extracting DFF.zip to ${UNZIP_DEST_DFF}..."
  mkdir -p "${UNZIP_DEST_DFF}"
  python3 "${EXTRACT_SCRIPT}" "${FILE1}" "${UNZIP_DEST_DFF}"
else
  echo "DFF directory already exists and is not empty. Skipping extraction."
fi

if [ ! -d "${UNZIP_DEST_NIBG}" ] || [ -z "$(ls -A "${UNZIP_DEST_NIBG}")" ]; then
  echo "Extracting NIBG.zip to ${UNZIP_DEST_NIBG}..."
  mkdir -p "${UNZIP_DEST_NIBG}"
  python3 "${EXTRACT_SCRIPT}" "${FILE2}" "${UNZIP_DEST_NIBG}"
else
  echo "NIBG directory already exists and is not empty. Skipping extraction."
fi

echo "Step 2 complete."
echo

#------------------------------------------------------------------------------
# Step 3: Extract audio and visual features from the video files.
#------------------------------------------------------------------------------
echo "--- Step 3: Extracting audio and visual features ---"

PROJECT_DIR="${CINEPHILE_DATA_DIR}/wise-project/cinephile"
STATE_DIR="${CINEPHILE_DATA_DIR}/state/wise-project/cinephile"
FEATURE_SET1_EXTRACTION_SUCCESS_FILE="${STATE_DIR}/feature-set1-extraction.success"

# Create the state directory if it doesn't exist
mkdir -p "${STATE_DIR}"

# Check if the process has already completed successfully.
if [ -f "${FEATURE_SET1_EXTRACTION_SUCCESS_FILE}" ]; then
    echo "Feature extraction has already completed successfully. Skipping."
else
    # If the success file is missing, check if the project directory exists.
    # Its existence implies a previously failed or interrupted run.
    if [ -d "${PROJECT_DIR}" ]; then
        echo "Incomplete project directory found. Deleting it to restart feature extraction."
        rm -rf "${PROJECT_DIR}"
    fi

    echo "Extracting features..."
    python3 /wise/extract-features.py \
        "/data/cinephile/videos/" \
        --media-include "*.mp4" \
        --shard-maxcount 4096 \
        --shard-maxsize 20971520 \
        --num-workers 0 \
        --feature-store webdataset \
        --audio-feature-id "microsoft/clap/2023/four-datasets" \
        --video-feature-id "mlfoundations/open_clip/ViT-L-16-SigLIP2-512/webli" \
        --video-feature-id "deepinsight/insightface/buffalo_l/_unknown" \
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