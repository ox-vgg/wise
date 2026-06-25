#!/bin/bash

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
# Step 1: Download the ZIP file containing videos and metadata
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Step 2: Extract the archives using a custom script to handle unicode filenames.
#------------------------------------------------------------------------------
echo "--- Step 1 + 2: Downloading and Extracting video archives ---"
URL1="https://hessenbox.uni-marburg.de/dl/fiP3skhbYUzNcsnsQoS8ND/DFF.dir"
URL2="https://hessenbox.uni-marburg.de/dl/fiDW4TuWzSmqmeLPsNYgGf/NIBG.dir"

TEMP_DIR="${CINEPHILE_DATA_DIR}/temp"
TARGET_DIR="${TEMP_DIR}/videos_zip"
mkdir -p "${TARGET_DIR}"

UNZIP_DEST_DFF="${CINEPHILE_DATA_DIR}/videos/DFF"
UNZIP_DEST_NIBG="${CINEPHILE_DATA_DIR}/videos/NIBG"
EXTRACT_SCRIPT="scripts/cinephile/extract_matched_video_json_pairs.py"

if [ ! -d "${UNZIP_DEST_DFF}" ] || [ -z "$(ls -A "${UNZIP_DEST_DFF}")" ]; then
  # Download the zip if it doesn't already exist
  # TODO checksum
  FILE1="${TARGET_DIR}/DFF.zip"
  if [ ! -f "${FILE1}" ]; then
    echo "Downloading DFF.zip (21GB, takes ~15 minutes) ..."
    curl -L -o "${FILE1}" "${URL1}"
  fi
  echo "Extracting DFF.zip to ${UNZIP_DEST_DFF}..."
  mkdir -p "${UNZIP_DEST_DFF}"
  python "${EXTRACT_SCRIPT}" "${FILE1}" "${UNZIP_DEST_DFF}"
else
  echo "DFF directory already exists and is not empty. Skipping extraction."
fi

if [ ! -d "${UNZIP_DEST_NIBG}" ] || [ -z "$(ls -A "${UNZIP_DEST_NIBG}")" ]; then
  # Download the zip if it doesn't already exist
  # TODO checksum
  FILE2="${TARGET_DIR}/NIBG.zip"
  if [ ! -f "${FILE2}" ]; then
    echo "Downloading NIBG.zip (18GB, takes ~12 minutes) ..."
    curl -L -o "${FILE2}" "${URL2}"
  fi
  echo "Extracting NIBG.zip to ${UNZIP_DEST_NIBG}..."
  mkdir -p "${UNZIP_DEST_NIBG}"
  python "${EXTRACT_SCRIPT}" "${FILE2}" "${UNZIP_DEST_NIBG}"
else
  echo "NIBG directory already exists and is not empty. Skipping extraction."
fi

echo "Step 1 + 2 complete."
echo
echo "Data download complete. Videos and metadata are available in ${CINEPHILE_DATA_DIR}/videos/"
echo
