#!/bin/bash

# Notes:
# 1. CINEPHILE_DATA_DIR points to `/data/cinephile/` inside the container and is 
#    set inside the Dockerfile. This folder is volume mapped to a local folder
#    (e.g. /path/to/local/folder) where all data related to the Cinephile project
#    will be stored.
# 2. The project computed by VGG is downloaded and stored in the CINEPHILE_DATA_DIR to save time

set -euo pipefail

#------------------------------------------------------------------------------
# Step 1: Download the ZIP file containing the features and metadata
#------------------------------------------------------------------------------
WISE_PROJECT_URL="https://thor.robots.ox.ac.uk/wise/assets/cinephile/wise-cinephile-project-2025-08-28.tar.gz"
WISE_PROJECT_FILE=${WISE_PROJECT_URL##*/}
TEMP_DIR="${CINEPHILE_DATA_DIR}/temp"

WISE_PROJECT_DIR="${CINEPHILE_DATA_DIR}/wise-project/"

if [ ! -d "${WISE_PROJECT_DIR}" ] || [ -z "$(ls -A "${WISE_PROJECT_DIR}")" ]; then
  # Download the zip if it doesn't already exist
  # TODO checksum
  TAR_FILE="${TEMP_DIR}/${WISE_PROJECT_FILE}"
  if [ ! -f "${TAR_FILE}" ]; then
    echo "Downloading pre-computed wise project ..."
    mkdir -p ${TEMP_DIR}
    curl -L "${WISE_PROJECT_URL}" -o "${TAR_FILE}"
  fi
  echo "Extracting ${WISE_PROJECT_FILE} to ${WISE_PROJECT_DIR}..."
  mkdir -p "${WISE_PROJECT_DIR}"
  tar -zxvf "${TAR_FILE}" -C "${WISE_PROJECT_DIR}"
else
  echo "WISE Cinephile project directory already exists and is not empty. Skipping extraction."
fi

echo "WISE project download complete. Project files are available in ${WISE_PROJECT_DIR}/cinephile/"
echo