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
PROJECT_DIR="${CINEPHILE_DATA_DIR}/wise-project/cinephile"
STATE_DIR="${CINEPHILE_DATA_DIR}/state/wise-project/cinephile"
FEATURE_SET2_EXTRACTION_SUCCESS_FILE="${STATE_DIR}/feature-set2-extraction.success"
VIDEO_FEATURE_ID3="transformers/owlv2/google/owlv2-large-patch14-ensemble"
FAISS_INDEX_TYPE="IndexIVFFlat"

# Create the state directory if it doesn't exist
mkdir -p "${STATE_DIR}"

# Check if the process has already completed successfully.
if [ -f "${FEATURE_SET2_EXTRACTION_SUCCESS_FILE}" ]; then
    echo "Feature extraction has already completed successfully. Skipping."
else
    echo "Extracting object features ... (takes ~7 hours)"
    python extract-features.py \
        --yes \
        --media-include "*.mp4" \
        --shard-maxcount 4096 \
        --num-workers 0 \
        --no-thumbnails \
        --use-shots \
        --enable-autocast \
        --video-feature-id "${VIDEO_FEATURE_ID3}" \
        --project-dir "${PROJECT_DIR}"

    # If the python script completes successfully, create the success file.
    if [ $? -eq 0 ]; then
        echo "Feature extraction completed successfully."
        touch "${FEATURE_SET2_EXTRACTION_SUCCESS_FILE}"
    else
        echo "Feature extraction failed."
        # The script will exit here because of `set -e`
    fi
fi

echo "Step 3 complete."
echo

#------------------------------------------------------------------------------
# Step 2: Create search index
#------------------------------------------------------------------------------
echo "--- Step 2: Create search index for objects ---"
FAISS_INDEX_TYPE="IndexFlatIP"
VIDEO_INDEX_FILENAME3="${PROJECT_DIR}/store/${VIDEO_FEATURE_ID3}/index/video-${FAISS_INDEX_TYPE}.faiss"

if [ ! -f "${VIDEO_INDEX_FILENAME3}" ]; then
    echo "Creating index (takes about 5 min.) ..."
    python create-index.py \
           --media-type "video" \
           --feature-id "${VIDEO_FEATURE_ID3}" \
           --index-type "${FAISS_INDEX_TYPE}" \
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
