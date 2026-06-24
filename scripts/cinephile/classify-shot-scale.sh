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
#
# Assumptions:
# 1. The `create-wise-project.sh` script has already been executed
# 2. A valid WISE project already exists in `${CINEPHILE_DATA_DIR}/wise-project/cinephile`.

set -euo pipefail

#------------------------------------------------------------------------------
# Sanity checks
#------------------------------------------------------------------------------
# Check if the environment is activated
if [[ -z "${CONDA_PREFIX}" || ! "${CONDA_PREFIX}" == *"/shot-scale-classifier-env" ]]; then
    echo "Micromamba environment 'shot-scale-classifier-env' could not be activated. Exiting."
    exit 1
fi

if [ -z "${CINEPHILE_DATA_DIR}" ]; then
    echo "CINEPHILE_DATA_DIR is not set. Exiting."
    exit 1
fi

#------------------------------------------------------------------------------
# Download the shot_scale_ckpt.pth model from
# https://github.com/Jyxarthur/shot-by-shot/blob/main/film_grammar/shot_scale_classifier.py
#------------------------------------------------------------------------------
WEIGHTS_FILE_URL="https://thor.robots.ox.ac.uk/wise/assets/cinephile/shot_scale_ckpt.pth"
WEIGHTS_BASEDIR="/tmp/cache/shot-scale-classifier/checkpoints"
# check if the size of /tmp/shot-scale-classifier/checkpoints/shot_scale_ckpt.pth is less than 200 bytes
if [ ! -f "${WEIGHTS_BASEDIR}/shot_scale_ckpt.pth" ]; then
    echo "Downloading weights ..."
    mkdir -p "${WEIGHTS_BASEDIR}"
    curl -L "${WEIGHTS_FILE_URL}" -o "${WEIGHTS_BASEDIR}/shot_scale_ckpt.pth"
    if [ $(stat -c%s "${WEIGHTS_BASEDIR}/shot_scale_ckpt.pth") -ne 510614103 ]; then
        echo "Downloaded weights are invalid."
        exit 1
    fi
fi

#------------------------------------------------------------------------------
# Extract shot scale for all videos
#------------------------------------------------------------------------------
TEMP_DIR="${CINEPHILE_DATA_DIR}/temp"
PROJECT_DIR="${CINEPHILE_DATA_DIR}/wise-project/cinephile"
THUMBNAIL_SHOT_SCALE_FILENAME="${CINEPHILE_DATA_DIR}/wise-project/cinephile/thumbs-shot-scale.csv"

# check if shot boundaries file exists
if [ ! -f "${THUMBNAIL_SHOT_SCALE_FILENAME}" ]; then
    echo "Finding shot scales ... (take ~ 1 hour)"
    python3 classify_shot_scale.py \
        --batch-size 64 \
        --num-workers 4 \
        --resume_path ${WEIGHTS_BASEDIR}/shot_scale_ckpt.pth \
        --out-csv $THUMBNAIL_SHOT_SCALE_FILENAME \
        --project-dir $PROJECT_DIR
fi

# # check if shot_scale column of shots table in internal.db has values set or if all of them are empty
# Cannot do it here - must be done in WISE container
# if [ -z "$(sqlite3 ${PROJECT_DIR}/metadata/internal.db "SELECT shot_scale FROM shots WHERE shot_scale IS NOT NULL AND shot_scale != '' LIMIT 1;")" ]; then
#     echo "No shot_scale values found. Importing from CSV ..."
#     cd /wise
#     python3 media-metadata.py \
#          import-shot-scale \
#          --project-dir $PROJECT_DIR \
#          --from-csv $THUMBNAIL_SHOT_SCALE_FILENAME
# fi