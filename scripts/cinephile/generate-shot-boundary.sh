#!/bin/bash

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

eval "$(micromamba shell hook -s bash)"
micromamba activate shot-detection-env

#------------------------------------------------------------------------------
# Sanity checks
#------------------------------------------------------------------------------
# Check if the environment is activated
if [[ -z "${CONDA_PREFIX}" || ! "${CONDA_PREFIX}" == *"/shot-detection-env" ]]; then
    echo "Micromamba environment 'shot-detection-env' could not be activated. Exiting."
    exit 1
fi

if [ -z "${CINEPHILE_DATA_DIR}" ]; then
    echo "CINEPHILE_DATA_DIR is not set. Exiting."
    exit 1
fi

#------------------------------------------------------------------------------
# Download transnetv2 model
#------------------------------------------------------------------------------
WEIGHTS_FILE_URL="https://thor.robots.ox.ac.uk/wise/assets/cinephile/transnetv2-weights.tar.gz"
WEIGHTS_BASEDIR="/tmp/shot-detection/transnetv2/inference/"
# check if the size of /tmp/shot-detection/transnetv2/inference/transnetv2-weights/saved_model.pb is less than 200 bytes
if [ -f "${WEIGHTS_BASEDIR}/transnetv2-weights/saved_model.pb" ] && [ $(stat -c%s "${WEIGHTS_BASEDIR}/transnetv2-weights/saved_model.pb") -lt 200 ]; then
    echo "Downloading weights ..."
    rm -rf "${WEIGHTS_BASEDIR}/transnetv2-weights"
    curl -L "${WEIGHTS_FILE_URL}" -o "${WEIGHTS_BASEDIR}/transnetv2-weights.tar.gz"
    tar -zxvf "${WEIGHTS_BASEDIR}/transnetv2-weights.tar.gz" -C "${WEIGHTS_BASEDIR}"
    rm -f "${WEIGHTS_BASEDIR}/transnetv2-weights.tar.gz"
    if [ $(stat -c%s "${WEIGHTS_BASEDIR}/transnetv2-weights/saved_model.pb") -ne 5933260 ]; then
        echo "Downloaded weights are invalid."
        exit 1
    fi
fi

#------------------------------------------------------------------------------
# Extract shot boundaries for all videos
#------------------------------------------------------------------------------
TEMP_DIR="${CINEPHILE_DATA_DIR}/temp"
PROJECT_DIR="${CINEPHILE_DATA_DIR}/wise-project/cinephile"
SHOT_BOUNDARIES_FILENAME="${CINEPHILE_DATA_DIR}/wise-project/cinephile/shot-boundaries.csv"
SHOT_TEMPDIR="${TEMP_DIR}/shot-boundaries"
mkdir -p $SHOT_TEMPDIR

# check if shot boundaries file exists
if [ ! -f "${SHOT_BOUNDARIES_FILENAME}" ]; then
    echo "Generating shot boundaries ... (take ~ ? hours)"
    cd /tmp/shot-detection
    export PYTHONPATH="transnetv2/inference"
    start_time=$(date +%s)
    echo "Started shot boundary detection at: $(date)"
    python cli.py \
      detect-and-convert \
      $PROJECT_DIR \
      --save-to $SHOT_TEMPDIR
    end_time=$(date +%s)
    echo "Finished shot boundary detection at: $(date)"
    duration=$((end_time - start_time))
    echo "Shot boundary detection took $duration seconds."
    # ensure that /tmp/shot-detection/shots.csv file has been created
    if [ ! -f "/tmp/shot-detection/shots.csv" ]; then
        echo "Failed to generate shot boundaries."
        exit 1
    fi
    mv /tmp/shot-detection/shots.csv $SHOT_BOUNDARIES_FILENAME
    echo "Saved shot boundaries to ${SHOT_BOUNDARIES_FILENAME}"
fi