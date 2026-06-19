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

if [ "$#" -lt 1 ]; then
    echo "This script tests segment-level video feature extractors (e.g. Qwen3-VL-Embedding)"
    echo "using the Wikimedia Commons 25 dataset."
    echo ""
    echo "WARNING: This test requires a GPU with sufficient VRAM (>= 18 GB for 8B, >= 8 GB for 2B)."
    echo ""
    echo "Usage: bash ${0} TMP_DIR [MODEL_VARIANT]"
    echo ""
    echo "  TMP_DIR         Directory for test data, project files, etc."
    echo "  MODEL_VARIANT   '2B' or '8B' (default: 2B)"
    echo ""
    echo "Assumptions:"
    echo "  - The virtual environment containing all the required python packages is already activated."
    echo "  - A CUDA GPU with sufficient VRAM is available."
    echo ""
    echo "Example:"
    echo "    bash ${0} /tmp/wise-test-segment/ 2B"
    exit
fi

# uncomment the following line to produce verbose output and enable debugging
#set -euxo pipefail

TEST_ID="wikimedia-commons-25"
MODEL_VARIANT="${2:-2B}"
VIDEO_FEATURE_ID="hf/Qwen/Qwen3-VL-Embedding/${MODEL_VARIANT}"
FAISS_INDEX_TYPE="IndexFlatIP"
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10002"
MAX_POLL_SERVER_COUNT=25
NUM_WORKERS=0

WISE_CODE_DIR=$(pwd)
TMP_DIR=$(realpath "${1}")
OUTDIR="${TMP_DIR}/wise-test-segment/"
mkdir -p "${OUTDIR}"

## Ensure we are running the version in the repo instead of some other
## version of wise installed.
export PYTHONPATH="$WISE_CODE_DIR/src:$PYTHONPATH"

WISE_PKG_PATH=$(python -c 'import wise; print(wise.__path__[0])')
if [ "$WISE_CODE_DIR/src/wise" != "$WISE_PKG_PATH" ]; then
    echo "ERROR: WISE in path '$WISE_PKG_PATH' is not the current repo" >&2
    exit 1
fi

DATA_DIR="${OUTDIR}/test-data"
TEST_DATA_DIR="${DATA_DIR}/${TEST_ID}/"
TEST_DATA_DOWNLOAD_URL="https://thor.robots.ox.ac.uk/wise/assets/test/${TEST_ID}.zip"
WISE_PROJECT_DIR="${OUTDIR}/wise-project/${TEST_ID}-segment-${MODEL_VARIANT}/"

# check if required tools exist
REQUIRED_TOOLS=(ffmpeg sqlite3 jq curl unzip)
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        echo "$tool package not found, install the $tool software using your distribution package manager"
        exit 1
    fi
done

# check if required python packages exist
REQUIRED_PYTHON_PACKAGES=(torch torchvision torchaudio transformers faiss qwen_vl_utils)
for package in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    if ! python3 -c "import importlib.util; exit(0) if importlib.util.find_spec('${package}') else exit(1)"; then
        echo "$package package not found, please install the $package python package"
        exit 1
    fi
done

# check if a CUDA GPU is available
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"; then
    echo "ERROR: No CUDA GPU found. Segment-level models require a GPU."
    exit 1
fi

start=$(date +%s)
echo "Starting segment-level model tests for ${TEST_ID} with ${VIDEO_FEATURE_ID} ..."

## Task 1: Download test dataset
if [ ! -d "${TEST_DATA_DIR}" ]; then
    echo "Downloading test dataset to ${DATA_DIR} ..."
    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}"
    curl -sLO "$TEST_DATA_DOWNLOAD_URL"
    unzip -q "${TEST_ID}.zip" -d "${DATA_DIR}"
    rm "${TEST_ID}.zip"
    if [ ! -d "${TEST_DATA_DIR}" ]; then
        echo "Failed to download and extract the test dataset in ${TEST_DATA_DIR}"
        exit 1
    fi
else
    echo "Skipping test dataset download (already exists)"
fi

## Task 2: Extract features using segment-level model
if [ ! -d "${WISE_PROJECT_DIR}" ]; then
    echo "Extracting segment-level features using ${VIDEO_FEATURE_ID} ..."
    cd "${WISE_CODE_DIR}"
    python -m wise extract-features \
           "${TEST_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 512 \
           --num-workers $NUM_WORKERS \
           --video-feature-id "${VIDEO_FEATURE_ID}" \
           --no-thumbnails \
           --project-dir "$WISE_PROJECT_DIR"
    if [ $? -ne 0 ]; then
        echo "Feature extraction FAILED"
        exit 1
    fi
fi

## Test 2.1: check that feature store was created
FEATURE_STORE_DIR="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID}/"
if [ ! -d "${FEATURE_STORE_DIR}" ]; then
    echo "Test 2.1 FAILED: feature store directory ${FEATURE_STORE_DIR} does not exist"
    exit 1
else
    echo "Test 2.1 PASSED: feature store directory exists"
fi

## Task 3: Import media metadata
METADATA_DB_FILE="${WISE_PROJECT_DIR}metadata/internal.db"
METADATA_TABLE_NAME="metadata-${TEST_ID}"
RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
    echo "Importing metadata from ${TEST_DATA_DIR}/media-metadata.csv ..."
    python3 -m wise media-metadata import \
            --metadata-id "${TEST_ID}" \
            --from-csv "${TEST_DATA_DIR}/media-metadata.csv" \
            --metadata-type "media" \
            --project-dir "$WISE_PROJECT_DIR"
fi

## Task 4: Create search index
VIDEO_INDEX_FILENAME="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID}/index/video-${FAISS_INDEX_TYPE}.faiss"
if [ ! -f "${VIDEO_INDEX_FILENAME}" ]; then
    echo "Creating index ..."
    cd "${WISE_CODE_DIR}"
    python -m wise create-index \
           --index-type "${FAISS_INDEX_TYPE}" \
           --project-dir "$WISE_PROJECT_DIR"
fi

## Test 4.1: check that the video index file exists
if [ ! -f "${VIDEO_INDEX_FILENAME}" ]; then
    echo "Test 4.1 FAILED: video index file ${VIDEO_INDEX_FILENAME} does not exist"
    exit 1
else
    echo "Test 4.1 PASSED: video index exists"
fi

## Task 5: Start WISE server and run search tests
cleanup() {
    echo -e "\nShutting down server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}

if [ ! -d "frontend/dist" ]; then
    mkdir -p frontend/dist && \
        (cd frontend && npm ci && npm run build)
fi

echo "Starting WISE server on ${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT} ..."
cd "${WISE_CODE_DIR}"
LISTEN_ADDRESS=$HTTP_SERVER_HOST PORT=$HTTP_SERVER_PORT python -m wise serve \
        --index-type "${FAISS_INDEX_TYPE}" \
        --project-dir "$WISE_PROJECT_DIR" &
SERVER_PID=$!
trap cleanup SIGINT
trap cleanup SIGTERM
trap cleanup EXIT

SERVER_URL="http://${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT}/${TEST_ID}-segment-${MODEL_VARIANT}/"
PROJECT_INFO_URL="${SERVER_URL}info"
SLEEP_DURATION=5

for ((i=1; i<=MAX_POLL_SERVER_COUNT; i++)); do
    if curl -s --head --request GET "${PROJECT_INFO_URL}" | grep "200 OK" > /dev/null; then
        echo "Server is running at ${PROJECT_INFO_URL}"
        break
    else
        echo "Waiting for ${SLEEP_DURATION} sec. before checking again (${i}/${MAX_POLL_SERVER_COUNT}) ..."
        sleep $SLEEP_DURATION
    fi
    if [ "$i" -eq "$MAX_POLL_SERVER_COUNT" ]; then
        echo "Server did not start within the expected time."
        cleanup
        exit 1
    fi
done

sleep 5
echo "Server started successfully."

## Test 5.1: check that the server is running
if curl -s --head --request GET "${SERVER_URL}" | grep "200 OK" > /dev/null; then
    echo "Test 5.1 PASSED"
else
    echo "Test 5.1 FAILED: server is not running at ${SERVER_URL}"
    exit 1
fi

## Test 5.2: check project info lists the segment-level feature extractor
response=$(curl -s -X GET -H "Content-Type: application/json" "${PROJECT_INFO_URL}")
video_search_targets=($(echo "$response" | jq -r '.search_targets.video[]'))

found=false
for target in "${video_search_targets[@]}"; do
    if [ "$target" = "$VIDEO_FEATURE_ID" ]; then
        found=true
        break
    fi
done

if [ "$found" = true ]; then
    echo "Test 5.2 PASSED: ${VIDEO_FEATURE_ID} found in search targets"
else
    echo "Test 5.2 FAILED: ${VIDEO_FEATURE_ID} not found in search targets"
    echo "Actual: ${video_search_targets[*]}"
    exit 1
fi

validate_visual_response () {
    local test_id="${1}"
    local response="${2}"
    local expected_json="${3}"
    local response_selected_json=$(echo "$response" | jq -c ' . as $root | {
        merged_windows: [
        .video_results.merged_windows[]
        | .media_id as $id
        | {filename: $root.video_results.videos[$id].filename, source_url: $root.video_results.videos[$id].external_metadata.source_url}
        ]
    }')
    if diff <(echo "$expected_json" | jq -S .) <(echo "$response_selected_json" | jq -S .) > /dev/null; then
        echo "Test ${test_id} PASSED"
    else
        echo "Test ${test_id} FAILED: unexpected search results"
        echo "Expected:"
        echo "$expected_json" | jq .
        echo "Actual:"
        echo "$response_selected_json" | jq .
        exit 1
    fi
}

## Test 5.3: visual query "a horse jumping over fence"
SEARCH_QUERY="a+horse+jumping+over+fence"
RESULT_COUNT=60
SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID}&text_queries=${SEARCH_QUERY}"
response=$(curl -s -X POST -H "Content-Type: application/json" "${SEARCH_URL}")

# Verify the top-ranking unmerged_window is Learning_to_jump.mp4 at ts=20.0, te=27.5
top_result=$(echo "$response" | jq -c '
    . as $root |
    .video_results.unmerged_windows[0] |
    {
        filename: $root.video_results.videos[.media_id].filename,
        ts: .ts,
        te: .te
    }
')
expected_top='{"filename":"Learning_to_jump.mp4","ts":20,"te":27.5}'

if [ "$(echo "$top_result" | jq -Sc .)" = "$(echo "$expected_top" | jq -Sc .)" ]; then
    echo "Test 5.3 PASSED: top result is Learning_to_jump.mp4 [ts=20.0, te=27.5]"
else
    echo "Test 5.3 FAILED: unexpected top result for 'a horse jumping over fence'"
    echo "Expected: $(echo "$expected_top" | jq .)"
    echo "Actual:   $(echo "$top_result" | jq .)"
    exit 1
fi

end_time=$(date +%s)
elapsed_time=$((end_time-start))
echo ""
echo "*** All segment-level model tests for ${TEST_ID} (${VIDEO_FEATURE_ID}) completed in ${elapsed_time} sec. ***"
echo ""

cleanup
