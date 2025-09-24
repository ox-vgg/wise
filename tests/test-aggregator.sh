#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "This script tests the functionality of WISE2 software's aggregator feature which"
    echo "enables WISE to aggregate search response from multiple standalone WISE projects."
    echo ""
    echo "Usage: bash tests/${0} TMP_DIR"
    echo ""
    echo "where, the assumptions are:"
    echo "  - The virtual environment containing all the required python packages is already activated."
    echo "  - The WISE2 code is already cloned to the current directory."
    echo ""
    echo "For example, if you have cloned the WISE2 repository to $HOME/wise, run the following commands:"
    echo "    1. cd $HOME/wise/"
    echo "    2. source .../bin/activate"
    echo "    3. bash ${0} /tmp/wise-test/"
    echo ""
    echo "The TMP_DIR will contain everything (test data, wise project, etc.) required by this script to run the tests."
    echo "In the final stage, this script will start the WISE2 server and run a series of tests to verify the installation."
    exit
fi

# uncomment the following line to produce verbose output and enable debugging
#set -euxo pipefail

# Set these variables to the appropriate values
# for your environment
TEST_ID="aggregator-3"
VIDEO_FEATURE_ID1="mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"
VIDEO_FEATURE_ID2="deepinsight/insightface/buffalo_l/_unknown"
VIDEO_FEATURE_ID3="transformers/owlv2/google/owlv2-large-patch14-ensemble"
AUDIO_FEATURE_ID="microsoft/clap/2023/four-datasets"
FAISS_INDEX_TYPE="IndexFlatIP"
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10001"
MAX_POLL_SERVER_COUNT=15
GPU_ID=1
NUM_WORKERS=2

# to enable triton server, export an environment variable FEATURE_EXTRACTOR_CONFIG
# containing a JSON string with the URL of the triton server. For example:
# $ export FEATURE_EXTRACTOR_CONFIG="{\"default\": {\"url\": \"localhost:8801\"}}"
triton_url=`echo "${FEATURE_EXTRACTOR_CONFIG}" | jq -e '.default.url'`
if [ ! -z "$triton_url" ]; then
    NUM_WORKERS=0 # set to 0 only if default.url is not defined in FEATURE_EXTRACTOR_CONFIG
fi

WISE_CODE_DIR=`pwd`
TMP_DIR=$(realpath ${1})
OUTDIR="${TMP_DIR}/wise-test/"
mkdir -p "${OUTDIR}"

DATA_DIR="${OUTDIR}/test-data"
TEST_DATA_DIR="${DATA_DIR}/${TEST_ID}/"
TEST_DATA_DOWNLOAD_URL="https://thor.robots.ox.ac.uk/wise/assets/test/${TEST_ID}.zip"
WISE_PROJECT_BASEDIR="${OUTDIR}/wise-project/"
WISE_ALL_PROJECT_DIR="${WISE_PROJECT_BASEDIR}/123/"
WISE_PROJECT1_DIR="${WISE_PROJECT_BASEDIR}/1/"
WISE_PROJECT2_DIR="${WISE_PROJECT_BASEDIR}/2/"
WISE_PROJECT3_DIR="${WISE_PROJECT_BASEDIR}/3/"

QUERY_DATA_DIR="${OUTDIR}/test-query/${TEST_ID}/"
mkdir -p "${QUERY_DATA_DIR}"

# check if required tools exist
REQUIRED_TOOLS=(ffmpeg sqlite3 jq curl unzip)
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        echo "$tool package not found, install the $tool software using your distribution package manager"
        exit 1
    fi
done

# check if required python scripts exist
REQUIRED_PYTHON_SCRIPTS=(extract-features.py media-metadata.py create-index.py serve.py)
for script in "${REQUIRED_PYTHON_SCRIPTS[@]}"; do
    if [ ! -f "${script}" ]; then
        echo "$script not found, please run this script from the WISE2 code directory"
        exit 1
    fi
done

# check if required python packages exist
REQUIRED_PYTHON_PACKAGES=(torch torchvision torchaudio transformers faiss webdataset msclap open_clip)
for package in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    if ! python3 -c "import importlib.util; exit(0) if importlib.util.find_spec('${package}') else exit(1)"; then
        echo "$package package not found, please install the $package python package"
        exit 1
    fi
done

start=`date +%s`
echo "Starting tests for ${TEST_ID} ..."

## Task: 1. Download test dataset
if [ ! -d "${TEST_DATA_DIR}" ]; then
    echo "Downloading test dataset to ${DATA_DIR} ..."
    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}"
    curl -sLO $TEST_DATA_DOWNLOAD_URL
    unzip -q "${TEST_ID}.zip" -d "${DATA_DIR}"
    rm "${TEST_ID}.zip"
    if [ ! -d "${TEST_DATA_DIR}" ]; then
        echo "Failed to download and extract the test dataset in ${TEST_DATA_DIR}"
        exit 1
    fi
else
    echo "Skipping test dataset download"
fi

## Task: 2. Create a WISE project corresponding to each subset
for subset_id in 1 2 3; do
    SUBSET_DIR="${TEST_DATA_DIR}${subset_id}/"
    WISE_PROJECT_SUBSET_DIR="${OUTDIR}/wise-project/${subset_id}/"

    ## Task 2.1 : Extract features from videos in the subset
    if [ ! -d "${WISE_PROJECT_SUBSET_DIR}" ]; then
        echo "Extracting features from videos in subset ${subset_id} using ${NUM_WORKERS} workers (takes about 8 min.) ..."
        cd "${WISE_CODE_DIR}"
        python extract-features.py \
               "${SUBSET_DIR}" \
               --media-include "*.mp4" \
               --shard-maxcount 4096 \
               --shard-maxsize 20971520 \
               --num-workers $NUM_WORKERS \
               --feature-store webdataset \
               --audio-feature-id "${AUDIO_FEATURE_ID}" \
               --video-feature-id "${VIDEO_FEATURE_ID1}" \
               --video-feature-id "${VIDEO_FEATURE_ID2}" \
               --video-feature-id "${VIDEO_FEATURE_ID3}" \
               --project-dir "$WISE_PROJECT_SUBSET_DIR"
    fi

    ## Task 2.2 : Import media metadata
    METADATA_DB_FILE="${WISE_PROJECT_SUBSET_DIR}metadata/internal.db"
    METADATA_TABLE_NAME="metadata-${subset_id}"
    RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
    if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
        echo "Importing metadata from ${SUBSET_DIR}media-metadata.csv (takes few seconds) ..."
        python3 media-metadata.py import \
                --metadata-id "${subset_id}" \
                --from-csv "${SUBSET_DIR}media-metadata.csv" \
                --metadata-type "media" \
                --project-dir "$WISE_PROJECT_SUBSET_DIR"
    fi

    ## Test 2.3 : check if the metadata table exists
    TABLE_NAME=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
    if [ "$TABLE_NAME" != "$METADATA_TABLE_NAME" ]; then
        echo "Test 2.3 FAILED: metadata table $TABLE_NAME does not exist in $METADATA_DB_FILE"
        exit 1
    else
        echo "Test 2.3 PASSED"
    fi

    ## Test 2.4 : check if all the metadata rows are imported
    ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_TABLE_NAME';")
    TRUE_ROW_COUNT=$(wc -l < "${SUBSET_DIR}media-metadata.csv")
    TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
    if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
        echo "Test 2.4 FAILED: metadata table $TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
        exit 1
    else
        echo "Test 2.4 PASSED"
    fi

    ## Test 2.5 : create index
    NUM_INDEX_FILES=$(find "${WISE_PROJECT_SUBSET_DIR}store/" -type f -name "*.faiss" | wc -l)
    if [ "$NUM_INDEX_FILES" -eq 0 ]; then
        echo "Creating FAISS index of type ${FAISS_INDEX_TYPE} for video feature ${VIDEO_FEATURE_ID1} (takes about 1 min.) ..."
        cd "${WISE_CODE_DIR}"
        python create-index.py \
            --index-type "IndexFlatIP" \
            --project-dir "$WISE_PROJECT_SUBSET_DIR"
    fi
    NUM_INDEX_FILES=$(find "${WISE_PROJECT_SUBSET_DIR}store/" -type f -name "*.faiss" | wc -l)
    if [ "$NUM_INDEX_FILES" -ne 4 ]; then
        echo "Test 2.5 FAILED: expected 4 but only $NUM_INDEX_FILES indices found in ${WISE_PROJECT_SUBSET_DIR}store/"
        exit 1
    else
        echo "Test 2.5 PASSED"
    fi
done

## Task: 3.1 Create a WISE project containing all the 3 subsets
if [ ! -d "${WISE_ALL_PROJECT_DIR}" ]; then
    echo "Extracting features from videos using ${NUM_WORKERS} workers (takes about 20 min.) ..."
    cd "${WISE_CODE_DIR}"
    python extract-features.py \
           "${TEST_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 4096 \
           --shard-maxsize 20971520 \
           --num-workers $NUM_WORKERS \
           --feature-store webdataset \
           --audio-feature-id "${AUDIO_FEATURE_ID}" \
           --video-feature-id "${VIDEO_FEATURE_ID1}" \
           --video-feature-id "${VIDEO_FEATURE_ID2}" \
           --video-feature-id "${VIDEO_FEATURE_ID3}" \
           --project-dir "$WISE_ALL_PROJECT_DIR"
fi

## Task 3.2 : Import media metadata for the WISE project containing all the 3 subsets
METADATA_DB_FILE="${WISE_ALL_PROJECT_DIR}metadata/internal.db"
METADATA_TABLE_NAME="metadata-123"
RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
    echo "Importing metadata from ${TEST_DATA_DIR}all-media-metadata.csv (takes few seconds) ..."
    python3 media-metadata.py import \
            --metadata-id "123" \
            --from-csv "${TEST_DATA_DIR}media-metadata.csv" \
            --metadata-type "media" \
            --project-dir "$WISE_ALL_PROJECT_DIR"
fi

## Test 3.3 : check if the metadata table exists
TABLE_NAME=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$TABLE_NAME" != "$METADATA_TABLE_NAME" ]; then
    echo "Test 3.3 FAILED: metadata table $TABLE_NAME does not exist in $METADATA_DB_FILE"
    exit 1
else
    echo "Test 3.3 PASSED"
fi

## Test 3.4 : check if all the metadata rows are imported
ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_TABLE_NAME';")
TRUE_ROW_COUNT=$(wc -l < "${TEST_DATA_DIR}media-metadata.csv")
TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
    echo "Test 3.4 FAILED: metadata table $TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
    exit 1
else
    echo "Test 3.4 PASSED"
fi

## Test 3.5 : create index
NUM_INDEX_FILES=$(find "${WISE_ALL_PROJECT_DIR}store/" -type f -name "*.faiss" | wc -l)
if [ "$NUM_INDEX_FILES" -eq 0 ]; then
    echo "Creating FAISS index of type ${FAISS_INDEX_TYPE} for video feature ${VIDEO_FEATURE_ID1} (takes about 5 min.) ..."
    cd "${WISE_CODE_DIR}"
    python create-index.py \
        --index-type "IndexFlatIP" \
        --project-dir "$WISE_ALL_PROJECT_DIR"
fi
NUM_INDEX_FILES=$(find "${WISE_ALL_PROJECT_DIR}store/" -type f -name "*.faiss" | wc -l)
if [ "$NUM_INDEX_FILES" -ne 4 ]; then
    echo "Test 3.5 FAILED: expected 4 but only $NUM_INDEX_FILES indices found in ${WISE_ALL_PROJECT_DIR}store/"
    exit 1
else
    echo "Test 3.5 PASSED"
fi

## Helper function to start multiple WISE servers
BIND_ADDRESS="0.0.0.0"
BIND_BASE_PORT=10000
PIDS=()
cleanup() {
    echo "Shutting down all background processes..."
    for pid in "${PIDS[@]}"; do
        kill -- -"$pid" 2>/dev/null
    done
    wait
    exit 0
}
trap cleanup SIGINT

# Task 4.1 : Start WISE server for each of the 3 subsets
PROJECT_LIST=("1" "2" "3")
REMOTE_PROJECTS='['
for i in "${!PROJECT_LIST[@]}"; do
    PORT=$((BIND_BASE_PORT + 1 + i))
    LISTEN_ADDRESS=$BIND_ADDRESS PORT=$PORT CUDA_VISIBLE_DEVICES=$GPU_ID python serve.py \
        --index-type IndexFlatIP \
        --project-dir "$WISE_PROJECT_BASEDIR/${PROJECT_LIST[$i]}/" &
    PIDS+=($!)
    REMOTE_PROJECTS="${REMOTE_PROJECTS}\"http://localhost:${PORT}/${PROJECT_LIST[$i]}/\","
    echo "Started server for ${PROJECT_LIST[$i]} on port $PORT with PID ${PIDS[-1]}"
    echo "###### PIDS: ${PIDS[@]}"
done
# remove last comma from REMOTE_PROJECTS and add a closing bracket
REMOTE_PROJECTS="${REMOTE_PROJECTS%,}]"

# Task 4.2 : Start WISE server for the combined project
COMBINED_PROJECT_PORT=$((BIND_BASE_PORT + 1 + ${#PROJECT_LIST[@]}))
LISTEN_ADDRESS=$BIND_ADDRESS PORT=$COMBINED_PROJECT_PORT CUDA_VISIBLE_DEVICES=$GPU_ID python serve.py \
    --index-type IndexFlatIP \
    --project-dir "$WISE_PROJECT_BASEDIR/123/" &
PIDS+=($!)
echo "Started server for combined project on port $COMBINED_PROJECT_PORT with PID ${PIDS[-1]}"
echo "###### PIDS: ${PIDS[@]}"
echo "********************************** REMOTE_PROJECTS=${REMOTE_PROJECTS}"

# Task 4.3 : Wait for all http endpoints to be available
echo "Waiting for all servers to be available ..."
SLEEP_DURATION=5

PROJECTS_TO_CHECK=("${PROJECT_LIST[@]}" "123")
PORTS_TO_CHECK=()
for i in "${!PROJECT_LIST[@]}"; do
    PORTS_TO_CHECK+=($((BIND_BASE_PORT + 1 + i)))
done
PORTS_TO_CHECK+=($COMBINED_PROJECT_PORT)

for ((poll_count=1; poll_count<=MAX_POLL_SERVER_COUNT; poll_count++)); do
    all_servers_up=true
    for i in "${!PROJECTS_TO_CHECK[@]}"; do
        project_id="${PROJECTS_TO_CHECK[$i]}"
        port="${PORTS_TO_CHECK[$i]}"
        url="http://localhost:${port}/${project_id}/info"

        if ! curl -s --head --request GET "${url}" | grep "200 OK" > /dev/null; then
            echo "Server for project ${project_id} on port ${port} is not yet available."
            all_servers_up=false
            break # break from inner loop and sleep
        else
            echo "Server for project ${project_id} on port ${port} is running."
        fi
    done

    if $all_servers_up; then
        echo "All servers are up and running."
        break
    fi

    if [ "$poll_count" -eq "$MAX_POLL_SERVER_COUNT" ]; then
        echo "Timeout: Not all servers started within the expected time."
        cleanup
        exit 1
    fi

    echo "Waiting for ${SLEEP_DURATION} sec. before checking again (${poll_count}/${MAX_POLL_SERVER_COUNT}) ..."
    sleep $SLEEP_DURATION
done

PORT=$BIND_BASE_PORT REMOTE_PROJECTS=$REMOTE_PROJECTS python3 serve.py \
    --project-dir tmp/123/

# TODO:
# [1] aggregator server running on PORT=$BIND_BASE_PORT
# [2] combined project server running on PORT=$COMBINED_PROJECT_PORT
#
# The search results from both [1] and [2] should be identical for any query

wait