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

if [ "$#" -ne 1 ]; then
    echo "This script tests the functionality of WISE software's aggregator feature which"
    echo "enables WISE to aggregate search response from multiple standalone WISE projects."
    echo ""
    echo "Usage: bash tests/${0} TMP_DIR"
    echo ""
    echo "where, the assumptions are:"
    echo "  - The virtual environment containing all the required python packages is already activated."
    echo "  - The WISE code is already cloned to the current directory."
    echo ""
    echo "For example, if you have cloned the WISE repository to $HOME/wise, run the following commands:"
    echo "    1. cd $HOME/wise/"
    echo "    2. source .../bin/activate"
    echo "    3. bash ${0} /tmp/wise-test/"
    echo ""
    echo "The TMP_DIR will contain everything (test data, wise project, etc.) required by this script to run the tests."
    echo "In the final stage, this script will start the WISE server and run a series of tests to verify the installation."
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
WISE_MERGED_PROJECT_DIR="${WISE_PROJECT_BASEDIR}/merged/"

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
        echo "$script not found, please run this script from the WISE code directory"
        exit 1
    fi
done

# check if required python packages exist
REQUIRED_PYTHON_PACKAGES=(torch torchvision torchaudio transformers faiss msclap open_clip)
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
    # ensure that the sha256 checksum of the downloaded zip file matches the expected value
    EXPECTED_CHECKSUM="50af20db7ac9c56d21b84b373e2d8008417a3a8c7ea12c1ef26733d4c7ca7394"
    DOWNLOADED_CHECKSUM=$(sha256sum "${TEST_ID}.zip" | awk '{print $1}')
    if [ "$DOWNLOADED_CHECKSUM" != "$EXPECTED_CHECKSUM" ]; then
        echo "Checksum verification failed for the downloaded test dataset. Expected: ${EXPECTED_CHECKSUM}, Got: ${DOWNLOADED_CHECKSUM}"
        exit 1
    fi

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
    WISE_PROJECT_SUBSET_DIR="${WISE_PROJECT_BASEDIR}/${subset_id}/"

    ## Task 2.1 : Extract features from videos in the subset
    if [ ! -d "${WISE_PROJECT_SUBSET_DIR}" ]; then
        echo "[${subset_id}] Extracting features from videos in subset ${subset_id} using ${NUM_WORKERS} workers (takes about 8 min.) ..."
        cd "${WISE_CODE_DIR}"
        python extract-features.py \
               "${SUBSET_DIR}" \
               --media-include "*.mp4" \
               --shard-maxcount 4096 \
               --num-workers $NUM_WORKERS \
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
        echo "[${subset_id}] Importing metadata from ${SUBSET_DIR}media-metadata.csv (takes few seconds) ..."
        python3 media-metadata.py import \
                --metadata-id "${subset_id}" \
                --from-csv "${SUBSET_DIR}media-metadata.csv" \
                --metadata-type "media" \
                --project-dir "$WISE_PROJECT_SUBSET_DIR"
    fi

    ## Test 2.3 : check if the metadata table exists
    TABLE_NAME=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
    if [ "$TABLE_NAME" != "$METADATA_TABLE_NAME" ]; then
        echo "[${subset_id}] Test 2.3 FAILED: metadata table $TABLE_NAME does not exist in $METADATA_DB_FILE"
        exit 1
    else
        echo "[${subset_id}] Test 2.3 PASSED"
    fi

    ## Test 2.4 : check if all the metadata rows are imported
    ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_TABLE_NAME';")
    TRUE_ROW_COUNT=$(python -c "import csv; print(sum(1 for i in csv.reader(open('"${SUBSET_DIR}media-metadata.csv"'))))")
    TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
    if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
        echo "[${subset_id}] Test 2.4 FAILED: metadata table $TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
        exit 1
    else
        echo "[${subset_id}] Test 2.4 PASSED"
    fi

    ## Test 2.5 : create index
    FTS_CONFIG_FILE="${WISE_PROJECT_SUBSET_DIR}fts-config.json"
    echo "{ \"metadata-${subset_id}\": [ \"description\", \"source_url\" ] }" > "${FTS_CONFIG_FILE}"
    NUM_INDEX_FILES=$(find "${WISE_PROJECT_SUBSET_DIR}store/" -type f -path '*/index/*' -name "*.faiss" | wc -l)
    if [ "$NUM_INDEX_FILES" -eq 0 ]; then
        echo "[${subset_id}] Creating FAISS index of type ${FAISS_INDEX_TYPE} for video feature ${VIDEO_FEATURE_ID1} (takes about 1 min.) ..."
        cd "${WISE_CODE_DIR}"
        python create-index.py \
            --index-type "IndexFlatIP" \
            --fts-config "${FTS_CONFIG_FILE}" \
            --project-dir "$WISE_PROJECT_SUBSET_DIR"
    fi
    NUM_INDEX_FILES=$(find "${WISE_PROJECT_SUBSET_DIR}store/" -type f -path '*/index/*' -name "*.faiss" | wc -l)
    if [ "$NUM_INDEX_FILES" -ne 4 ]; then
        echo "[${subset_id}] Test 2.5 FAILED: expected 4 but only $NUM_INDEX_FILES indices found in ${WISE_PROJECT_SUBSET_DIR}store/"
        exit 1
    else
        echo "[${subset_id}] Test 2.5 PASSED"
    fi
done

## Task: 3.1 Create a WISE project containing all the 3 subsets
if [ ! -d "${WISE_ALL_PROJECT_DIR}" ]; then
    echo "[123] Extracting features from videos using ${NUM_WORKERS} workers (takes about 20 min.) ..."
    cd "${WISE_CODE_DIR}"
    python extract-features.py \
           "${TEST_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 4096 \
           --num-workers $NUM_WORKERS \
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
    echo "[123] Importing metadata from ${TEST_DATA_DIR}media-metadata.csv (takes few seconds) ..."
    python3 media-metadata.py import \
            --metadata-id "123" \
            --from-csv "${TEST_DATA_DIR}media-metadata.csv" \
            --metadata-type "media" \
            --project-dir "${WISE_ALL_PROJECT_DIR}"
fi

if [ ! -d ${WISE_MERGED_PROJECT_DIR} ]; then
    echo "[merged] Merging the 3 subsets into a single WISE project (takes about 6 min.) ..."
    cd "${WISE_CODE_DIR}"
    python3 -m src.wise_project merge \
       --into "${WISE_MERGED_PROJECT_DIR}" \
       "${WISE_PROJECT1_DIR}" \
       "${WISE_PROJECT2_DIR}" \
       "${WISE_PROJECT3_DIR}"
fi

# NOTE: This metadata import only works because filenames are unique across the 3 subsets
METADATA_DB_FILE="${WISE_MERGED_PROJECT_DIR}metadata/internal.db"
METADATA_TABLE_NAME="metadata-123"
RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
    ## Task 2.2 : Import media metadata
    echo "[${subset_id}] Importing metadata from ${TEST_DATA_DIR}media-metadata.csv (takes few seconds) ..."
    TMPFILE=$(mktemp)
    cat ${TEST_DATA_DIR}1/media-metadata.csv | sed -e '$a\' > $TMPFILE
    tail -n +2 ${TEST_DATA_DIR}2/media-metadata.csv | sed -e '$a\' >> $TMPFILE
    tail -n +2 ${TEST_DATA_DIR}3/media-metadata.csv | sed -e '$a\' >> $TMPFILE
    python3 media-metadata.py import \
            --metadata-id "123" \
            --from-csv ${TMPFILE} \
            --metadata-type "media" \
            --project-dir "${WISE_MERGED_PROJECT_DIR}"

    rm -f $TMPFILE
fi

for PROJECT in "${WISE_ALL_PROJECT_DIR}" "${WISE_MERGED_PROJECT_DIR}"; do
    METADATA_DB_FILE="${PROJECT}metadata/internal.db"
    METADATA_TABLE_NAME="metadata-123"
    ## Test 3.3 : check if the metadata table exists
    TABLE_NAME=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
    if [ "$TABLE_NAME" != "$METADATA_TABLE_NAME" ]; then
        echo "[123] Test 3.3 FAILED: metadata table $TABLE_NAME does not exist in $METADATA_DB_FILE"
        exit 1
    else
        echo "[123] Test 3.3 PASSED"
    fi

    ## Test 3.4 : check if all the metadata rows are imported
    ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_TABLE_NAME';")
    TRUE_ROW_COUNT=$(python -c "import csv; print(sum(1 for i in csv.reader(open('"${TEST_DATA_DIR}media-metadata.csv"'))))")
    TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
    if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
        echo "[123] Test 3.4 FAILED: metadata table $TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
        exit 1
    else
        echo "[123] Test 3.4 PASSED"
    fi

    ## Test 3.5 : create index
    FTS_CONFIG_FILE="${PROJECT}fts-config.json"
    echo "{ \"metadata-123\": [ \"description\", \"source_url\" ] }" > "${FTS_CONFIG_FILE}"
    NUM_INDEX_FILES=$(find "${PROJECT}store/" -type f -path '*/index/*' -name "*.faiss" | wc -l)
    if [ "$NUM_INDEX_FILES" -eq 0 ]; then
        echo "Creating FAISS index of type ${FAISS_INDEX_TYPE} (takes about 5 min.) ..."
        cd "${WISE_CODE_DIR}"
        python create-index.py \
            --index-type "IndexFlatIP" \
            --fts-config "${FTS_CONFIG_FILE}" \
            --project-dir "${PROJECT}"
    fi
    NUM_INDEX_FILES=$(find "${PROJECT}store/" -type f -path '*/index/*' -name "*.faiss" | wc -l)
    if [ "$NUM_INDEX_FILES" -ne 4 ]; then
        echo "[123] Test 3.5 FAILED: expected 4 but only $NUM_INDEX_FILES indices found in ${PROJECT}store/"
        exit 1
    else
        echo "[123] Test 3.5 PASSED"
    fi

done
end=`date +%s`

## Helper function to start multiple WISE servers
BIND_ADDRESS="0.0.0.0"
BIND_BASE_PORT=10000
PIDS=()
cleanup() {
    echo "Shutting down all background processes..."
    for pid in "${PIDS[@]}"; do
        kill -- -"$pid" 2>/dev/null
    done
    echo "Press Ctrl + C to force terminate background processes ..."
    wait
    echo "All background processes terminated."
    exit ${1:-0}
}
trap "cleanup 1" SIGINT

# Task 4.1 : Start WISE server for each of the 3 subsets
PROJECT_LIST=("1" "2" "3")
PROJECTS_TO_CHECK=("${PROJECT_LIST[@]}" "123" "merged")

REMOTE_PROJECTS='['
PORTS_TO_CHECK=()
for i in "${!PROJECT_LIST[@]}"; do
    PORT=$((BIND_BASE_PORT + 1 + i))
    PORTS_TO_CHECK+=($PORT)
    LISTEN_ADDRESS=$BIND_ADDRESS PORT=$PORT CUDA_VISIBLE_DEVICES=$GPU_ID python serve.py \
        --index-type IndexFlatIP \
        --project-dir "$WISE_PROJECT_BASEDIR/${PROJECT_LIST[$i]}/" &
    PIDS+=($!)
    REMOTE_PROJECTS="${REMOTE_PROJECTS}\"http://localhost:${PORT}/${PROJECT_LIST[$i]}/\","
    echo "Started server for ${PROJECT_LIST[$i]} on port $PORT with PID ${PIDS[-1]}"
done
# remove last comma from REMOTE_PROJECTS and add a closing bracket
REMOTE_PROJECTS="${REMOTE_PROJECTS%,}]"

# Task 4.2 : Start WISE server for the merged project
MERGED_PROJECT_PORT=$((BIND_BASE_PORT + ${#PROJECT_LIST[@]}))
for PROJECT in "${WISE_ALL_PROJECT_DIR}" "${WISE_MERGED_PROJECT_DIR}"; do
    MERGED_PROJECT_PORT=$((MERGED_PROJECT_PORT + 1))
    PORTS_TO_CHECK+=($MERGED_PROJECT_PORT)
    LISTEN_ADDRESS=$BIND_ADDRESS PORT=$MERGED_PROJECT_PORT CUDA_VISIBLE_DEVICES=$GPU_ID python serve.py \
        --index-type IndexFlatIP \
        --project-dir "${PROJECT}" &
    PIDS+=($!)
    echo "Started server for $(basename ${PROJECT}) on port $MERGED_PROJECT_PORT with PID ${PIDS[-1]}"
done
echo "PIDS: ${PIDS[@]}"
echo "REMOTE_PROJECTS=${REMOTE_PROJECTS}"

# Task 4.3 : Wait for all http endpoints to be available
echo "Waiting for all servers to be available ..."
SLEEP_DURATION=5

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
        cleanup 1
    fi

    echo "Waiting for ${SLEEP_DURATION} sec. before checking again (${poll_count}/${MAX_POLL_SERVER_COUNT}) ..."
    sleep $SLEEP_DURATION
done

# Task 5 : Start the WISE aggregator server
PORT=$BIND_BASE_PORT REMOTE_PROJECTS=$REMOTE_PROJECTS python3 serve.py --project-dir tmp/123/ &
PIDS+=($!)
AGGREGATOR_URL="http://localhost:${BIND_BASE_PORT}/123/"
COMBINED_URL="http://localhost:$((MERGED_PROJECT_PORT-1))/123/"
MERGED_URL="http://localhost:${MERGED_PROJECT_PORT}/merged/"

echo "Waiting for the aggregator server to be available ..."
for ((poll_count=1; poll_count<=MAX_POLL_SERVER_COUNT; poll_count++)); do
    if curl -s --head --request GET "${AGGREGATOR_URL}info" | grep "200 OK" > /dev/null; then
        echo "Aggregator server is up and running."
        break
    fi
    if [ "$poll_count" -eq "$MAX_POLL_SERVER_COUNT" ]; then
        echo "Timeout: Aggregator server did not start within the expected time."
        cleanup 1
    fi
    echo "Waiting for ${SLEEP_DURATION} sec. before checking again (${poll_count}/${MAX_POLL_SERVER_COUNT}) ..."
    sleep $SLEEP_DURATION
done

# Task 6 : Run tests to compare results from the aggregator and merged project
TEST_RESULTS=()
ANY_TEST_FAILED=0
ASSERT_EQUAL() {
    URL1=$1
    URL2=$2
    TEST_ID=$3
    DESC=$4

    RESPONSE1=$(curl -s -X POST -H "Content-Type: application/json" "$URL1")
    RESPONSE2=$(curl -s -X POST -H "Content-Type: application/json" "$URL2")

    # Use jq to delete the 'total_duration' field (if it exists) and then sort keys for consistent comparison
    SORTED1=$(echo "$RESPONSE1" | jq -S 'del(.total_duration?)')
    SORTED2=$(echo "$RESPONSE2" | jq -S 'del(.total_duration?)')

    if [ "$SORTED1" == "$SORTED2" ]; then
        RESULT="Test ${TEST_ID} PASSED: ${DESC}"
        echo "$RESULT"
        TEST_RESULTS+=("$RESULT")
    else
        RESULT="Test ${TEST_ID} FAILED: ${DESC}"
        echo "$RESULT"
        TEST_RESULTS+=("$RESULT")
        ANY_TEST_FAILED=1
        echo "URL1: $URL1"
        echo "URL2: $URL2"
        echo "Response from URL1 (after removing total_duration):"
        echo "$SORTED1" | jq .
        echo "Response from URL2 (after removing total_duration):"
        echo "$SORTED2" | jq .
        echo "Diff:"
        diff <(echo "$SORTED1") <(echo "$SORTED2")
    fi
}

ASSERT_SAME_FILENAME_LIST() {
    URL1=$1
    URL2=$2
    TEST_ID=$3
    DESC=$4
    RESULT_KEY=$5
    TOP_K=$6

    RESPONSE1=$(curl -s -X POST -H "Content-Type: application/json" "$URL1")
    RESPONSE2=$(curl -s -X POST -H "Content-Type: application/json" "$URL2")

    JQ_EXTRACT_FILENAMES=".${RESULT_KEY} | .videos as \$videos | (.merged_windows // [])[] | \$videos[.media_id].filename"

    FILENAMES1_RAW=$(echo "$RESPONSE1" | jq -r "$JQ_EXTRACT_FILENAMES")
    FILENAMES2_RAW=$(echo "$RESPONSE2" | jq -r "$JQ_EXTRACT_FILENAMES")

    FILENAMES2_PREFIX_REMOVED=$(echo "$FILENAMES2_RAW" | sed 's/^[0-9]*\///')

    FILENAMES1_TRUNCATED_SORTED=$(echo "$FILENAMES1_RAW" | head -n $TOP_K | sort)
    FILENAMES2_TRUNCATED_SORTED=$(echo "$FILENAMES2_PREFIX_REMOVED" | head -n $TOP_K | sort)

    if [ "$FILENAMES1_TRUNCATED_SORTED" == "$FILENAMES2_TRUNCATED_SORTED" ]; then
        RESULT="Test ${TEST_ID} PASSED: ${DESC}"
        echo "$RESULT"
        TEST_RESULTS+=("$RESULT")
    else
        RESULT="Test ${TEST_ID} FAILED: ${DESC}"
        echo "$RESULT"
        TEST_RESULTS+=("$RESULT")
        ANY_TEST_FAILED=1
        echo "--- Debug Info for Test ${TEST_ID}: ${DESC} ---"
        echo "URL1: $URL1"
        echo "URL2: $URL2"
        echo "Result Key: $RESULT_KEY"
        echo "Top K: $TOP_K"
        echo "Filenames from URL1 (truncated to $TOP_K, sorted):"
        echo "$FILENAMES1_TRUNCATED_SORTED"
        echo "Filenames from URL2 (prefix removed, truncated to $TOP_K, sorted):"
        echo "$FILENAMES2_TRUNCATED_SORTED"
        echo "---------------------------------------------"

        echo "Diff:"
        diff <(echo "$FILENAMES1_TRUNCATED_SORTED") <(echo "$FILENAMES2_TRUNCATED_SORTED")
    fi
}

ASSERT_TOPK_EQUAL() {
    URL1=$1
    URL2=$2
    TEST_ID=$3
    DESC=$4
    RESULT_KEY=$5
    TOP_K=$6
    FILE_PATH=$7

    if [ -n "$FILE_PATH" ]; then
        RESPONSE1=$(curl -s -X POST "$URL1" -F "image_file_queries=@${FILE_PATH}")
        RESPONSE2=$(curl -s -X POST "$URL2" -F "image_file_queries=@${FILE_PATH}")
    else
        RESPONSE1=$(curl -s -X POST -H "Content-Type: application/json" "$URL1")
        RESPONSE2=$(curl -s -X POST -H "Content-Type: application/json" "$URL2")
    fi

    # Construct the jq filter dynamically to extract relevant fields
    JQ_EXTRACT_FIELDS=".${RESULT_KEY} | .videos as \$videos | (.merged_windows // [])[] | \"\\(\$videos[.media_id].filename)|\\(.distance)|\\(.ts)|\\(.te)\""

    FIELDS1_RAW=$(echo "$RESPONSE1" | jq -r "$JQ_EXTRACT_FIELDS")
    FIELDS2_RAW=$(echo "$RESPONSE2" | jq -r "$JQ_EXTRACT_FIELDS")

    # Remove shard-id prefix from URL2 filenames
    FIELDS2_PREFIX_REMOVED=$(echo "$FIELDS2_RAW" | sed 's/^[0-9]*\///')

    # Truncate results from both to TOP_K
    FIELDS1_TRUNCATED=$(echo "$FIELDS1_RAW" | head -n $TOP_K)
    FIELDS2_TRUNCATED=$(echo "$FIELDS2_PREFIX_REMOVED" | head -n $TOP_K)

    if [ "$FIELDS1_TRUNCATED" == "$FIELDS2_TRUNCATED" ]; then
        RESULT="Test ${TEST_ID} PASSED: ${DESC}"
        echo "$RESULT"
        TEST_RESULTS+=("$RESULT")
    else
        RESULT="Test ${TEST_ID} FAILED: ${DESC}"
        echo "$RESULT"
        TEST_RESULTS+=("$RESULT")
        ANY_TEST_FAILED=1
        echo "--- Debug Info for Test ${TEST_ID}: ${DESC} ---"
        echo "URL1: $URL1"
        echo "URL2: $URL2"
        echo "Result Key: $RESULT_KEY"
        echo "Top K: $TOP_K"
        if [ -n "$FILE_PATH" ]; then
            echo "File Path: $FILE_PATH"
        fi
        echo "Fields from URL1 (filename|distance|ts|te, truncated to $TOP_K results):"
        echo "$FIELDS1_TRUNCATED"
        echo "Fields from URL2 (filename|distance|ts|te, prefix removed, truncated to $TOP_K results):"
        echo "$FIELDS2_TRUNCATED"
        echo "---------------------------------------------"

        echo "Diff:"
        diff <(echo "$FIELDS1_TRUNCATED") <(echo "$FIELDS2_TRUNCATED")
    fi
}

TEST_EQUIVALENCE() {
    local MERGED_URL=$1
    local TEST_MAJOR_NUM=$2

    # Task ${TEST_MAJOR_NUM}.1 : ensure the /info endpoint for both the aggregator and merged project return identical results
    AGGREGATOR_INFO_URL="${AGGREGATOR_URL}info"
    MERGED_INFO_URL="${MERGED_URL}info"
    ASSERT_EQUAL "$AGGREGATOR_INFO_URL" "$MERGED_INFO_URL" "${TEST_MAJOR_NUM}.1" "identical values in /info endpoints"

    # Task ${TEST_MAJOR_NUM}.2 : ensure the /search endpoint has identical video search results for both the aggregator and merged project
    SEARCH_QUERY="panda"
    RESULT_COUNT=1000 # needs to be sufficiently large in order to get all the relevant video segments
    TOP_K=6           # we know there are only 6 videos in the test dataset that match the query
    SEARCH_URL_SUFFIX="search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID1}&text_queries=${SEARCH_QUERY}"
    AGGREGATOR_SEARCH_URL="${AGGREGATOR_URL}${SEARCH_URL_SUFFIX}"
    MERGED_SEARCH_URL="${MERGED_URL}${SEARCH_URL_SUFFIX}"
    ASSERT_TOPK_EQUAL "$AGGREGATOR_SEARCH_URL" "$MERGED_SEARCH_URL" "${TEST_MAJOR_NUM}.2" "identical results for video search query '${SEARCH_QUERY}'" "video_results" $TOP_K

    # Task ${TEST_MAJOR_NUM}.3 : ensure the /search endpoint has identical audio search results for both the aggregator and merged project
    SEARCH_QUERY="gunshot"
    RESULT_COUNT=1000
    TOP_K=5
    SEARCH_URL_SUFFIX="search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=av&feature_extractor_id=${AUDIO_FEATURE_ID}&text_queries=${SEARCH_QUERY}"
    AGGREGATOR_SEARCH_URL="${AGGREGATOR_URL}${SEARCH_URL_SUFFIX}"
    MERGED_SEARCH_URL="${MERGED_URL}${SEARCH_URL_SUFFIX}"
    ASSERT_TOPK_EQUAL "$AGGREGATOR_SEARCH_URL" "$MERGED_SEARCH_URL" "${TEST_MAJOR_NUM}.3" "identical results for audio search query '${SEARCH_QUERY}'" "video_audio_results" $TOP_K

    # Task ${TEST_MAJOR_NUM}.4 : ensure the /search endpoint has identical object search results for both the aggregator and merged project
    SEARCH_QUERY="boat"
    RESULT_COUNT=1000
    TOP_K=5
    SEARCH_URL_SUFFIX="search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID3}&text_queries=${SEARCH_QUERY}"
    AGGREGATOR_SEARCH_URL="${AGGREGATOR_URL}${SEARCH_URL_SUFFIX}"
    MERGED_SEARCH_URL="${MERGED_URL}${SEARCH_URL_SUFFIX}"
    ASSERT_TOPK_EQUAL "$AGGREGATOR_SEARCH_URL" "$MERGED_SEARCH_URL" "${TEST_MAJOR_NUM}.4" "identical results for object search query '${SEARCH_QUERY}'" "video_results" $TOP_K

    # Task ${TEST_MAJOR_NUM}.5 : ensure the /search endpoint has identical face search results for both the aggregator and merged project
    FACE_IMG_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/500px-President_Barack_Obama.jpg"
    FACE_IMG_FILE="${QUERY_DATA_DIR}/President_Obama_wikipedia_384x480.jpg"
    if [ ! -f "${FACE_IMG_FILE}" ]; then
        echo "Downloading face image to ${FACE_IMG_FILE} ..."
        curl -sLO "${FACE_IMG_URL}"
        mv "$(basename "${FACE_IMG_URL}")" "${FACE_IMG_FILE}"
    fi
    if [ ! -f "${FACE_IMG_FILE}" ]; then
        echo "Failed to download face image from ${FACE_IMG_URL}"
        exit 1
    fi
    RESULT_COUNT=500
    TOP_K=6
    SEARCH_URL_SUFFIX="search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID2}"
    AGGREGATOR_SEARCH_URL="${AGGREGATOR_URL}${SEARCH_URL_SUFFIX}"
    MERGED_SEARCH_URL="${MERGED_URL}${SEARCH_URL_SUFFIX}"
    ASSERT_TOPK_EQUAL "$AGGREGATOR_SEARCH_URL" "$MERGED_SEARCH_URL" "${TEST_MAJOR_NUM}.5" "identical results for face search query" "video_results" $TOP_K "$FACE_IMG_FILE"

    # Task ${TEST_MAJOR_NUM}.6 : ensure the /search endpoint has identical metadata search results for both the aggregator and merged project
    RESULT_COUNT=500
    TOP_K=4
    SEARCH_URL_SUFFIX="search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=wise/metadata&text_queries=president"
    AGGREGATOR_SEARCH_URL="${AGGREGATOR_URL}${SEARCH_URL_SUFFIX}"
    MERGED_SEARCH_URL="${MERGED_URL}${SEARCH_URL_SUFFIX}"
    ASSERT_SAME_FILENAME_LIST "$AGGREGATOR_SEARCH_URL" "$MERGED_SEARCH_URL" "${TEST_MAJOR_NUM}.6" "identical results for metadata search query" "video_results" $TOP_K
}
TEST_EQUIVALENCE ${MERGED_URL} 6
TEST_EQUIVALENCE ${COMBINED_URL} 7

echo ""
echo "------ Test Summary ------"
printf "%s\n" "${TEST_RESULTS[@]}"
echo "--------------------------"

if [ "$ANY_TEST_FAILED" -eq 1 ]; then
    echo "One or more tests failed."
    cleanup 1
else
    echo "All tests passed."
    cleanup 0
fi
