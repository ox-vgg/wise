#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "This script tests the functionality of WISE2 software using the Wikimedia Commons 25 dataset"
    echo "which contains 25 videos taken from the Wikimedia Commons repository."
    echo ""
    echo "Usage: ${0} WISE2-CODE-DIR TEST-DATA-DIR"
    echo "    WISE2-CODE-DIR : is the path to the WISE2 code directory"
    echo "    TEST-DATA-DIR  : is the path to the WISE2 test data directory"
    echo ""
    echo "This script is designed to be run from the WISE2 tests directory."
    echo "For example, if you have cloned the WISE2 repository to $HOME/wise, run the following commands:"
    echo "    1. cd $HOME/wise/tests/"
    echo "    2. ${0} $HOME/wise/ $HOME/wise-test-data/"
    echo ""
    echo "The TEST-DATA_DIR will contain everything (wise code, test data, dependencies, etc.) required by this script to run the tests."
    echo "In the final stage, this script will start the WISE2 server and run a series of tests to verify the installation."
    exit
fi

# uncomment the following line to produce verbose output and enable debugging
#set -euxo pipefail

# Set these variables to the appropriate values
# for your environment
TEST_ID="wikimedia-commons-25"
VIDEO_FEATURE_ID="mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"
AUDIO_FEATURE_ID="microsoft/clap/2023/four-datasets"
FAISS_INDEX_TYPE="IndexFlatIP"
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10001"
MAX_POLL_SERVER_COUNT=15


OUTDIR=$(realpath ${2})
ENV_DIR="${OUTDIR}/wise-dep"
HUGGINGFACE_HOME="${OUTDIR}/huggingface-home"
DATA_DIR="${OUTDIR}/wise-data"
TEST_DATA_DIR="${OUTDIR}/wise-data/${TEST_ID}"
TEST_DATA_DOWNLOAD_URL="https://thor.robots.ox.ac.uk/wise/assets/test/${TEST_ID}.zip"
PROJECT_BASEDIR="${OUTDIR}/wise-projects"
WISE_PROJECT_DIR="${PROJECT_BASEDIR}/${TEST_ID}/"
WISE_CODE_DIR="${OUTDIR}/wise-code/"

# required tools
REQUIRED_TOOLS=(ffmpeg sqlite3 jq curl rsync unzip)
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        echo "$tool package not found, install the $tool software using your distribution package manager"
        exit 1
    fi
done

start=`date +%s`

## Task: 1. Create a copy of WISE2 code
WISE_CODE_TO_TEST="${1}"
if [[ "${WISE_CODE_TO_TEST}" != */ ]]; then
    WISE_CODE_TO_TEST="${WISE_CODE_TO_TEST}/" # add trailing slash if not present
fi

if [ ! -d "${WISE_CODE_TO_TEST}" ]; then
    echo "The provided WISE2 code directory does not exist"
    exit 1
fi

# create a copy of WISE2 code
echo "Creating a copying of WISE2 code to ${WISE_CODE_DIR} ..."
mkdir -p "${WISE_CODE_DIR}"
rsync -rltv --no-perms --no-owner --no-group --exclude='.git/' --exclude='**/__pycache__/'\
    "${WISE_CODE_TO_TEST}"\
    "${WISE_CODE_DIR}"

## Task: 2. Install WISE2 dependencies
export HF_HOME=$HUGGINGFACE_HOME
if [ ! -d "${HF_HOME}" ]; then
    echo "Creating huggingface cache directory in ${HF_HOME} ..."
    mkdir -p $HF_HOME
fi

if [ ! -d "${ENV_DIR}" ]; then
    echo "Creating python venv in ${ENV_DIR} ..."
    python3 -m venv "$ENV_DIR"
fi
echo "Ensuring dependencies are installed..."
source "${ENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
pip install -r "${WISE_CODE_DIR}requirements.txt" -r "${WISE_CODE_DIR}torch-faiss-requirements.txt"
pip install --no-deps msclap==1.3.3


## Task: 3. Download test dataset
if [ ! -d "${TEST_DATA_DIR}" ]; then
    echo "Downloading test dataset to ${DATA_DIR} ..."
    mkdir -p "${TEST_DATA_DIR}"
    cd "${DATA_DIR}"
    curl -sLO $TEST_DATA_DOWNLOAD_URL
    unzip -q "${TEST_ID}.zip" -d "${DATA_DIR}"
    rm "${TEST_ID}.zip"
else
    echo "Skipping test dataset download"
fi

## Task: 4.1 Extract features
if [ ! -d "${WISE_PROJECT_DIR}" ]; then
    echo "Extracting features from videos (takes about 3 min.) ..."
    cd "${WISE_CODE_DIR}"
    python extract-features.py \
           "${TEST_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 4096 \
           --shard-maxsize 20971520 \
           --num-workers 0 \
           --feature-store webdataset \
           --video-feature-id "${VIDEO_FEATURE_ID}" \
           --audio-feature-id "${AUDIO_FEATURE_ID}" \
           --project-dir "$WISE_PROJECT_DIR"
fi

## Task: 4.2 Import media metadata
METADATA_DB_FILE="${WISE_PROJECT_DIR}metadata/internal.db"
METADATA_TABLE_NAME="metadata-${TEST_ID}"
RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
    echo "Importing metadata from ${TEST_DATA_DIR}/media-metadata.csv (takes few seconds) ..."
    cd "${WISE_CODE_DIR}"
    python3 media-metadata.py import \
            --metadata-id "${TEST_ID}" \
            --from-csv "${TEST_DATA_DIR}/media-metadata.csv" \
            --metadata-type "media" \
            --project-dir "$WISE_PROJECT_DIR"
fi

## Test 4.2.1 : check if the metadata table exists
TABLE_NAME=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$TABLE_NAME" != "$METADATA_TABLE_NAME" ]; then
    echo "Test 4.2.1 FAILED: metadata table $TABLE_NAME does not exist in $METADATA_DB_FILE"
    exit 1
else
    echo "Test 4.2.1 PASSED"
fi

## Test 4.2.2 : check if all the metadata rows are imported
ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_TABLE_NAME';")
TRUE_ROW_COUNT=$(wc -l < "${TEST_DATA_DIR}/media-metadata.csv")
TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
    echo "Test 4.2.2 FAILED: metadata table $TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
    exit 1
else
    echo "Test 4.2.2 PASSED"
fi

## 4.3 Create search index for features and metadata
VIDEO_INDEX_FILENAME="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID}/index/video-${FAISS_INDEX_TYPE}.faiss"
AUDIO_INDEX_FILENAME="${WISE_PROJECT_DIR}store/${AUDIO_FEATURE_ID}/index/audio-${FAISS_INDEX_TYPE}.faiss"
if [ ! -f "${VIDEO_INDEX_FILENAME}" ] || [ ! -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Creating index (takes about 1 min.) ..."
    cd "${WISE_CODE_DIR}"
    python create-index.py \
           --index-type "${FAISS_INDEX_TYPE}" \
           --project-dir "$WISE_PROJECT_DIR"
fi
# Test 4.3.1 : check if the video index files exist
if [ ! -f $VIDEO_INDEX_FILENAME ]; then
    echo "Test 4.3.1 FAILED: video index file ${VIDEO_INDEX_FILENAME} does not exist"
    exit 1
else
    echo "Test 4.3.1 PASSED"
fi

# Test 4.3.2 : check if the audio index files exist
if [ ! -f $AUDIO_INDEX_FILENAME ]; then
    echo "Test 4.3.2 FAILED: audio index file ${AUDIO_INDEX_FILENAME} does not exist"
    exit 1
else
    echo "Test 4.3.2 PASSED"
fi

## 5 Start WISE2 server
# Define cleanup function to run on Ctrl+C
cleanup() {
    echo -e "\nCaught Ctrl+C. Shutting down server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}

# Update WISE2 http server config
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10001"
WISE2_CONFIG_FILE="${WISE_CODE_DIR}config.py"
sed -i "s/^    listen_address: str = \".*\"/    listen_address: str = \"$HTTP_SERVER_HOST\"/" "$WISE2_CONFIG_FILE"
sed -i "s/^    port: int = .*/    port: int = $HTTP_SERVER_PORT/" "$WISE2_CONFIG_FILE"

echo "Starting WISE2 server (takes about 1 min.) ..."
cd "${WISE_CODE_DIR}"
python serve.py \
        --index-type "${FAISS_INDEX_TYPE}" \
        --project-dir "$WISE_PROJECT_DIR" & # to start the server in the background
SERVER_PID=$!
trap cleanup SIGINT
trap cleanup SIGTERM
trap cleanup EXIT

SERVER_URL="http://${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT}/${TEST_ID}/"
PROJECT_INFO_URL="${SERVER_URL}info"
SLEEP_DURATION=5
# Wait for the server to start  
# poll server every 5 seconds for 30 seconds
for ((i=1; i<=MAX_POLL_SERVER_COUNT; i++)); do
    if curl -s --head --request GET "${PROJECT_INFO_URL}" | grep "200 OK" > /dev/null; then
        echo "Checking if server is running at ${PROJECT_INFO_URL}"
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

sleep 5 # give the server some time to start
echo "Server started successfully."

# Test 5.1 : check if the server is running
if curl -s --head --request GET "${SERVER_URL}" | grep "200 OK" > /dev/null; then
    echo "Test 5.1 PASSED"
else
    echo "Test 5.1 FAILED: server is not running at ${SERVER_URL}"
    exit 1
    cleanup
fi

# Test 5.2 : check project info
response=$(curl curl -s -X GET -H "Content-Type: application/json" "${PROJECT_INFO_URL}")
project_name=$(echo "$response" | jq -r '.project_name')
if [ "$project_name" == "$TEST_ID" ]; then
    echo "Test 5.2 PASSED"
else
    echo "Test 5.2 FAILED : project name is $project_name, expected $TEST_ID"
    exit 1
    cleanup
fi

# Test 5.3 : check if the server returns correct results (including metadata) for query on video
SEARCH_QUERY="bees"
RESULT_COUNT=60
SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&text_queries=${SEARCH_QUERY}"
response=$(curl curl -s -X POST -H "Content-Type: application/json" "${SEARCH_URL}")

if [ "$TEST_ID" == "wikimedia-commons-25" ]; then
    # The WISE server's JSON response for search query is as follows:
    # {
    #     ...
    #     "video_results": {
    #         "total": 300,
    #         "unmerged_windows": [ ... ],
    #         "merged_windows": [
    #             {
    #                 "media_id": "5",
    #                 "distance": 0.13519121706485748,
    #                 "ts": 0,
    #                 "te": 22.5,
    #                 ...
    #             }, ...
    #         ],
    #         "videos": {
    #             "5": {
    #                 "filename": "Bees_drinking-finding_minerals_-_video.mp4",
    #                 "external_metadata": {
    #                     "description": "...",
    #                     "source_url": "...",
    #                     "author": "...",
    #                     ...
    #                 },
    #                 ...
    #             },
    #             ...
    #         }
    #     }
    # }
    response_selected_json=$(echo "$response" | jq -c ' . as $root | {
        merged_windows: [
        .video_results.merged_windows[]
        | .media_id as $id
        | {filename: $root.video_results.videos[$id].filename, source_url: $root.video_results.videos[$id].external_metadata.source_url}
        ]
    }')

    expected_json='{
      "merged_windows": [
        {
          "filename": "Bees_drinking-finding_minerals_-_video.mp4",
          "source_url": "https://commons.wikimedia.org/wiki/File:Bees_drinking-finding_minerals_-_video.webm"
        },
        {
          "filename": "Bee_wash_and_brush_up._Andrena_dorsata_video.mp4",
          "source_url": "https://commons.wikimedia.org/wiki/File:Bee_wash_and_brush_up._Andrena_dorsata_video.webm"
        }
      ]
    }'

    if diff <(echo "$expected_json" | jq -S .) <(echo "$response_selected_json" | jq -S .) > /dev/null; then
        echo "Test 5.3 PASSED"
    else
        echo "Test 5.3 FAILED: unexpected search results"
        echo "Expected:"
        echo "$expected_json" | jq .
        echo "Actual:"
        echo "$response_selected_json" | jq .
        exit 1
    fi
fi

# Test 5.4 : check if the server returns correct results (including metadata) for query on audio
SEARCH_QUERY="fire+engine+siren"
RESULT_COUNT=1
SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=av&text_queries=${SEARCH_QUERY}"
response=$(curl curl -s -X POST -H "Content-Type: application/json" "${SEARCH_URL}")

if [ "$TEST_ID" == "wikimedia-commons-25" ]; then
    response_selected_json=$(echo "$response" | jq -c '{
        merged_windows: [
        .video_audio_results.merged_windows[] as $mw
        | {
            filename: .video_audio_results.videos[$mw.media_id].filename,
            source_url: .video_audio_results.videos[$mw.media_id].external_metadata.source_url,
            description: .video_audio_results.videos[$mw.media_id].external_metadata.description
        }
        ]
    }')

    expected_json='{
      "merged_windows": [
        {
          "filename": "Fire_Department_Railroad_Street_Saint_Johnsbury_VT_July_2022.mp4",
          "source_url": "https://commons.wikimedia.org/wiki/File:Fire_Department_Railroad_Street_Saint_Johnsbury_VT_July_2022.webm",
          "description": "Fire Department to the rescue on Railroad Street in Saint Johnsbury, Vermont."
        }
      ]
    }'
    if diff <(echo "$expected_json" | jq -S .) <(echo "$response_selected_json" | jq -S .) > /dev/null; then
        echo "Test 5.4 PASSED"
    else
        echo "Test 5.4 FAILED: unexpected search results"
        echo "Expected:"
        echo "$expected_json" | jq .
        echo "Actual:"
        echo "$response_selected_json" | jq .
        exit 1
    fi
fi

end_time=`date +%s`
elapsed_time=$((end_time-start))
echo ""
echo "*** All tests for ${TEST_ID} completed in ${elapsed_time} sec. ***"
echo ""

cleanup