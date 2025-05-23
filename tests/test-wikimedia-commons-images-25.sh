#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "This script tests the functionality of WISE2 software using the Wikimedia Commons Images 25 dataset"
    echo "which contains 25 images taken from the Wikimedia Commons repository."
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
TEST_ID="wikimedia-commons-images-25"
IMAGE_FEATURE_ID1="mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"
IMAGE_FEATURE_ID2="deepinsight/insightface/buffalo_l/_unknown"
IMAGE_FEATURE_ID3="transformers/owlv2/google/owlv2-large-patch14-ensemble"
FAISS_INDEX_TYPE="IndexFlatIP"
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10001"
MAX_POLL_SERVER_COUNT=15
CUDA_VISIBLE_DEVICES=1

WISE_CODE_DIR=`pwd`
TMP_DIR=$(realpath ${1})
OUTDIR="${TMP_DIR}/wise-test/"
mkdir -p "${OUTDIR}"

DATA_DIR="${OUTDIR}/test-data"
TEST_DATA_DIR="${DATA_DIR}/${TEST_ID}/"
TEST_DATA_DOWNLOAD_URL="https://thor.robots.ox.ac.uk/wise/assets/test/${TEST_ID}.zip"
WISE_PROJECT_DIR="${OUTDIR}/wise-project/${TEST_ID}/"

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

## Task: 2. Extract features
if [ ! -d "${WISE_PROJECT_DIR}" ]; then
    echo "Extracting features from videos (takes about 3 min.) ..."
    cd "${WISE_CODE_DIR}"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python extract-features.py \
           "${TEST_DATA_DIR}" \
           --media-include "*.jpg" \
           --shard-maxcount 4096 \
           --shard-maxsize 20971520 \
           --num-workers 0 \
           --feature-store webdataset \
           --image-feature-id "${IMAGE_FEATURE_ID1}" \
           --image-feature-id "${IMAGE_FEATURE_ID2}" \
           --image-feature-id "${IMAGE_FEATURE_ID3}" \
           --project-dir "$WISE_PROJECT_DIR"
fi

## Task: 3. Import media metadata
METADATA_DB_FILE="${WISE_PROJECT_DIR}metadata/internal.db"
METADATA_TABLE_NAME="metadata-${TEST_ID}"
RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$RESULT" != "$METADATA_TABLE_NAME" ]; then
    echo "Importing metadata from ${TEST_DATA_DIR}/media-metadata.csv (takes few seconds) ..."
    python3 media-metadata.py import \
            --metadata-id "${TEST_ID}" \
            --from-csv "${TEST_DATA_DIR}/media-metadata.csv" \
            --metadata-type "media" \
            --project-dir "$WISE_PROJECT_DIR"
fi

## Test 3.1 : check if the metadata table exists
TABLE_NAME=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$METADATA_TABLE_NAME';")
if [ "$TABLE_NAME" != "$METADATA_TABLE_NAME" ]; then
    echo "Test 3.1 FAILED: metadata table $TABLE_NAME does not exist in $METADATA_DB_FILE"
    exit 1
else
    echo "Test 3.1 PASSED"
fi

## Test 3.2 : check if all the metadata rows are imported
ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_TABLE_NAME';")
TRUE_ROW_COUNT=$(wc -l < "${TEST_DATA_DIR}/media-metadata.csv")
TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
    echo "Test 3.2 FAILED: metadata table $TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
    exit 1
else
    echo "Test 3.2 PASSED"
fi

## Task 4. Create search index for features and metadata
IMAGE_INDEX_FILENAME1="${WISE_PROJECT_DIR}store/${IMAGE_FEATURE_ID1}/index/image-${FAISS_INDEX_TYPE}.faiss"
IMAGE_INDEX_FILENAME2="${WISE_PROJECT_DIR}store/${IMAGE_FEATURE_ID2}/index/image-${FAISS_INDEX_TYPE}.faiss"
IMAGE_INDEX_FILENAME3="${WISE_PROJECT_DIR}store/${IMAGE_FEATURE_ID3}/index/image-${FAISS_INDEX_TYPE}.faiss"

if [ ! -f "${IMAGE_INDEX_FILENAME1}" ] || [ ! -f "${IMAGE_INDEX_FILENAME2}" ] || [ ! -f "${IMAGE_INDEX_FILENAME3}" ]; then
    echo "Creating index (takes about 1 min.) ..."
    cd "${WISE_CODE_DIR}"
    FTS_CONFIG_FILE="${WISE_PROJECT_DIR}metadata-fts-config.json"
    echo "{ \"${METADATA_TABLE_NAME}\": [ \"description\", \"date\", \"source_url\", \"author\" ] }" > "${FTS_CONFIG_FILE}"
    python create-index.py \
        --media-type "image" \
        --media-type "metadata" \
        --fts-config "${FTS_CONFIG_FILE}" \
        --index-type "${FAISS_INDEX_TYPE}" \
        --project-dir "$WISE_PROJECT_DIR"
fi
# Test 4.1 : check if the image index files exist
if [ ! -f "${IMAGE_INDEX_FILENAME1}" ] && [ ! -f "${IMAGE_INDEX_FILENAME2}" ] && [ ! -f "${IMAGE_INDEX_FILENAME3}" ]; then
    echo "Test 4.1 FAILED: image index files ${IMAGE_INDEX_FILENAME1}, ${IMAGE_INDEX_FILENAME2}, and ${IMAGE_INDEX_FILENAME3} do not exist"
    exit 1
else
    echo "Test 4.1 PASSED"
fi

## Test 4.2 : check if all the metadata full text search index (FTS) files exist
METADATA_FTS_TABLE_NAME="metadata_fts"
ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$METADATA_FTS_TABLE_NAME';")
TRUE_ROW_COUNT=$(wc -l < "${TEST_DATA_DIR}/media-metadata.csv")
TRUE_ROW_COUNT=$((TRUE_ROW_COUNT - 1)) # subtract 1 for the header row
if [ "$ROW_COUNT" -ne "$TRUE_ROW_COUNT" ]; then
    echo "Test 4.2 FAILED: metadata FTS table $METADATA_FTS_TABLE_NAME has $ROW_COUNT rows, expected $TRUE_ROW_COUNT rows"
    exit 1
else
    echo "Test 4.2 PASSED"
fi

## 5 Start WISE2 server
# Define cleanup function to run on Ctrl+C
cleanup() {
    echo -e "\nCaught Ctrl+C. Shutting down server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}

echo "Starting WISE2 server on ${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT} (takes about 1 min.) ..."
cd "${WISE_CODE_DIR}"
LISTEN_ADDRESS=$HTTP_SERVER_HOST PORT=$HTTP_SERVER_PORT python serve.py \
        --index-type "${FAISS_INDEX_TYPE}" \
        --search-target video:open_clip \
        --search-target face:insightface \
        --search-target audio:clap \
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
response=$(curl -s -X GET -H "Content-Type: application/json" "${PROJECT_INFO_URL}")
expected_json='{
  "project_name": "wikimedia-commons-images-25",
  "search_targets": {
    "image": [
      "mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli",
      "deepinsight/insightface/buffalo_l/_unknown",
      "transformers/owlv2/google/owlv2-large-patch14-ensemble",
      "wise/metadata"
    ]
  }
}'
response_selected_json=$(jq '{project_name, search_targets}' <<< "$response")
if diff <(jq -S . <<< "$response_selected_json") <(jq -S . <<< "$expected_json") >/dev/null; then
    echo "Test 5.2 PASSED"
else
    echo "Test 5.2 FAILED: project info does not match expected values"
    echo "Expected: $expected_json"
    echo "Actual: $response_selected_json"
    exit 1
    cleanup
fi

# Test 5.3 : check if the server returns correct results for metadata search
METADATA_SEARCH_QUERY="church"
RESULT_COUNT=3
SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=image&feature_extractor_id=wise/metadata&text_queries=${METADATA_SEARCH_QUERY}"
response=$(curl -s -X POST -H "Content-Type: application/json" "${SEARCH_URL}")
response_selected_json=$(echo "$response" | jq -c '{
  media_ids: [.image_results.vectors[].media_id],
  images: (
    .image_results.images | 
    to_entries | 
    map({
      id: .value.id,
      filename: .value.filename,
      external_metadata: .value.external_metadata
    })
  )
}')
expected_json='{
  "media_ids": [
    "16"
  ],
  "images": [
    {
      "id": "16",
      "filename": "960px-St_Nikolaus_Mittelberg_South_Tyrol_Rainbow.jpg",
      "external_metadata": {
        "asr_segments": [],
        "description": " The picture shows the St. Nikolaus <b>Church</b> in Mittelberg on the Ritten plateau, just before a huge thunderstorm. My first images had the strong rainbow, but with the <b>church</b> still in the shadow, but luckily the sky opened a bit further, placing the <b>church</b> in the limelight.",
        "date": "26 April 2019",
        "source_url": "https://commons.wikimedia.org/wiki/File:St_Nikolaus_Mittelberg_South_Tyrol_Rainbow.jpg",
        "author": "C-M"
      }
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

end_time=`date +%s`
elapsed_time=$((end_time-start))
echo ""
echo "*** All tests for ${TEST_ID} completed in ${elapsed_time} sec. ***"
echo ""

cleanup