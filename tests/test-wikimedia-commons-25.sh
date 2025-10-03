#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "This script tests the functionality of WISE2 software using the Wikimedia Commons 25 dataset"
    echo "which contains 25 videos taken from the Wikimedia Commons repository."
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
TEST_ID="wikimedia-commons-25"
VIDEO_FEATURE_ID1="mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"
VIDEO_FEATURE_ID2="deepinsight/insightface/buffalo_l/_unknown"
AUDIO_FEATURE_ID="microsoft/clap/2023/four-datasets"
FAISS_INDEX_TYPE="IndexFlatIP"
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10001"
MAX_POLL_SERVER_COUNT=15
NUM_WORKERS=2

FEATURE_STORE="webdataset" # options are: faiss, webdataset, numpy
EXTENSION="tar" # options are: faiss, tar, npy

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
    echo "Extracting features from videos using ${NUM_WORKERS} workers (takes about 3 min.) ..."
    cd "${WISE_CODE_DIR}"
    python extract-features.py \
           "${TEST_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 4096 \
           --shard-maxsize 20971520 \
           --num-workers $NUM_WORKERS \
           --feature-store ${FEATURE_STORE} \
           --video-feature-id "${VIDEO_FEATURE_ID1}" \
           --video-feature-id "${VIDEO_FEATURE_ID2}" \
           --audio-feature-id "${AUDIO_FEATURE_ID}" \
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
VIDEO_INDEX_FILENAME1="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID1}/index/video-${FAISS_INDEX_TYPE}.faiss"
VIDEO_INDEX_FILENAME2="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID2}/index/video-${FAISS_INDEX_TYPE}.faiss"
AUDIO_INDEX_FILENAME="${WISE_PROJECT_DIR}store/${AUDIO_FEATURE_ID}/index/audio-${FAISS_INDEX_TYPE}.faiss"
if [ ! -f "${VIDEO_INDEX_FILENAME1}" ] || [ ! -f "${VIDEO_INDEX_FILENAME2}" ] || [ ! -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Creating index (takes about 1 min.) ..."
    cd "${WISE_CODE_DIR}"
    python create-index.py \
           --index-type "${FAISS_INDEX_TYPE}" \
           --project-dir "$WISE_PROJECT_DIR"
fi
# Test 4.1 : check if the video index files exist
if [ ! -f "${VIDEO_INDEX_FILENAME1}" ] && [ ! -f "${VIDEO_INDEX_FILENAME2}" ]; then
    echo "Test 4.1 FAILED: video index files ${VIDEO_INDEX_FILENAME1} and ${VIDEO_INDEX_FILENAME2} do not exist"
    exit 1
else
    echo "Test 4.1 PASSED"
fi

# Test 4.2 : check if the audio index files exist
if [ ! -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Test 4.2 FAILED: audio index file ${AUDIO_INDEX_FILENAME} does not exist"
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
if [ ! -d "frontend/dist" ]; then
    mkdir -p frontend/dist && \
        (cd frontend && npm ci && npm run build)
fi

echo "Starting WISE2 server on ${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT} (takes about 1 min.) ..."
cd "${WISE_CODE_DIR}"
LISTEN_ADDRESS=$HTTP_SERVER_HOST PORT=$HTTP_SERVER_PORT python serve.py \
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
response=$(curl -s -X GET -H "Content-Type: application/json" "${PROJECT_INFO_URL}")
expected_project_name="${TEST_ID}"
expected_video_targets=(
    "mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"
    "deepinsight/insightface/buffalo_l/_unknown"
)
expected_audio_targets=(
    "microsoft/clap/2023/four-datasets"
)

project_name=$(echo "$response" | jq -r '.project_name')
video_search_targets=($(echo "$response" | jq -r '.search_targets.video[]'))
audio_search_targets=($(echo "$response" | jq -r '.search_targets.audio[]'))

if [ "$project_name" = "$expected_project_name" ]; then
    echo "Test 5.2a PASSED"
else
    echo "Test 5.2a FAILED: project_name does not match expected value"
    echo "Expected: $expected_project_name"
    echo "Actual: $project_name"
    exit 1
    cleanup
fi

# Function to check if two arrays have the same elements (order-independent)
arrays_equal_any_order() {
    local -n arr1=$1
    local -n arr2=$2
    local sorted1 sorted2
    sorted1=($(printf "%s\n" "${arr1[@]}" | sort))
    sorted2=($(printf "%s\n" "${arr2[@]}" | sort))
    if [ "${#sorted1[@]}" -ne "${#sorted2[@]}" ]; then
        return 1
    fi
    for i in "${!sorted1[@]}"; do
        if [ "${sorted1[$i]}" != "${sorted2[$i]}" ]; then
            return 1
        fi
    done
    return 0
}

if arrays_equal_any_order video_search_targets expected_video_targets; then
    echo "Test 5.2b PASSED"
else
    echo "Test 5.2b FAILED: video search_targets do not match expected values"
    echo "Expected: ${expected_video_targets[*]}"
    echo "Actual: ${video_search_targets[*]}"
    exit 1
    cleanup
fi

if arrays_equal_any_order audio_search_targets expected_audio_targets; then
    echo "Test 5.2c PASSED"
else
    echo "Test 5.2c FAILED: audio search_targets do not match expected values"
    echo "Expected: ${expected_audio_targets[*]}"
    echo "Actual: ${audio_search_targets[*]}"
    exit 1
    cleanup
fi

# Test 5.3 : check if the server returns correct results (including metadata) for query on video
if [ "$VIDEO_FEATURE_ID1" == "mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli" ]; then
    SEARCH_QUERY="bees"
    RESULT_COUNT=60
    SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID1}&text_queries=${SEARCH_QUERY}"
    response=$(curl -s -X POST -H "Content-Type: application/json" "${SEARCH_URL}")

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
if [ "$AUDIO_FEATURE_ID" == "microsoft/clap/2023/four-datasets" ]; then
    SEARCH_QUERY="fire+engine+siren"
    RESULT_COUNT=1
    SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=av&feature_extractor_id=${AUDIO_FEATURE_ID}&text_queries=${SEARCH_QUERY}"
    response=$(curl -s -X POST -H "Content-Type: application/json" "${SEARCH_URL}")

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

# Test 5.5 : check if the server returns correct results (including metadata) for query on face
if [ "$VIDEO_FEATURE_ID2" == "deepinsight/insightface/buffalo_l/_unknown" ]; then
    FACE_IMG_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Christy_Turlington_and_Edward_Burns_at_the_2024_Toronto_International_Film_Festival_%28cropped%29.jpg/250px-Christy_Turlington_and_Edward_Burns_at_the_2024_Toronto_International_Film_Festival_%28cropped%29.jpg"
    FACE_IMG_FILE="${QUERY_DATA_DIR}/Christy_Turlington_wikipedia_180x240.jpg"
    if [ ! -f "${FACE_IMG_FILE}" ]; then
        echo "Downloading face image to ${FACE_IMG_FILE} ..."
        curl -sLO "${FACE_IMG_URL}"
        mv "$(basename "${FACE_IMG_URL}")" "${FACE_IMG_FILE}"
    fi
    if [ ! -f "${FACE_IMG_FILE}" ]; then
        echo "Failed to download face image from ${FACE_IMG_URL}"
        exit 1
    fi
    RESULT_COUNT=3
    SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID2}"
    response=$(curl -s -X POST "${SEARCH_URL}" -F "image_file_queries=@${FACE_IMG_FILE}")
    echo $response
    response_selected_json=$(echo "$response" | jq -c '{
    results: [
        .video_results.merged_windows[]
        as $w
        | {
            filename: .video_results.videos[$w.media_id].filename,
            ts: $w.ts,
            te: $w.te
        }
    ]
    }')
    expected_json='{
    "results": [
        {
        "filename": "Celebrities_Against_Smoking_Christy_Turlington_PSA.mp4",
        "ts": 28,
        "te": 33.5
        },
        {
        "filename": "Celebrities_Against_Smoking_Christy_Turlington_PSA.mp4",
        "ts": 16,
        "te": 20
        }
    ]
    }'
    if diff <(echo "$expected_json" | jq -S 'sort_by(.filename) | sort_by(.ts) | sort_by(.te)' .) <(echo "$response_selected_json" | jq -S 'sort_by(.filename) | sort_by(.ts) | sort_by(.te)' .) > /dev/null; then
        echo "Test 5.5 PASSED"
    else
        echo "Test 5.5 FAILED: unexpected search results"
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