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
    echo "This script tests the functionality of WISE software using the Wikimedia Commons Edited Videos dataset"
    echo "which contains edited videos (i.e. with shots) taken from the Wikimedia Commons repository. This test is"
    echo "is designed to mainly test: (a) incremental extraction of features, and (b) sampling video frames based on the shots."
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

TEST_ID="wikimedia-commons-edited-videos"
VIDEO_FEATURE_ID1="mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli"
VIDEO_FEATURE_ID2="deepinsight/insightface/buffalo_l/_unknown"
AUDIO_FEATURE_ID="microsoft/clap/2023/four-datasets"
FAISS_INDEX_TYPE="IndexFlatIP"
HTTP_SERVER_HOST="0.0.0.0"
HTTP_SERVER_PORT="10001"
MAX_POLL_SERVER_COUNT=60
NUM_WORKERS=2

EXTENSION="faiss" # options are: faiss, tar, npy

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
    unzip -q "${TEST_ID}.zip" -d "${DATA_DIR}"
    rm "${TEST_ID}.zip"
    if [ ! -d "${TEST_DATA_DIR}" ]; then
        echo "Failed to download from ${TEST_DATA_DOWNLOAD_URL} and extract to test dataset in ${TEST_DATA_DIR}"
        exit 1
    fi
else
    echo "Skipping test dataset download"
fi

## Task: 2. Extract audio features
if [ ! -d "${WISE_PROJECT_DIR}" ]; then
    echo "Extracting audio features from videos with ${NUM_WORKERS} workers (takes about 3 min.) ..."
    cd "${WISE_CODE_DIR}"
    python extract-features.py \
        "${TEST_DATA_DIR}" \
        --media-include "*.mp4" \
        --shard-maxcount 16 \
        --num-workers $NUM_WORKERS \
        --no-thumbnails \
        --audio-feature-id "${AUDIO_FEATURE_ID}" \
        --project-dir "$WISE_PROJECT_DIR"
    if [ $? -ne 0 ]; then
        echo "Audio feature extraction failed, please check the logs for more details"
        exit 1
    fi
fi

AUDIO_FEATURE_STORE="${WISE_PROJECT_DIR}store/${AUDIO_FEATURE_ID}/"
# Test 2.1 : check if the audio feature store directory exists
if [ ! -d "${AUDIO_FEATURE_STORE}" ]; then
    echo "Test 2.1 FAILED: audio feature store directory ${AUDIO_FEATURE_STORE} does not exist"
    exit 1
else
    echo "Test 2.1 PASSED"
fi
# Test 2.2 : check that the audio feature store directory contains more than 0 *.${EXTENSION} files
AUDIO_TAR_COUNT=$(find "${AUDIO_FEATURE_STORE}" -type f -name "*.${EXTENSION}" | wc -l)
if [ "$AUDIO_TAR_COUNT" -eq 0 ]; then
    echo "Test 2.2 FAILED: audio feature store directory ${AUDIO_FEATURE_STORE} does not contain any *.${EXTENSION} files"
    exit 1
else
    echo "Test 2.2 PASSED: audio feature store directory ${AUDIO_FEATURE_STORE} contains ${AUDIO_TAR_COUNT} *.${EXTENSION} files"
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
    if [ $? -ne 0 ]; then
        echo "Failed to import media metadata, please check the logs for more details"
        exit 1
    fi
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

## Task 4. Import shots metadata
SHOTS_TABLE_NAME="shots"
RESULT=$(sqlite3 "$METADATA_DB_FILE" "SELECT name FROM sqlite_master WHERE type='table' AND name='$SHOTS_TABLE_NAME';")
if [ "$RESULT" != "$SHOTS_TABLE_NAME" ]; then
    echo "Test 4.1 FAILED: ${SHOTS_TABLE_NAME} does not exist in $METADATA_DB_FILE"
else
    echo "Test 4.1 PASSED"
fi

# ensure that the table is empty
ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$SHOTS_TABLE_NAME';")
if [ "$ROW_COUNT" -eq 0 ]; then
    # import shots metadata
    python3 media-metadata.py \
        import-shots \
        --project-dir "$WISE_PROJECT_DIR" \
        --from-csv "${TEST_DATA_DIR}/shots.csv"
    if [ $? -ne 0 ]; then
        echo "Test 4.2 FAILED: shots metadata import failed, please check the logs for more details"
        exit 1
    else
        echo "Test 4.2 PASSED: shots metadata imported successfully"
    fi
fi

ROW_COUNT=$(sqlite3 "$METADATA_DB_FILE" "SELECT COUNT(*) FROM '$SHOTS_TABLE_NAME';")
EXPECTED_ROW_COUNT=60 # 60 shots are contained in the shots.csv file
if [ "$ROW_COUNT" -ne "$EXPECTED_ROW_COUNT" ]; then
    echo "Test 4.3 FAILED: ${SHOTS_TABLE_NAME} contains $ROW_COUNT rows, expected $EXPECTED_ROW_COUNT rows"
    exit 1
else
    echo "Test 4.3 PASSED"
fi

## Task: 5. Extract the remaining features (video and face)
FEATURE_STORE1="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID1}/"
if [ ! -d "${FEATURE_STORE1}" ]; then
    cd "${WISE_CODE_DIR}"
    echo "Extracting features from videos (takes about 3 min.) ..."
    python extract-features.py \
        --yes \
        --media-include "*.mp4" \
        --shard-maxcount 32 \
        --num-workers $NUM_WORKERS \
        --no-thumbnails \
        --use-shots \
        --video-feature-id "${VIDEO_FEATURE_ID1}" \
        --project-dir "$WISE_PROJECT_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to extract features from videos, please check the logs for more details"
        exit 1
    fi
fi

FEATURE_STORE2="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID2}/"
if [ ! -d "${FEATURE_STORE2}" ]; then
    cd "${WISE_CODE_DIR}"
    echo "Extracting face features from videos (takes about 3 min.) ..."
    python extract-features.py \
        --yes \
        --media-include "*.mp4" \
        --shard-maxcount 64 \
        --num-workers 0 \
        --thumbnails \
        --use-shots \
        --video-feature-id "${VIDEO_FEATURE_ID2}" \
        --project-dir "$WISE_PROJECT_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to extract face features from videos, please check the logs for more details"
        exit 1
    fi
fi

# Test 5.1 : check if the video feature store directories exist
if [ ! -d "${FEATURE_STORE1}" ] || [ ! -d "${FEATURE_STORE2}" ]; then
    echo "Test 5.1 FAILED: video feature store directories ${FEATURE_STORE1} and ${FEATURE_STORE2} do not exist"
    exit 1
else
    echo "Test 5.1 PASSED"
fi
# Test 5.2 : check that the video feature store directories contains more than 0 *.${EXTENSION} files
VIDEO_TAR_COUNT1=$(find "${FEATURE_STORE1}" -type f -name "*.${EXTENSION}" | wc -l)
VIDEO_TAR_COUNT2=$(find "${FEATURE_STORE2}" -type f -name "*.${EXTENSION}" | wc -l)
if [ "$VIDEO_TAR_COUNT1" -eq 0 ] || [ "$VIDEO_TAR_COUNT2" -eq 0 ]; then
    echo "Test 5.2 FAILED: video feature store directories ${FEATURE_STORE1} and ${FEATURE_STORE2} do not contain any *.${EXTENSION} files"
    exit 1
else
    echo "Test 5.2 PASSED: video feature store directories ${FEATURE_STORE1} and ${FEATURE_STORE2} contain ${VIDEO_TAR_COUNT1} and ${VIDEO_TAR_COUNT2} *.${EXTENSION} files, respectively"
fi

## Task 6. Create search index for features and metadata
VIDEO_INDEX_FILENAME1="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID1}/index/video-${FAISS_INDEX_TYPE}.faiss"
VIDEO_INDEX_FILENAME2="${WISE_PROJECT_DIR}store/${VIDEO_FEATURE_ID2}/index/video-${FAISS_INDEX_TYPE}.faiss"
AUDIO_INDEX_FILENAME="${WISE_PROJECT_DIR}store/${AUDIO_FEATURE_ID}/index/audio-${FAISS_INDEX_TYPE}.faiss"
FTS_CONFIG_FILE="${WISE_PROJECT_DIR}fts-config.json"
echo "{ \"metadata-${TEST_ID}\": [ \"description\", \"date\", \"source_url\", \"author\" ] }" > "${FTS_CONFIG_FILE}"

if [ ! -f "${VIDEO_INDEX_FILENAME1}" ] || [ ! -f "${VIDEO_INDEX_FILENAME2}" ] || [ ! -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Creating search index (takes about 1 min.) ..."
    cd "${WISE_CODE_DIR}"
    python create-index.py \
           --index-type "${FAISS_INDEX_TYPE}" \
           --fts-config "${FTS_CONFIG_FILE}" \
           --project-dir "$WISE_PROJECT_DIR"
fi

# Test 6.1 : check if the video index files exist
if [ ! -f "${VIDEO_INDEX_FILENAME1}" ] && [ ! -f "${VIDEO_INDEX_FILENAME2}" ]; then
    echo "Test 6.1 FAILED: video index files ${VIDEO_INDEX_FILENAME1} and ${VIDEO_INDEX_FILENAME2} do not exist"
    exit 1
else
    echo "Test 6.1 PASSED"
fi

# Test 6.2 : check if the audio index files exist
if [ ! -f "${AUDIO_INDEX_FILENAME}" ]; then
    echo "Test 6.2 FAILED: audio index file ${AUDIO_INDEX_FILENAME} does not exist"
    exit 1
else
    echo "Test 6.2 PASSED"
fi

## 7 Start WISE server
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

echo "Starting WISE server on ${HTTP_SERVER_HOST}:${HTTP_SERVER_PORT} ( takes about 1 min.) ..."
cd "${WISE_CODE_DIR}"
USE_SHOTS=1 LISTEN_ADDRESS=$HTTP_SERVER_HOST PORT=$HTTP_SERVER_PORT python serve.py \
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

# Test 7.1 : check if the server is running
if curl -s --head --request GET "${SERVER_URL}" | grep "200 OK" > /dev/null; then
    echo "Test 7.1 PASSED"
else
    echo "Test 7.1 FAILED: server is not running at ${SERVER_URL}"
    exit 1
    cleanup
fi

# Test 7.2 : check project info
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
    echo "Test 7.2 PASSED"
else
    echo "Test 7.2 FAILED: project_name does not match expected value"
    echo "Expected: $expected_project_name"
    echo "Actual: $project_name"
    exit 1
    cleanup
fi

# Test 8.1 : check if the server returns correct results (including metadata) for query on audio
if [ "$AUDIO_FEATURE_ID" == "microsoft/clap/2023/four-datasets" ]; then
    RESULT_COUNT=1
    SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=av&feature_extractor_id=${AUDIO_FEATURE_ID}"
    response=$(curl -s -X POST -H "Content-Type:multipart/form-data" \
                    -F 'query_term={"term_id": "x", "is_negative": false, "txt": "trumpet"}' \
                    "${SEARCH_URL}")
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
          "filename": "Presidente_acompanha_a_troca_da_guarda_presidencial_no_Planalto.mp4",
          "source_url": "https://commons.wikimedia.org/wiki/File:Presidente_acompanha_a_troca_da_guarda_presidencial_no_Planalto.webm",
          "description": "O presidente da República, Jair Bolsonaro, participou de cerimônia da troca da guarda das residências oficiais do presidente e do vice-presidente. O ato ocorreu na rampa do Palácio do Planalto, com a presença também do vice-presidente, Hamilton Mourão."
        }
      ]
    }'
    if diff <(echo "$expected_json" | jq -S .) <(echo "$response_selected_json" | jq -S .) > /dev/null; then
        echo "Test 8.1 PASSED"
    else
        echo "Test 8.1 FAILED: unexpected search results"
        echo "Expected:"
        echo "$expected_json" | jq .
        echo "Actual:"
        echo "$response_selected_json" | jq .
        exit 1
    fi
fi

# Test 8.2 : check if the server returns correct results (including metadata) for query on face
if [ "$VIDEO_FEATURE_ID2" == "deepinsight/insightface/buffalo_l/_unknown" ]; then
    FACE_IMG_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/HarryStylesWembley170623_%2865_of_93%29_%2852982678051%29_%28cropped_2%29.jpg/500px-HarryStylesWembley170623_%2865_of_93%29_%2852982678051%29_%28cropped_2%29.jpg"
    FACE_IMG_FILE="${QUERY_DATA_DIR}/Harry_Styles_wikipedia_360x480.jpg"
    if [ ! -f "${FACE_IMG_FILE}" ]; then
        echo "Downloading face image to ${FACE_IMG_FILE} ..."
        curl -sLO "${FACE_IMG_URL}"
        mv "$(basename "${FACE_IMG_URL}")" "${FACE_IMG_FILE}"
    fi
    if [ ! -f "${FACE_IMG_FILE}" ]; then
        echo "Failed to download face image from ${FACE_IMG_URL}"
        exit 1
    fi
    RESULT_COUNT=1
    SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID2}"
    response=$(curl -s -X POST -H "Content-Type:multipart/form-data" \
                    -F 'query_term={"term_id": "x", "is_negative": false, "src": null, "qtype": "visual"}' \
                    -F "query_file=@${FACE_IMG_FILE};filename=x" \
                    "${SEARCH_URL}")
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
        "filename": "Can_I_Offer_You_A_Scotch,_Officer-_-_My_Policeman_-_Prime_Video.mp4",
        "ts": 15.349,
        "te": 18.085
        }
    ]
    }'
    if diff <(echo "$expected_json" | jq -S 'sort_by(.filename) | sort_by(.ts) | sort_by(.te)' .) <(echo "$response_selected_json" | jq -S 'sort_by(.filename) | sort_by(.ts) | sort_by(.te)' .) > /dev/null; then
        echo "Test 8.2 PASSED"
    else
        echo "Test 8.2 FAILED: unexpected search results"
        echo "Expected:"
        echo "$expected_json" | jq .
        echo "Actual:"
        echo "$response_selected_json" | jq .
        exit 1
    fi
fi

# Test 8.3 : check if the server returns correct results (including metadata) for query on video
if [ "$VIDEO_FEATURE_ID1" == "mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli" ]; then
    RESULT_COUNT=1
    SEARCH_URL="${SERVER_URL}search?start=0&end=${RESULT_COUNT}&thumbs=0&search_in=video&feature_extractor_id=${VIDEO_FEATURE_ID1}"
    response=$(curl -s -X POST -H "Content-Type:multipart/form-data" \
                    -F 'query_term={"term_id": "x", "is_negative": false, "txt": "a person pulling a rope"}' \
                    "${SEARCH_URL}")
    response_selected_json=$(echo "$response" | jq -c '
    . as $root |
    {
        merged_windows: [
        .video_results.merged_windows[]
        | .media_id as $id
        | {
            filename: $root.video_results.videos[$id].filename,
            source_url: $root.video_results.videos[$id].external_metadata.source_url,
            ts: .ts,
            te: .te
            }
        ]
    }
    ')

    expected_json='{
      "merged_windows": [
        {
          "filename": "Tasnim_News_Agency_-_Iranian_navy_personnel_talking_about_anti-piracy_missions.mp4",
          "source_url": "https://commons.wikimedia.org/wiki/File:Tasnim_News_Agency_-_Iranian_navy_personnel_talking_about_anti-piracy_missions.webm",
          "ts": 2.28,
          "te": 5.96
        }
      ]
    }'

    if diff <(echo "$expected_json" | jq -S .) <(echo "$response_selected_json" | jq -S .) > /dev/null; then
        echo "Test 8.3 PASSED"
    else
        echo "Test 8.3 FAILED: unexpected search results"
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
