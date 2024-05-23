#!/bin/bash

set -x

if [ "$#" -ne 1 ]; then
    echo "Usage: ${0} OUTDIR"
    echo "where, OUTDIR is the path for temporary storeage"
    exit
fi

OUTDIR=$1

ENV_DIR="${OUTDIR}/wise-dep/"
HUGGINGFACE_HOME="${OUTDIR}/huggingface-home/"
DATA_DIR="${OUTDIR}/wise-data/"
KINETICS_DATA_DIR="${OUTDIR}/wise-data/Kinetics-6/"
PROJECT_DIR="${OUTDIR}/wise-projects/"
CODE_BASEDIR="${OUTDIR}/wise-code/"
CODE_DIR="${CODE_BASEDIR}/wise/"
TEST_DIR="${OUTDIR}/wise-test/"

## 1. fetch WISE code
if [ ! -d "${CODE_DIR}" ]; then
    echo "Downloading WISE2 code to ${CODE_DIR} ..."
    mkdir -p "${CODE_BASEDIR}"
    cd "${CODE_BASEDIR}"
    git clone -b wise2-integration https://gitlab.com/vgg/wise/wise.git
else
    echo "Updating WISE2 code in ${CODE_DIR} ..."
    cd "${CODE_DIR}"
    git pull origin
fi

if ! dpkg --get-selections | grep -q "^ffmpeg[[:space:]]*install$" >/dev/null; then
    echo "ffmpeg package not found, install the ffmpeg software using your distribution package manager"
    exit
fi

## 2. Install WISE2 dependencies
export HF_HOME=$HUGGINGFACE_HOME
if [ ! -d "${ENV_DIR}" ]; then
    echo "Installing WISE2 dependencies using python venv in ${ENV_DIR} ..."
    python3 -m venv $ENV_DIR
    source "${ENV_DIR}/bin/activate"
    python3 -m pip install --upgrade pip
    pip install -r "${CODE_DIR}/requirements.txt"
    pip install --no-deps msclap==1.3.3
    pip install -r "${CODE_DIR}/torch-faiss-requirements.txt"
else
    echo "Skipping installation of WISE2 dependencies"
fi

## 3. Download Kinetics-6 dataset
if [ ! -d "${KINETICS_DATA_DIR}" ]; then
    echo "Downloading Kinetics-6 dataset to ${DATA_DIR} ..."
    mkdir -p "${KINETICS_DATA_DIR}"
    cd "${DATA_DIR}"
    curl -sLO "https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/Kinetics-6.tar.gz"
    tar -zxvf Kinetics-6.tar.gz -C "${KINETICS_DATA_DIR}"
else
    echo "Skipping Kinetics-6 dataset download"
fi

## 4. Run WISE2 tests
#if [ -d "${TEST_DIR}" ]; then
#    echo "WISE2 test data directory already exists"
#    echo "To run tests, delete the folder ${TEST_DIR}"
#    exit
#fi
mkdir -p "${TEST_DIR}"

KINETICS_PROJECT_DIR="${PROJECT_DIR}/Kinetics-6/"

## 4.1 Extract features
if [ ! -d "${KINETICS_PROJECT_DIR}" ]; then
    echo "Extracting features from videos (takes about 3 min.) ..."
    cd "${CODE_DIR}"
    source "${ENV_DIR}/bin/activate"
    python extract-features.py \
           --media-dir "${KINETICS_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 4096 \
           --shard-maxsize 20971520 \
           --num-workers 0 \
           --feature-store webdataset \
           --video-feature-id "mlfoundations/open_clip/ViT-H-14-quickgelu/dfn5b" \
           --audio-feature-id "microsoft/clap/2023/four-datasets" \
           --project-dir $KINETICS_PROJECT_DIR
fi

## 4.2 Index features
if [ ! -d "${KINETICS_PROJECT_DIR}store/microsoft/clap/2023/four-datasets/index/" ]; then
    echo "Creating index (takes about 1 min.) ..."
    cd "${CODE_DIR}"
    source "${ENV_DIR}/bin/activate"
    python create-index.py \
           --index-type IndexFlatIP \
           --project-dir $KINETICS_PROJECT_DIR
fi

## 4.3 Export results in CSV format
SEARCH_RESULT_GND_FN="${TEST_DIR}/cooking-music-GND-TRUTH.csv"
echo "query_id,query_text,media_type,rank,filename,start_time,end_time,score" > $SEARCH_RESULT_GND_FN
echo '0,"cooking",video,0,"frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4",0.5,9.5,0.354' >> $SEARCH_RESULT_GND_FN
echo '0,"cooking",video,1,"frying-vegetables/mwkOrWZxvrU_000006_000016.mp4",0.0,1.5,0.349' >> $SEARCH_RESULT_GND_FN
echo '0,"cooking",video,2,"frying-vegetables/5E20wCGF6Ig_000122_000132.mp4",9.5,9.5,0.336' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,0,"frying-vegetables/hxK9mej0_zw_000086_000096.mp4",0.0,8.0,0.256' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,1,"jogging/OmWoDAQM1kk_000000_000010.mp4",0.0,8.0,0.237' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,2,"singing/vdnskiY-DRc_000023_000033.mp4",0.0,8.0,0.237' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,3,"singing/GO5DhmRmHco_000112_000122.mp4",0.0,8.0,0.206' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,4,"singing/arBpk6QCVFs_000064_000074.mp4",0.0,8.0,0.184' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,5,"singing/WKSxT9T-P_U_000157_000167.mp4",0.0,8.0,0.183' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,6,"shouting/9NdaqLe2gIs_000022_000032.mp4",0.0,8.0,0.181' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,7,"singing/I6NDj1EcP6w_000073_000083.mp4",4.0,8.0,0.163' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,8,"jogging/UQsA-W-q3oA_000002_000012.mp4",4.0,8.0,0.145' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,9,"frying-vegetables/5E20wCGF6Ig_000122_000132.mp4",0.0,8.0,0.143' >> $SEARCH_RESULT_GND_FN
echo '1,"music",audio,10,"jogging/QY8RJBxbLnA_000116_000126.mp4",0.0,8.0,0.139' >> $SEARCH_RESULT_GND_FN
echo '0-1,"cooking and music",video and audio,0,"frying-vegetables/5E20wCGF6Ig_000122_000132.mp4",0.0,9.5,0.479' >> $SEARCH_RESULT_GND_FN

SEARCH_RESULT_CSV_FN="${TEST_DIR}/cooking-music.csv"
if [ ! -d "${SEARCH_RESULT_CSV_FN}" ]; then
    echo "Search 1 (takes about 1 min.) ..."
    cd "${CODE_DIR}"
    source "${ENV_DIR}/bin/activate"
    python search.py \
           --query "cooking" --in video \
           --query "music" --in audio \
           --topk 20 \
           --index-type IndexFlatIP \
           --result-format csv \
           --save-to-file $SEARCH_RESULT_CSV_FN \
           --project-dir $KINETICS_PROJECT_DIR
    if [ cmp --silent $SEARCH_RESULT_GND_FN $SEARCH_RESULT_CSV_FN ]; then
        echo "Search results are unexpected, compare ${SEARCH_RESULT_CSV_FN} and ${SEARCH_RESULT_GND_FN}"
    else
        echo "Search passed"
    fi
fi
