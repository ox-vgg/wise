#!/bin/bash

set -euxo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: ${0} OUTDIR [NO-GIT-PULL]"
    echo "where, OUTDIR is the path for temporary storeage"
    exit
fi

OUTDIR=$(realpath ${1})

ENV_DIR="${OUTDIR}/wise-dep"
HUGGINGFACE_HOME="${OUTDIR}/huggingface-home"
DATA_DIR="${OUTDIR}/wise-data"
KINETICS_DATA_DIR="${OUTDIR}/wise-data/Kinetics-6b"
KINETICS6_DOWNLOAD_URL="https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6b.tar.gz"
PROJECT_DIR="${OUTDIR}/wise-projects"
CODE_BASEDIR="${OUTDIR}/wise-code"
CODE_DIR="${CODE_BASEDIR}/wise"
TEST_DIR="${OUTDIR}/wise-test"

start=`date +%s`

## 1. fetch WISE code
if [ ! -d "${CODE_DIR}" ]; then
    echo "Downloading WISE2 code to ${CODE_DIR} ..."
    mkdir -p "${CODE_BASEDIR}"
    cd "${CODE_BASEDIR}"
    git clone -b wise2-integration https://gitlab.com/vgg/wise/wise.git
else
    echo "Updating WISE2 code in ${CODE_DIR} ..."
    cd "${CODE_DIR}"
    if [ "$#" -eq 2 ] && [ "${2}" = "NO-GIT-PULL" ]; then
        echo "Skipping git pull on user request"
    else
        git pull origin
    fi
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg package not found, install the ffmpeg software using your distribution package manager"
    exit
fi

## 2. Install WISE2 dependencies
export HF_HOME=$HUGGINGFACE_HOME
if [ ! -d "${ENV_DIR}" ]; then
    echo "Creating python venv in ${ENV_DIR} ..."
    python3 -m venv "$ENV_DIR"
fi
echo "Ensuring dependencies are installed..."
source "${ENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
pip install -r "${CODE_DIR}/requirements.txt" -r "${CODE_DIR}/torch-faiss-requirements.txt"
pip install --no-deps msclap==1.3.3


## 3. Download Kinetics-6 dataset
if [ ! -d "${KINETICS_DATA_DIR}" ]; then
    echo "Downloading Kinetics-6 dataset to ${DATA_DIR} ..."
    mkdir -p "${KINETICS_DATA_DIR}"
    curl -sLO $KINETICS6_DOWNLOAD_URL
    tar -zxvf Kinetics-6b.tar.gz -C "${KINETICS_DATA_DIR}"
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
    python extract-features.py \
           --media-dir "${KINETICS_DATA_DIR}" \
           --media-include "*.mp4" \
           --shard-maxcount 4096 \
           --shard-maxsize 20971520 \
           --num-workers 0 \
           --feature-store webdataset \
           --video-feature-id "mlfoundations/open_clip/ViT-L-16-SigLIP-384/webli" \
           --audio-feature-id "microsoft/clap/2023/four-datasets" \
           --project-dir "$KINETICS_PROJECT_DIR"
fi

## 4.2 Import metadata
if [ ! -d "${KINETICS_PROJECT_DIR}metadata/Kinetics/6b.sqllite" ]; then
    echo "Importing metadata (takes few seconds) ..."
    cd "${CODE_DIR}"
    python3 metadata.py import \
            --from-csv "${KINETICS_DATA_DIR}/metadata.csv" \
            --metadata-id "Kinetics/6b/video_categories" \
            --col-metadata-id metadata_id \
            --col-filename "{metadata}/{filename}" \
            --col-starttime starttime \
            --col-stoptime stoptime \
            --col-metadata metadata \
            --project-dir "$KINETICS_PROJECT_DIR"
fi

## 4.3 Create search index for features and metadata
if [ ! -d "${KINETICS_PROJECT_DIR}store/microsoft/clap/2023/four-datasets/index/" ]; then
    echo "Creating index (takes about 1 min.) ..."
    cd "${CODE_DIR}"
    python create-index.py \
           --index-type IndexFlatIP \
           --project-dir $KINETICS_PROJECT_DIR
fi

##
## Test 1 : test audiovisual search and export to CSV
##
TEST1_GND_FN="${TEST_DIR}/cooking-music-GND-TRUTH.csv"
cat << EOF > ${TEST1_GND_FN}
query,rank,filename,start_time,end_time,score
"""cooking"" in video",0,"frying-vegetables/mwkOrWZxvrU_000006_000016.mp4",0.0,8.5,0.102
"""cooking"" in video",1,"frying-vegetables/hxK9mej0_zw_000086_000096.mp4",1.5,8.0,0.090
"""cooking"" in video",2,"frying-vegetables/lUyXiF6KfgU_000296_000306.mp4",5.0,9.5,0.088
"""cooking"" in video",3,"frying-vegetables/lUyXiF6KfgU_000296_000306.mp4",0.0,0.0,0.084
"""music"" in audio",0,"frying-vegetables/hxK9mej0_zw_000086_000096.mp4",0.0,8.0,0.256
"""music"" in audio",1,"jogging/OmWoDAQM1kk_000000_000010.mp4",0.0,8.0,0.237
"""music"" in audio",2,"singing/vdnskiY-DRc_000023_000033.mp4",0.0,8.0,0.237
"""music"" in audio",3,"singing/GO5DhmRmHco_000112_000122.mp4",0.0,8.0,0.206
"""music"" in audio",4,"singing/arBpk6QCVFs_000064_000074.mp4",0.0,8.0,0.184
"""music"" in audio",5,"singing/WKSxT9T-P_U_000157_000167.mp4",0.0,8.0,0.183
"""music"" in audio",6,"shouting/9NdaqLe2gIs_000022_000032.mp4",0.0,8.0,0.181
"""music"" in audio",7,"singing/I6NDj1EcP6w_000073_000083.mp4",4.0,8.0,0.163
"""music"" in audio",8,"jogging/UQsA-W-q3oA_000002_000012.mp4",4.0,8.0,0.145
"""music"" in audio",9,"frying-vegetables/5E20wCGF6Ig_000122_000132.mp4",0.0,8.0,0.143
"""music"" in audio",10,"jogging/QY8RJBxbLnA_000116_000126.mp4",0.0,8.0,0.139
"""cooking"" in video and ""music"" in audio",0,"frying-vegetables/hxK9mej0_zw_000086_000096.mp4",0.0,8.0,0.346
EOF

TEST1_RESULT_FN="${TEST_DIR}/cooking-music.csv"
if [ ! -d "${TEST1_RESULT_FN}" ]; then
    echo "Test 1 (takes about 3 min.) ..."
    cd "${CODE_DIR}"
    python search.py \
           --query "cooking" --in video \
           --query "music" --in audio \
           --topk 20 \
           --index-type IndexFlatIP \
           --result-format csv \
           --save-to-file $TEST1_RESULT_FN \
           --project-dir $KINETICS_PROJECT_DIR
    end=`date +%s`
    elapsed_time=$((end-start))
    if ! cmp -s ${TEST1_GND_FN} ${TEST1_RESULT_FN}; then
        echo "Test 1 FAILED because search results are unexpected"
        diff "${TEST1_RESULT_FN}" "${TEST1_GND_FN}"
    else
        echo "Test 1 PASSED (completed in ${elapsed_time} sec.)"
    fi
fi

##
## Test 2 : test metadata search and --not-in flag
##
TEST2_GND_FN="${TEST_DIR}/music-singing-GND-TRUTH.csv"
cat << EOF > ${TEST2_GND_FN}
query,rank,filename,start_time,end_time,score
"""music"" in audio and ""singing"" not in metadata",0,"frying-vegetables/hxK9mej0_zw_000086_000096.mp4",0.0,4.0,0.256
"""music"" in audio and ""singing"" not in metadata",1,"jogging/OmWoDAQM1kk_000000_000010.mp4",0.0,8.0,0.237
"""music"" in audio and ""singing"" not in metadata",2,"shouting/9NdaqLe2gIs_000022_000032.mp4",0.0,4.0,0.181
EOF

TEST2_RESULT_FN="${TEST_DIR}/music-singing.csv"
if [ ! -d "${TEST2_RESULT_FN}" ]; then
    echo "Test 2 (takes about 3 min.) ..."
    cd "${CODE_DIR}"
    python search.py \
           --query "music" --in audio \
           --query "singing" --not-in metadata \
           --topk 10 \
           --index-type IndexFlatIP \
           --result-format csv \
           --save-to-file $TEST2_RESULT_FN \
           --project-dir $KINETICS_PROJECT_DIR
    end=`date +%s`
    elapsed_time=$((end-start))
    if ! cmp -s ${TEST2_GND_FN} ${TEST2_RESULT_FN}; then
        echo "Test 2 FAILED because search results are unexpected"
        diff "${TEST2_RESULT_FN}" "${TEST2_GND_FN}"
    else
        echo "Test 2 PASSED (completed in ${elapsed_time} sec.)"
    fi
fi


##
## Test 3 : test --queries-from flag (Note: ground truth is same as Test2)
##
TEST3_GND_FN="${TEST_DIR}/queries-from-GND-TRUTH.csv"
cat << EOF > ${TEST3_GND_FN}
query,rank,filename,start_time,end_time,score
"""music"" in audio and ""singing"" not in metadata",0,"frying-vegetables/hxK9mej0_zw_000086_000096.mp4",0.0,4.0,0.256
"""music"" in audio and ""singing"" not in metadata",1,"jogging/OmWoDAQM1kk_000000_000010.mp4",0.0,8.0,0.237
"""music"" in audio and ""singing"" not in metadata",2,"shouting/9NdaqLe2gIs_000022_000032.mp4",0.0,4.0,0.181
EOF

TEST3_RESULT_FN="${TEST_DIR}/music-singing.csv"
if [ ! -d "${TEST3_RESULT_FN}" ]; then
    echo "Test 3 (takes about 3 min.) ..."
    cd "${CODE_DIR}"
    python search.py \
           --queries-from "${KINETICS_DATA_DIR}/sample_queries.csv" \
           --topk 10 \
           --index-type IndexFlatIP \
           --result-format csv \
           --save-to-file $TEST3_RESULT_FN \
           --project-dir $KINETICS_PROJECT_DIR
    end=`date +%s`
    elapsed_time=$((end-start))
    if ! cmp -s ${TEST3_GND_FN} ${TEST3_RESULT_FN}; then
        echo "Test 3 FAILED because search results are unexpected"
        diff "${TEST3_RESULT_FN}" "${TEST3_GND_FN}"
    else
        echo "Test 3 PASSED (completed in ${elapsed_time} sec.)"
    fi
fi
