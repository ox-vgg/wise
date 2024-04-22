# Shot detection using TransNetV2

When searching through a video, it is highly likely that frames that are close the matching frame also matches the search query. This allows us to combine the results and show a representative frame in the UI and play the segment when the user wants to view a particular result. This helps reduce the clutter in the UI, as well as group related and relevant frames together.

Edited videos contain shots - video segments captured by different cameras, but put together in the final video in editing stage. We can either avoid aggregating across shot boundaries when merging relevant segments or retrieve the shot when any frame matches the search query (we use option 2)

To get these shot boundaries, TransNetV2 model is used. ([Paper](https://arxiv.org/pdf/2008.04838), [Code](https://github.com/soCzech/TransNetV2))

## Steps

1. Clone shot-detection project and set it up

```bash
git clone git@gitlab.com:vgg/wise/shot-detection.git
cd shot-detection
git submodule update --init --recurisve

# Output folder
mkdir -p output

# Conda / Mamba
conda create -f environment.yml
conda activate shot-detection

# If using micromamba, it doesnt respect the PYTHONPATH set in environment.yml, so export it manually
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="transnetv2/inference:$PYTHONPATH"

```

2. Export video locations from wise as CSV, along with the ids. (Edit the CSV / create a symlink to the directories with the videos files here)

```bash
# PATH TO WISE PROJECT METADATA DB
WISE_METADATA_DB=path/to/metadata.db
sqlite3 ${WISE_METADATA_DB} <<EOF
.headers on
.mode csv
.output wise_videos.csv
select m.id as id, (s.location || "/" || m.path as location) from media as m INNER JOIN source_collections as s ON m.source_collection_id = s.id;
EOF
```

3. Run the CLI to process videos

```bash
# Help
# python3 cli.py --help

# Detect shots
 CUDA_VISIBLE_DEVICES=0 python3 cli.py \
    detect \
    wise_videos.csv \
    --save-to output

# Convert shot predictions to table format
python3 cli.py \
    convert \
    wise_videos.csv \
    --predictions output

# Will output shots.csv in current directory

# Import shots.csv into dummy table and
# test querying
# Delete old tables before doing this
mv shots.db "shots-$(date '+%Y-%m-%dT%H_%M_%S').db"
alembic upgrade head
sqlite3 shots.db <<EOF
.import --csv --skip 1 wise_videos.csv media
.import --csv --skip 1 shots.csv shots
EOF

# Test querying
python3 cli.py query --media-id MEDIA_ID --t TIMESTAMP
```
