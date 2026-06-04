# Processing Large Datasets

This document describes a workflow for processing a large collection (e.g. 10,000 hours)
of videos using the WISE software tool.

> **Warning:** This document is still in draft mode and work is ongoing. Some sections may be incomplete or subject to change.

## 1. Resize the videos

- A video may contain multiple audio streams. The English language audio stream is selected using the `-map 0:a:m:language:eng` flag.
- Videos may have non-square pixels. For more details, see [this discussion](https://gitlab.com/vgg/wise/wise/-/issues/175). The `-vf "scale=trunc($VIDEO_HEIGHT*dar/2)*2:$VIDEO_HEIGHT,setsar=1"` flag resizes videos to a height of `512px` and sets the pixel aspect ratio to `1.0`.

```bash
export VIDEO_HEIGHT=512
ffmpeg -threads 8 -fflags +genpts+discardcorrupt -err_detect ignore_err \
    -i "input.mp4" \
    -map 0:v:0 \
    -map 0:a:m:language:eng \
    -vf "scale=trunc($VIDEO_HEIGHT*dar/2)*2:$VIDEO_HEIGHT,setsar=1,format=yuv420p" \
    -ac 2 \
    -c:v libx264 -preset slow -crf 23 \
    -c:a aac -b:a 128k \
    -movflags +faststart \
    -f mp4 \
    "output.mp4"
```

## 2. Extract audio features

We first extract the audio features because it is faster to extract audio features as compared to visual features. The generated WISE project (containing only audio features) is then used to compute the shot boundaries.

```bash
CUDA_VISIBLE_DEVICES=0 python -m wise extract-features \
    "/data/all_videos/" \
    --media-include "*.mp4" \
    --shard-maxcount 4096 \
    --num-workers 0 \
    --no-thumbnails \
    --audio-feature-id "microsoft/clap/2023/four-datasets" \
    --project-dir /data/wise-project/
```

## 3. Compute shot boundaries (only for edited videos)
```bash
cd ~/code
git clone git@gitlab.com:vgg/wise/shot-detection.git
cd ~/code/shot-detection
git submodule update --init --recursive
# copy weights from https://github.com/soCzech/TransNetV2/tree/master/inference/transnetv2-weights
# to "transnetv2/inference/transnetv2-weights/"
micromamba env create --channel-priority flexible -f environment.yml -n shot-detection
micromamba activate shot-detection

cd ~/code/shot-detection
mkdir -p output
export PYTHONPATH="transnetv2/inference:$PYTHONPATH"
python3 cli.py \
    detect-and-convert \
    /data/wise-project/ \
    --save-to output
cp shots.csv /data/wise-project/

cd ~/wise
micromamba activate wise
python3 -m wise media-metadata \
    import-shots \
    --project-dir /data/wise-project/ \
    --from-csv /data/wise-project/shots.csv
```

## 4. Extract visual features
```bash
CUDA_VISIBLE_DEVICES=0 python -m wise extract-features \
    --yes \
    --media-include "*.mp4" \
    --shard-maxcount 4096 \
    --num-workers 0 \
    --thumbnails \
    --use-shots \
    --video-feature-id "mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli" \
    --project-dir /data/wise-project/
```

## 5. Classify each shot into following shot scale types {'extreme close-up', 'close-up', 'medium shot', 'full shot', 'long shot'}
```bash
git clone https://gitlab.com/vgg/wise/shot-scale-classifier.git
cd $HOME/shot-scale-classifier
micromamba create -n shot-scale-classifier python=3.12 -y
micromamba activate shot-scale-classifier
pip install -r requirements.txt

# Download "shot_scale_ckpt.pth" from https://drive.google.com/drive/folders/1HKqaw5aPpeTfuHkqU9Xr7TLc3va2noM1?usp=sharing
# If the commands shown below fail, download it manually using a web browser

python3 classify_shot_scale.py \
    --batch-size 8 \
    --num-workers 2 \
    --resume_path checkpoints/shot_scale_ckpt.pth \
    --out-csv /data/a-wise-project/thumbs-shot-scale.csv \
    --project-dir /data/a-wise-project/
micromamba deactivate

cd ~/wise
micromamba activate wise
python3 -m wise media-metadata \
    import-shot-scale \
    --project-dir /data/a-wise-project/ \
    --from-csv /data/a-wise-project/thumbs-shot-scale.csv
```

## 6. Use Triton Inference Server for Feature Extraction

See [Using Triton Inference Server](using-triton-inference-server.md) to understand how better GPU memory utilisation can be achieved using the Triton Inference Server (maintained by NVIDIA) and running inference on optimised models (e.g. ONNX, TensorRT).

## 7. Aggregator Mode
WISE can operate in aggregator mode, where a large audiovisual collection is split up across multiple standalone nodes each covering a different subset. A central instance distributes search queries to all standalone nodes, gathers their results, and presents a unified response to the user. See [tests/test-aggregator.sh](tests/test-aggregator.sh) to understand the aggregator mode available in WISE.

For the purpose of illustration, let us assume that we have split a large video dataset into two sub-sets called `shard1` and `shard2` each containing around 1000 videos and independently processed using WISE.

```bash
# We assume that a triton inference server is running at localhost:8801
# 1. Start shard1
FEATURE_EXTRACTOR_CONFIG='{"default": {"url": "localhost:8801"}}' \
  PORT="10001" \
  python3 -m wise serve \
  --project-dir /data/wise-projects/shard1

# 2. Start shard2
FEATURE_EXTRACTOR_CONFIG='{"default": {"url": "localhost:8801"}}' \
  PORT="10002" \
  python3 -m wise serve \
  --project-dir /data/wise-projects/shard2

# 3. Serve both shard1 and shard2 using aggregator
REMOTE_PROJECTS='["http://localhost:10001/shard1/", "http://localhost:10002/shard2/"]' \
  FEATURE_EXTRACTOR_CONFIG='{"default": {"url": "localhost:8801"}}' \
  PORT=10000 \
  python3 -m wise serve \
  --project-dir combined_shards
```

Now visit `http://localhost:10000/combined_shards` to search on across both shards.

## 8. Adjust threshold for Object Feature Extractor

The OWLv2 object feature extractor has the parameter `objectness_threshold` to control the number
of extracted objects that get stored in the database. By default, the `objectness_threshold=0.02`
which often results in a very large number of objects thereby increasing storage costs.
These costs can be reduced by setting `objectness_threshold=0.10` which is known to give reasonably good
coverage of objects. The parameter can be set as follows:

```bash
FEATURE_EXTRACTOR_CONFIG='{"transformers/owlv2/google/owlv2-large-patch14-ensemble":{"objectness_threshold":0.10}}' \
  python3 -m wise extract-features \
  --enable-autocast \          # this reduces memory and compute by using fp16
  --video-feature-id "transformers/owlv2/google/owlv2-large-patch14-ensemble" \
  ...
```
