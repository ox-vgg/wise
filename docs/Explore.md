# WISE Explore

> **Note:** The features described in this document are still being developed and therefore are not yet stable or ready for production usage.

WISE Explore is a set of tools that allows exploration of a collection of
videos using anchors such as face, objects, acoustic events. We create
a WISE project based on the wise-test-dataset (private dataset
with 236 videos) as follows.

```bash
FEATURE_EXTRACTOR_CONFIG='{"transformers/owlv2/google/owlv2-large-patch14-ensemble":{"objectness_threshold":0.13}}' \
CUDA_VISIBLE_DEVICES=0 python extract-features.py \
    "/data/videos/wise-test-dataset/mp4/v1" \
    --shard-maxcount 16384 \
    --num-workers 1 \
    --feature-store faiss \
    --audio-feature-id "microsoft/clap/2023/four-datasets" \
    --video-feature-id "mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli" \
    --video-feature-id "deepinsight/insightface/buffalo_l/_unknown" \
    --video-feature-id "transformers/owlv2/google/owlv2-large-patch14-ensemble" \
    --project-dir /data/wise-projects/wise-test-dataset/
  Feature extraction completed in 22881 sec (381.35 min)

python3 create-index.py \
  --index-type IndexIVFFlat \
  --project-dir /data/wise-projects/wise-test-dataset/
```

## Face Facet
A media collection can be explored using human faces based anchors as
shown below.

```bash
# 1. Fetch code and install dependencies
cd $HOME
git clone https://gitlab.com/vgg/wise/wise.git
cd $HOME/wise/scripts/explore
pip install -r requirements.txt

# 2. Automatically cluster faces
cd $HOME/wise/
python3 scripts/explore/cluster_faces.py \
  --project-dir /data/wise-projects/wise-test-dataset/\
  --feature-extractor-id "deepinsight/insightface/buffalo_l/_unknown"

# 3. Manually review the face clusters and set their status to "Reviewed"
# if the cluster is well formed. Add other metadata such as "description"
# "reference", etc. to the cluster and press "Publish" button in the
# top-right corner. This creates the following three tables in the
# $project_dir/metadata/internal.db SQLite database:
#     facets, cluster_metadata, facet_metadata
cd $HOME/wise/scripts/explore/frontend
npm install && npm run build # needs to be done only once

cd $HOME/wise/
python3 scripts/explore/explore.py \
  --project-dir /data/wise-projects/wise-test-dataset/ \
  --port 10101

# 4. Serve project with facets enabled
ENABLE_FACETS=true PORT=10101 python3 serve.py \
  --project-dir /data/wise-projects/wise-test-dataset/
```

Click on the link titled "Facets" that appears beside the search input panel
to view the face facet. Human annotators can continue to review and publish
face clusters using the Explore web interface created in Step 3. The updates
get instantly reflected to users viewing the facets using web interface
created in Step 4.

# Object Facet
TODO

# Acoustic Event Facet
TODO