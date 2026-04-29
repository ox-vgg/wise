# WISE Explore

> **Note:** The features described in this document are still being developed and therefore are not yet stable or ready for production usage.

WISE Explore is a set of tools that allows exploration of a collection of
videos using anchors such as face, objects, acoustic events. 

We demonstrate this feature using the [`wikimedia-commons-25`](http://thor.robots.ox.ac.uk/wise/assets/test/wikimedia-commons-25.zip) dataset which contains 25 videos taken from the Wikimedia Commons repository. First, we run the [tests/test-wikimedia-commons-25.sh](tests/test-wikimedia-commons-25.sh) script to create a sample WISE project based on this dataset. This script automatically downloads the dataset, extracts features from the videos and creates a WISE project that can be used as a visual search engine for these videos.

```bash
cd $HOME
git clone https://gitlab.com/vgg/wise/wise.git
cd $HOME/wise
./tests/test-wikimedia-commons-25.sh /tmp/wise/
```

The resulting WISE project is stored in `/tmp/wise/wise-test/wise-project/wikimedia-commons-25/` folder which will be used for illustrations below.

## Face Facet
A media collection can be explored using human faces based anchors as
shown below. See [scripts/explore/README.md](../scripts/explore/README.md)
for more details about the semi-supervised, iterative clustering approach
to group face embeddings into distinct identities

```bash
# 1. Install dependencies (assuming $HOME/wise/ contains WISE source)
cd $HOME/wise/scripts/explore
pip install -r requirements.txt

# 2. Automatically cluster faces
cd $HOME/wise/
python3 scripts/explore/cluster_faces.py \
  --project-dir /tmp/wise/wise-test/wise-project/wikimedia-commons-25/ \
  --feature-extractor-id "deepinsight/insightface/buffalo_l/_unknown"\
  --k-neighbors 50 \
  --similarity-threshold 0.7

# 3. Manually review the face clusters and set their status to "Reviewed"
# if the cluster is well formed. Add other metadata such as "description"
# "reference", etc. to the cluster and press "Publish" button in the
# top-right corner. This creates the following three tables in the
# /tmp/wise/wise-test/wise-project/wikimedia-commons-25/metadata/internal.db
# SQLite database:
#     facets, cluster_metadata, facet_metadata
cd $HOME/wise/scripts/explore/frontend
npm install && npm run build # needs to be done only once

cd $HOME/wise/
python3 scripts/explore/explore.py \
  --project-dir /tmp/wise/wise-test/wise-project/wikimedia-commons-25/ \
  --port 10101

# [Optional] 3b. Re-run the clustering algorithm to improve clusters based
# on manually reviewed clusters.
cd $HOME/wise/
python3 scripts/explore/cluster_faces.py \
  --project-dir /tmp/wise/wise-test/wise-project/wikimedia-commons-25/ \
  --feature-extractor-id "deepinsight/insightface/buffalo_l/_unknown"\
  --k-neighbors 50 \
  --similarity-threshold 0.7

# [Optional] 3c. Review updated face clusters and see if new face clusters
# can be published.
python3 scripts/explore/explore.py \
  --project-dir /tmp/wise/wise-test/wise-project/wikimedia-commons-25/ \
  --port 10101

# 4. [Optional] Repeat Steps 3b and 3c as required.
# See scripts/explore/README.md for more details about the iterative workflow

# 5. Serve project with facets enabled
ENABLE_FACETS=true PORT=10102 python3 serve.py \
  --project-dir /tmp/wise/wise-test/wise-project/wikimedia-commons-25/
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