# WISE Search Index

A search index allows us to search through millions of images in less
than one second. This document describes how we build and evaluate the
performance of the search index used in our [online
demo](https://meru.robots.ox.ac.uk/wikimedia/) based on the images
contained in the Wikimedia Commons repository.

For evaluation purpose, we use a subset containing 55070776 images
taken from the Wikimedia Commons. A 768 dimensional vector
representation these images is created using the
[OpenClip](https://github.com/mlfoundations/open_clip) model. This
representation allows each image to be represented by a point in a 768
dimensional space. Visual search of these images involves finding the
points (i.e. images) in the high dimensional space that are nearest to
the query point.

A naive approach for search involves computing the distance between
the query point and all other points in the high dimensional
space. This approach leads to most accurate search results but is
slow. For example, for the 55070776 images taken from the Wikimedia
Commons (henceforth referred to as wikimedia-55M), the naive approach
leads to a search time of nearly 40 seconds which in turn would result
in a frustrating experience for users. The naive approach is useful
for establising the top-line performance that can then be used to
benchmark the performance of other approximate methods for nearest
neighbour search. We create a pre-defined list of search queries
(e.g. "pencil drawing of lion") and use the naive approach to search
for establishing the ground truth search results for those queries on
the wikimedia-55M set.

```
## We assume that a WISE server based on naive approach to search
## is serving on http://localhost:9670/wikimedia-55M/
##
## Such a WISE server can be started by running the following command
##
## python3 wise.py serve wikimedia-55M \
##  --index-type IndexFlatIP \
##  --theme-asset-dir ./www/imgrid/

cd $HOME/code/wise/
python3 scripts/query-search-engine.py \
  --search-queries-fn data/index/search-queries.txt \
  --search-queries-count 60 \
  --wise-server-url http://localhost:9670/wikimedia-55M/ \
  --save-results-to data/index/wikimedia-55M/IndexFlatIP.json \
  --results-to-return 100
```

Next, we create a search index that is designed to perform fast
nearest neighbour search at the expense of some loss in accuracy. Such
search index is created using approximate methods
(e.g. [LSH](https://www.pinecone.io/learn/locality-sensitive-hashing/),
[HNSW](https://www.pinecone.io/learn/hnsw/), etc.) for nearest
neighbour search. This tutorial illustrates the performance of an [IndexIVFFlat](https://www.pinecone.io/learn/faiss-tutorial/)
search index created using the following parameters.

```
## We assume that a WISE server based on HNSW search index
## is serving on http://localhost:9670/wikimedia-55M/
##
## Such a WISE server can be started by running the following command
##
## python3 wise.py serve wikimedia-55M \
##  --index-type IndexIVFFlat \
##  --theme-asset-dir ./www/imgrid/
##

python3 scripts/query-search-engine.py \
  --search-queries-fn data/index/search-queries.txt \
  --search-queries-count 60 \
  --wise-server-url http://jupiter.robots.ox.ac.uk:9670/wikimedia5/ \
  --wise-server-description "IndexIVFFlat-nprobe1024" \
  --save-results-to data/index/wikimedia-55M/IndexIVFFlat-nprobe1024.json \
  --results-to-return 100
```

The search retrieval performance of a search index (e.g. IVFFlat) can be compared against
the search results obtained using the naive approach to nearest neighbour search as follows.

```
python3 scripts/eval-search-engine-perf.py \
  --perf-data-dir data/index/wikimedia-55M/ \
  --ref-perf-filename IndexFlatIP.json
```

## Table Showing Performance of Some Search Indices

The naive approach to nearest neighbour search is taken as the
reference for evaluating the performance of approximate nearest
neighbour search methods as shown in the table below.

```
|------------+--------------------+-------+-------+-------|
| Index-Type | Index-Parameters   | Size  |   MAP | Time  |
|------------+--------------------+-------+-------+-------|
| Naive      | IndexFlatIP        | 158GB |   1.0 | 52.8s |
| IVF        | nprobe=1024        | 159GB | 0.495 | 1.0s  |
|            | nprobe=512         |       | 0.386 | 0.2s  |
|            | nprobe=256         |       | 0.324 | 0.8s  |
|------------+--------------------+-------+-------+-------|
```
