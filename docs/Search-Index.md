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
Recall0@K   ( R0@K   ) = ( ref[0:K-1] INTERSECT ann[0:K-1] ) / K

Recall1@N,K ( R1@N,K ) = ( ref[0:N-1] INTERSECT ann[0:K-1] ) / N

where,
  ref[0:N-1] : denote the top-N results from exhaustive search (i.e. compare against all)
  ann[0:K-1] : denote the top-K results from approximate nearest neighbour (ANN) search
  N          : size of result set obtained from exhaustive search
  K          : size of result set obtained from ANN search

|--------+------------------+-------+-------+--------+----------+-----------+-------|
| Index  | Index-Parameters | Size  | R0@20 | R0@100 | R1@20,30 | R1@20,100 |  Time |
|--------+------------------+-------+-------+--------+----------+-----------+-------|
| Naive  | IndexFlatIP      | 158G  |   1.0 |    1.0 |      1.0 |       1.0 |  52.8 |
| IVF    | nlist=74160      | 159G  | 0.954 |  0.951 |    0.961 |     0.961 | 1.018 |
| IVF+PQ | {  8, 8, 32768}  | 937M  | 0.007 |  0.019 |    0.008 |     0.025 | 0.058 |
| IVF+PQ | { 16, 8, 32768}  | 1.3G  | 0.011 |  0.025 |    0.012 |     0.042 | 0.056 |
| IVF+PQ | { 24, 8, 32768}  | 1.7G  | 0.014 |  0.032 |    0.017 |     0.060 | 0.062 |
| IVF+PQ | { 48, 8, 32768}  | 3.0G  | 0.028 |  0.051 |    0.036 |     0.101 | 0.057 |
| IVF+PQ | { 64, 8, 65536}  | 3.9G  | 0.069 |  0.097 |    0.078 |     0.188 | 0.073 |
| IVF+PQ | {128, 8, 65536}  | 7.2G  | 0.259 |  0.294 |    0.313 |     0.576 | 0.074 |
| IVF+PQ | {192, 8, 65536}  | 10.4G | 0.486 |  0.545 |    0.578 |     0.865 | 0.078 |
| IVF+PQ | {256, 8, 65536}  | 13.7G | 0.642 |  0.687 |    0.766 |     0.950 | 0.087 |
| IVF+PQ | {384, 8, 65536}  | 20.3G | 0.799 |  0.831 |    0.923 |     0.968 | 0.084 |
| IVF+PQ | {768, 8, 65536}  | 40.0G | 0.904 |  0.919 |    0.963 |     0.968 | 0.098 |
|--------+------------------+-------+-------+--------+----------+-----------+-------|
```

Notes:

* Recall0@N (or R0@N) is a more stricter measure of recall which
measures the proportion of search results that are common in top N
results from the reference index (i.e. exhaustive search) and the
Approximate nearest neighbour (ANN) search index. For example,
`R0@20=0.904` indicates that `90.4%` percent of images retrieved by
ANN search index matches the images retrieved by the exhaustive search
method.

* Recall1@N,K (or R1@N,K) is a more lenient measure of recall which
measures the proportion of top N results from the reference index
(i.e. exhaustive search) contained in the top K results from the ANN
search index. For example, `R1@20,30=0.963` indicates that `96.3%` of
the top 30 images retrived by the ANN search index matches the top 20
images retrieved by the exhaustive search method. The [product
quantization paper (2010)](https://ieeexplore.ieee.org/abstract/document/5432202) uses
Recall1@N=1,K metric for reporting performance of PQ based ANN
methods.

* The reported recall values were obtained by averaging the recall
values for the [following manually selected 60 search queries](data/index/search-queries.txt).

* For index IVF+PQ, the Index Parameters are given as a tuple
`{m, nbits, nlist}` which indicates that each feature vector gets split
into `m` sub-vectors which is quantized using `2^nbits` centroids. The
search space is divided into `nlist` Voronoi cells. All the IVF based
search indices visit `1024` nearby Voronoi cells (i.e. `nprobe=1024`)
for each search operation.

* The search index contains 55070776 images from Wikimedia Commons
image repository. Each image is represented by a `768` dimensional
feature vector extracted by the `ViT-L-14:laion2b_s32b_b82k`
[OpenClip](https://github.com/mlfoundations/open_clip) model. The
"Time" column measures the average search query response time in
seconds.