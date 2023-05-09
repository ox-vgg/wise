This folder contains various scripts used by WISE developers.

## Comparing the Retrieval Performance of Wikimedia and WISE search engine 

We compare the performance of Wikimedia Commons (based on text
metadata) and WISE Image Search Engine (based on image content) by
manually checking the search results obtained for a common set of
search queries (e.g. toy aeroplane, pencil drawing of lion, etc.). The
[compare-wikimedia-wise-retrieval-perf.py](compare-wikimedia-wise-retrieval-perf.py)
script queries the two search engines using a common set of search
queries and generates manual annotation projects that can be reviewed
using the [LISA](https://gitlab.com/vgg/lisa) annotation tool. A human
annotator goes through all the results and marks them as being
correct or incorrect. This allows computation of performance metrics
(e.g. precision) of the two search engines.

```
$ cd wise/scripts
$ python3 compare-wikimedia-wise-retrieval-perf.py \
  --out-lisa-dir=/data/tmp \
  --wikimedia-server-url="https://commons.wikimedia.org/w/api.php?" \
  --wise-server-url="https://meru.robots.ox.ac.uk/wikimedia/" \
  --wise-username=demo \
  --wise-password=HIDDEN
```

## Evaluate Performance of an Approximate Nearest Neighbour (ANN) Search Against Exhaustive Search

We use the retrieval results obstained from exhaustive search (hence
slower search speed and higher storage costs) to evaluate the
performance of a search index based on approximate nearest neighbour
(hence faster search speed and lower storage costs) search
strategy. The exhaustive search defines the topline performance that
can achieved by an approximate nearest neighbour method.

```
$ cd wise/script
$ python3 export-exhaustive-search-results.py \
  --wise-server-url http://jupiter.robots.ox.ac.uk:9670/wikimedia5/ \
  --out-fn /ssd/adutta/code/wise/data/index/wikimedia5-55M-exhaustive-search-results.json

toy aeroplane [completed in 29.627 sec.]
pencil drawing of lion [completed in 75.313 sec.]
a train in mountains near waterfall [completed in 41.607 sec.]
cheetah running [completed in 63.233 sec.]
hot air balloon above a mountain [completed in 55.701 sec.]
dolphin playing with ball [completed in 42.323 sec.]
penguin with wings raised [completed in 74.861 sec.]
running on a hill [completed in 28.462 sec.]
people on a roller coaster [completed in 70.833 sec.]
car with a bicycle on top [completed in 47.316 sec.]
...


## TODO
$ python3 eval-ann-search-perf.py \
  --wise-server-url http://jupiter.robots.ox.ac.uk:9671/wikimedia/ \
  --exhaustive-search-results=/ssd/adutta/code/wise/data/index/wikimedia5-55M-exhaustive-search-results.json
```