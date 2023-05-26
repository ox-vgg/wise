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
See [docs/Search-Index.md](docs/Search-Index.md).
