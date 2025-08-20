# Retrieval Performance

The organisers of the 
[Cinephile-2005 challenge](https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html)
have released the
[ground_truth_validation_dataset.json](https://thor.robots.ox.ac.uk/wise/assets/cinephile/ground_truth_validation_dataset.json)
file that contains 22 search queries and a list of video filenames
that should be retrieved by a video retrieval system as the top
matching results. The [`evaluate-performance.py`](evaluate-performance.py)
script is used to compute the number of videos contained in the ground
truth (i.e. the JSON file) that can be correctly retrieved the 
[WISE Search Engine (WISE)](https://meru.robots.ox.ac.uk/cinephile/).

```
$ python scripts/cinephile/evaluate-performance.py \
  --wise-url https://meru.robots.ox.ac.uk/cinephile/ \
  --ground-truth-url https://thor.robots.ox.ac.uk/wise/assets/cinephile/ground_truth_validation_dataset.json

...
|-------+-------------------------------------------------+--------|
| top-k | Update to the original query                    | Recall |
|-------+-------------------------------------------------+--------|
| 1000  | Find videos -> Old photos                       | 0.47   |
| 1000  | Find videos -> Photo                            | 0.42   |
| 1000  | Find videos -> ""                               | 0.42   |
| 1000  | Find videos -> German and Dutch archival photos | 0.40   |
| 1000  | None (i.e. original query was used as it is)    | 0.40   |
| 1000  | videos -> photos                                | 0.39   |
| 1000  | Find videos -> Old video frames                 | 0.38   |
| 1000  | videos -> images                                | 0.35   |
|-------+-------------------------------------------------+--------|
```
