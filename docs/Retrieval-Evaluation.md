# Evaluation of Video Retrieval Performance

We evaluate the WISE video retrieval performance using the
[Multi-Instance Retrieval](https://epic-kitchens.github.io/2024#challenge-action-retrieval)
challenge introduced in the following [research paper](https://arxiv.org/pdf/2006.13256.pdf).

We do not use the training subset of this dataset as WISE uses
pre-trained vision language models. We evaluate performance using the
test subset which contains `9668` video segments defined over 700
videos. The evaluation is based on `3842` text queries (e.g. take
plate, wash hands, etc.).

```
|-----------------+------------------+--------+----------+-----------+---------------------------+-------|
| Method          | Dataset          | Subset | # Videos | # Queries | FeatureExtractor          |   mAP |
|-----------------+------------------+--------+----------+-----------+---------------------------+-------|
| WISE2           | EpicKitchens-100 | Test   |     9668 |      3842 | ViT-H-14-quickgelu:dfn5b  | 0.413 |
| WISE2           | EpicKitchens-100 | Test   |     9668 |      3842 | xlm-..ViT-H-14:..laion5b..| 0.413 |
| WISE2           | EpicKitchens-100 | Test   |     9668 |      3842 | ViT-L-16-SigLIP-384:webli | 0.412 |
| Baseline: JPoSE | EpicKitchens-100 | Test   |     9668 |      3842 |                           | 0.381 |
| Baseline: MLP   | EpicKitchens-100 | Test   |     9668 |      3842 | 2 layer MLP, triplet loss |  0.34 |
|-----------------+------------------+--------+----------+-----------+---------------------------+-------|
```

The `FeatureExtractor` column corresponds to the model used by WISE to
extract features from video frames. More details about various feature
extractors are available at the [mlfoundation's openclip
page](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_retrieval_results.csv). The
baseline performance (i.e. MLP and JPoSE) are taken from the [research
paper](https://arxiv.org/pdf/2006.13256.pdf) introducing the
multi-instance action retrieval challenge.

The performance evaluation was carried out as follows.

```
python search.py \
  --queries-from "epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test_sentence.csv" \
  --in video \
  --topk 1000 \
  --merge-tolerance-video 4   --merge-rank-tolerance 10 \
  --index-type IndexIVFFlat \
  --result-format csv \
  --save-to-file eval/EpicKitchens-100/retrieval_test_topk1000_tol4_ranktol10_IndexIVFFlat.csv \
  --project-dir wise/EpicKitchens-100/webli/
Processed 3842 queries in 1178.92 sec. or 19.65 min.

python3 scripts/eval/EpicKitchens-100/retrieval_eval.py \
  --epic-video-segments "epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv" \
  --relevancy-mat-file "temp/wise/eval/EpicKitchens-100/EPIC_100_retrieval_test_relevancy.pkl" \
  --wise-query "epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test_sentence.csv" \
  --wise-search-results "temp/wise/eval/EpicKitchens-100/retrieval_test_topk1000_tol4_ranktol10_IndexIVFFlat.csv"

loaded 3842 queries
loaded 9668 video segments
Computing 9668x3842 similarity matrix
Loading 9668x3842 relevancy matrix
mAP = 0.412
```