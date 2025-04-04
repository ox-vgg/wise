# Evaluation of WISE Audiovisual Search

WISE Search Engine (WISE) can retrieve results from an audiovisual collection.
This page shows the retrieval performance based on standard benchmark datasets.

## EpicKitchens multi-instance retrieval challenge
The EpicKitchens-100 multi-instance retrieval challenge contains 3843 query
sentences and 9668 videos segments. More details about this challenge are available 
in the [EpicKitchens website](https://epic-kitchens.github.io/2024#challenge-action-retrieval).

```
# 1. Generate relevancy matrix
git clone https://github.com/mwray/Joint-Part-of-Speech-Embeddings.git
cd Joint-Part-of-Speech-Embeddings
export PYTHONPATH=src/
python -m scripts.create_relevancy_files\
  epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.pkl

# 2. Run evaluation
python3 text-to-video-retrieval.py\
  --path-epic-annotations "epic-kitchens-100-annotations/retrieval_annotations/"\
  --project-dir "temp/wise/EpicKitchens-100/webli/"\
  --relevancy-matrix "Joint-Part-of-Speech-Embeddings/caption_relevancy_EPIC_100_retrieval_test.pkl"\
  --out-dir "temp/wise/eval/EpicKitchens-100/webli/"\
  --binary-relevancy\
  --export-visulisation
 ```

 This script generates a `index.html`, `retrieval-data-binary_relevancy.js` (or `retrieval-data-noun_verb_relevancy.js`) and a `thumbs/` folder in `--out-dir`. Open the `index.html` file in a web browser to view query specific ranked results and the computed average precision. 
 
 We ran this evaluation on three different vision-language models and the results of this evaluation are shown below.

 ```
|----------------------------+--------------------------+------------+-------------|
| Model                      | Training Dataset         | mAP-R{0,1} | mAP-R[0..1] |
|----------------------------+--------------------------+------------+-------------|
| xlm-roberta-large-ViT-H-14 | frozen_laion5b_s13b_b90k |      5.47% |      15.49% |
| ViT-H-14-quickgelu         | dfn5b                    |      5.72% |      16.03% |
| ViT-B-16-SigLIP            | webli                    |      4.47% |      14.04% |
|----------------------------+--------------------------+------------+-------------|
 ```

where `R{0,1}` corresponds to binary relevancy and `R[0..1]` corresponds to a softer
version of relevancy based on Intersection-over-Union (IoU) of noun and verbs.