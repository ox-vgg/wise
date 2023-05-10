# WISE Image Search Engine (WISE)

WISE Image Search Engine (WISE) is an open-source image search engine which leverages recent advances in machine learning and vision-language models that enable search based on image content using natural language. The expressive power of natural language allows the user to flexibly describe what they are looking for.

> The code and documentation contained in this repository is not stable yet for production usage. We are working to make it production ready.

## Installation and Usage

```bash
## setup environment and install dependency libraries
cd $HOME
git clone git@gitlab.com:vgg/wise/wise.git
cd $HOME/wise
conda env create -f environment.yml
conda activate wise
python3 app.py --help

## initialise a project with a batch of nearly 80 million images from 
## the Wikimedia Commons stored in the WebDataset format
python3 app.py init wikimedia \
  --batch-size 128 --model "ViT-L-14:laion2b_s32b_b82k" \
  --store-in "/data/hdd/wikimedia/wise-store" \
  --source "/data/hdd1/wikimedia/data/batch{001..999}/{00000..00007}.tar"
  
## add some new images stored in a folder to the project 
python3 app.py update wikimedia \
  --batch-size 128 \
  --source "/data/hdd1/wikimedia/data/new-images-uploaded-in-2023/"

## create a search index which based on approximate nearest neighbour search
## (for exhaustive search, use --index-type IndexFlatIP)
python3 app.py index wikimedia --index-type IndexIVFFlat

## serve the visual search engine over web
python3 app.py serve wikimedia \
  --index-type IndexIVFFlat \
  --theme-asset-dir ./www/dynamic/
# you can optionally provide a query blocklist (i.e. a list of queries that users should be blocked from searching) using `--query-blocklist /path/to/blocklist.txt`
```

## Test
WISE contains some automated tests to verify the software's
functionality.

```
conda activate wise
python -m pytest -s tests
```

## Contact
For any queries or feedback related to the WISE software, contact [Prasanna Sridhar](mailto:prasanna@robots.ox.ac.uk), [Horace Lee](mailto:horacelee@robots.ox.ac.uk) or [Abhishek Dutta](mailto:adutta@robots.ox.ac.uk).

## Acknowledgements
Development and maintenance of WISE software has been supported by the following grant: Visual AI: An Open World Interpretable Visual Transformer (UKRI Grant [EP/T028572/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/T028572/1))
