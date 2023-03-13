# WISE Image Search Engine (WISE)

WISE Image Search Engine (WISE) is an open-source image search engine
for efficiently searching through large collections of images.

## Installation and Usage

```bash

## setup environment and install dependency libraries
cd $HOME
git clone git@gitlab.com:vgg/wise/wise.git
cd $HOME/wise
conda env create -f environment.yml
conda activate wise
python3 app.py --help

## initialise a project
python3 app.py init wikimedia \
  --batch-size 8 --model "ViT-B/32" \
  --store-in "/data/wikimedia/wise-store" \
  --source "/data/wikimedia/data/batch001.tar" \
  --source "/data/wikimedia/data/batch{005..009}.tar" \

## create a search index
python3 app.py index wikimedia

## enable access of search engine using a web browser
python3 app.py serve wikimedia \
  --theme-asset-dir ./www/[CHOSEN-THEME]/
```

## Test
WISE contains some automated tests to verify the software's
functionality.

```
conda activate wise
python -m pytest -s tests
```
