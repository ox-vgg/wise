# WISE2

WISE search engine enables visual search of large-scale
audiovisual data set using natural language queries. The
WISE open source software is being developed and maintained
by the Visual Geometry Group ([VGG](https://www.robots.ox.ac.uk/~vgg/software/wise/)).

**Note**: This software is under active development and is not suitable for general
users or production usage.

## Demo

```bash
## 1. Get the code
git clone -b wise2-integration https://gitlab.com/vgg/wise/wise.git
cd wise

## 2. Install software dependencies
## There are two ways to install software dependencies.
## 2.1 Using Conda
## 2.2 Using Python venv (requires Python >= 3.10)

## 2.1 Installing software dependencies using Conda
conda env create -f environment.yml
conda activate wise
pip install --no-deps msclap==1.3.3  # avoids installing conflicting version of torch

## 2.2 Alternatively, you can also use Python's venv module which
# supports creating lightweight “virtual environments”. The lines
# below are commented out to avoid confusion.
# python3 --version                  # must be >= 3.10
# sudo apt install ffmpeg            # ffmpeg is required to load videos
# python3 -m venv wise-dep/          # create virtual environment
# source wise-dep/bin/activate
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# pip install --no-deps msclap==1.3.3
# pip install -r torch-faiss-requirements.txt

## 3. Download some sample videos
mkdir -p wise-data/
curl -sLO "https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/CondensedMovies-10.tar.gz"
tar -zxvf CondensedMovies-10.tar.gz -C wise-data/

## 4. Extract features
mkdir -p wise-projects/
python3 extract-features.py \
  --media-dir wise-data/CondensedMovies-10/ \
  --project-dir wise-projects/CondensedMovies-10/

## 5. Create search index
python3 create-index.py \
  --project-dir wise-projects/CondensedMovies-10/

## 6. Search from command line
# simple use cases
python3 search.py \
  --query fighting --in video \
  --query punching --in audio \
  --project-dir wise-projects/CondensedMovies-10/

## 7. Search using web interface (TODO)
python3 serve.py \
  --project-dir wise-projects/CondensedMovies-10/

## 8. Add or remove media (TODO)
python3 add-media.py \
  --media-source PATH \
  --project-dir wise-projects/CondensedMovies-10/

python3 del-media.py \
  --media-id ... \
  --project-dir wise-projects/CondensedMovies-10/
```

For advanced users, we provide the `wise.py` script that
provides more control over all the tasks related to a
WISE project.

```bash
python3 wise.py [init|index|search|serve|add|del|] ...
```

## Test

Individual tests can be executed as follows.

```bash
python -m unittest src/feature/test_feature_extractor.py
python -m unittest src/feature/store/test_feature_store.py
```

All tests can be discovered and run as follows.

```bash
python -m unittest discover -s src/
```
