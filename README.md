# WISE2
WISE search engine enables visual search of large-scale
audiovisual data set using natural language queries. The
WISE open source software is being developed and maintained
by the Visual Geometry Group ([VGG](https://www.robots.ox.ac.uk/~vgg/software/wise/)).

**Note**: This software is under active development and is not suitable for general
users or production usage.

## Demo

```
## 1. Get the code
cd $HOME
git clone https://gitlab.com/vgg/wise/wise.git
cd $HOME/wise
git checkout wise2-integration

## 2. Install software dependencies
python3 -m venv $HOME/wise-dep/
source $HOME/wise-dep/bin/activate
cd $HOME/wise
pip install -r requirements.txt
pip install -r torch-faiss-requirements.txt

## 3. Download some sample videos
mkdir -p $HOME/wise-data/test/
cd $HOME/wise-data/
wget https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/CondensedMovies-10.tar.gz
tar -zxvf CondensedMovies-10.tar.gz

## 4. Extract features
mkdir -p $HOME/wise-projects/
cd $HOME/wise
python3 extract-features.py \
  --media-dir $HOME/wise-data/CondensedMovies-10/ \
  --project-dir $HOME/wise-projects/CondensedMovies-10/

## 5. Create search index (TODO)
python3 create-index.py \
  --project-dir $HOME/wise-projects/CondensedMovies-10/

## 6. Search from command line (TODO)
python3 search.py \
  --query-video "..." \
  --query-audio "..." \
  --project-dir $HOME/wise-projects/CondensedMovies-10/

## 7. Search using web interface (TODO)
python3 serve.py \
  --project-dir $HOME/wise-projects/CondensedMovies-10/

## 8. Add or remove media
python3 add-media.py \
  --media-source PATH \
  --project-dir $HOME/wise-projects/CondensedMovies-10/

python3 del-media.py \
  --media-id ... \
  --project-dir $HOME/wise-projects/CondensedMovies-10/
```

For advanced users, we provide the `wise.py` script that
provides more control over all the tasks related to a
WISE project.
```
python3 wise.py [init|index|search|serve|add|del|] ...
```

## Test

All tests can be discovered and run as follows.
```
cd $HOME/wise
python -m unittest discover -s src/
```

Individual tests can be executed as follows.
```
cd $HOME/wise
python -m unittest src/feature/test_feature_extractor.py
```
