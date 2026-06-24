# Cinephile Challenge 2025
These instructions describe the process to reproduce the [WISE Search Engine (WISE)](https://meru.robots.ox.ac.uk/cinephile/) operating on 40 hours of 787 videos released by the [Cinephile Challenge 2025](https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html).

## Pre-requisites

- Storage space of 120GB are required to build and run the audiovisual search engine

- NVIDIA GPU with 11 GB to [re-use precomputed features](#create-audiovisual-search-engine-simple) (or) 24 GB of memory for reproducing the results [from scratch](#create-audiovisual-search-engine-advanced)
  - Note: GPU must be Volta Architecture or newer. Kepler, Maxwell and Pascal Architectures are not supported with this release. Please contact us if you need a version that works with these unsupported GPUs

- Docker with GPU support
  - First, ensure docker desktop / docker engine is installed on your system - [instructions](https://docs.docker.com/desktop/)
  - To enable GPU support,
    - Linux:
      - Install NVIDIA Driver for your GPU - [installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)
      - Install NVIDIA Container Toolkit - [installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    - Windows
      - Please follow the steps in this [guide](https://docs.docker.com/desktop/features/gpu/)

  - To test if GPU support works, run the following. It should complete without error and print your GPU device and the FLOPs observed.
  ```bash
  docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
  ```
- gettext (for envsubst command in Unix / Linux, optional)

We tested everything on a server running Ubuntu Linux. While everything should work as is on other platforms, we haven't explicitly tested them ourselves. We recommend running the script on a Linux server to reproduce the results.


## Computing and Storage Estimates
The computing time reported in this document are based on the following hardware and software configurations.

 * OS: Ubuntu 22.04.5 LTS
 * CPU: Intel Xeon Silver 4216 CPU @ 2.10GHz with 64 cores and 377GB of RAM
 * GPU: Nvidia RTX A6000 (49140MB RAM)

```
|------------------------------------+--------+---------|
| Step                               | Time   | Storage |
|------------------------------------+--------+---------|
| 1. Download video and metadata zip | 27min  | 41GB    |
| 2. Unzip video and metadata        | -      | -       |
| 3. Extract audiovisual features    | 13.6hr | 8GB     |
| 4. Find shot boundaries            |  3.3hr | 70MB    |
| 5. Classify scale of each shot     |   ~1hr | 3MB     |
| 6. Import shot and media metadata  | -      | -       |
| 7. Extract object features         |  6.6hr | 5.4GB   |
| 8. Create search index             | -      | -       |
| 9. Serve search engine over web    | -      | -       |
|------------------------------------+--------+---------|
```

After step 3, a basic audiovisual search engine is ready. The remaining steps are to extract features for objects, classify shot scale and enable media metadata search.

## Create Audiovisual Search Engine (Simple)
Uses pre-computed data and therefore does not demand huge computational and storage resources.

```bash
# Clone the code
git clone -b cinephile2025 https://gitlab.com/vgg/wise/wise.git
cd wise

# Setup env vars
## NOTE: If you are running on windows or if envsubst is unavailable
## Please copy .env.template to .env and manually set the values for the HOST_UID and HOST_GID,
## matching your current user id and group id. It can usually be set to 1000 when using docker desktop
## on windows and mac.
HOST_UID=$(id -u $USER) HOST_GID=$(id -g $USER) envsubst < .env.template > .env

export COMPOSE_FILE="scripts/cinephile/compose.yml"

# folder to store all data
# change this if you want to store the data elsewhere
# NOTE: Must be either an absolute path or be relative to the COMPOSE_FILE
export CINEPHILE_DATA_DIR="$PWD/data/cinephile/"
mkdir -p ${CINEPHILE_DATA_DIR}

# pull pre-built docker image
docker compose pull wise

# Download data and features
docker compose run --rm -it wise scripts/cinephile/download-data.sh
docker compose run --rm -it wise scripts/cinephile/download-project.sh
```

At this stage, the `CINEPHILE_DATA_DIR/wise-project/cinephile` folder
contains a WISE project that can be used to search audio, video, face
and metadata corresponding to all the 787 videos in the Cinephile dataset.
A WISE search engine based on this project can be made available to users
as follows:

```bash
# Must run the following commands everytime you start a new shell
# make sure to change the CINEPHILE_DATA_DIR if you have a different path
# cd /path/to/wise/
# export COMPOSE_FILE="scripts/cinephile/compose.yml"
# export CINEPHILE_DATA_DIR="$PWD/data/cinephile/"

# Serve
docker compose up wise
```

You will see an output as follows when the service is up and running
```
...
2025-08-19 13:04:11,422 (MainThread): api - INFO - Loading html user interface from frontend/dist
2025-08-19 13:04:11,422 (MainThread): api - INFO - Open http://0.0.0.0:10001/cinephile/ in your browser
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10001 (Press CTRL+C to quit)
...
```

Now the search engine can be accessed by visiting [http://0.0.0.0:10001/cinephile/](http://0.0.0.0:10001/cinephile/) using a web browser. We recommand using the Google Chrome (or Chromium) browser as they have better handling for web pages containing a large number of videos.

## Create Audiovisual Search Engine (Advanced)

Step by step instructions to reproduce the project from scratch

```bash
export BASEDIR=$HOME
cd $BASEDIR
git clone https://gitlab.com/vgg/wise/wise.git
cd $BASEDIR/wise/
HOST_UID=$(id -u $USER) HOST_GID=$(id -g $USER) envsubst < .env.template > .env
export COMPOSE_FILE=scripts/cinephile/compose.yml

# folder to store all data
# change this if you want to store the data elsewhere
# NOTE: Must be either an absolute path or be relative to the COMPOSE_FILE
export CINEPHILE_DATA_DIR="$PWD/data/cinephile/"
mkdir -p ${CINEPHILE_DATA_DIR}

# Build docker image for WISE (requires 11.4GB, takes 13min)
# if sudo is required, the use `sudo -E HOST_UID=$(id -u) ...`
docker compose build wise

# Create WISE project containing audiovisual features (requires 8GB, takes 13.6 hours)
docker compose run -it wise scripts/cinephile/create-wise-project.sh
```

At this stage, the `${CINEPHILE_DATA_DIR}/wise-project/cinephile` folder
contains a WISE project that can be used to search audio, video, face
and metadata corresponding to all the 787 videos in the Cinephile dataset.
A WISE search engine based on this project can be made available to users
as follows:

```bash
docker compose up wise
```

Once the app is up and running you will see a log similar to one shown below

```
...
2025-08-19 13:04:11,422 (MainThread): api - INFO - Loading html user interface from frontend/dist
2025-08-19 13:04:11,422 (MainThread): api - INFO - Open http://0.0.0.0:10001/cinephile/ in your browser
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10001 (Press CTRL+C to quit)
...
```
Now the search engine can be accessed by visiting [http://0.0.0.0:10001/cinephile/](http://0.0.0.0:10001/cinephile/) using a web browser. We recommand using the Google Chrome (or Chromium) browser as they have better handling for web pages containing a large number of videos.

We now proceed to detect the shot boundaries (i.e. shot video segments).

```bash
# Create docker container for detecting shot boundaries (requires 12GB, takes 7 min)
docker compose build shot-detection

# Find shot boundaries for visual content (requires 70MB, takes 3.3 hours)
docker compose run -it shot-detection bash generate-shot-boundary.sh

# Import shot boundaries in the WISE project
docker compose run -it wise \
  python media-metadata.py \
    import-shots \
    --project-dir "/data/cinephile/wise-project/cinephile/" \
    --from-csv "/data/cinephile/wise-project/cinephile/shot-boundaries.csv"
```

Next, we detect the scale (i.e. close-up, full shot, etc) of each shot.

```bash
# Build docker container for classifying shot scale
docker compose build shot-scale-classifier

# Classify scale of each shot and import as metadata (requires 3Mb, takes ~ 1 hour)
docker compose run -it shot-scale-classifier bash classify-shot-scale.sh

# import shot scale
docker compose run -it wise \
  python3 media-metadata.py \
    import-shot-scale \
    --project-dir "/data/cinephile/wise-project/cinephile/" \
    --from-csv "/data/cinephile/wise-project/cinephile/thumbs-shot-scale.csv"
```

We now proceed to detect individual objects in the dataset.

```bash
# Extract object features. (requires 5.4GB, takes 6.6 hours)
# To reduce the storage requirements, you can set `objectness_threshold=0.1` in line 190 of `src/feature/transformers_owlv2.py`
docker compose run -it wise scripts/cinephile/extract-objects.sh

# Re-run the import-shots command to ensure that vectors_to_shots_map is contains new vector
docker compose run -it wise \
  python media-metadata.py \
    import-shots \
    --project-dir "/data/cinephile/wise-project/cinephile/" \
    --from-csv "/data/cinephile/wise-project/cinephile/shot-boundaries.csv"
```

As before, you can run the whole app with the following command to search based on all features of WISE

```bash
# Serve search engine over web
docker compose up wise
```


## Methodology

The WISE search engine has the following five search modes: Visual, Face, Object, Metadata and Audio.
The Visual search mode is based on vision-language models (e.g. [CLIP](https://github.com/mlfoundations/open_clip/)) and has already been described in [1].
The Audio search mode relies on audio-language model (e.g. [CLAP](https://github.com/microsoft/CLAP)) for feature representation of audio content and operates in a way similar to the Visual mode.
The Face search mode uses [Insightface](https://github.com/deepinsight/insightface) model to represent each automatically detected face region using a feature vector followed by a nearest neighbour search (e.g. using [faiss](https://github.com/facebookresearch/faiss) library) to find other matching faces in a large collection of images or videos.
The Object search mode relies on the [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2) open-vocabulary object detection model, which allows users to search for objects in images and video frames using natural language descriptions instead of fixed category labels.
Finally, the Metadata mode uses the full text search capabilities of SQL database engine (e.g. [SQLite](https://sqlite.org/fts5.html)) to search through text metadata (e.g. title, description, year, etc.) associated with each media file.
The Visual, Audio, Face and Object search modes can be combined with Metadata mode to filter audiovisual search results based on metadata constraints.

References:

[1] Sridhar, Prasanna, Horace Lee, Abhishek Dutta, and Andrew Zisserman. "WISE image search engine (WISE)." In Wiki workshop, virtual event. 2023.

## Frequently Asked Questions (FAQ)

### Docker compose commands fail with `no space left on device` error

Use `docker system prune` command to free up space before running the `docker compose` command.

### After the videos are extracted, is it safe to delete the downloaded ZIP files (size=40GB) containing the videos?

Yes, it safe to delete the ZIP files to free up storage space.

### What models are being used for visual, face, object and audio searches?

The following models are being used for audiovisual search engine publicly available at [https://meru.robots.ox.ac.uk/cinephile/](https://meru.robots.ox.ac.uk/cinephile/).

| Search Mode | Model |
|-------------|-------|
| Visual   | [OpenCLIP ViT-L-16-SigLIP2-512](https://github.com/mlfoundations/open_clip/) trained on [webli](https://research.google/blog/pali-scaling-language-image-learning-in-100-languages/) dataset |
| Face     | [InsightFace buffalo_l](https://github.com/deepinsight/insightface) model trained on [various face datasets](https://github.com/deepinsight/insightface#datasets) |
| Object   | [Google owlv2-large-patch14-ensemble](https://huggingface.co/google/owlv2-large-patch14-ensemble) trained on [various datasets](https://arxiv.org/abs/2306.09683) |
| Audio    | [Microsoft CLAP](https://github.com/microsoft/CLAP) model trained on [four datasets](https://arxiv.org/abs/2309.05767) |
| Metadata | [SQLite](https://sqlite.org/) database engine's full text search module |
