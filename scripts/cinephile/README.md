# Cinephile Challenge 2025
These instructions describe the process to reproduce the [WISE Search Engine (WISE)](https://meru.robots.ox.ac.uk/cinephile/) operating on 40 hours of 787 videos released by the [Cinephile Challenge 2025](https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html).


## Time and Storage Requirements
The results are based on the following software and hardware configurations.

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
@TODO

## Create Audiovisual Search Engine (Advanced)
```
export BASEDIR=$HOME
cd $BASEDIR
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd $BASEDIR/wise/
docker system prune -a                      # to free docker storage
export CINEPHILE_DATA_DIR=/data/cinephile/  # folder to store all data

# Build docker image for WISE (requires 11.4GB, takes 13min)
# if sudo is required, the use `sudo -E HOST_UID=$(id -u) ...`
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml \
  build wise

# Run wise container in background
HOST_UID=$(id -u) HOST_GID=$(id -g) CINEPHILE_DATA_DIR="/data/cinephile/" docker compose -f scripts/cinephile/compose.yml up -d wise

# Create WISE project containing audiovisual features (requires 8GB, takes 13.6 hours)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile wise \
  /wise/scripts/cinephile/create-wise-project.sh
```

At this stage, the `CINEPHILE_DATA_DIR/wise-project/cinephile` folder
contains a WISE project that can be used to search audio, video, face
and metadata corresponding to all the 787 videos in the Cinephile dataset.
A WISE search engine based on this project can be made available to users
as follows:

```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile wise \
  bash -c '
    LISTEN_ADDRESS="0.0.0.0" \
    PORT="10001" \
    python serve.py \
      --index-type IndexFlatIP \
      --project-dir /data/cinephile/wise-project/cinephile/
  '
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f docs/cinephile/compose.yml exec \
  wise bash -c 'LISTEN_ADDRESS="0.0.0.0" PORT="10001" /env/bin/python serve.py --index-type 
     IndexFlatIP --project-dir /data/cinephile/wise-project/cinephile/'

...
2025-08-19 13:04:11,422 (MainThread): api - INFO - Loading html user interface from frontend/dist
2025-08-19 13:04:11,422 (MainThread): api - INFO - Open http://0.0.0.0:10001/cinephile/ in your browser
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10001 (Press CTRL+C to quit)
...
```
Now the search engine can be accessed by visiting [http://0.0.0.0:10001/cinephile/](http://0.0.0.0:10001/cinephile/) using a web browser. We recommand using the Google Chrome (or Chromium) browser as they have better handling for web pages containing a large number of videos.

We now detect the shot boundaries (i.e. shot video segments).

```
# Create docker container for detecting shot boundaries (requires 12GB, takes 7 min)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml \
  build shot-detection

HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml \
  up -d shot-detection

# Find shot boundaries for visual content (requires 70MB, takes 3.3 hours)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile shot-detection \
  /wise/scripts/cinephile/generate-shot-boundary.sh

# Import shot boundaries in the WISE project
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile wise \
  bash -c '
    /opt/conda/envs/wise-env/bin/python media-metadata.py \
    import-shots \
    --project-dir "/data/cinephile/wise-project/cinephile/" \
    --from-csv "/data/cinephile/wise-project/cinephile/shot-boundaries.csv"
  '
```


Next, we detect the scale (i.e. close-up, full shot, etc) of each shot.

```
# Build docker container for classifying shot scale (requires ?, takes ? hours)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml \
  build shot-scale-classifier

# Classify scale of each shot and import as metadata (requires ?, takes ? hours)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile \
  -e CUDA_VISIBLE_DEVICES=1 \
  shot-scale-classifier \
  /wise/scripts/cinephile/classify-shot-scale.sh

# Extract object features. (requires 5.4GB, takes 6.6 hours)
# To reduce the storage requirements, you can set `objectness_threshold=0.1` in line 190 of `src/feature/transformers_owlv2.py`
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile wise \
  /wise/scripts/cinephile/extract-objects.sh

# Re-run the import-shots command to ensure that vectors_to_shots_map is contains new vector
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile wise \
  bash -c '
    /opt/conda/envs/wise-env/bin/python media-metadata.py \
    import-shots \
    --project-dir "/data/cinephile/wise-project/cinephile/" \
    --from-csv "/data/cinephile/wise-project/cinephile/shot-boundaries.csv"
  '

# Serve search engine over web
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml exec \
  -e CINEPHILE_DATA_DIR=/data/cinephile wise \
  LISTEN_ADDRESS="0.0.0.0" PORT="10001" /wise/serve.py \
  --index-type IndexFlatIP\
  --project-dir /home/tlm/data/wise/projects/cinephile/
```

## Frequently Asked Questions (FAQ)

- docker compose commands fail with `no space left on device` error

Use `docker system prune` command to free up space before running the `docker compose` command.