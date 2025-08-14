# Cinephile Challenge 2025
These instruction describe the process to reproduce the WISE Search Engine (WISE)
operating on 40 hours of videos released by the [Cinephile Challenge 2025](https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html).

## Create Audiovisual Search Engine
```
export BASEDIR=$HOME
cd $BASEDIR
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd $BASEDIR/wise/
docker system prune -a                      # to free docker storage
export CINEPHILE_DATA_DIR=/data/cinephile/  # folder to store all data

# Build docker image for WISE (requires 11.4GB, takes 13min)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose \
  -f scripts/cinephile/compose.yml \
  build wise

# Run wise container in background
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f scripts/cinephile/compose.yml up -d wise

# Create WISE project containing audiovisual features (requires ?, takes ? hours)
docker compose -f scripts/cinephile/compose.yml \
  exec -e CINEPHILE_DATA_DIR=/data/cinephile \
  wise \
  /wise/scripts/cinephile/create-wise-project.sh

# Find shot boundaries for visual content (requires ?, takes ? hours)
...

# Classify scale of each shot (requires ?, takes ? hours)
...

# Import shot and media metadata (requires ?, takes ? hours)
...

# Extract object features. (requires ?, takes ? hours)
...

# Create search index (requires ?, takes ? hours)
...

# Serve search engine over web
...
```

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
| 3. Extract audiovisual features    |        |         |
| 4. Find shot boundaries            |        |         |
| 5. Classify scale of each shot     |        |         |
| 6. Import shot and media metadata  |        |         |
| 7. Extract object features.        |        |         |
| 8. Create search index             |        |         |
| 9. Serve search engine over web    |        |         |
|------------------------------------+--------+---------|
```

After step 3, a basic audiovisual search engine is ready. The remaining steps are
to extract features for objects, classify shot scale and enable media metadata search.

## Frequently Asked Questions (FAQ)

- docker compose commands fail with `no space left on device` error

Use `docker system prune` command to free up space before running the `docker compose` command.