# Cinephile Challenge 2025
These instruction describe the process to reproduce the WISE Search Engine (WISE)
operating on 40 hours of videos released by the [Cinephile Challenge 2025](https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html).

```
export BASEDIR=$HOME
cd $BASEDIR
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd wise/scripts/cinephile
export CINEPHILE_DATA_DIR=/scratch/local/nvme/adutta/d/cinephile/docker 

CINEPHILE_RECREATE_ENV=true docker compose up --build -d
docker compose exec wise /bin/bash /wise/scripts/cinephile/create-wise-project.sh

```