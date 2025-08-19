# Template for the operating system, environment and libraries
# to prepare an audiovisual search engine for the Cinephile 2025
# challenge using the WISE Search Engine (WISE) software tool.
#
# More details about the Cinephile 2025 challenge are available at:
# https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html

ARG MAMBA_IMAGE=mambaorg/micromamba:1.5.10
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.10-slim

# For holding huggingface cache, and accessible by any user in the system who runs the container
ARG CACHE_DIR=/tmp/tmp

# Can be wise / wise-cpu
ARG APP=wise

FROM ${MAMBA_IMAGE} AS wise-base
ARG APP
ARG CACHE_DIR
ENV APP=${APP} \
    CACHE_DIR=${CACHE_DIR} \
    HF_HOME=${CACHE_DIR}/hfcache  \
    PYTHONPYCACHEPREFIX=${CACHE_DIR}/pycache

USER root
RUN mkdir -p ${HF_HOME} ${PYTHONPYCACHEPREFIX} && chmod 3777 -R /tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential && \
    rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

# Add ARGs for user and group IDs
ARG HOST_UID=1000
ARG HOST_GID=1000

# Create a non-root user with specified UID and GID
RUN groupadd -g ${HOST_GID} -o nonroot && useradd --no-log-init -u ${HOST_UID} -g nonroot -o -m nonroot

USER ${MAMBA_USER}
WORKDIR /tmp

FROM wise-base AS wise-env
# Setup micromamba environment for WISE
COPY --chown=${MAMBA_USER}:${MAMBA_USER} "docker/wise.yml" ./
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/home/${MAMBA_USER}/.cache,uid=${MAMBA_USER_ID},gid=${MAMBA_USER_GID} \
    export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cpu' && \
    export ONNXRUNTIME_PACKAGE='onnxruntime' && \
    if [[ ${APP} == 'wise' ]]; then \
        export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cu124' && \
        export ONNXRUNTIME_PACKAGE='onnxruntime-gpu[cuda,cudnn]'; \
    fi && \
    echo "Using Pip index url: ${PIP_EXTRA_INDEX_URL} ONNX: ${ONNXRUNTIME_PACKAGE} " && \
    micromamba create --always-copy --yes -n wise-env -f "wise.yml" && \
    ${MAMBA_ROOT_PREFIX}/envs/wise-env/bin/python3 -m pip install "${ONNXRUNTIME_PACKAGE}"

# Setup micromamba environment for shot-detection
FROM wise-base AS shot-detection-env
RUN git clone https://gitlab.com/vgg/wise/shot-detection.git && \
    cd shot-detection && \
    git submodule update --init --recursive && \
    micromamba create --always-copy --yes -n shot-detection-env -f "environment.yml"

# Setup micromamba environment for shot-scale-classifier
FROM wise-base AS shot-scale-classifier-env
RUN git clone https://gitlab.com/vgg/wise/shot-scale-classifier.git && \
    cd shot-scale-classifier && \
    git submodule update --init --recursive && \
    micromamba create --always-copy --yes -c conda-forge -n shot-scale-classifier-env python=3.10 && \
    ${MAMBA_ROOT_PREFIX}/envs/shot-scale-classifier-env/bin/python3 -m pip install -r requirements.txt && \
    ${MAMBA_ROOT_PREFIX}/envs/shot-scale-classifier-env/bin/python3 -m pip install torch torchvision torchaudio pandas tqdm omegaconf torchmetrics decord

# Setup dependencies for WISE frontend
FROM ${NODE_IMAGE} AS wise-frontend

RUN mkdir -p /wise/frontend/node_modules && \
    chown -R node:node /wise/frontend
COPY frontend/package*.json /wise/frontend/
WORKDIR /wise/frontend
USER node
RUN npm ci
COPY --chown=node:node frontend .
RUN npm run build

FROM wise-base AS wise
COPY --from=wise-env --chown=nonroot:nonroot \
    /opt/conda/envs/wise-env /opt/conda/envs/wise-env

# Since we run out of space in `/var/lib/docker` while creating this container,
# we decided to create three separate stages for each environment.
#
#COPY --from=shot-detection-env --chown=nonroot:nonroot \
#    /opt/conda/envs/shot-detection-env /opt/conda/envs/shot-detection-env
#COPY --from=shot-scale-classifier-env --chown=nonroot:nonroot \
#    /opt/conda/envs/shot-scale-classifier-env /opt/conda/envs/shot-scale-classifier-env


WORKDIR /wise
COPY --from=wise-env --chmod=7777 --chown=nonroot:nonroot  \
    ${CACHE_DIR}/ /tmp

COPY --chown=nonroot:nonroot . .

USER root
# Change ownership of the application files to the non-root user
RUN chown -R nonroot:nonroot /wise
# Create the directory that will be used as a volume mount point
RUN mkdir -p /data/cinephile
# Change ownership of the cinephile files to the non-root user
RUN chown -R nonroot:nonroot /data/cinephile
# Mark scripts as executable
RUN chmod +x /wise/scripts/cinephile/*.sh

COPY --from=wise-frontend --chown=nonroot:nonroot --chmod=3775 \
    /wise/frontend/dist/ /wise/frontend/dist/

# Hack to allow the rewrite of index.html when we serve the project
# The user who runs the container is possibly not nonroot, so we need a+rw equivalent
COPY --from=wise-frontend --chown=nonroot:nonroot --chmod=0666 \
    /wise/frontend/dist/index.html /wise/frontend/dist/index.html

# For libmagic to find the database in a non standard location
ENV MAGIC='/opt/conda/envs/wise-env/share/misc/magic' \
    LD_LIBRARY_PATH="/opt/conda/envs/wise-env/lib/:${LD_LIBRARY_PATH}"

# Declare the volume that will hold the environment and data
VOLUME [ "/data/cinephile" ]

USER nonroot


