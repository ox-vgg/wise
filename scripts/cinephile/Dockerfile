## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

# Template for the operating system, environment and libraries
# to prepare an audiovisual search engine for the Cinephile 2025
# challenge using the WISE Search Engine (WISE) software tool.
#
# More details about the Cinephile 2025 challenge are available at:
# https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html

ARG MAMBA_IMAGE=mambaorg/micromamba:1.5.10
ARG NODE_IMAGE=node:22-slim

# For holding huggingface cache, and accessible by any user in the system who runs the container
ARG CACHE_DIR=/tmp/cache

# Can be wise / wise-cpu
ARG APP=wise

FROM ${MAMBA_IMAGE} AS wise-base
ARG APP
ARG CACHE_DIR
# non-root user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

COPY --from=tianon/gosu /gosu /usr/local/bin/

ENV APP=${APP} \
    CACHE_DIR=${CACHE_DIR} \
    HF_HOME=${CACHE_DIR}/hfcache  \
    PYTHONPYCACHEPREFIX=${CACHE_DIR}/pycache \
    USERNAME=${USERNAME} \
    USER_UID=${USER_UID} \
    USER_GID=${USER_GID} \
    PATH="/usr/local/bin/:${PATH}"

USER root
RUN mkdir -p ${HF_HOME} ${PYTHONPYCACHEPREFIX} && chmod 3777 -R ${CACHE_DIR}
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential && \
    rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

# Create the user
RUN groupadd -o --gid $USER_GID $USERNAME \
    && useradd -o --uid $USER_UID --gid $USER_GID \
        --create-home --home "/home/$USERNAME" \
        --comment "$USERNAME" $USERNAME

USER ${MAMBA_USER}
WORKDIR /app

# Setup micromamba environment for shot-detection
FROM wise-base AS shot-detection-env
WORKDIR /app
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/home/${MAMBA_USER}/.cache,uid=${MAMBA_USER_ID},gid=${MAMBA_USER_GID} \
    git clone https://gitlab.com/vgg/wise/shot-detection.git && \
    cd shot-detection && \
    git submodule update --init --recursive && \
    micromamba create --always-copy --yes -n shot-detection-env -f "environment.yml" && \
    ${MAMBA_ROOT_PREFIX}/envs/shot-detection-env/bin/python3 -m pip install "tensorflow[and-cuda]>=2.15.0"

WORKDIR /app/shot-detection
COPY scripts/cinephile/generate-shot-boundary.sh .
USER root
COPY scripts/cinephile/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/entrypoint.sh"]
CMD ["bash"]

# Setup micromamba environment for shot-scale-classifier
FROM wise-base AS shot-scale-classifier-env
WORKDIR /app
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/home/${MAMBA_USER}/.cache,uid=${MAMBA_USER_ID},gid=${MAMBA_USER_GID} \
    git clone https://gitlab.com/vgg/wise/shot-scale-classifier.git && \
    cd shot-scale-classifier && \
    git submodule update --init --recursive && \
    micromamba create --always-copy --yes -c conda-forge -n shot-scale-classifier-env python=3.10 && \
    ${MAMBA_ROOT_PREFIX}/envs/shot-scale-classifier-env/bin/python3 -m pip install -r requirements.txt && \
    ${MAMBA_ROOT_PREFIX}/envs/shot-scale-classifier-env/bin/python3 -m pip install torch torchvision torchaudio pandas tqdm omegaconf torchmetrics decord

WORKDIR /app/shot-scale-classifier
COPY scripts/cinephile/classify-shot-scale.sh .
USER root
COPY scripts/cinephile/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/entrypoint.sh"]
CMD ["bash"]


FROM wise-base AS wise-env
# Setup micromamba environment for WISE
COPY --chown=${MAMBA_USER}:${MAMBA_USER} "environment.yml" "requirements.txt" ./
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/home/${MAMBA_USER}/.cache,uid=${MAMBA_USER_ID},gid=${MAMBA_USER_GID} \
    export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cpu' && \
    if [[ ${APP} == 'wise' ]]; then \
        export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cu124'; \
    fi && \
    echo "Using Pip index url: ${PIP_EXTRA_INDEX_URL}" && \
    micromamba create --always-copy --yes -n wise-env -f "environment.yml"

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
ARG CACHE_DIR
ARG USERNAME

USER root
# Create the directory that will be used as a cache mount point
RUN mkdir -p ${CACHE_DIR} && chmod 3777 -R ${CACHE_DIR}

# Create the directory that will be used as a volume mount point
RUN mkdir -p /data/cinephile && chmod 777 /data/cinephile

COPY --from=wise-env --chmod=7777 ${CACHE_DIR}/ ${CACHE_DIR}

COPY --from=wise-env --chown=${MAMBA_USER}:${MAMBA_USER} \
    /opt/conda/envs/wise-env /opt/conda/envs/wise-env

WORKDIR /app
COPY --from=wise-frontend  --chown=${USERNAME}:${USERNAME} --chmod=3775 \
    /wise/frontend/dist/ frontend/dist/

COPY --chown=${USERNAME}:${USERNAME} . .

# Mark scripts as executable
RUN chmod +x scripts/cinephile/*.sh

# For libmagic to find the database in a non standard location
ENV MAGIC='/opt/conda/envs/wise-env/share/misc/magic' \
    LD_LIBRARY_PATH="/opt/conda/envs/wise-env/lib/:${LD_LIBRARY_PATH}" \
    HOME=/tmp \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    MPLCONFIGDIR=/tmp/mplconfig

# Declare the volume that will hold the environment and data
VOLUME [ "/data/cinephile", "${CACHE_DIR}" ]

COPY scripts/cinephile/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/entrypoint.sh"]
CMD ["bash"]
