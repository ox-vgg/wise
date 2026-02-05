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

# Based on
# https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html
# https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html
ARG MAMBA_IMAGE=mambaorg/micromamba:1.5.10
ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.10-slim

# For holding huggingface cache, and accessible by any user in the system who runs the container
ARG CACHE_DIR=/tmp/tmp

# Can be wise / wise-cpu
ARG APP=wise

FROM ${MAMBA_IMAGE} AS wise-env
ARG APP
ARG CACHE_DIR
ENV APP=${APP} \
    CACHE_DIR=${CACHE_DIR} \
    HF_HOME=${CACHE_DIR}/hfcache  \
    PYTHONPYCACHEPREFIX=${CACHE_DIR}/pycache


USER root
RUN mkdir -p ${HF_HOME} ${PYTHONPYCACHEPREFIX} && chmod 3777 -R /tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential && \
    rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

USER ${MAMBA_USER}
WORKDIR /tmp

COPY --chown=${MAMBA_USER}:${MAMBA_USER} "environment.yml" "requirements.txt" ./
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/home/${MAMBA_USER}/.cache,uid=${MAMBA_USER_ID},gid=${MAMBA_USER_GID} \
    export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cpu' && \
    if [[ ${APP} == 'wise' ]]; then \
        export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cu124'; \
    fi && \
    echo "Using Pip index url: ${PIP_EXTRA_INDEX_URL}" && \
    micromamba create --always-copy --yes -n wise-env -f "environment.yml"

FROM ${NODE_IMAGE} AS wise-frontend

RUN mkdir -p /wise/frontend/node_modules && \
    chown -R node:node /wise/frontend
COPY frontend/package*.json /wise/frontend/
WORKDIR /wise/frontend
USER node
RUN npm ci
COPY --chown=node:node frontend .
RUN npm run build

FROM gcr.io/distroless/base-debian12 AS wise
ARG APP
ARG CACHE_DIR
ENV APP=${APP} \
    PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    HF_HOME=/tmp/hfcache \
    PATH="/env/bin/:${PATH}"

COPY --from=wise-env --chown=nonroot:nonroot \
    /opt/conda/envs/wise-env /env

WORKDIR /wise
COPY --from=wise-env --chmod=7777 --chown=nonroot:nonroot  \
    ${CACHE_DIR}/ /tmp

COPY --chown=nonroot:nonroot . .

COPY --from=wise-frontend --chown=nonroot:nonroot --chmod=3775 \
    /wise/frontend/dist/ /wise/frontend/dist/

# Hack to allow the rewrite of index.html when we serve the project
# The user who runs the container is possibly not nonroot, so we need a+rw equivalent
COPY --from=wise-frontend --chown=nonroot:nonroot --chmod=0666 \
    /wise/frontend/dist/index.html /wise/frontend/dist/index.html

# For libmagic to find the database in a non standard location
ENV MAGIC='/env/share/misc/magic' \
    LD_LIBRARY_PATH="/env/lib/:${LD_LIBRARY_PATH}"

# You can modify the CMD statement as needed....
ENTRYPOINT ["python3"]
VOLUME [ "/tmp" ]
CMD ["extract-features.py", "--help"]
