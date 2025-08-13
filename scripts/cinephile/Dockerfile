# Template for the operating system, environment and libraries
# to prepare an audiovisual search engine for the Cinephile 2025
# challenge using the WISE Search Engine (WISE) software tool.
#
# More details about the Cinephile 2025 challenge are available at:
# https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html

ARG NODE_IMAGE=node:22-slim
ARG BASE_IMAGE=python:3.10-slim
ARG APP=wise

# Stage to build the frontend
FROM ${NODE_IMAGE} AS wise-frontend
RUN mkdir -p /wise/frontend/node_modules && \
    chown -R node:node /wise/frontend
COPY frontend/package*.json /wise/frontend/
WORKDIR /wise/frontend
USER node
RUN npm ci
COPY --chown=node:node frontend . 
RUN npm run build

# Final image
FROM ${BASE_IMAGE}
ARG APP
ENV APP=${APP} \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}" \
    CINEPHILE_DATA_DIR="/data/cinephile"

# Install gosu, curl, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      gosu curl build-essential \
    && rm -rf /var/lib/apt/lists/*



# Create a non-root user to own the files and run the application
RUN useradd --create-home --shell /bin/bash nonroot

# Copy the application source code
WORKDIR /wise
COPY . .


# Copy the built frontend
COPY --from=wise-frontend --chown=nonroot:nonroot \
    /wise/frontend/dist/ /wise/frontend/dist/
COPY --from=wise-frontend --chown=nonroot:nonroot --chmod=0666 \
    /wise/frontend/dist/index.html /wise/frontend/dist/index.html

# Copy the entrypoint script and make it executable
COPY scripts/cinephile/setup.sh /usr/local/bin/setup.sh
RUN chmod +x /usr/local/bin/setup.sh

# Change ownership of the application files to the non-root user
RUN chown -R nonroot:nonroot /wise

# Declare the volume that will hold the environment and data
VOLUME [ "/data/cinephile" ]

ENTRYPOINT ["/usr/local/bin/setup.sh"]
