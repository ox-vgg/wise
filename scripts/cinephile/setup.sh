#!/bin/bash

# Setup the environment and install dependencies to run the 
# WISE Search Engine (WISE) software tool.

set -e

# Install micromamba if it's not already present
if [ ! -f "/usr/local/bin/micromamba" ]; then
    echo "Micromamba not found. Installing it directly..."
    # Download the static binary to the desired location
    curl -L "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64" -o /usr/local/bin/micromamba
    # Make it executable
    chmod +x /usr/local/bin/micromamba
fi

# Define all persistent paths
MAMBA_ROOT_PREFIX="/data/cinephile/deps/micromamba"
CACHE_DIR="/data/cinephile/deps/cache"
TMP_DIR="/data/cinephile/deps/tmp" # For both install and runtime

ENV_NAME="wise-env"
ENV_PATH="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
SUCCESS_FILE="${MAMBA_ROOT_PREFIX}/.install_complete"

TORCH_HOME="${CACHE_DIR}/torch"
HF_HOME="${CACHE_DIR}/huggingface"
MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
WIDS_CACHE="${CACHE_DIR}/wids"
PIP_CACHE_DIR="${CACHE_DIR}/pip"

# If CINEPHILE_RECREATE_ENV is true, wipe the environment first.
if [ "${CINEPHILE_RECREATE_ENV}" = "true" ]; then
    echo "CINEPHILE_RECREATE_ENV is set. Forcing environment rebuild."
    rm -rf "${MAMBA_ROOT_PREFIX}"
fi

# Create all necessary directories as root and set ownership for the nonroot user.
# This is safe to run every time.
mkdir -p "${MAMBA_ROOT_PREFIX}" "${TMP_DIR}" "${TORCH_HOME}" "${HF_HOME}" "${MPLCONFIGDIR}" "${WIDS_CACHE}" "${PIP_CACHE_DIR}"
chown -R nonroot:nonroot "/data/cinephile"

# As the nonroot user, check if the environment is valid. If not, create it.
gosu nonroot env APP="${APP}" TMPDIR="${TMP_DIR}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" bash <<'EOF'
set -e

MAMBA_ROOT_PREFIX="/data/cinephile/deps/micromamba"
ENV_NAME="wise-env"
ENV_PATH="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
SUCCESS_FILE="${MAMBA_ROOT_PREFIX}/.install_complete"

# We only install if the success file is missing.
if [ ! -f "${SUCCESS_FILE}" ]; then
    # If the env path exists without a success file, it's a broken install. Clean it.
    if [ -d "${ENV_PATH}" ]; then
        echo "Incomplete installation detected. Removing previous environment."
        rm -rf "${ENV_PATH}"
    fi

    echo "Micromamba environment not found or incomplete. Creating it..."
    echo "Using temporary directory for pip: ${TMPDIR}"
    echo "Using cache directory for pip: ${PIP_CACHE_DIR}"

    export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cpu'
    export CUDNN_PACKAGE=''
    export ONNXRUNTIME_PACKAGE='onnxruntime'
    if [[ "${APP}" == 'wise' ]]; then
        export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cu124'
        export CUDNN_PACKAGE='cudnn'
        export ONNXRUNTIME_PACKAGE='onnxruntime-gpu'
    fi
    echo "Using Pip index url: ${PIP_EXTRA_INDEX_URL} CUDNN: ${CUDNN_PACKAGE:-none} ONNX: ${ONNXRUNTIME_PACKAGE}"
    
    # Create env from file, install cudnn package if needed
    /usr/local/bin/micromamba create --root-prefix "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" -f "/wise/docker/wise.yml" -y ${CUDNN_PACKAGE}
    # Install onnxruntime
    /usr/local/bin/micromamba run --root-prefix "${MAMBA_ROOT_PREFIX}" --prefix "${ENV_PATH}" python3 -m pip install "${ONNXRUNTIME_PACKAGE}"

    # Create success file *inside this block* on success.
    touch "${SUCCESS_FILE}"
    echo "Environment creation successful."
else
    echo "Using existing micromamba environment."
fi
EOF

echo "Executing command as nonroot: $@"

# Execute the given command within the environment as the nonroot user,
# ensuring all cache and temp directories are correctly set.
exec gosu nonroot env \
  TMPDIR="${TMP_DIR}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  TORCH_HOME="${TORCH_HOME}" \
  HF_HOME="${HF_HOME}" \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  WIDS_CACHE="${WIDS_CACHE}" \
  MAGIC="${ENV_PATH}/share/misc/magic" \
  LD_LIBRARY_PATH="${ENV_PATH}/lib/:${LD_LIBRARY_PATH}" \
  /usr/local/bin/micromamba run --root-prefix "${MAMBA_ROOT_PREFIX}" --prefix "${ENV_PATH}" "$@"




