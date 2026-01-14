# Using WISE with docker

Starting 2.1.0, WISE releases will have a corresponding docker image, allowing developers to quickly try the app where all the dependencies are installed.

## Pre-requisites

To use WISE with docker, please make sure you have [Docker](https://docs.docker.com/get-started/get-docker/) running on your system. To enable NVIDIA GPU support, please install the [Driver](https://www.nvidia.com/en-us/drivers/) and the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) from NVIDIA. 

On Windows, GPU support for docker containers is available through the WSL2 backend - See [here](https://docs.docker.com/desktop/features/gpu/) to check the pre-requisites and links with details on how to enable it.

## Setup

```bash
git clone https://gitlab.com/vgg/wise/wise.git
cd wise
HOST_UID=$(id -u $USER) HOST_GID=$(id -g $USER) envsubst < .env.template > .env
```

See `.env` to set other variables - VERSION, WISE_PROJECT_DIR, etc. If you are running in an environment where envsubst may not be available, please do this step manually - copy `.env.template` to `.env` and fill the values based on your user id and group id. This is required to persist the project and model files with the correct permissions.

## Usage

To use the docker image, set the project name and run the scripts as shown below. Please refer to the details documentation for clearer instructions for each script. You can also pass `--help` to the scripts to get more info on the arguments.

```bash
export WISE_PROJECT="projects/PROJECT_NAME"
docker compose run --rm -it wise extract-features.py --project-dir "${WISE_PROJECT}" ...
docker compose run --rm -it wise create-index.py --project-dir "${WISE_PROJECT}" ...
docker compose run --rm -it wise serve.py --project-dir "${WISE_PROJECT}" ...
```

To test if WISE Docker can use the GPU, run the following command - it must print `True`

```bash
docker compose run -it wise -- -c 'import torch; print(torch.cuda.is_available())'
```

The default build includes the GPU libraries. While it can run on all machines, it may not be useful to download a very large image when running in a GPU-less environment. A smaller image `wise-cpu` is also available.

To use the CPU only version, run `docker compose -f compose.yaml -f compose.cpu.yaml` instead in the above commands.

To make things simpler, consider setting an alias in bash. Add these to your bashrc to make it persistent. 
```bash
alias wise='docker compose run --rm -it wise'
alias wise-cpu='docker compose -f compose.yaml -f compose.cpu.yaml run --rm -it wise'
```
TODO: Add windows powershell instructions

Once the alias is set, WISE can also be run as follows
```bash
export WISE_PROJECT="projects/PROJECT_NAME"
wise extract-features.py --project-dir "${WISE_PROJECT}" ...
wise create-index.py --project-dir "${WISE_PROJECT}" ...
wise serve.py --project-dir "${WISE_PROJECT}" ...
```

The projects created by WISE will be stored in the `projects/PROJECT_NAME` folder in the current working directory (or in `WISE_PROJECT_DIR` if set in .env)

## Future plans

- Publish multiarch images for better compatibility
- Add a docker based development environment (devcontainer) and VSCode extensions recommendations

