# Triton Inference Server with WISE

> **Note:** The features described in this document are [still being finalized](https://gitlab.com/vgg/wise/wise/-/merge_requests/96) and may change. They are not yet stable.

WISE supports running the multi-modal models separately using [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (maintained by NVIDIA). This allows better utilisation of GPU memory and opens doors for running inference optimised models (eg. ONNX, TensorRT, Quantization) with WISE

## Pre-requisites

1. Python 3.12 (if not using Python 3.12, you may have to change the triton image in the compose file)
1. You have a working installation of WISE with venv (or conda)
1. [venv-pack](https://jcristharif.com/venv-pack/). You maybe able to use conda with [conda-pack](https://conda.github.io/conda-pack/) if you use correct python version and triton image combination.

The default triton server docker image in `compose.triton.yaml` assumes python3.12 is being used. If you use a different python version, adjust the triton image version to match your python version (see [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html) - example, last image with python3.10 as default is 24.10) or follow the [instructions here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#building-custom-python-backend-stub) to compile a stub compatible with your version.

For venv,
```bash
source PATH_TO_WISE_ENV/bin/activate
pip install venv-pack
venv-pack -o triton/wise-env.tar
```

For conda
```bash
conda activate wise2
export PYTHONNOUSERSITE=1
conda install -c conda-forge conda-pack
conda-pack -o triton/wise-env.tar
```

Once you have the environment packaged, extract it to the `triton/wise-env` folder
```bash
tar -xf triton/wise-env.tar -C triton/wise-env
```

_Note:_ If using conda, run `conda-unpack` as an additional step
```
./triton/wise-env/bin/conda-unpack
```

## Getting started

To get started with Triton, we need to create a folder under `triton/models` corresponding to the feature extractor you are interested in.

Triton requires your models directories to follow a specific structure and doesnt support models deep inside other folders. It needs to be flat.

With WISE, we follow a convention similar to Huggingface - we replace `/` with `--`

For example, if you would like to run `deepinsight/insightface/buffalo_l/_unknown` models with triton, you create a folder `deepinsight--insightface--buffalo_l--_unknown--{MODALITY}` per modality, where modality is one of `image`, `audio`, `text` based on what the model supports. We provide some of the models to make it easy to get started. You can copy paste the config from a similar model with the same type of feature extractor and adjust to your needs.

We will provide a CLI / migrate to [PyTriton](https://triton-inference-server.github.io/pytriton/latest/) in the future to make this easier.

```bash
FEATURE_ID='deepinsight/insightface/buffalo_l/_unknown'

MODEL_DIR="triton/models/${FEATURE_ID//\//--}--image"

mkdir -p ${MODEL_DIR}/1
(cd ${MODEL_DIR}/1 && ln -sf ../../model.py .)

cat <<EOF > ${MODEL_DIR}/config.pbtxt
name: "${FEATURE_ID//\//--}--image"
backend: "python"
max_batch_size: 0

input [
    {
        name: "image"
        data_type: TYPE_UINT8
        dims: [ -1, -1, 3 ]
    }
]

output [
    {
        name: "embeddings"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
    },
    {
        name: "scores"
        data_type: TYPE_FP32
        dims: [ -1 ]
    },
    {
        name: "boxes"
        data_type: TYPE_FP32
        dims: [ -1, 4 ]
    },
    {
        name: "age"
        data_type: TYPE_INT32
        dims: [ -1 ]
    },
    {
        name: "is_male"
        data_type: TYPE_BOOL
        dims: [ -1 ]
    }
]

parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "/wise-env/"}
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
EOF
```

To run triton with this model

```bash
# create a cache dir for triton to store any temporary data
mkdir -p triton/cache
chmod 2777 triton/cache

docker compose -f compose.triton.yaml up -d
```

Now, with WISE, prefix the feature id with `triton:///` - this will automatically use the triton client
in WISE to forward the requests to `localhost:8001`. If you are running this on a separate machine, you can pass it as
`triton://HOST:PORT/${FEATURE_ID}`

You can provide feature extractor config by providing `${MODEL_DIR}/1/config.yaml` (optional - useful for open_clip / HF model overrides)

## Advanced (WIP)

With Triton you can swap the current `model.py` with optimised versions of models exported in ONNX / TensorRT / TorchScript with little to no change

TODO: Add more details on how to use the model export utility for open_clip, clap, owlv2 included in `src/wise/feature/__main__.py`, tests and FAQ

```bash
python3 -m wise.feature --feature-extractor mlfoundations/open_clip/ViT-L-16-SigLIP2-512/webli onnx_models --verify --export --batch_size 1
python3 -m wise.feature --feature-extractor microsoft/clap/2023/four-datasets onnx_models --verify --export --batch_size 1
python3 -m wise.feature --feature-extractor transformers/owlv2/google/owlv2-large-patch14-ensemble output --verify --export
```

## FAQ

**1. Getting "No space left on device" error when running `venv-pack` / `conda-pack`**

This usually happens when the default temporary directory does not have sufficient space to hold the packed env tar.

The workaround is to set the environment variable `TMPDIR` (or `TEMP` or `TMP`) to point to a folder that has space.

```bash
mkdir -p /PATH/TO/FOLDER/WITH/ENOUGH/SPACE

TMPDIR=/PATH/TO/FOLDER/WITH/ENOUGH/SPACE conda-pack ... # or venv-pack
```

*Note: Make sure to create the directory before setting TMPDIR to it, otherwise it will silently switch back to default directory*

**2. CondaPackError: Files managed by conda were found to hav ebeen deleted / overwritten in the following packages**

This happens when pip dependencies sometimes clobber the conda environment.

The fix usually is to pack the env as-is and not let conda-pack do any kind of magic

So if you see the following error
```
Collecting packages...
CondaPackError:
Files managed by conda were found to have been deleted/overwritten in the
following packages:

- PACKAGE A X.Y.Z:
    file1
    file2
    + N others
- PACKAGE B U.V.W:
    ...
...

This is usually due to `pip` uninstalling or clobbering conda managed files,
resulting in an inconsistent environment. Please check your environment for
conda/pip conflicts using `conda list`, and fix the environment by ensuring
only one version of each package is installed (conda preferred).
```

Then try running the `conda-pack` command with `--ignore-missing-files` option. If it still fails, then try the `venv-pack` option
