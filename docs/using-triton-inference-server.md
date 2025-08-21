# Triton Inference Server with WISE

WISE supports running the multi-modal models separately using [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (maintained by NVIDIA). This allows better utilisation of GPU memory and opens doors for running inference optimised models (eg. ONNX, TensorRT, Quantization) with WISE

## Pre-requisites

1. Python 3.12 (if not using Python 3.12, you may have to change the triton image in the compose file)
1. You have a working installation of WISE with venv (You maybe able to use conda with [conda-pack](https://conda.github.io/conda-pack/) if you use correct python version and triton image combination)
1. [venv-pack](https://jcristharif.com/venv-pack/)

The default triton server docker image in `compose.triton.yaml` assumes python3.12 is being used. If you use a different python version, adjust the triton image version to match your python version (see [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html) - example, last image with python3.10 as default is 24.10) or follow the [instructions here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#building-custom-python-backend-stub) to compile a stub compatible with your version.


```bash
source PATH_TO_WISE_ENV/bin/activate
pip install venv-pack
venv-pack -o triton/wise-env.tar

# delete existing env
tar -xf triton/wise-env.tar -C triton/wise-env
```

## Getting started

To get started with Triton, we need to create a folder under `triton_model` corresponding to the feature extractor you are interested in. 

Triton requires your models directories to follow a specific structure and doesnt support models deep inside other folders. It needs to be flat.

With WISE, we follow a convention similar to Huggingface - we replace `/` with `--`

For example, if you would like to run `deepinsight/insightface/buffalo_l/_unknown` models with triton, you create a folder `deepinsight--insightface--buffalo_l--_unknown--{MODALITY}` per modality, where modality is one of `image`, `audio`, `text` based on what the model supports. We provide some of the models to make it easy to get started. You can copy paste the config from a similar model with the same type of feature extractor and adjust to your needs.

We will provide a CLI / migrate to [PyTriton](https://triton-inference-server.github.io/pytriton/latest/) in the future to make this easier.

```bash
FEATURE_ID='deepinsight/insightface/buffalo_l/_unknown'

MODEL_DIR="triton_models/${FEATURE_ID#/#--}--image"

mkdir -p ${MODEL_DIR}/1
(cd ${MODEL_DIR}/1 && ln -sf ../../model.py .)

cat <<EOF > ${MODEL_DIR}/config.pbtxt
name: "${FEATURE_ID#/#--}--image"
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

_TODO: Add other models to `triton_models`_


To run triton with this model

```bash

# activate your environment if not already done
# source /path/to/env/bin/activate
export WISE_ENV=${VIRTUAL_ENV} # or $CONDA_PREFIX if using that

# create a cache dir for triton to store any temporary data
mkdir -p cache
chmod 2777 cache

docker compose -f compose.triton.yaml up -d
```

Now, with WISE, prefix the feature id with `triton:///` - this will automatically use the triton client
in WISE to forward the requests to `localhost:8001`. If you are running this on a separate machine, you can pass it as 
`triton://HOST:PORT/${FEATURE_ID}`

You can provide feature extractor config by providing `${MODEL_DIR}/1/config.yaml` (optional - useful for open_clip / HF model overrides)

## Advanced (WIP)

With Triton you can swap the current `model.py` with optimised versions of models exported in ONNX / TensorRT / TorchScript with little to no change

TODO: Add more details on how to use the model export utility for open_clip, clap, owlv2 included in `src/feature/__main__.py`, tests and FAQ

```bash
python3 -m src.feature --feature-extractor mlfoundations/open_clip/ViT-L-16-SigLIP2-512/webli onnx_models --verify --export --batch_size 1
python3 -m src.feature --feature-extractor microsoft/clap/2023/four-datasets onnx_models --verify --export --batch_size 1
python3 -m src.feature --feature-extractor transformers/owlv2/google/owlv2-large-patch14-ensemble output --verify --export
```