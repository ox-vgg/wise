#!/usr/bin/env python3

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

import logging
import os


logger = logging.getLogger(__name__)

try:
    # must import torch tensorrt before compile calls
    import torch_tensorrt
except:
    logger.warning(
        "torch_tensorrt is not installed. Models will be compiled with inductor backend."
    )
    pass

# TODO move this to pydantic settings and inject it
default_triton_url = os.environ.get("WISE_TRITON_URL", "localhost:8001")

def _get_triton_url_and_model_id(_id: str) -> tuple[str | None, str]:
    """
    Extracts the Triton URL and model ID from the feature extractor ID if it starts with "triton://".
    If the ID does not start with "triton://", it returns None for url.
    """
    if not _id.startswith("triton://"):
        return None, _id

    model_id = _id[len("triton://") :]
    url, model_id = model_id.split("/", 1)

    return url, model_id


def get_canonical_feature_extractor_id(_id: str) -> str:
    """
    Returns a canonical form of the feature extractor ID.
    This is useful for ensuring consistent naming when using the feature
    extractor in different contexts, such as in file names or database entries,
    with or without triton
    """
    _, model_id = _get_triton_url_and_model_id(_id)
    return model_id


def get_triton_url_from_id(_id: str) -> str | None:
    """
    Extracts the Triton URL from the feature extractor ID if it starts with "triton://".
    If the ID does not start with "triton://", it returns None.
    """
    url, _ = _get_triton_url_and_model_id(_id)
    return url


def get_feature_extractor_class(_id: str):

    if _id.startswith("mlfoundations/open_clip/"):
        from .mlfoundation_openclip import MlfoundationOpenClipFeatureExtractor

        return MlfoundationOpenClipFeatureExtractor

    if _id.startswith("microsoft/clap/"):
        from .microsoft_clap import MicrosoftClapFeatureExtractor

        return MicrosoftClapFeatureExtractor

    if _id.startswith("transformers/owlv2/"):
        from .transformers_owlv2 import TransformersOWLv2FeatureExtractor

        return TransformersOWLv2FeatureExtractor

    if _id.startswith("deepinsight/insightface/"):
        from .insightface import InsightFaceFeatureExtractor

        return InsightFaceFeatureExtractor

    if _id.startswith("deepinsight/insightface-average/"):
        from .insightface_average import InsightFaceAverageFeatureExtractor

        return InsightFaceAverageFeatureExtractor

    if _id.startswith("hf/"):
        from .hf_feature_extractor import HFMultiModalFeatureExtractor

        return HFMultiModalFeatureExtractor

    raise ValueError(f"Unknown feature extractor id {_id}")


def FeatureExtractorFactory(id, config: dict[str, dict] = {}):
    """
    Extract features (e.g. a vector of length 256) from images, videos and audio.

    Parameters
    ----------
    id : str
        To uniquely identify a pre-trained model and its pre-training dataset.
        The id string is formatted as:

        USER_OR_ORGANIZATION / REPOSITORY_NAME / MODEL_NAME / TRAINING_DATASET

        The purpose of this id is to simplify the management of a WISE project
        containing features extracted by different models operating on different
        modalities (e.g. images, videos, audio).

        Here are some examples showing the id assigned to some of models:
        (a) The model "ViT-B-16-SigLIP-256" trained on the "webli" dataset
        that has been made available at https://github.com/mlfoundations/open_clip/
        can use the following id: "mlfoundations/open_clip/ViT-B-16-SigLIP-256/webli"

        (b) The audio-language model trained on a combination of four datasets and
        made available at https://github.com/microsoft/CLAP/ can be assigned
        the id "microsoft/clap/2023/four-datasets/". The model release year "2023"
        is being used to identify different versions of the model architecture.

        Notes:
        - use "_unknown" to indicate that the model's training dataset is not known
            (e.g. deepinsight/insightface/buffalo_l/_unknown)
        - if both REPOSITORY_NAME and NAME_NAME are same (e.g. CLAP), use model release year
            (e.g. microsoft/clap/2023/four-datasets/)
    """
    url, model_id = _get_triton_url_and_model_id(id)
    model_config = {} | config.get("default", {}) | config.get(model_id, {})

    # If url is None, search for it in the model config
    # else if it is empty string, use the default Triton URL
    url = model_config.get("url", None) if url is None else url
    is_triton = url is not None

    if len(model_id.split("/")) != 4:
        raise ValueError('''Feature extractor name must be formatted as
              USER_OR_ORGANIZATION / REPOSITORY_NAME / MODEL_NAME / TRAINING_DATASET
            For example, use "mlfoundations/open_clip/ViT-B-16-SigLIP-256/webli" for extracting features using ViT
            model trained on the Web Language Image (WebLI) dataset.
            ''')

    cls = get_feature_extractor_class(model_id)
    if is_triton:
        from .triton_runner import make_triton_feature_extractor
        url = url or default_triton_url
        cls = make_triton_feature_extractor(cls)
        model_config["url"] = url

    logger.info(
        f"Creating{is_triton and ' Triton-enabled ' or ' '}feature extractor "
        f"for {model_id} with config: {model_config}"
    )
    return cls.from_config(model_id, config=model_config)
