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

import itertools
import logging
from functools import cached_property
from typing import Literal, Type

import numpy as np
import torch
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype

from wise.feature.feature_extractor import FeatureExtractor, MultiModalModel


logger = logging.getLogger(__name__)


def get_config_and_metadata(client, model_name: str):
    try:
        client.is_model_ready(model_name)
    except grpcclient.InferenceServerException as e:
        raise RuntimeError(
            f"Model '{model_name}' is not available in the triton server: {e}"
        ) from e

    """Retrieves the model configuration and metadata from the Triton server."""
    # Get the input / output schema
    logger.debug(
        f"model metadata: {client.get_model_metadata(model_name, as_json=True)}"
    )
    model_metadata = client.get_model_metadata(model_name)

    logger.debug(f"Model config: {client.get_model_config(model_name, as_json=True)}")
    model_config = client.get_model_config(model_name).config

    max_batch_size = model_config.max_batch_size
    inputs = {
        input.name: {"shape": input.shape, "dtype": input.datatype}
        for input in model_metadata.inputs
    }
    outputs = {
        output.name: {"shape": output.shape, "dtype": output.datatype}
        for output in model_metadata.outputs
    }
    return max_batch_size, inputs, outputs


class TritonModel(MultiModalModel):
    def __init__(self, model_id: str, url: str, debug: bool = False, **kwargs):
        """Initializes the Triton feature extractor and sets up the client."""
        super().__init__(model_id, **kwargs)
        self._client = grpcclient.InferenceServerClient(
            url=url,
            verbose=debug,  # Set to True for debugging
        )

        # Ensure the server is reachable and ready
        # Ensure the model is ready
        try:
            self._client.is_server_live()
            self._client.is_server_ready()
        except grpcclient.InferenceServerException as e:
            raise RuntimeError(
                f"Failed to connect to Triton server at {url}: {e}"
            ) from e

        logger.debug(f"Connected to Triton server at {url}")

        self._model = model_id
        self._triton_configs = {}

    @property
    def input_image_size(self):
        model_name = f"{self._model}--image"
        if "image" not in self._triton_configs:
            self._triton_configs["image"] = get_config_and_metadata(
                self._client, model_name
            )
        _, inputs, _ = self._triton_configs["image"]
        # NOTE Assuming the first input is always the image in our case
        first_key = next(iter(inputs))
        shape = inputs[first_key]["shape"]
        return tuple(shape[-2:])

    def _get_features(
        self, _type: Literal["image", "audio", "text"], input_params: dict
    ):
        """Sends a tensor to the Triton server and returns the features.

        This method should be implemented to handle the logic for sending a
        generic tensor (like an image or audio tensor) to Triton.
        """
        logger.debug(
            f"({self._model}) Getting {_type} features with params: {input_params}"
        )
        model_name =  f"{self._model}--{_type}"
        if _type not in self._triton_configs:
            self._triton_configs[_type] = get_config_and_metadata(
                self._client, model_name
            )

        max_batch_size, inputs, outputs = self._triton_configs[_type]
        # Send input params to triton inference server using the client

        def _prepare_chunked_input(name, value):

            if name not in inputs:
                raise ValueError(
                    f"Input '{name}' not found in model inputs: {inputs.keys()}"
                )

            if not isinstance(value, (torch.Tensor, np.ndarray)):
                raise TypeError(
                    f"Input '{name}' must be a torch.Tensor or np.ndarray, got {type(value)}"
                )

            input_info = inputs[name]

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()

            np_dtype = triton_to_np_dtype(input_info["dtype"])
            if value.dtype != np_dtype:
                logger.debug(
                    "Converting input dtype from {} to {}".format(value.dtype, np_dtype)
                )
                value = value.astype(np_dtype)

            # TODO shape validation
            if len(value.shape) != len(input_info["shape"]):
                raise ValueError(
                    f"Input '{name}' has shape {value.shape}, expected {input_info['shape']}"
                )

            bs = value.shape[0]
            input_batch_size = input_info["shape"][0]

            chunks = []
            if max_batch_size == 0:
                if input_batch_size != -1:
                    if bs != input_batch_size:
                        raise ValueError(
                            f"Input '{name}' has batch size {bs}, expected {input_batch_size}"
                        )

                # bs = input_batch_size or input_batch_size = -1
                # pass full
                chunks = [
                    grpcclient.InferInput(
                        name, value.shape, input_info["dtype"]
                    ).set_data_from_numpy(value)
                ]
            else:
                if input_batch_size != -1:
                    raise NotImplementedError(
                        f"not implemented for fixed batch size {input_batch_size} with max_batch_size {max_batch_size}"
                    )
                # chunk
                for i in range(0, bs, max_batch_size):
                    v = value[i : i + max_batch_size]
                    chunks.append(
                        grpcclient.InferInput(
                            name, v.shape, input_info["dtype"]
                        ).set_data_from_numpy(v)
                    )

            return chunks

        inputs_data = list(
            zip(*itertools.starmap(_prepare_chunked_input, input_params.items()))
        )

        outputs_data = []
        for chunk in inputs_data:
            logger.debug(f"Sending chunk with ({len(chunk)}) inputs")
            _outputs = [
                grpcclient.InferRequestedOutput(name) for name in outputs.keys()
            ]

            response = self._client.infer(model_name, inputs=chunk, outputs=_outputs)
            if not response:
                raise RuntimeError("Failed to get response from Triton server")

            # Process the response
            infer_output = {
                name: torch.from_numpy(response.as_numpy(name).copy())
                for name in outputs.keys()
            }
            logger.debug(f"Received features: {infer_output.keys()}")
            outputs_data.append(infer_output)

        # Concatenate the outputs
        concatenated_outputs = {}
        if len(outputs_data) == 0:
            raise RuntimeError("No outputs received from Triton server")
        if len(outputs_data) == 1:
            concatenated_outputs = outputs_data[0]
        else:  # Concatenate the outputs along the batch dimension
            for name in outputs.keys():
                concatenated_outputs[name] = torch.cat(
                    [output[name] for output in outputs_data], axis=0
                )

        if len(outputs.keys()) == 1:
            # If there's only one output, return it directly
            return next(iter(concatenated_outputs.values()))
        return concatenated_outputs

    def get_image_features(self, **kwargs):
        return self._get_features("image", kwargs)

    def get_text_features(self, **kwargs):
        return self._get_features("text", kwargs)

    def get_audio_features(self, **kwargs):
        return self._get_features("audio", kwargs)


class _InitializeParameterized(object):
    """
    When called with the cls as the only argument, returns an
    un-initialized instance of the parameterized class. Subsequent __setstate__
    will be called by pickle.
    """

    def __call__(
        self,
        cls,
    ):
        # make a simple object which has no complex __init__
        obj = _InitializeParameterized()
        obj.__class__ = make_triton_feature_extractor(cls)
        return obj


def make_triton_feature_extractor(cls: Type[FeatureExtractor]):
    """A class factory that creates a Triton-enabled feature extractor.

    This function takes a `FeatureExtractor` subclass and dynamically
    creates a new class that overrides its feature extraction methods. The new
    methods are designed to communicate with a Triton Inference Server to
    perform the actual inference, rather than running the model locally.

    Args:
        cls: The `FeatureExtractorBase` subclass to wrap.

    Returns:
        A new class that is a Triton-enabled version of the input class.
    """

    class TritonFeatureExtractor(cls):
        """A Triton-enabled feature extractor class."""
        class Config(cls.Config):
            """Configuration for the Triton feature extractor."""

            url: str
            debug: bool = False

        def __init__(self, model_id, config: Config, **kwargs):
            """Initializes the Triton feature extractor with the given ID and configuration."""

            self.__model_name = model_id.replace("/", "--")
            self.config = config
            super().__init__(model_id, config=config, **kwargs)

        @cached_property
        def model(self):
            """Returns the Triton model client."""
            return TritonModel(
                self.__model_name, self.config.url, debug=self.config.debug
            )

        def __getstate__(self):
            logger.debug(f"getstate: {self.__dict__}")
            state = self.__dict__.copy()
            # Remove unpicklable entries.
            entries = {"model", "tokenizer", "processor"}
            for entry in entries:
                if entry in state:
                    del state[entry]

            if "config" in state:
                # dump model config as dict, so that we can reload it in __setstate__
                config_state = state["config"].model_dump()
                state["config"] = config_state

            if "_TritonFeatureExtractor__model_name" in state:
                # Ensure the model name is a string
                state["model_id"] = state.pop("_TritonFeatureExtractor__model_name")
            return state

        def __setstate__(self, state):
            logger.debug(f"setstate: {state}")
            # Restore instance attributes
            self.__dict__.update(state)
            # Re-initialize the model property
            if "config" in state and not isinstance(state["config"], self.Config):
                self.config = self.Config(**state["config"])
            if (
                "_TritonFeatureExtractor__model_name" not in state
                and "model_id" in state
            ):
                self.__model_name = state["model_id"].replace("/", "--")

        def __reduce__(self):
            # Ensure the object can be pickled
            return (
                _InitializeParameterized(),
                (cls,),
                self.__getstate__(),
            )

    return TritonFeatureExtractor
