from functools import cached_property
import itertools
import logging
from typing import Type, Literal

from .feature_extractor import FeatureExtractor
import torch
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype

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


class TrtitonModel(object):
    def __init__(self, model: str, url: str, debug: bool = False):
        """Initializes the Triton feature extractor and sets up the client."""

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

        self._model = model
        self._triton_configs = {}

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

            super().__init__(model_id, config=config, **kwargs)
            self.__model_name = model_id.replace('/', '--')

            self.config = config

        @cached_property
        def model(self):
            """Returns the Triton model client."""
            return TrtitonModel(
                self.__model_name, self.config.url, debug=self.config.debug
            )

    return TritonFeatureExtractor
