#!/usr/bin/env python3

import json
from pathlib import Path
import yaml

import triton_python_backend_utils as pb_utils
import torch
from torch.utils.dlpack import from_dlpack
import wise.src.feature as wise_feature


class TritonPythonModel:
    def initialize(self, args):
        # Load the Hugging Face model and processor
        # read the current model name
        # parse the name

        kind = args['model_instance_kind']
        device_id = args['model_instance_device_id']

        device = 'cpu'
        if kind == 'GPU' and torch.cuda.is_available():
            device = f"cuda:{device_id}"

        repository = args['model_repository']
        model_name = args['model_name']
        model_version = args['model_version']

        config_path = Path(repository) / model_version / 'config.yaml'
        config = {}
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
        except Exception as e:
            print(f"Unable to read config at {config_path} - {e}, using empty config.")

        self.enable_autocast = config.pop("enable_autocast", False)
        self.device = device

        self.config = config
        default_config = config.get('default', {})
        default_config['device'] = device
        self.config["default"] = default_config

        model_name = model_name.replace('--', '/')
        model_id, feature_type = model_name.rsplit('/', 1)

        triton_model_config = json.loads(args['model_config'])
        print(triton_model_config)

        self.inputs = [
            input['name'] for input in triton_model_config['input']
        ]
        self.outputs = [
            output['name'] for output in triton_model_config['output']
        ]
        feature_extractor = wise_feature.FeatureExtractorFactory(model_id, self.config)
        self.model = feature_extractor.model
        self.feature_type = feature_type

    def execute(self, requests):
        logger = pb_utils.Logger
        responses = []
        for request in requests:
            inputs = {
                k: from_dlpack(pb_utils.get_input_tensor_by_name(request, k).to_dlpack())
                for k in self.inputs
            }
            if self.feature_type == 'image':
                fn = self.model.get_image_features
            elif self.feature_type == 'text':
                fn = self.model.get_text_features
            elif self.feature_type == 'audio':
                fn = self.model.get_audio_features
            else:
                raise ValueError(f"Unsupported feature type: {self.feature_type}")

            with torch.autocast(
                device_type=self.device,
                enabled=self.enable_autocast,
            ):
                outputs = fn(**inputs)

            if not isinstance(outputs, dict):
                if len(self.outputs) == 1:
                    outputs = {self.outputs[0]: outputs}
                else:
                    raise ValueError("Outputs must be a dictionary when multiple outputs are defined.")

            output_tensors = [
                pb_utils.Tensor.from_dlpack(k, v)
                for k, v in outputs.items()
            ]

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        # Cleanup if necessary
        pass
