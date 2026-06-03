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

import collections
import logging
import re
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from msclap import CLAP

from wise.feature.feature_extractor import (
    FeatureExtractor,
    MultiModalModel,
    get_torch_device,
)


logger = logging.getLogger(__name__)

# Copied from clap to make it pickleable
np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class MicrosoftClapModel(MultiModalModel):

    @cached_property
    def _clap_wrapper(self):
        use_cuda = self.DEVICE.type == 'cuda'
        logger.info(
            f"Initialising microsoft/clap (version={self.model_id}, use_cuda={use_cuda})"
        )
        instance = CLAP(version=self.model_id, use_cuda=use_cuda)
        instance.clap.to(self.DEVICE)
        # TODO get it from config along with options?
        if self.compile:
            available_backends = torch._dynamo.list_backends()
            backend = "inductor"
            if "tensorrt" in available_backends:
                backend = "tensorrt"
            logger.info(f"Compiling model with backend {backend}")
            instance.clap.compile(mode="reduce-overhead", backend=backend)
        return instance

    @property
    def model(self):
        return self._clap_wrapper.clap

    @torch.inference_mode()
    def get_audio_features(self, **kwargs) -> torch.Tensor:
        """Extract image features from the model."""
        audio = kwargs.get("audio", None)
        if not isinstance(audio, torch.Tensor):
            raise ValueError(
                "Audio tensor input is required for audio feature extraction."
            )
        return self.model.audio_encoder(audio.to(self.DEVICE))[0].float()

    @torch.inference_mode()
    def get_text_features(self, **kwargs) -> torch.Tensor:
        """Extract text features from the model."""
        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if input_ids is None or attention_mask is None:
            raise ValueError(
                "Input IDs and attention mask are required for text feature extraction."
            )
        x = {
            "input_ids": input_ids.to(self.DEVICE),
            "attention_mask": attention_mask.to(self.DEVICE),
        }

        return self.model.caption_encoder(x).float()

    get_image_features = None  # CLAP does not support image features

    @torch.inference_mode()
    def export_to_onnx(
        self, save_path: Path, audio_inputs: tuple, text_inputs: tuple, **kwargs
    ):
        """Export the model to ONNX format."""
        model = self.model
        model.eval()

        logger.info(f"Exporting audio model")
        output_path = Path(f"{save_path}--audio")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "model.onnx"

        class CustomAudioEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.audio_encoder = model.audio_encoder

            def forward(self, audio):
                x = self.audio_encoder(audio)
                return x[0]

        audio_encoder = CustomAudioEncoder(model)
        audio_encoder.eval()

        torch.onnx.export(
            audio_encoder,
            audio_inputs,
            output_path,
            input_names=["audio"],
            output_names=[
                "embeddings",
            ],
            dynamo=True,
            verify=True,
            do_constant_folding=True,
            opset_version=20,
            report=True,
        )
        logger.info(f"Successfully exported audio model to {output_path}")

        class CustomTextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.caption_encoder = model.caption_encoder

            def forward(self, input_ids, attention_mask):
                x = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                return self.caption_encoder(x)

        logger.info(f"Exporting text model")
        output_path = Path(f"{save_path}--text")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "model.onnx"

        text_encoder = CustomTextEncoder(model)
        torch.onnx.export(
            text_encoder,
            text_inputs,
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=[
                "embeddings",
            ],
            do_constant_folding=True,
            opset_version=20,
            dynamo=True,
            verify=True,
        )
        logger.info(f"Successfully exported text model to {output_path}")

class MicrosoftClapFeatureExtractor(FeatureExtractor):
    """
    Audio feature extractors created by Microsoft's CLAP project
    see https://github.com/microsoft/CLAP/

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'microsoft/clap/'
    DESCRIPTION = 'See https://github.com/microsoft/CLAP'

    ## CLAP supports text and audio (no image)
    preprocess_image = None
    extract_image_features = None
    extract_image_region_features = None

    class Config(FeatureExtractor.Config):
        pass

    def __init__(
        self,
        id,
        *,
        device: str | torch.device | None = None,
        warmup: bool = False,
        config: Config = Config(),
        **kwargs,
    ):
        super().__init__(id, device=device)
        id_tokens = id.split('/')

        assert len(id_tokens) == 4
        if id_tokens[2] not in CLAP.model_name or "clapcap" in id_tokens[2]:
            raise ValueError(
                f'Model version {id_tokens[2]} is not available. Available models are {[x for x in CLAP.model_name.keys() if "clapcap" not in x]}'
            )

        self.model_id = id_tokens[2]
        self.DEVICE = get_torch_device(device)
        self.compile = config.compile

        # we instantiate the mode, but we do not keep the CLAP model, as we load it lazily on the requested device
        # later
        _model = CLAP(version=self.model_id, use_cuda=False)
        with torch.no_grad():
            self.logit_scale = _model.clap.logit_scale.detach().clone()

        del _model

        if warmup:
            self.warmup()

    @cached_property
    def processor(self):
        _processor = CLAP(version=self.model_id, use_cuda=False)
        _processor.clap = None
        del _processor.clap

        return _processor

    @cached_property
    def model(self):
        return MicrosoftClapModel(
            model_id=self.model_id,
            device=self.DEVICE,
            compile=self.compile
        )

    @staticmethod
    def preprocess_audio(audio: torch.Tensor) -> torch.Tensor:
        # CLAP accepts (1xN_samples)
        if audio.shape[0] > 2:
            audio = torch.transpose(audio, 0, 1)
        # the CLAP model only accepts single channel audio
        if audio.shape[0] != 1:
            audio = torch.mean(audio, 0, keepdim=True)
        return default_collate([audio])

    def preprocess_text(self, text: str) -> str:
        return self.processor.preprocess_text(text)

    @torch.inference_mode()
    def extract_audio_features(self, preprocessed_audio: torch.Tensor) -> np.ndarray:
        preprocessed_audio = preprocessed_audio.reshape(
            preprocessed_audio.shape[0], preprocessed_audio.shape[2]
        )
        audio_embeddings = self.model.get_audio_features(audio=preprocessed_audio)
        audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
        return audio_embeddings.cpu().numpy()

    @torch.inference_mode()
    def extract_text_features(self, text: list[str]) -> np.ndarray:
        preprocessed_text = self.preprocess_text(text)
        text_embeddings = self.model.get_text_features(**preprocessed_text)
        text_embeddings = text_embeddings / torch.norm(
            text_embeddings, dim=-1, keepdim=True
        )
        return text_embeddings.cpu().numpy()

    def warmup(self):
        logger.info("Warming up model")
        random_audio = torch.rand((1, 192_000))
        preprocessed_audio = self.preprocess_audio(random_audio)
        audio_embedding = self.extract_audio_features(preprocessed_audio)
        text_embedding = self.extract_text_features(["some random text"])
        assert audio_embedding.shape[1] == text_embedding.shape[1]
        return

    def transform_faiss_distances_hook(self, dist: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dist_tensor = torch.from_numpy(dist)
            dist_tensor = dist_tensor * self.logit_scale.exp()
            return dist_tensor.detach().numpy()
