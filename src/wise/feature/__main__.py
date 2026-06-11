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

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

## import torch before onnxruntime so both use same cuDNN, see
## https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch
import torch

from wise.feature.feature_extractor_factory import FeatureExtractorFactory


parser = argparse.ArgumentParser(
    description="Test feature extractor and export model to ONNX"
)
parser.add_argument(
    "--feature-extractor",
    type=str,
    required=True,
    help="feature id for WISE.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to run the model on (e.g., 'cpu' or 'cuda').",
)
parser.add_argument(
    "onnx_path",
    type=str,
    default=None,
    help="Path to save the exported ONNX models.",
)
parser.add_argument(
    "--export",
    action="store_true",
    help="Export the model to ONNX format.",
)
parser.add_argument(
    "--verify",
    action="store_true",
    help="Verify the model export by running inference on a dummy input.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Batch size for the dummy input (default: 2).",
)
args = parser.parse_args()
if hasattr(torch.backends, 'mha'):
    torch.backends.mha.set_fastpath_enabled(False)

# Example usage
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)8s %(threadName)15s %(name)s "
        "[%(filename)s:%(lineno)d %(funcName)s] - %(message)s"
    ),
)
logger = logging.getLogger(__name__)

default_config = {
    "default": {
        "device": args.device,
        "compile": False,
    },
    "transformers/owlv2/google/owlv2-large-patch14-ensemble": {
        "objectness_threshold": 1e-3,
    },
}
feature_extractor = FeatureExtractorFactory(args.feature_extractor, default_config)

image_tensor = torch.randn(args.batch_size, 3, 1008, 1008)  # Example image tensor
audio_tensor = torch.randn(args.batch_size, 192_000)  # Example audio tensor
text_query = ["This is a test sentence."] * args.batch_size

preprocessed_audio = None
preprocessed_image = None
preprocessed_text = None

with torch.inference_mode():
    # Example audio and text inputs
    preprocessed_text = feature_extractor.preprocess_text(text_query)

    text_features = feature_extractor.extract_text_features(text_query)
    logger.info(f"Text Features Shape: {text_features.shape}")

    if feature_extractor.preprocess_audio is not None:
        preprocessed_audio = torch.cat(
            [feature_extractor.preprocess_audio(x.unsqueeze(0)) for x in audio_tensor],
            dim=0,
        )
        print(preprocessed_audio.shape)
        audio_features = feature_extractor.extract_audio_features(preprocessed_audio)
        logger.info(f"Audio Features Shape: {audio_features.shape}")

    if feature_extractor.preprocess_image is not None:
        preprocessed_image = feature_extractor.preprocess_image(image_tensor)
        image_features = feature_extractor.extract_image_features(preprocessed_image)
        logger.info(
            f"Image Features Shape: {len(image_features)} x {image_features[0].vectors.shape}"
        )

if args.onnx_path is None:
    logger.info("No ONNX path provided, skipping export / verify")
    exit(0)

output = Path(args.onnx_path)

if output.exists() and not output.is_dir():
    raise ValueError(f"Output path {output} must be a directory.")

output /= args.feature_extractor.replace("/", "--")
if args.export:
    logger.info(f"Exporting models to {output}")
    model = feature_extractor.model

    visual_inputs = (preprocessed_image,) if preprocessed_image is not None else tuple()
    audio_inputs = (preprocessed_audio.squeeze(1),) if preprocessed_audio is not None else tuple()
    text_inputs = (preprocessed_text['input_ids'],)
    text_inputs += (preprocessed_text["attention_mask"],) if "attention_mask" in preprocessed_text else tuple()

    with torch.inference_mode():
        model.export_to_onnx(
            output,
            visual_inputs=visual_inputs,
            text_inputs=text_inputs,
            audio_inputs=audio_inputs
        )

if not args.verify:
    logger.info("Skipping ONNX model verification as --verify is not set.")
    exit(0)

# if not output.exists():
#     raise ValueError(
#         f"Output path {output} does not exist. Please export the model first."
#     )

providers = ['CPUExecutionProvider']
if args.device.startswith('cuda:'):
    provders = ['CUDAExecutionProvider']

def _get_normalizer():
    if feature_extractor.__class__.__name__ == 'TransformersOWLv2FeatureExtractor':
        def normalize(x):
            return x
    else:
        def normalize(x):
            return x / np.linalg.norm(x, axis=-1, keepdims=True)
    return normalize

def _run(session: ort.InferenceSession, inputs):
    ort_inputs = {name: value.detach().numpy() for name, value in inputs.items()}
    ort_outputs = session.run(None, ort_inputs)
    return ort_outputs

def _verify_outputs(ort_outputs, outputs):
    normalizer = _get_normalizer()
    for ort_output, (k, model_output) in zip(ort_outputs, outputs.items()):
        ort_output = normalizer(ort_output)

        print("Features shape:", ort_output.shape, model_output.shape)
        diff = np.abs(model_output - ort_output)

        dot_product = model_output @ ort_output.T
        logger.info(f"({k}) diff - mean: {diff.mean()}, min: {diff.min()}, max: {diff.max()}")
        logger.info(f"({k}) dot product: {dot_product}")

if preprocessed_image is not None:
    logger.info(f"Running inference on ONNX models - image")
    session_image = ort.InferenceSession(
        f'{output}--image/model.onnx',
        providers=providers,
    )
    if feature_extractor.__class__.__name__ == 'TransformersOWLv2FeatureExtractor':
         # Save original image sizes in a list before they get resized
        orig_sizes = [(image.shape[2], image.shape[1]) for image in image_tensor] # list of (width, height) tuples
        # Preprocess image (including resizing)
        images = feature_extractor.processor(images=image_tensor, return_tensors="pt")[
            "pixel_values"
        ]  # shape: (B, C, 960, 960)

    ort_outputs = _run(session_image, {
        "images": images,
    })

    if feature_extractor.__class__.__name__ == 'TransformersOWLv2FeatureExtractor':
        from wise.feature.transformers_owlv2 import sort_by_objectness
        scores_tensor = torch.from_numpy(ort_outputs[0])
        embeddings_tensor = torch.from_numpy(ort_outputs[1])
        boxes_tensor = torch.from_numpy(ort_outputs[2])
        scores, embeddings, boxes = sort_by_objectness(scores_tensor, embeddings_tensor, boxes_tensor)
        scores = scores.cpu().numpy()
        embeddings = embeddings.cpu().numpy()
        boxes = boxes.cpu().numpy()
        features = feature_extractor.construct_features(
            embeddings,
            scores,
            boxes,
            orig_sizes,
        )
        a = features[0]
        b = image_features[0]
        print(a.vectors, b.vectors)
        assert np.allclose(a.vectors, b.vectors, rtol=1e-5, atol=1e-3), f"Feature vectors do not match"
        assert len(a.metadata) == len(b.metadata), f"Metadata length mismatch: {len(a.metadata)} != {len(b.metadata)}"


        for c, d in zip(a.metadata, b.metadata):
            x_arr = np.array([
                c.objectness_score,
                c.bbox.x,
                c.bbox.y,
                c.bbox.w,
                c.bbox.h,
            ])
            y_arr = np.array([
                d.objectness_score,
                d.bbox.x,
                d.bbox.y,
                d.bbox.w,
                d.bbox.h,
            ])
            assert np.allclose(x_arr, y_arr, rtol=1e-5, atol=1e-3), f"Metadata mismatch: {c} != {d}"

    else:
        _verify_outputs(ort_outputs, {
            "embeddings": np.concatenate([x.vectors for x in image_features]),
        })

if preprocessed_audio is not None:
    logger.info(f"Running inference on ONNX models - audio")
    session_audio = ort.InferenceSession(
        f'{output}--audio/model.onnx',
        providers=providers,
    )
    ort_outputs = _run(session_audio, {
        "audio": preprocessed_audio.squeeze(1),  # Ensure audio is 2D
    })
    _verify_outputs(ort_outputs, {
        "embeddings": audio_features,
    })

logger.info(f"Running inference on ONNX models - text")
session_text = ort.InferenceSession(
    f'{output}--text/model.onnx',
    providers=providers,
)
ort_outputs = _run(session_text, preprocessed_text)
_verify_outputs(ort_outputs, {
    "embeddings": text_features,
})
