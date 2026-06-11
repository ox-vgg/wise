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

import os
import unittest


# isort: off
## InsightFace, which we will import below, imports albumentations
## which by default checks on PyPI if the user is running the last
## version and prints a warning if not.  This is a silly default, and
## we want to disable it.  Upstream recommends libraries to disable
## the check.  Refer to
## https://github.com/albumentations-team/albumentations/issues/2206#issuecomment-2585905414
## and https://github.com/deepinsight/insightface/pull/2849
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable version check
# isort: on

import insightface.data
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_sample_image

from wise.feature.feature_extractor import BBoxXYWH
from wise.feature.feature_extractor_factory import FeatureExtractorFactory
from wise.feature.transformers_owlv2 import (
    TransformersOWLv2FeatureExtractor,
    owlv2_bbox_to_xywh,
    sort_by_objectness,
)


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def test_image_feature(self):
        featureExtractor = FeatureExtractorFactory('mlfoundations/open_clip/ViT-L-14/openai')
        input_image_size = featureExtractor.input_image_size
        self.assertEqual(input_image_size, (224,224))

        TEST_DATA_COUNT = 8
        TEST_DATA = []
        for i in range(0, TEST_DATA_COUNT):
            TEST_DATA.append( Image.new('RGB', input_image_size) )

        self.assertEqual(len(TEST_DATA), TEST_DATA_COUNT, 'Malformed test data')
        self.assertTrue( isinstance(TEST_DATA[0], Image.Image) )
        self.assertEqual(TEST_DATA[0].size, input_image_size, 'Malformed image')

        # preprocess images
        preprocessed_data = featureExtractor.preprocess_image(TEST_DATA)

        # extract features
        extracted_features = featureExtractor.extract_image_features(preprocessed_data)

        self.assertEqual(preprocessed_data.shape[0], len(extracted_features))
        self.assertTrue(all([x.vectors.shape == (1, 768) for x in extracted_features]))
        self.assertTrue(all([x.metadata == None for x in extracted_features]))

    def test_audio_feature(self):
        featureExtractor = FeatureExtractorFactory('microsoft/clap/2023/Not-Applicable')
        audio_time_series = torch.rand((1,408700)) # 2 sec. random audio
        preprocessed_audio = featureExtractor.preprocess_audio(audio_time_series)
        audio_embeddings = featureExtractor.extract_audio_features(preprocessed_audio)
        self.assertEqual(audio_embeddings.shape[1], 1024)

    def tearDown(self):
        pass


class TestInsightFaceFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self._extractor = FeatureExtractorFactory("deepinsight/insightface/buffalo_l/_")

    def _preprocess_and_extract_features(self, images):
        return self._extractor.extract_image_features(
            self._extractor.preprocess_image(images)
        )

    def _preprocess_and_extract_region_features(self, image, region):
        return self._extractor.extract_image_region_features(
            self._extractor.preprocess_image_region(image, region), region
        )
    def _get_tom_hanks_grayscale_tensor(self) -> torch.Tensor:
        np_img = insightface.data.get_image("Tom_Hanks_54745", to_rgb=True)
        assert (np.all(np_img[:,:,0] == np_img[:,:,1])
                and np.all(np_img[:,:,0] == np_img[:,:,2]))
        np_img = np_img.transpose([2, 0, 1])  # H,W,C -> C,H,W
        return torch.tensor(np_img[0:1,:,:].copy())

    def _get_t1_rgb_tensor(self) -> torch.Tensor:
        np_img = insightface.data.get_image("t1", to_rgb=True)
        np_img = np_img.transpose([2, 0, 1])  # H,W,C -> C,H,W
        return torch.tensor(np_img.copy())

    def _get_t1_rgb_pil(self) -> Image.Image:
        np_img = insightface.data.get_image("t1", to_rgb=True)
        return Image.fromarray(np_img, mode="RGB")

    def _test_with_t1_images(self, images, n_images):
        ## The t1 image distributed with InsightFace is a photo from
        ## Friends (the TV show) showing the six characters.  The
        ## model should find 6 faces, 3 male and 3 female.
        assert n_images > 0
        features = self._preprocess_and_extract_features(images)
        ## Check vectors
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), n_images)
        self.assertIsInstance(features[0].vectors, np.ndarray)
        self.assertTupleEqual(features[0].vectors.shape, (6, 512))
        self.assertTrue(all([x.vectors.shape == (6, 512) for x in features]))
        ## Check metadata (just check presence of one of the attributes)
        self.assertTrue(all([isinstance(x.metadata, list) for x in features]))
        for f in features:
            self.assertEqual(len(f.metadata), 6)
            self.assertEqual([x.is_male for x in f.metadata].count(True), 3)

    def _test_region_on_ross(self, image):
        ## This region selects all of Ross face and part of Monica's
        ## face.  It purposely large to cover part of another face and
        ## test the selection of face with highest IoU.
        ross_xt = BBoxXYWH(353/1280, 183/886, 311/1280, 292/886)
        ross_bbox = [0.36, 0.30, 0.08, 0.17]  # ~ expected from InsightFace
        features = self._preprocess_and_extract_region_features(image, ross_xt)
        self.assertTrue(len(features.metadata), 1)
        self.assertTrue(features.metadata[0].is_male)
        np.testing.assert_allclose(
            features.metadata[0].bbox, ross_bbox, atol=0.01
        )

    def test_with_one_element_list(self):
        images = [self._get_t1_rgb_pil()]
        self._test_with_t1_images(images, 1)

    def test_with_one_image_tensor(self):
        images = torch.unsqueeze(self._get_t1_rgb_tensor(), dim=0)
        self._test_with_t1_images(images, 1)

    def test_region_with_pil(self):
        self._test_region_on_ross(self._get_t1_rgb_pil())

    def test_region_with_tensor(self):
        self._test_region_on_ross(self._get_t1_rgb_tensor())

    def test_with_n_elements_list(self):
        images = [
            self._get_t1_rgb_pil(),
            self._get_t1_rgb_pil(),
            self._get_t1_rgb_pil(),
            self._get_t1_rgb_pil(),
        ]
        self._test_with_t1_images(images, 4)

    def test_with_n_images_tensor(self):
        images = torch.stack(
            [
                self._get_t1_rgb_tensor(),
                self._get_t1_rgb_tensor(),
                self._get_t1_rgb_tensor(),
                self._get_t1_rgb_tensor(),
            ]
        )
        self._test_with_t1_images(images, 4)

    def test_grayscale_torch_image(self):
        images = torch.unsqueeze(self._get_tom_hanks_grayscale_tensor(), dim=0)
        with self.assertRaisesRegex(Exception, 'RGB in NCHW order'):
            self._preprocess_and_extract_features(images)

    def test_with_empty_list(self):
        images = []
        features = self._preprocess_and_extract_features(images)
        self.assertListEqual(features, [])

    def test_with_empty_tensor(self):
        images = torch.empty([0, 3, 768, 1024])
        features = self._preprocess_and_extract_features(images)
        self.assertListEqual(features, [])


class TestInsightFaceWithAuraFace(TestInsightFaceFeatureExtractor):
    def setUp(self):
        self._extractor = FeatureExtractorFactory("deepinsight/insightface/fal/AuraFace-v1")


class TestOWLv2BBoxConversion(unittest.TestCase):
    ## Test values picked so they can be visualised on a 20x20 grid.
    def test_original_square_image(self):
        owlv2_bbox = np.array([0.15, 0.35, 0.20, 0.10])
        xywh = owlv2_bbox_to_xywh(owlv2_bbox, 20, 20)
        np.testing.assert_allclose(xywh, np.array([0.05, 0.30, 0.20, 0.10]))

    def test_original_landscape_image(self):
        owlv2_bbox = np.array([0.15, 0.35, 0.20, 0.10])
        xywh = owlv2_bbox_to_xywh(owlv2_bbox, 20, 10)
        np.testing.assert_allclose(xywh, np.array([0.05, 0.6, 0.20, 0.20]))

    def test_original_portrait_image(self):
        owlv2_bbox = np.array([0.15, 0.35, 0.20, 0.10])
        xywh = owlv2_bbox_to_xywh(owlv2_bbox, 10, 20)
        np.testing.assert_allclose(xywh, np.array([0.1, 0.30, 0.40, 0.10]))


class TestOWLv2PatchSorting(unittest.TestCase):
    ## Test values picked so they can be visualised on a 20x20 grid.
    def test_1_batch_5_predictions_sorting(self):
        # Use integers and fake embedding with length 3 for simplicity
        scores = torch.Tensor([[0, 3, 2, 4, 1]])
        embeds = torch.Tensor(
            [[[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]]
        )
        boxes = torch.Tensor(
            [
                [
                    [5, 5, 5, 5],
                    [6, 6, 6, 6],
                    [7, 7, 7, 7],
                    [8, 8, 8, 8],
                    [9, 9, 9, 9],
                ]
            ]
        )
        scores, embeds, boxes = sort_by_objectness(scores, embeds, boxes)
        torch.testing.assert_close(scores, torch.Tensor([[4, 3, 2, 1, 0]]))
        torch.testing.assert_close(
            embeds,
            torch.Tensor(
                [[[8, 8, 8], [6, 6, 6], [7, 7, 7], [9, 9, 9], [5, 5, 5]]]
            ),
        )
        torch.testing.assert_close(
            boxes,
            torch.Tensor(
                [
                    [
                        [8, 8, 8, 8],
                        [6, 6, 6, 6],
                        [7, 7, 7, 7],
                        [9, 9, 9, 9],
                        [5, 5, 5, 5],
                    ]
                ]
            ),
        )

    def test_2_batch_5_predictions_sorting(self):
        # Use integers and fake embedding with length 2 for simplicity
        scores = torch.Tensor([[0, 3, 2, 4, 1], [5, 7, 6, 9, 8]])
        embeds = torch.Tensor(
            [
                [
                    [5, 5, 5],
                    [6, 6, 6],
                    [7, 7, 7],
                    [8, 8, 8],
                    [9, 9, 9],
                ],
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                ],
            ],
        )
        boxes = torch.Tensor(
            [
                [
                    [5, 5, 5, 5],
                    [6, 6, 6, 6],
                    [7, 7, 7, 7],
                    [8, 8, 8, 8],
                    [9, 9, 9, 9],
                ],
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                ],
            ]
        )
        scores, embeds, boxes = sort_by_objectness(scores, embeds, boxes)
        torch.testing.assert_close(
            scores, torch.Tensor([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]])
        )
        torch.testing.assert_close(
            embeds,
            torch.Tensor(
                [
                    [
                        [8, 8, 8],
                        [6, 6, 6],
                        [7, 7, 7],
                        [9, 9, 9],
                        [5, 5, 5],
                    ],
                    [
                        [3, 3, 3],
                        [4, 4, 4],
                        [1, 1, 1],
                        [2, 2, 2],
                        [0, 0, 0],
                    ],
                ],
            ),
        )
        torch.testing.assert_close(
            boxes,
            torch.Tensor(
                [
                    [
                        [8, 8, 8, 8],
                        [6, 6, 6, 6],
                        [7, 7, 7, 7],
                        [9, 9, 9, 9],
                        [5, 5, 5, 5],
                    ],
                    [
                        [3, 3, 3, 3],
                        [4, 4, 4, 4],
                        [1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
        )


class TestOWLv2FeatureExtractor(unittest.TestCase):
    def setUp(self):
        # use an objectness threshold of 0.0 to avoid filtering out any boxes in the output
        model_config = TransformersOWLv2FeatureExtractor.Config(
            objectness_threshold=0.0,
        )
        self._extractor = TransformersOWLv2FeatureExtractor(
            "transformers/owlv2/google/owlv2-base-patch16-ensemble",
            config=model_config,
        )

    def test_feature_extractor_factory_setup(self):
        # try setting up using the feature extractor factory (just to check no errors were raised)
        FeatureExtractorFactory("transformers/owlv2/google/owlv2-base-patch16-ensemble")

    def test_text_encoder(self):
        text_input = ['some random text']
        text_features = self._extractor.extract_text_features(text_input)
        self.assertTrue(text_features.shape == (1, 513))

    def _preprocess_and_extract_features(self, images):
        return self._extractor.extract_image_features(
            self._extractor.preprocess_image(images)
        )

    def _get_sample_image_tensor(self):
        np_img = load_sample_image('flower.jpg')
        np_img = np_img.transpose([2, 0, 1])  # H,W,C -> C,H,W
        return torch.tensor(np_img.copy())

    def _get_sample_image_pil(self):
        np_img = load_sample_image('flower.jpg')
        return Image.fromarray(np_img, 'RGB')

    def _test_images(self, images, n_images):
        assert n_images > 0
        features = self._preprocess_and_extract_features(images)
        ## Check vectors
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), n_images)
        self.assertTrue(all([isinstance(x.vectors, np.ndarray) for x in features]))
        self.assertTrue(all([x.vectors.shape == (3600, 513) for x in features]))
        ## Check metadata
        self.assertTrue(all([isinstance(x.metadata, list) for x in features]))
        for f in features:
            self.assertEqual(len(f.metadata), 3600)
            self.assertTrue(all([
                isinstance(m.objectness_score, float)
                and isinstance(m.bbox.x, float)
                and isinstance(m.bbox.y, float)
                and isinstance(m.bbox.w, float)
                and isinstance(m.bbox.h, float)
                for m in f.metadata
            ]))

            ## Check if top detected objects (in terms of objectness score) are the same as expected
            top_objects = [m for m in f.metadata if m.objectness_score > 0.2]
            self.assertEqual(len(top_objects), 2) # there should only be 2 objects with an objectness score above 0.2

            # object #1: check objectness score and bbox coordinates
            np.testing.assert_allclose(top_objects[0].objectness_score, 0.475717157125473, rtol=1e-5)
            np.testing.assert_allclose(np.array(top_objects[0].bbox), np.array([0.26258963346481323, 0.20182788232450463, 0.4356532692909241, 0.6405366723375522]), rtol=1e-5)

            # object #2: check objectness score and bbox coordinates
            np.testing.assert_allclose(top_objects[1].objectness_score, 0.2692972719669342, rtol=1e-5)
            np.testing.assert_allclose(np.array(top_objects[1].bbox), np.array([0.4490808695554733, 0.8730822033848641, 0.2807672321796417, 0.12563285559625204]), rtol=1e-5)

    def test_with_one_element_list(self):
        images = [self._get_sample_image_pil()]
        self._test_images(images, 1)

    def test_with_one_image_tensor(self):
        self._test_images(self._get_sample_image_tensor().unsqueeze(0), 1)

    def test_with_n_elements_list(self):
        images = [self._get_sample_image_pil() for _ in range(4)]
        self._test_images(images, 4)

    def test_with_n_images_tensor(self):
        image_batch = torch.stack([
            self._get_sample_image_tensor() for _ in range(4)
        ])
        self._test_images(image_batch, 4)


if __name__ == '__main__':
    unittest.main()
