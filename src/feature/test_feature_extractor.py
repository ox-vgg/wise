import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from .feature_extractor_factory import FeatureExtractorFactory

## Typically, imports from external libraries come before local
## imports.  But insightface needs some care to import, already done
## in ..feature.insightface, so we are importing it at the end to
## avoid duplicating that mess.
import insightface.data  # isort: skip


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def test_image_feature(self):
        featureExtractor = FeatureExtractorFactory('mlfoundations/open_clip/ViT-L-14/openai')
        input_image_size = featureExtractor.get_input_image_size()
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


class TestInsigthFaceFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self._extractor = FeatureExtractorFactory("insightface/_/buffalo_l/_")

    def _preprocess_and_extract_features(self, images):
        return self._extractor.extract_image_features(
            self._extractor.preprocess_image(images)
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

    def test_with_one_element_list(self):
        images = [self._get_t1_rgb_pil()]
        self._test_with_t1_images(images, 1)

    def test_with_one_image_tensor(self):
        images = torch.unsqueeze(self._get_t1_rgb_tensor(), dim=0)
        self._test_with_t1_images(images, 1)

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


if __name__ == '__main__':
    unittest.main()
