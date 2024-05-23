import unittest
import torch
import tempfile
from PIL import Image
import numpy as np

from .feature_extractor_factory import FeatureExtractorFactory

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

        self.assertEqual(preprocessed_data.shape[0], extracted_features.shape[0])
        self.assertEqual(extracted_features.shape[1], 768)

    def test_audio_feature(self):
        featureExtractor = FeatureExtractorFactory('microsoft/clap/2023/Not-Applicable')
        audio_time_series = [ torch.rand((1,408700)) ] # 2 sec. random audio
        preprocessed_audio = featureExtractor.preprocess_audio(audio_time_series)
        audio_embeddings = featureExtractor.extract_audio_features(preprocessed_audio)
        self.assertEqual(audio_embeddings.shape[1], 1024)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
