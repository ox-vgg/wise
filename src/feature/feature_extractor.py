from typing import List
from PIL import Image
import torch
import numpy as np

class FeatureExtractor:
    def __init__(self):
        raise NotImplementedError
 
    def preprocess_image(self, images: List[Image.Image]) -> torch.Tensor:
        """ Preprocess media to prepare it for feature extraction

        Parameters
        ----------
        images : a list of PIL.Image where each element represents an image

        Returns
        -------
        torch.Tensor
            a torch tensor representing pre-processed images
        """
        raise NotImplementedError

    def extract_image_features(self, images: torch.Tensor) -> np.ndarray:
        """ Extracts features from pre-processed images

        Parameters
        ----------
        images : a torch.Tensor containing pre-processed images

        Returns
        -------
        np.ndarray
            a numpy ndarray containing extracted feature vectors
        """
        raise NotImplementedError

    def extract_text_features(self, text_query: List[str]) -> np.ndarray:
        """ Extracts features from text

        Parameters
        ----------
        text_query : a list of strings

        Returns
        -------
        np.ndarray
            a numpy ndarray containing extracted feature vectors
        """
        raise NotImplementedError
