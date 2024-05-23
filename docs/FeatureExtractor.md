# Feature Extractor

WISE depends on feature extractors to compute a vector representation of an image,
a set of video frames or audio samples. The [src/feature] folder contains
implementation of various types of feature extractors. For example, the 
[src/feature/mlfoundation_openclip.py] file implements the [open_clip](https://github.com/mlfoundations/open_clip)
feature extractor. Any new implementation of feature extractor must contain an
implementation for `preprocess_image(), extract_image_features(), ...` and
various other methods that are defined in the [feature/feature_extractor.py].

## Implement a New Feature Extractor
WISE can be extended to use a new feature extractor. In this tutorial, we
describe a new feature extractor called `RandomFeatures` which generates a
random 512 dimensional feature for any type of input. While this feature
extractor has no practical benefit, it is useful for illustrating the process
of using a new feature extractor in WISE.

First, we create a new file [src/feature/random_features.py] and create an
implementation of a feature extractor that returns a random 512 dimensional 
vector for any input.

```
# File: src/feature/random_features.py
import torch
import numpy as np
from typing import List, Union
from PIL import Image
import torchvision.transforms.functional as F
from collections.abc import Iterable

from .feature_extractor import FeatureExtractor

class RandomFeatures(FeatureExtractor):
    """
    Feature extractors that generates a random 512 dimensional feature for
    any type of input

    see FeatureExtractor.py for documentation of API
    """

    ID_PREFIX = 'vgg/random_features/'
    DESCRIPTION = 'See https://gitlab.com/vgg/wise/wise/-/blob/main/docs/FeatureExtractor.md'

    def __init__(self, id):
        if not id.startswith(self.ID_PREFIX):
            raise ValueError(f'feature id cannot start with {id} and must start with {self.ID_PREFIX}')
        id_tokens = id.split('/')

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Models and preprocessing functions are initialized here
        # For this tutorial, we do not need a model as we simply return a
        # random 512 dimensional feature
        self.output_dim = 512
        self.input_image_size = (224, 224)

    def preprocess(self, image):
        # For this tutorial, we return the input image as it is but in feature
        # extractors of practical value, one would apply some processing to the
        # input image
        return image

    def get_output_dim(self):
        return self.output_dim

    def get_input_image_size(self):
        return self.input_image_size

    def preprocess_image(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            result = torch.stack([self.preprocess(im) for im in images], dim=0).to(device=self.DEVICE)
            return result
        elif isinstance(images, torch.Tensor) and len(images.shape) == 4:
            result = torch.stack([self.preprocess(F.to_pil_image(im)) for im in images], dim=0).to(device=self.DEVICE)
            return result

        else:
            raise ValueError('all input to preprocess_image() must be an instance of torch.Tensor or PIL.Image')

    def extract_image_features(self, images: torch.Tensor) -> np.ndarray:
        return torch.rand((1, self.output_dim), dtype=np.float32).numpy()

    def extract_text_features(self, text_query: List[str]) -> np.ndarray:
        return torch.rand((1, self.output_dim), dtype=np.float32).numpy()
```

Next, we register this newly created feature extractor in WISE by updating the
[src/feature/feature_extractor_factory.py] as follows.

```
# File: src/feature/feature_extractor_factory.py
...
from .random_features import RandomFeatures

def FeatureExtractorFactory(id):
    ...
    if id.startswith('mlfoundations/open_clip/'):
        return MlfoundationOpenClip(id)
    elif id.startswith('microsoft/clap/'):
        return MicrosoftClap(id)
    elif id.startswith('vgg/random/'):
        return RandomFeatures(id)
    else:
        raise ValueError(f'Unknown feature extractor id {id}')
```

Now, you can use this newly created feature extractor in the WISE software. Here
is an example of how we can use the `RandomFeatures` feature extractor in a 
new WISE project.

```
# Assumption: The WISE software dependencies are already installed

## 1. Download sample videos
mkdir -p wise-data/ wise-projects
curl -sLO "https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/CondensedMovies-10.tar.gz"
tar -zxvf CondensedMovies-10.tar.gz -C wise-data/

python extract-features.py \
  --media-dir wise-data/CondensedMovies-10/ \
  --video-feature-id "vgg/random/2024/04" \
  --project-dir wise-projects/CondensedMovies-10/

```

This will invoke the newly created random feature extractor for computing the
feature vector for video frames extracted from input videos. The features will
be stored in `wise-projects/CondensedMovies-10/store/vgg/random/2024/04/features`.