from .mlfoundation_openclip import MlfoundationOpenClip
from .microsoft_clap import MicrosoftClap
from .transformers_owlv2 import TransformersOWLv2
from .insightface import InsightFaceFeatureExtractor

def FeatureExtractorFactory(id):
    """
    Extract features (e.g. a vector of length 256) from images, videos and audio.

    Parameters
    ----------
    id : str
        To uniquely identify a pre-trained model and its pre-training dataset.
        The id string is formatted as:

        USER_OR_ORGANIZATION / REPOSITORY_NAME / MODEL_NAME / TRAINING_DATASET

        The purpose of this id is to simplify the management of a WISE project
        containing features extracted by different models operating on different
        modalities (e.g. images, videos, audio).

        Here are some examples showing the id assigned to some of models:
        (a) The model "ViT-B-16-SigLIP-256" trained on the "webli" dataset
        that has been made available at https://github.com/mlfoundations/open_clip/
        can use the following id: "mlfoundations/open_clip/ViT-B-16-SigLIP-256/webli"

        (b) The audio-language model trained on a combination of four datasets and
        made available at https://github.com/microsoft/CLAP/ can be assigned
        the id "microsoft/clap/2023/four-datasets/". The model release year "2023"
        is being used to identify different versions of the model architecture.

        Notes:
        - use "_unknown" to indicate that the model's training dataset is not known
            (e.g. deepinsight/insightface/buffalo_l/_unknown)
        - if both REPOSITORY_NAME and NAME_NAME are same (e.g. CLAP), use model release year
            (e.g. microsoft/clap/2023/four-datasets/)
    """
    if len(id.split('/')) != 4:
        raise ValueError(f'''Feature extractor name must be formatted as
              USER_OR_ORGANIZATION / REPOSITORY_NAME / MODEL_NAME / TRAINING_DATASET
            For example, use "mlfoundations/open_clip/ViT-B-16-SigLIP-256/webli" for extracting features using ViT
            model trained on the Web Language Image (WebLI) dataset.
            ''')
    if id.startswith('mlfoundations/open_clip/'):
        return MlfoundationOpenClip(id)
    elif id.startswith('microsoft/clap/'):
        return MicrosoftClap(id)
    elif id.startswith('transformers/owlv2/'):
        return TransformersOWLv2(id)
    elif id.startswith('deepinsight/insightface/'):
        return InsightFaceFeatureExtractor(id)
    else:
        raise ValueError(f'Unknown feature extractor id {id}')
