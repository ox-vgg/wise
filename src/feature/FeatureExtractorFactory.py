from MlFoundationOpenClip import MlFoundationOpenClip

def FeatureExtractorFactory(id):
    """
    Extract features (e.g. a vector of length 256) from images, videos and audio.

    Parameters
    ----------
    id : str
         The id string is formatted as MODEL_CREATOR:MODEL_NAME:PRETRAINING_DATASET
         e.g. "mlfoundations/open_clip:ViT-B-16-SigLIP-256:webli" for extracting features using ViT
         model trained on the Web Language Image (WebLI) dataset.
         Use "__RANDOM_768__" for generating random 768 dimensional features for debugging.
    """
    if len(id.split(':')) != 3:
        raise ValueError(f'''Feature extractor name must be formatted as
              MODEL_CREATOR : MODEL_NAME : PRETRAINING_DATASET
            For example, use "mlfoundations/open_clip:ViT-B-16-SigLIP-256:webli" for extracting features using ViT
            model trained on the Web Language Image (WebLI) dataset.
            ''')
    if id.startswith('mlfoundations/open_clip:'):
        return MlFoundationOpenClip(id)
    else:
        raise ValueError(f'Unknown feature extractor id {id}')

        '''
        elif id.startswith('microsoft/CLAP:'):
            return MicrosoftClap(id)
        elif id.startswith('RandomFeatures'):':
            return RandomFeatures(id)
        '''
