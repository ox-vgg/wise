import enum
import logging
from pathlib import Path
from .dataset import get_media_metadata, AVDataset
import numpy as np
import torch
import torch.utils.data as torch_data
import torchvision.transforms.v2 as transforms_v2
import open_clip
import typer
from tqdm import tqdm

app = typer.Typer()
app_state = {"verbose": True}
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = open_clip.list_pretrained(as_str=True) + ["internvideo"]


class _CLIPModel(str, enum.Enum):
    pass


CLIPModel = _CLIPModel("CLIPModel", {x: x for x in AVAILABLE_MODELS})

INTERNVIDEO_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
INTERNVIDEO_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def get_input_transform_for_model(clip_model: CLIPModel):
    if clip_model == "internvideo":
        # Internvideo preprocessing
        return transforms_v2.Compose(
            [
                # Convert chunk to tensor to overcome issue with
                # .numpy() method in tensor subclasses
                transforms_v2.Resize(224),
                transforms_v2.CenterCrop(224),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(
                    mean=INTERNVIDEO_MEAN.tolist(), std=INTERNVIDEO_STD.tolist()
                ),
                # C x B x H x W
                lambda x: x.permute(1, 0, 2, 3),
            ]
        )

    model_name, _ = clip_model.value.split(":", 1)
    logger.info(f"Loading CLIP (model: {model_name})...")
    model = open_clip.create_model(model_name, None)
    preprocess = open_clip.transform.image_transform_v2(
        open_clip.transform.PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return transforms_v2.Compose(
        [
            # Convert chunk to tensor to overcome issue with
            # .numpy() method in tensor subclasses
            lambda x: x.squeeze(0),
            transforms_v2.ToPILImage(),
            preprocess,
            lambda x: x.unsqueeze(0),
        ]
    )


@app.callback()
def base(verbose: bool = False):
    """
    Dataloader demo app

    Usage: python3 -m src INPUT_FILE --model CLIP_MODEL
    TODO: Fill this
    """
    app_state["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    global logger
    logger = logging.getLogger()


@app.command()
def run(
    input_file: Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Path to input file / folder of videos",
    ),
    model: CLIPModel = typer.Option(
        ..., help="Pass in a open_clip model string (or) internvideo"
    ),
    thumbnails: bool = typer.Option(True, help="Flag to control thumbnail extraction"),
):
    """
    Dummy CLI to test the dataloader

    Example:
    # Loading one video based on clip preprocessing
    python3 -m src run data/Shazam.mkv --model "ViT-L-14:openai"

    # With a directory of videos
    python3 -m src run data/ --model "ViT-L-14:openai"

    # With internvideo
    python3 -m src run data/ --model "internvideo"
    """

    logger.debug("Getting preprocessing function")
    # get preprocessing function from feature extractor
    preprocess = get_input_transform_for_model(model)

    # Dummy filtering, will be replaced in the main code
    video_extensions = ("mkv", "mp4", "webm")
    if input_file.is_file() and input_file.suffix in video_extensions:
        logger.debug(f'"{input_file}"is a file, adding to sources')
        sources = [input_file]
    elif input_file.is_dir():
        logger.debug(f'"{input_file}"is a dir, globbing for videos')
        sources = list(x for y in video_extensions for x in input_file.rglob(f"*.{y}"))
    else:
        raise ValueError(f'Unrecognized file "{input_file}"')

    # Get metadata to write into the media table
    metadata = (get_media_metadata(str(x)) for x in sources)
    input_files = {x.id: y for x, y in zip(metadata, sources)}

    # Define output stream options based on model.
    # Every 0.5 seconds, we read 8 frames chunk for internvideo, and 1 for clip

    # Construct the dataset
    audio_sampling_rate = 48_000  # (48 kHz)

    video_frame_rate = 2  # fps
    video_frames_per_chunk = 8 if model == "internvideo" else 1  # frames
    segment_length = video_frames_per_chunk / video_frame_rate  # frames / fps = seconds

    # If this is not an integer, may cause drift?
    audio_frames_per_chunk = int(
        round(audio_sampling_rate * segment_length)
    )  # audio frames spanning the same segment length as video

    stream = AVDataset(
        input_files,
        video_frames_per_chunk=video_frames_per_chunk,
        video_frame_rate=video_frame_rate,
        video_preprocessing_function=preprocess,
        audio_samples_per_chunk=audio_frames_per_chunk,
        audio_sample_rate=audio_sampling_rate,
        audio_preprocessing_function=None,
        offset=None,
        thumbnails=thumbnails,
    )

    # Construct the dataloader
    loader = torch_data.DataLoader(stream, batch_size=None, num_workers=0)
    logger.info(f"Iterating over {len(input_files)} file(s)")
    for mid, *chunks in tqdm(loader):
        logger.debug(f"{mid}, {[x and (x.tensor.shape, x.pts) for x in chunks]}")
        pass


app()
