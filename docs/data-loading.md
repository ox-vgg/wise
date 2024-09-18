# Data loading in WISE

WISE implements the a small wrapper around the [`StreamingMediaDecoder`](https://pytorch.org/audio/2.2.0/generated/torio.io.StreamingMediaDecoder.html#torio.io.StreamingMediaDecoder) (from `torchaudio`) and [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) (from `torch.utils.data`)

## Interface

`get_media_metadata`: Function that takes in a path to the media file and returns the metadata associated with the default video and audio streams, in the format expected by WISE

```python
# Media file could be image, video, audio
metadata = get_media_metadata("/path/to/media/file.ext")
```

`MediaDataset`: A sub class of PyTorch IterableDataset. Takes in a `input_files: list[Path] | dict [str, Path]`, output stream options (that define the sampling rate, width, height, format etc), and the corresponding transformation to apply to each media stream chunk. Each chunk is returned along with index / key from `input_files`. Can optionally return thumbnails if needed.

`AudioDataset`: A convenience interface that accepts a list of audio files and sets up the arguments for `MediaDataset` (no frame level data)

`ImageDataset`: A convenience interface that accepts a list of image files and sets up the arguments for `MediaDataset`. (not for video, audio)

`VideoDataset`: A convenience interface that accepts a list of video files and sets up the arguments for `MediaDataset`, such that it returns the frame data (no audio)

`AVDataset`: A convenience interface that accepts a list of audio files and sets up the arguments for `MediaDataset`, such that it returns the tensors for both audio and video.

```python

# input_files = [ List of media file paths ]

# Set the output fps of the video stream
video_frame_rate = 2

# number of frames to aggregate as a chunk
video_frames_per_chunk = 8

# Segment length (in seconds)
segment_length = video_frames_per_chunk / video_frame_rate

# Sampling rate of the output audio stream
audio_sampling_rate = 48_000  # (48 kHz)

# calculate the audio_frames_per_chunk to align with video
# TODO: If this is not an integer, may cause drift?
audio_frames_per_chunk = int(
    round(audio_sampling_rate * segment_length)
)  # audio frames spanning the same segment length as video

stream = AVDataset(
    input_files,
    video_frames_per_chunk=video_frames_per_chunk,
    video_frame_rate=video_frame_rate,
    video_preprocessing_function=None,
    audio_samples_per_chunk=audio_frames_per_chunk,
    audio_sample_rate=audio_sampling_rate,
    audio_preprocessing_function=None,
    offset=None,
    thumbnails=True,
)
# optionally, you can add a dataloader if you want to read the files
# in parallel
# Caveats:
#
# 1. Set the multiprocessing method to 'spawn' instead of 'fork' before starting the workers
# import torch
# torch.multiprocessing.set_start_method('spawn')
#
# 2. The preprocessing functions should be pickleable
# No nested functions, python lambdas (I think)
#
# Keep batch_size None, as we havent implemented collation
# TODO: Implement collation
# loader = torch_data.DataLoader(stream, batch_size=None, num_workers=num_workers)
for id, chunks in stream:
    video, audio, thumbnails = chunks.get(MediaChunkType.VIDEO), chunks.get(MediaChunkType.AUDIO), chunks.get(MediaChunkType.THUMBNAILS)
    print(f"{id}:")
    print(f"\tvideo: {(video.tensor.shape, video.pts) if video else None}")
    print(f"\taudio: {(audio.tensor.shape, audio.pts) if audio else None}")
    print(f"\tthumbnails: {(f'list length: {len(thumbnails.tensor)}', thumbnails.pts) if thumbnails else None}")    
```

# Outputs

`MediaChunk`: A wrapper around `ChunkedTensor` from `StreamingMediaDecoder`. Data structure to represent an output tensor, along with its timestamp in the media file. The tensor shape depends on the input parameters to MediaDataset. Needed primarily to work around a bug in torch, where transforms are not applied to instances of subclasses of Tensor.

## Code organisation

- [`__init__.py`](../src/dataloader/__init__.py): Defines the public methods and classes exported by the module

- [`__main__.py`](../src/dataloader/__init__.py): Module script that provides a CLI to test the features of the dataloader in isolation.

- [dataset.py](../src/dataloader/dataset.py): Wrapper around the torch IterableDataset. Provides an interface to read data from the media files and apply the required transformations. The data is returned as torch Tensors, along with the presentation timestamp. Convenience wrappers based on modality (image, video, audio, audio-visual) are also provided.

- [streamreader.py](../src/dataloader/streamreader.py): Wrapper around StreamReader (a.k.a StreamingMediaDecoder) functionality. Provides functions for reading metadata about the media file and creating an instance of StreamReader with the provided output stream options.

- [utils.py](../src/dataloader/utils.py): Utility functions used internally
