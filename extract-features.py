import argparse
import glob
import os
import json
import configparser

import torch.utils.data as torch_data
from tqdm import tqdm

from src.dataloader import AVDataset
from src.feature.feature_extractor_factory import FeatureExtractorFactory
from src.feature.store.webdataset_store import WebdatasetStore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract-features',
                                     description='Initialise a WISE project by extractng features from images, audio and videos.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')
    parser.add_argument('--media-dir',
                        required=True,
                        action='append',
                        dest='media_dir_list',
                        type=str,
                        help='source images and video from this folder')

    parser.add_argument('--media-include',
                        required=False,
                        action='append',
                        dest='media_include_list',
                        default=['*.mp4'],
                        type=str,
                        help='regular expression to include certain media files')

    parser.add_argument('--shard_maxcount',
                        required=False,
                        type=int,
                        default=2048,
                        help='max number of entries in each shard of webdataset tar')

    parser.add_argument('--shard_maxsize',
                        required=False,
                        type=int,
                        default=20*1024*1024, # tar overheads results in 25MB shards
                        help='max size (in bytes) of each shard of webdataset tar')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()
    store_dir = os.path.join(args.project_dir, 'store')
    if not os.path.exists(args.project_dir):
        os.makedirs(args.project_dir)
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    
    ## 1. Create a list of input files
    media_filelist = []
    for media_dir in args.media_dir_list:
        for media_include in args.media_include_list:
            media_search_dir = os.path.join(media_dir, '**/' + media_include)
            for media_path in glob.iglob(pathname=media_search_dir, recursive=True):
                media_filelist.append(media_path)
    print(f'Extracting features from {len(media_filelist)} files')

    ## 2. Initialise feature extractor
    video_feature_extractor_id = 'mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k'
    video_feature_extractor = FeatureExtractorFactory(video_feature_extractor_id)
    print(f'Using {video_feature_extractor_id} for extracting features from video frames')
    
    ## 3. Initialise feature store
    feature_store_dir = store_dir
    for feature_id_tok in video_feature_extractor_id.split('/'):
        feature_store_dir = os.path.join(feature_store_dir, feature_id_tok)
    feature_metadata_dir = os.path.join(feature_store_dir, 'metadata')
    feature_index_dir = os.path.join(feature_store_dir, 'index')
    feature_store_dir = os.path.join(feature_store_dir, 'features')
    if not os.path.exists(feature_store_dir):
        os.makedirs(feature_store_dir)
    if not os.path.exists(feature_metadata_dir):
        os.mkdir(feature_metadata_dir)
    if not os.path.exists(feature_index_dir):
        os.mkdir(feature_index_dir)

    video_feature_store = WebdatasetStore('video',
                                          feature_store_dir,
                                          args.shard_maxcount,
                                          args.shard_maxsize)

    ## 4. Initialise data loader
    audio_sampling_rate = 48_000  # (48 kHz)
    video_frame_rate = 2          # fps
    video_frames_per_chunk = 1    # frames
    segment_length = video_frames_per_chunk / video_frame_rate  # frames / fps = seconds
    audio_frames_per_chunk = int(round(audio_sampling_rate * segment_length))

    stream = AVDataset(
        media_filelist,
        video_frames_per_chunk=video_frames_per_chunk,
        video_frame_rate=video_frame_rate,
        video_preprocessing_function=video_feature_extractor.preprocess_image,
        audio_samples_per_chunk=audio_frames_per_chunk,
        audio_sample_rate=audio_sampling_rate,
        audio_preprocessing_function=None,
        offset=None,
    )

    ## 5. extract video and audio features
    av_data_loader = torch_data.DataLoader(stream, batch_size=None, num_workers=0)
    internal_metadata = {}
    feature_id = 0
    for mid, video, audio in tqdm(av_data_loader):
        if video is None or audio is None:
            continue
        #print(f"{mid}, {video.tensor.shape}, {video.pts}, {audio.tensor.shape}, {audio.pts}")
        if mid not in internal_metadata:
            internal_metadata[mid] = {
                'filename': media_filelist[mid],
                'video': { 'feature_id_list':[], 'pts':[] },
                'audio': { 'feature_id_list':[], 'pts':[] },
            }
        video_feature = video_feature_extractor.extract_image_features(video.tensor)
        video_feature_store.add(feature_id, video_feature)
        internal_metadata[mid]['video']['feature_id_list'].append(feature_id)
        internal_metadata[mid]['video']['pts'].append(video.pts)
        feature_id += 1
    video_feature_store.close()
    print(f'Saved {feature_id-1} features to {feature_store_dir}')

    ## 6. save internal metadata (TODO: replace with DB implementation)
    internal_metadata_fn = os.path.join(feature_metadata_dir, 'video-internal-metadata.json')
    with open(internal_metadata_fn, 'w') as f:
        json.dump(internal_metadata, f)
        print(f'Saved internal metadata in {internal_metadata_fn}')
