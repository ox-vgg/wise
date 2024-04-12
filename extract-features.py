import argparse
import glob
import os
import json
import configparser
import time

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

    parser.add_argument('--shard-maxcount',
                        required=False,
                        type=int,
                        default=2048,
                        help='max number of entries in each shard of webdataset tar')

    parser.add_argument('--shard-maxsize',
                        required=False,
                        type=int,
                        default=20*1024*1024, # tar overheads results in 25MB shards
                        help='max size (in bytes) of each shard of webdataset tar')

    parser.add_argument('--num-workers',
                        required=False,
                        type=int,
                        default=0,
                        help='number of workers used by data loader')

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

    start_time = time.time()

    ## 1. Create a list of input files
    media_filelist = []
    for media_dir in args.media_dir_list:
        for media_include in args.media_include_list:
            media_search_dir = os.path.join(media_dir, '**/' + media_include)
            for media_path in glob.iglob(pathname=media_search_dir, recursive=True):
                media_filelist.append(media_path)
    print(f'Extracting features from {len(media_filelist)} files')

    ## 2. Prepare for feature extraction and storage
    feature_extractor_id_list = {
        'audio': 'microsoft/clap/2023/four-datasets',
        'video': 'mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k'
    }
    feature_extractor_list = {}
    feature_store_dir_list = {}
    feature_store_list     = {}

    for media_type in feature_extractor_id_list:
        feature_extractor_id = feature_extractor_id_list[media_type]

        ## 2.1 Initialise feature extractor
        feature_extractor_list[media_type] = FeatureExtractorFactory(feature_extractor_id)
        print(f'Using {feature_extractor_id} for {media_type}')

        ## 2.2 Create folders to store features, metadata and search index
        feature_extractor_store_dir = store_dir
        for feature_extractor_id_tok in feature_extractor_id.split('/'):
            feature_extractor_store_dir = os.path.join(feature_extractor_store_dir, feature_extractor_id_tok)
        feature_store_dir_list[media_type] = {
            'root'             : feature_extractor_store_dir,
            'index'            : os.path.join(feature_extractor_store_dir, 'index'),
            'features'         : os.path.join(feature_extractor_store_dir, 'features')
        }
        for store_name in feature_store_dir_list[media_type]:
            feature_extractor_store_dir = feature_store_dir_list[media_type][store_name]
            if not os.path.exists(feature_extractor_store_dir):
                os.makedirs(feature_extractor_store_dir)

        ## 2.3 Initialise feature store to store features
        feature_store_list[media_type] = WebdatasetStore(media_type,
                                                         feature_store_dir_list[media_type]['features'])
        feature_store_list[media_type].enable_write(args.shard_maxcount,
                                                    args.shard_maxsize)

    ## 4. Initialise data loader
    audio_sampling_rate = 48_000  # (48 kHz)
    video_frame_rate = 2          # fps
    video_frames_per_chunk = 8    # frames
    segment_length = video_frames_per_chunk / video_frame_rate  # frames / fps = seconds
    audio_segment_length = 4      # seconds
    audio_frames_per_chunk = int(round(audio_sampling_rate * audio_segment_length))

    stream = AVDataset(
        media_filelist,
        video_frames_per_chunk=video_frames_per_chunk,
        video_frame_rate=video_frame_rate,
        video_preprocessing_function=feature_extractor_list['video'].preprocess_image,
        audio_samples_per_chunk=audio_frames_per_chunk,
        audio_sample_rate=audio_sampling_rate,
        audio_preprocessing_function=feature_extractor_list['audio'].preprocess_audio,
        offset=None,
    )

    ## FIXME: temporary code for managing internal metadata
    internal_metadata = {}
    feature_id = {}
    for media_type in feature_extractor_id_list:
        feature_id[media_type] = 0
        internal_metadata[media_type] = {}

    ## 5. extract video and audio features
    print(f'Initializing data loader with {args.num_workers} workers ...')
    av_data_loader = torch_data.DataLoader(stream, batch_size=None, num_workers=args.num_workers)
    #for mid, video, audio in tqdm(av_data_loader):
    for mid, video, audio in av_data_loader:
        #print(f'mid={mid}, filename={media_filelist[mid]}, {video and video.tensor.shape}, {video and video.pts}, {audio and audio.tensor.shape}, {audio and audio.pts}')

        media_segment = { 'video':video, 'audio':audio }
        
        for media_type in feature_extractor_id_list:
            if mid not in internal_metadata[media_type]:
                internal_metadata[media_type][mid] = {
                    'filename': media_filelist[mid],
                    'feature_id_list':[],
                    'pts':[]
                }
            if media_segment[media_type] is None:
                continue
            segment_tensor = media_segment[media_type].tensor
            segment_pts    = media_segment[media_type].pts
            
            if media_type == 'image' or media_type == 'video':
                segment_feature = feature_extractor_list[media_type].extract_image_features(segment_tensor)
            elif media_type == 'audio':
                if segment_tensor.shape[2] < audio_frames_per_chunk:
                    # we discard any malformed audio segments
                    continue
                segment_feature = feature_extractor_list[media_type].extract_audio_features(segment_tensor)
            print(f'[{feature_id[media_type]}] mid={mid}, filename={media_filelist[mid]}, {media_type}, {segment_tensor.shape}, {segment_pts}')
            feature_store_list[media_type].add(feature_id[media_type],
                                               segment_feature)
            internal_metadata[media_type][mid]['feature_id_list'].append( feature_id[media_type] )
            internal_metadata[media_type][mid]['pts'].append(segment_pts)
            feature_id[media_type] += 1

    ## 6. save internal metadata (TODO: replace with DB implementation)
    for media_type in feature_extractor_id_list:
        # close all feature store
        feature_store_list[media_type].close()
        print(f'Saved {feature_id[media_type]} {media_type} features to {feature_store_dir_list[media_type]["features"]}')

        internal_metadata_fn = os.path.join(feature_store_dir_list[media_type]['root'], 'internal-metadata.json')
        with open(internal_metadata_fn, 'w') as f:
            json.dump(internal_metadata[media_type], f)
            print(f'Saved internal metadata for {media_type} in {internal_metadata_fn}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Feature extraction completed in {int(elapsed_time)} sec. or {int(elapsed_time/60)} min.')
