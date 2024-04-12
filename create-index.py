import argparse
import os
import json
import configparser
import math
import sys

import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np
import webdataset as wds
import torch
import faiss

from src.dataloader import AVDataset
from src.feature.feature_extractor_factory import FeatureExtractorFactory
from src.feature.store.webdataset_store import WebdatasetStore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create-index',
                                     description='Create a nearest neighbour search index for features extracted from images and videos.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')
    parser.add_argument('--index-type',
                        required=False,
                        default='IndexFlatIP',
                        choices=['IndexFlatIP', 'IndexIVFFlat'],
                        type=str,
                        help='the type of faiss index')

    parser.add_argument('--force',
                        required=False,
                        action='store_true',
                        help='overwrite existing index file')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    ## 1. Load project data structures
    store_dir = os.path.join(args.project_dir, 'store')
    if not os.path.exists(args.project_dir):
        print('Initialise a WISE project first using extract-features.py')
        sys.exit(1)

    # 1.1 Define the receipe (i.e. features, media_type)
    feature_extractor_id_list = {
        'audio': 'microsoft/clap/2023/four-datasets',
        'video': 'mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k'
    }
    feature_store_dir_list = {}
    feature_store_list     = {}

    for media_type in feature_extractor_id_list:
        feature_extractor_id = feature_extractor_id_list[media_type]

        ## 1.2 store the path for each feature extractor
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
                raise ValueError(f'Missing folder {feature_extractor_store_dir}')

        ## 2.3 Initialise feature store in read only mode
        feature_store_list[media_type] = WebdatasetStore(media_type,
                                                         feature_store_dir_list[media_type]['features'])
        feature_store_list[media_type].enable_read(shard_shuffle=False)

    for media_type in feature_extractor_id_list:
        print(f'Creating index for {media_type} with features extracted by {feature_extractor_id_list[media_type]}')
        index_dir = feature_store_dir_list[media_type]['index']
        index_fn = os.path.join(index_dir, args.index_type + '.faiss')
        if os.path.exists(index_fn) and not args.force:
            print(f'  Skipping {media_type} : index exists at f{index_fn}')
            continue

        feature_count = feature_store_list[media_type].feature_count
        feature_dim = feature_store_list[media_type].feature_dim

        index = faiss.IndexFlatIP(feature_dim)
        if args.index_type == 'IndexIVFFlat':
            quantizer = index
            cell_count = 10 * round(math.sqrt(feature_count))
            train_count = min(feature_count, 100 * cell_count)
            index = faiss.IndexIVFFlat(quantizer, feature_dim, cell_count)

            print(f'  Loading a random sample of {train_count} features from {feature_count} features ...')
            shuffled_features = WebdatasetStore(media_type,
                                                feature_store_dir_list[media_type]['features'])
            shuffled_features.enable_read(shard_shuffle=True)

            train_features = np.ndarray((train_count, feature_dim), dtype=np.float32)
            feature_index = 0
            for feature_id, feature_vector in shuffled_features:
                train_features[feature_index,:] = feature_vector
                feature_index += 1

            assert not index.is_trained
            print(f'Training {args.index_type} faiss index with {train_count} features ...')
            index.train(train_features)
            assert index.is_trained

        with tqdm(total=feature_count) as pbar:
            for feature_id, feature_vector in feature_store_list[media_type]:
                index.add(feature_vector)
                pbar.update(1)

        faiss.write_index(index, index_fn)
        print(f'  Saved faiss index to {index_fn}')
