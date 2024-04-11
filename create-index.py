import argparse
import glob
import os
import json
import configparser
import io
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
    video_feature_extractor_id = 'mlfoundations/open_clip/ViT-L-14/openai'

    feature_store_dir = store_dir
    for feature_id_tok in video_feature_extractor_id.split('/'):
        feature_store_dir = os.path.join(feature_store_dir, feature_id_tok)
    feature_metadata_dir = os.path.join(feature_store_dir, 'metadata')
    feature_index_dir = os.path.join(feature_store_dir, 'index')
    feature_store_dir = os.path.join(feature_store_dir, 'features')
    if not os.path.exists(feature_store_dir):
        print('features extracted from images and videos are missing')
        sys.exit(1)
    index_fn = os.path.join(feature_index_dir, args.index_type + '.faiss')
    if os.path.exists(index_fn):
        print(f'Index already exists at f{index_fn}')
        sys.exit(1)

    ## 2. Load features
    wds_tar_pattern = os.path.join(feature_store_dir, 'video-*.tar')
    wds_tar_list = []
    for tar_file in glob.iglob(pathname=wds_tar_pattern, recursive=False):
        wds_tar_list.append(tar_file)

    ## 3. Compute feature dimension and number of features
    video_features = wds.WebDataset(wds_tar_list, shardshuffle=False)
    feature_count = 0
    feature_dim = -1
    for payload in video_features:
        feature_count += 1
    for payload in video_features:
        feature = np.load(io.BytesIO(payload['features.pyd']), allow_pickle=True)
        feature_dim = feature.shape[1]
        break

    ## 4. Train faiss index
    ## source: https://gitlab.com/vgg/wise/wise/-/blob/main/wise.py
    index = faiss.IndexFlatIP(feature_dim)
    if args.index_type == 'IndexIVFFlat':
        quantizer = index
        cell_count = 10 * round(math.sqrt(feature_count))
        train_count = min(feature_count, 100 * cell_count)
        index = faiss.IndexIVFFlat(quantizer, feature_dim, cell_count)

        print(f'Loading a random sample of {train_count} features from {feature_count} features ...')
        shuffled_features = wds.WebDataset(wds_tar_list, shardshuffle=True).shuffle(10000)
        train_features = np.ndarray((train_count, feature_dim), dtype=np.float32)
        feature_index = 0
        for shuffled_payload in shuffled_features:
            train_features[feature_index,:] = np.load(io.BytesIO(payload['features.pyd']), allow_pickle=True)
            feature_index += 1

        assert not index.is_trained
        print(f'Training {args.index_type} faiss index with {train_count} features ...')
        index.train(train_features)
        assert index.is_trained
    
    with tqdm(total=feature_count) as pbar:
        for payload in video_features:
            feature_id = payload['__key__']
            feature = np.load(io.BytesIO(payload['features.pyd']), allow_pickle=True)
            index.add(feature)
            pbar.update(1)

    faiss.write_index(index, index_fn)
    print(f'Saved faiss index to {index_fn}')
