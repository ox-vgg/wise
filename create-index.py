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
from src.search_index import SearchIndex
from src.wise_project import WiseProject

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

    parser.add_argument('--overwrite',
                        required=False,
                        action='store_true',
                        default=False,
                        help='overwrite existing index file')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()

    for media_type in project_assets:
        for feature_extractor_id in project_assets[media_type]:
            project.create_index_store(feature_extractor_id)
            feature_dir = project_assets[media_type][feature_extractor_id]['features_dir']
            index_dir = project_assets[media_type][feature_extractor_id]['index_dir']

            search_index = SearchIndex(media_type,
                                       feature_extractor_id,
                                       index_dir,
                                       feature_dir)
            print(f'Creating index for {media_type} using features extracted by [{feature_extractor_id}]')
            search_index.create_index(args.index_type, args.overwrite)
