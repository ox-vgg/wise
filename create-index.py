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

#from src.dataloader import AVDataset
#from src.search_index import SearchIndex
from src.wise_project import WiseProject
from src.index.search_index_factory import SearchIndexFactory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create-index',
                                     description='Create a nearest neighbour search index for features extracted from images and videos.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')
    parser.add_argument('--media-type',
                        required=False,
                        action='append',
                        choices=['audio', 'video', 'metadata', 'image'],
                        help='create search index only for these media types; applies to all by default ')

    parser.add_argument('--index-type',
                        required=False,
                        default='IndexFlatIP',
                        choices=['IndexFlatIP', 'IndexIVFFlat'],
                        type=str,
                        help='the type of faiss index for feature vectors, metadata defaults to full-text-search index')

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

    media_type_list = list(project_assets.keys())
    if args.media_type is not None:
        media_type_list = list(args.media_type)
    for media_type in media_type_list:
        for asset_id in project_assets[media_type]:
            asset = project_assets[media_type][asset_id]
            search_index = SearchIndexFactory(media_type, asset_id, asset)
            search_index.create_index(args.index_type, args.overwrite)
