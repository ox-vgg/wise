import argparse
from pathlib import Path
import json
import configparser
import math
import sys
import urllib.parse

import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np
import webdataset as wds
import torch
import faiss

from src.dataloader import AVDataset
from src.wise_project import WiseProject
from src.search_index import SearchIndex

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='search',
                                     description='Search images and videos using natural language.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')

    parser.add_argument('--query-video',
                        required=False,
                        type=str,
                        help='search image or video frames based on the text description of their visual content')

    parser.add_argument('--query-audio',
                        required=False,
                        action='append',
                        type=str,
                        help='search audio based on the text description of the audio content')

    parser.add_argument('--index-type',
                        required=False,
                        default='IndexFlatIP',
                        choices=['IndexFlatIP', 'IndexIVFFlat'],
                        type=str,
                        help='the type of faiss index to search')

    parser.add_argument('--topk',
                        required=False,
                        type=int,
                        default=5,
                        help='show only the topk search results')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    print(f'Searching {args.project_dir}')
    print(f'  query-video = {args.query_video}')
    print(f'  query-audio = {args.query_audio}')

    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()
    
    for media_type in project_assets:
        for feature_extractor_id in project_assets[media_type]:
            # 1. load internal metadata
            internal_metadata_fn = project_assets[media_type][feature_extractor_id]['features_root'] / 'internal-metadata.json'
            with open(internal_metadata_fn.as_posix(), 'r') as f:
                internal_metadata = json.load(f)
                reverse_internal_metadata = {}
            for mid in internal_metadata:
                for i in range(0, len(internal_metadata[mid]['feature_id_list'])):
                    feature_id = internal_metadata[mid]['feature_id_list'][i]
                    reverse_internal_metadata[feature_id] = {
                        'filename': internal_metadata[mid]['filename'],
                        'pts': internal_metadata[mid]['pts'][i]
                    }

            # 2. load search index
            index_dir = project_assets[media_type][feature_extractor_id]['index_dir']
            search_index = SearchIndex(media_type,
                                       feature_extractor_id,
                                       index_dir)
            search_index.load_index(args.index_type)

            # 3. Find nearest neighbours to the search query
            if hasattr(args, 'query_' + media_type):
                media_query_text = getattr(args, 'query_' + media_type)
            dist, ids = search_index.search(media_type, media_query_text, args.topk, query_type='text')

            print(f'\nShowing results for {media_type}')
            for i in range(0, len(ids)):
                feature_id = int(ids[i])
                distance = dist[i]
                filename = reverse_internal_metadata[feature_id]['filename']
                pts = reverse_internal_metadata[feature_id]['pts']
                print('  [%d] : pts=%s, file=%s' % (i, pts, filename))
    # TODO: Combine search results for different media types using set operations
