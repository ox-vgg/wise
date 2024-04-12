import argparse
import os
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
from src.feature.feature_extractor_factory import FeatureExtractorFactory
from src.feature.store.webdataset_store import WebdatasetStore

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
                        default=3,
                        help='show only the topk search results')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    ## 1. Load project data structures
    print(f'Searching {args.project_dir}')
    print(f'  query-video = {args.query_video}')
    print(f'  query-audio = {args.query_audio}')
    store_dir = os.path.join(args.project_dir, 'store')
    if not os.path.exists(args.project_dir):
        print('Initialise a WISE project first using extract-features.py')
        sys.exit(1)

    # 1.1 Define the receipe (i.e. features, media_type)
    feature_extractor_id_list = {
        'audio': 'microsoft/clap/2023/four-datasets',
        'video': 'mlfoundations/open_clip/xlm-roberta-large-ViT-H-14/frozen_laion5b_s13b_b90k'
    }
    if not args.query_video:
        del feature_extractor_id_list['video']
    if not args.query_audio:
        del feature_extractor_id_list['audio']
    if not feature_extractor_id_list:
        print('Query missing')
        sys.exit(1)

    feature_extractor_list = {}
    feature_store_dir_list = {}
    feature_store_list     = {}

    for media_type in feature_extractor_id_list:
        feature_extractor_id = feature_extractor_id_list[media_type]

        ## 1.2 Initialise feature extractor (to extract features from query-text or query-media)
        feature_extractor_list[media_type] = FeatureExtractorFactory(feature_extractor_id)
        print(f'Using {feature_extractor_id} for {media_type}')

        ## 1.3 store the path for each feature extractor
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


    ## 2. Load search index
    feature_search_index = {}
    feature_internal_metadata = {}
    for media_type in feature_extractor_id_list:
        print(f'Loading index for {media_type} with features extracted by {feature_extractor_id_list[media_type]}')
        index_dir = feature_store_dir_list[media_type]['index']
        index_fn = os.path.join(index_dir, args.index_type + '.faiss')
        if not os.path.exists(index_fn):
            printf(f'  Search index is missing')
            continue
        feature_search_index[media_type] = faiss.read_index(index_fn,
                                                            faiss.IO_FLAG_READ_ONLY)
        internal_metadata_fn = os.path.join(feature_store_dir_list[media_type]['root'], 'internal-metadata.json')
        with open(internal_metadata_fn, 'r') as f:
            internal_metadata = json.load(f)
            feature_internal_metadata[media_type] = {}
            for mid in internal_metadata:
                for i in range(0, len(internal_metadata[mid]['feature_id_list'])):
                    feature_id = internal_metadata[mid]['feature_id_list'][i]
                    feature_internal_metadata[media_type][feature_id] = {
                        'filename': internal_metadata[mid]['filename'],
                        'pts': internal_metadata[mid]['pts'][i]
                    }

    ## 3. Find nearest neighbours to the search query
    search_result = {}
    for media_type in feature_extractor_id_list:
        if hasattr(args, 'query_' + media_type):
            media_query_text = getattr(args, 'query_' + media_type)
            if media_type == 'audio':
                if len(media_query_text) == 0:
                    continue
                prompt = 'this is the sound of '
                media_query_text = [prompt + x for x in media_query_text]
            else:
                prompt = 'This is a photo of a '
                media_query_text = [ (prompt + media_query_text) ]
            print(f'Querying {media_type} with "{media_query_text}"')
            media_query_features = feature_extractor_list[media_type].extract_text_features(media_query_text)
            dist, ids  = feature_search_index[media_type].search(media_query_features, args.topk)

            search_result[media_type] = {
                'feature_id': [],
                'media_fn': [],
                'media_pts': [],
                'distance': []
            }
            for i in range(0, len(ids[0])):
                feature_id = int(ids[0][i])
                distance = dist[0][i]
                filename = feature_internal_metadata[media_type][feature_id]['filename']
                pts = feature_internal_metadata[media_type][feature_id]['pts']
                search_result[media_type]['feature_id'].append(feature_id)
                search_result[media_type]['media_fn'].append(filename)
                search_result[media_type]['media_pts'].append(pts)
                search_result[media_type]['distance'].append( ('%.4f' % distance) )
    # FIXME: for debugging
    print(json.dumps(search_result, indent=2))
    HTTP_PREFIX = 'https://meru.robots.ox.ac.uk/dset/video/CondensedMovies/'
    for media_type in feature_extractor_id_list:
        print(f'\nShowing results for {media_type}')
        for i in range(len(search_result[media_type]['media_fn'])):
            # FIXME: remove absolute path from replace()
            fn = search_result[media_type]['media_fn'][i].replace('/data/ssd/wise/data/CondensedMovies-3/', '')
            pts = search_result[media_type]['media_pts'][i]
            dist = search_result[media_type]['distance'][i]
            url = HTTP_PREFIX + urllib.parse.quote(fn) + ('#t=%.2f' % (pts))
            print('  [%d : %s] %s' % (i, dist, url))

    # TODO: combine results from different media_type
