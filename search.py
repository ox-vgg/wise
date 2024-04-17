import argparse
from pathlib import Path
import json
import configparser
import math
import sys
import urllib.parse
import time

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from src.dataloader import AVDataset
from src.wise_project import WiseProject
from src.search_index import SearchIndex

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='search',
                                     description='Search images and videos using natural language.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')

    parser.add_argument('--query',
                        required=True,
                        action='append',
                        help='search image or video frames based on the text description of their visual content')

    parser.add_argument('--in',
                        required=False,
                        action='append',
                        dest='media_type', # since "in" is a reserved keyword
                        choices=['audio', 'video'],
                        help='apply the search query term to these features; query applied to all features if --in argument is missing')

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

    ## Sanity check
    if len(args.query) > 1 and len(args.query) != len(args.media_type):
        print('Each --query argument must be followed by a --in argument. For example:')
        print('  $ python search.py --query people --in video --query shouting --in audio ...')
        sys.exit(0)

    ## if "--in" argments are missing, assume that the search query is
    ## to be applied on all possible media types
    if len(args.query) == 1 and args.media_type is None:
        setattr(args, 'media_type', ['audio', 'video'])
        only_query = args.query[0]
        setattr(args, 'query', [only_query, only_query])

    print(f'Searching {args.project_dir} for')
    for i in range(0, len(args.query)):
        print(f'  [{i}] "{args.query[i]}" in {args.media_type[i]}')
    print('\n')

    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()

    ## FIXME: develop a grammar to address all possible ways
    ## of describing the search query on images, audio and videos
    start_time = time.time()
    console = Console()
    search_result = []
    for query_index in range(0, len(args.query)):
        query_text = args.query[query_index]
        media_type = args.media_type[query_index]
        feature_extractor_id_list = list(project_assets[media_type].keys())
        feature_extractor_id = feature_extractor_id_list[0] # TODO: allow users to select the feature

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
        dist, ids = search_index.search(media_type, query_text, args.topk, query_type='text')
        results = []
        table = Table(title='Search results for "' + query_text + '" in ' + media_type,
                      show_lines=False,
                      show_edge=False,
                      box=None,
                      safe_box=True)
        table.add_column('Rank', justify='right', no_wrap=True)
        table.add_column('Filename', justify='left', no_wrap=True)
        table.add_column('Time', justify='left', no_wrap=True)
        #table.add_column('Link', justify='left', no_wrap=True)
        for rank in range(0, len(ids)):
            feature_id = int(ids[rank])
            filename = reverse_internal_metadata[feature_id]['filename']
            pts_str = '%.1f' % (reverse_internal_metadata[feature_id]['pts'])
            results.append({
                'rank':rank,
                'filename':filename,
                'pts':pts_str,
                'distance':dist[rank]
            })
            #file_link = '[link]file://' + filename + '#t=' + pts_str + 'View[/link]!'
            # FIXME: improve readability by removing the media_dir part from
            # filename that was provided to the extract_feature.py script.
            MAX_FILENAME_CHARS = 80
            filename_short = filename
            if len(filename_short) > MAX_FILENAME_CHARS:
                filename_short = '...' + filename[ len(filename)-MAX_FILENAME_CHARS : len(filename) ]
            table.add_row(str(rank),
                          filename_short,
                          pts_str)

        search_result.append(results)
        console.print(table)
        print('\n')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'\nSearch completed in {int(elapsed_time)} sec.')
