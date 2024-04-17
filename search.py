import argparse
from pathlib import Path
import json
import configparser
import math
import sys
import urllib.parse
import time
import itertools

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from src.dataloader import AVDataset
from src.wise_project import WiseProject
from src.search_index import SearchIndex

def to_hhmmss(sec):
    hh = int(sec / (60*60))
    remaining_sec = sec - hh*60*60
    mm = int(remaining_sec / 60)
    remaining_sec = int(remaining_sec - mm*60)
    return '%02d:%02d:%02d' % (hh, mm, remaining_sec)

def clamp_str(text, MAX_CHARS):
    if len(text) > MAX_CHARS:
        text_short = '...' + text[ len(text)-MAX_CHARS : len(text) ]
        return text_short
    else:
        return text

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

    parser.add_argument('--merge-tolerance',
                        required=False,
                        type=int,
                        default=10,
                        help='tolerance (in seconds) for merging search results')

    parser.add_argument('--max-filename-length',
                        required=False,
                        type=int,
                        default=50,
                        help='only show this many characters from the end in a filename')

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
        table = Table(title='Search results for "' + query_text + '" in ' + media_type,
                      show_lines=False,
                      show_edge=False,
                      box=None,
                      safe_box=True)
        table.add_column('Rank', justify='right', no_wrap=True)
        table.add_column('Filename', justify='left', no_wrap=True)
        table.add_column('Time', justify='left', no_wrap=True)
        #table.add_column('Link', justify='left', no_wrap=True)
        match_filename_list = []
        match_pts_list = []
        for rank in range(0, len(ids)):
            feature_id = int(ids[rank])
            filename = reverse_internal_metadata[feature_id]['filename']
            pts = reverse_internal_metadata[feature_id]['pts']
            pts_hhmmss = to_hhmmss(reverse_internal_metadata[feature_id]['pts'])
            match_filename_list.append(filename)
            match_pts_list.append(pts)

            # FIXME: improve readability by showing filenames relative to
            # the --media-dir argument provided to extract_feature.py script.
            table.add_row(str(rank),
                          clamp_str(filename, args.max_filename_length),
                          pts_hhmmss)

        search_result.append({
            'match_filename_list': match_filename_list,
            'match_pts_list': match_pts_list
        })
        console.print(table)
        print('\n')

    if len(search_result) != 1:
        # first merge based on filenames
        filename_merge = set(search_result[0]['match_filename_list'])
        for result_index in range(1, len(search_result)):
            result_index_set = set(search_result[result_index]['match_filename_list'])
            filename_merge = filename_merge.intersection(result_index_set)

        # now merge based on timestamp of common filenames
        merged_queries = []
        for query_index in range(0, len(args.query)):
            query_text = args.query[query_index]
            media_type = args.media_type[query_index]
            merged_queries.append('"' + query_text + '" in ' + media_type)
        table = Table(title='Search results for ' + ' and '.join(merged_queries),
                      show_lines=False,
                      show_edge=False,
                      box=None,
                      safe_box=True)
        table.add_column('Filename', justify='left', no_wrap=True)
        table.add_column('Time Range', justify='left', no_wrap=True)

        row_count = 0
        for filename in filename_merge:
            pts_merge = []
            for result_index in range(0, len(search_result)):
                result_pts = []
                for filename_index in range(0, len(search_result[result_index]['match_filename_list'])):
                    if filename == search_result[result_index]['match_filename_list'][filename_index]:
                        pts = search_result[result_index]['match_pts_list'][filename_index]
                        result_pts.append(pts)
                pts_merge.append(result_pts)

            # FIXME: Is there a more efficient way to do this?
            for i in range(0, len(pts_merge)):
                for j in range(i+1, len(pts_merge)):
                    for pts_pairs in itertools.product(pts_merge[i], pts_merge[j]):
                        del_pts = abs(pts_pairs[0] - pts_pairs[1])
                        if del_pts <= args.merge_tolerance:
                            if pts_pairs[0] > pts_pairs[1]:
                                time_range = '%s - %s' % (to_hhmmss(pts_pairs[1]), to_hhmmss(pts_pairs[0]))
                            else:
                                time_range = '%s - %s' % (to_hhmmss(pts_pairs[0]), to_hhmmss(pts_pairs[1]))
                            table.add_row(clamp_str(filename, args.max_filename_length),
                                          time_range)
                            row_count += 1
        if row_count:
            console.print(table)
            print('\n')
        else:
            print('No search results for ' + ' and '.join(merged_queries))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'\nSearch completed in {int(elapsed_time)} sec.')
