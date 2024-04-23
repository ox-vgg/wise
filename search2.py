import argparse
from pathlib import Path
import json
import configparser
import math
import sys
import urllib.parse
import time
import itertools
import readline
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from src.dataloader import AVDataset
from src.wise_project import WiseProject
from src.search_index import SearchIndex

from src import db
from src.data_models import (
    MediaMetadata,
    SourceCollection,
    ExtraMediaMetadata,
    VectorMetadata,
    MediaType,
    SourceCollectionType,
)
from src.repository import (
    SourceCollectionRepo,
    MediaRepo,
    VectorRepo,
    MediaMetadataRepo,
)

def to_hhmmss(sec):
    hh = int(sec / (60*60))
    remaining_sec = sec - hh*60*60
    mm = int(remaining_sec / 60)
    remaining_sec = int(remaining_sec - mm*60)
    ms = int(sec - (hh*60*60 + mm*60 + remaining_sec))
    return '%02d:%02d:%02d.%03d' % (hh, mm, remaining_sec, ms)

def clamp_str(text, MAX_CHARS):
    if len(text) > MAX_CHARS:
        text_short = '...' + text[ len(text)-MAX_CHARS : len(text) ]
        return text_short
    else:
        return text

def process_query(search_index_list, query_specs, args):
        search_result = []
        topk = args.topk
        if 'topk' in query_specs:
            topk = int(query_specs['topk'])
        for query_index in range(0, len(query_specs['query'])):
            start_time = time.time()
            query_text = query_specs['query'][query_index]
            media_type = query_specs['media_type'][query_index]

            # 2. Find nearest neighbours to the search query
            dist, ids = search_index_list[media_type].search(media_type,
                                                             query_text,
                                                             topk,
                                                             query_type='text')
            match_filename_list = []
            match_pts_list = []
            with db_engine.connect() as conn:
                for rank in range(0, len(ids)):
                    vector_id = int(ids[rank])
                    vector_metadata = VectorRepo.get(conn, vector_id)
                    media_metadata = MediaRepo.get(conn, vector_metadata.media_id)
                    filename = media_metadata.path
                    pts = vector_metadata.timestamp
                    pts_str = '%.3f' % pts
                    pts_hhmmss = to_hhmmss(pts)
                    match_filename_list.append(filename)
                    match_pts_list.append(pts)
            end_time = time.time()

            search_result.append({
                'match_filename_list': match_filename_list,
                'match_pts_list': match_pts_list,
                'search_time_sec': (end_time - start_time)
            })
        return search_result

def show_result(query_specs, result, args):
    console = Console()
    total_search_time = 0
    for query_index in range(0, len(query_specs['query'])):
        query_text = query_specs['query'][query_index]
        media_type = query_specs['media_type'][query_index]
        table = Table(title='Search results for "' + query_text + '" in ' + media_type,
                      show_lines=False,
                      show_edge=False,
                      box=None,
                      safe_box=True)
        table.add_column('Rank', justify='right', no_wrap=True)
        table.add_column('Filename', justify='left', no_wrap=True)
        table.add_column('Time', justify='left', no_wrap=True)

        for rank in range(0, len(result[query_index]['match_filename_list'])):
            pts_str = '%.3f' % (result[query_index]['match_pts_list'][rank])
            filename = result[query_index]['match_filename_list'][rank]
            table.add_row(str(rank),
                          clamp_str(filename, args.max_filename_length),
                          pts_str)
        console.print(table)
        print('\n')
        total_search_time += result[query_index]['search_time_sec']
    print('(search completed in %.3f sec.)' % (total_search_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='search',
                                     description='Search images and videos using natural language.',
                                     epilog='For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/')

    parser.add_argument('--query',
                        required=False,
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

    project = WiseProject(args.project_dir, create_project=False)
    project_assets = project.discover_assets()

    DB_SCHEME = "sqlite+pysqlite://"
    PROJECT_DIR = Path(args.project_dir)
    DB_URI = f"{DB_SCHEME}/{args.project_dir}/{PROJECT_DIR.stem}.db"
    db_engine = db.init_project(DB_URI, echo=False)

    ## load search assets
    search_index_list = {}
    for media_type in project_assets:
        feature_extractor_id_list = list(project_assets[media_type].keys())
        feature_extractor_id = feature_extractor_id_list[0] # TODO: allow users to select the feature

        # load search index
        index_dir = project_assets[media_type][feature_extractor_id]['index_dir']
        search_index_list[media_type] = SearchIndex(media_type,
                                                    feature_extractor_id,
                                                    index_dir)
        if not search_index_list[media_type].load_index(args.index_type):
            continue

    if hasattr(args, 'query') and args.query is not None:
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
        query_specs = {
            'query': args.query,
            'media_type': args.media_type
        }
        search_result = process_query(search_index_list, query_specs, args)
        show_result(query_specs, search_result, args)
    else:
        PROMPT = '> '
        print('Starting WISE search console ...')
        print('Some examples queries (press Ctrl + C or Ctrl + D to exit):')
        print('  1. find videos showing the action of taking a plate, show only 10 videos')
        print('     > --query take plate --in video --topk 10')
        print('  2. find videos showing the action of taking a plate with music playing in background')
        print('     > --query take plate --in video --query music --in audio')

        cmd = ''
        while cmd != 'quit':
            try:
                cmd = input(PROMPT)
                repl_args = { 'query':[], 'in':[] }
                # a basic parser
                tok_index = 0
                N = len(cmd)
                last_token_name = ''
                parse_token_name = False
                parse_token_value = False
                while tok_index < N:
                    if cmd[tok_index] == '-' and cmd[tok_index+1] == '-':
                        tok_index = tok_index + 2
                        parse_token_name = True
                        parse_token_value = False
                    elif parse_token_name and not parse_token_value:
                        next_space = cmd.find(' ', tok_index, N)
                        if next_space == -1:
                            token_name = cmd[tok_index:N]
                            tok_index = N
                        else:
                            token_name = cmd[tok_index:next_space]
                            tok_index = next_space
                        last_token_name = token_name.strip()
                        parse_token_name = False
                        parse_token_value = True
                    elif not parse_token_name and parse_token_value:
                        double_dash = cmd.find('--', tok_index, N)
                        if double_dash == -1:
                            token_value = cmd[tok_index:N]
                            tok_index = N
                        else:
                            token_value = cmd[tok_index:double_dash]
                            tok_index = double_dash
                        if last_token_name in repl_args:
                            repl_args[last_token_name].append(token_value.strip())
                        else:
                            repl_args[last_token_name] = token_value.strip()
                        parse_token_name = False
                        parse_token_value = False
                    else:
                        tok_index += 1
                if 'in' in repl_args:
                    # we avoid using 'in' as it is a reserved word in python
                    repl_args['media_type'] = repl_args['in']
                    del repl_args['in']
                search_result = process_query(search_index_list, repl_args, args)
                show_result(repl_args, search_result, args)
            except EOFError:
                print('Bye')
                break
            except KeyboardInterrupt:
                print('Bye')
                break
