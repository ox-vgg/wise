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

def parse_user_input(cmd):
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
    return repl_args

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
                pts_str = '%.1f' % pts
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
        if len(query_specs['query']) > 1:
            print('\n')
        total_search_time += result[query_index]['search_time_sec']
    print('(search completed in %.3f sec.)' % (total_search_time))

def export_result_as_csv(csv_filename, query_specs, search_result, args):
    with open(csv_filename, 'w') as f:
        f.write('query_text,media_type,rank,filename,timestamp\n')
        for query_index in range(0, len(query_specs['query'])):
            query_text = query_specs['query'][query_index]
            media_type = query_specs['media_type'][query_index]
            for rank in range(0, len(search_result[query_index]['match_filename_list'])):
                pts_str = '%.1f' % (search_result[query_index]['match_pts_list'][rank])
                filename = search_result[query_index]['match_filename_list'][rank]
                f.write(f'"{query_text}",{media_type},{rank},{filename},{pts_str}\n')

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

    parser.add_argument('--max-filename-length',
                        required=False,
                        type=int,
                        default=50,
                        help='only show this many characters from the end in a filename')

    parser.add_argument('--export-csv',
                        required=False,
                        type=str,
                        help='save results to this CSV file')

    parser.add_argument('--project-dir',
                        required=True,
                        type=str,
                        help='folder where all project assets are stored')

    args = parser.parse_args()

    project = WiseProject(args.project_dir, create_project=False)
    project_assets = project.discover_assets()
    if len(project_assets) == 0:
        print(f'failed to load assets from {args.project_dir}')
        sys.exit(1)
    db_engine = db.init_project(project.dburi, echo=False)

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
            print(f'failed to load {args.index_type} for {media_type}')
            del search_index_list[media_type]
            continue

    if len(search_index_list) == 0:
        print(f'search index missing from {args.project_dir}')
        sys.exit(1)

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
        if args.export_csv:
            export_result_as_csv(args.export_csv,
                                 query_specs,
                                 search_result,
                                 args)
            print(f'saved results to {args.export_csv}')

    else:
        print('Starting WISE search console ...')
        print('Some examples queries (press Ctrl + D to exit):')
        print('  1. find cooking videos with music playing in background')
        print('     > --query "cooking" --in video --query music --in audio')
        print('  2. find videos showing train, show only top 3 results and export results to a file')
        print('     > --query train --in video --topk 3 --export-csv train.csv')

        cmd_id = 0
        # Start the WISE search console Read-Evaluate-Print loop (REPL)
        while True:
            try:
                cmd = input('[%d] > ' % (cmd_id))
                repl_args = parse_user_input(cmd)
                search_result = process_query(search_index_list, repl_args, args)
                show_result(repl_args, search_result, args)
                if 'export-csv' in repl_args:
                    csv_filename = repl_args['export-csv']
                    export_result_as_csv(csv_filename,
                                         repl_args,
                                         search_result,
                                         args)
                    print(f'saved results to {csv_filename}')
                cmd_id += 1
            except EOFError:
                print('\nBye')
                break
            except KeyboardInterrupt:
                print('\nBye')
                break
