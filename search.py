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
import csv
import io

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

## Constants used by this script
EXPORT_CSV_HEADER = 'query_id,query_text,media_type,rank,filename,start_time,end_time,score'

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

"""A parser for user input obtained from the WISE search console
"""
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
    if not parse_token_name and parse_token_value:
        repl_args[last_token_name] = ''

    if 'in' in repl_args:
        # we avoid using 'in' as it is a reserved word in python
        repl_args['media_type'] = repl_args['in']
        del repl_args['in']
    return repl_args

""" Merge search results for each modality (i.e. audio, video, etc) separately.

    Parameters
    ----------
    search_index_list : a list containing instances of SearchIndex

    query_specs : a dictionary structured as shown in the example below
      {
        'query':[ "query-text-1", "query-text-2"],
        'media_type': [ "video", "audio"],
        'topk': 10
      }

    args : command line arguments

    Returns
    -------
    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...], 'search_time_sec': ... },
        { 'match_filename_list': [...], 'match_pts_list': [...], 'search_time_sec': ...  }
      ]
"""
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
        match_score_list = []
        with db_engine.connect() as conn:
            for rank in range(0, len(ids)):
                vector_id = int(ids[rank])
                # if faiss cannot return topk number of results, it marks
                # the end of result by setting ids to -1
                if vector_id == -1:
                    break
                vector_metadata = VectorRepo.get(conn, vector_id)
                media_metadata = MediaRepo.get(conn, vector_metadata.media_id)
                filename = media_metadata.path
                pts = vector_metadata.timestamp
                pts_hhmmss = to_hhmmss(pts)
                match_filename_list.append(filename)
                match_pts_list.append(pts)
                match_score_list.append(float(dist[rank]))
        end_time = time.time()
        search_result.append({
            'match_filename_list': match_filename_list,
            'match_pts_list': match_pts_list,
            'match_score_list': match_score_list,
            'search_time_sec': (end_time - start_time)
        })
    return search_result

""" Merge search results for each modality (i.e. audio, video, etc) separately.

    Parameters
    ----------
    query_specs : a dictionary structured as shown in the example below
      {
        'query':[ "query-text-1", "query-text-2"],
        'media_type': [ "video", "audio"],
        'topk': 10
      }

    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] },
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] }
      ]

    args : command line arguments

    Returns
    -------
    merged search result in the same format as the "result" argument
"""
def merge0(query_specs, result, args):
    for query_index in range(0, len(query_specs['query'])):
        media_type = query_specs['media_type'][query_index]
        merge_tolerance_id = 'merge_tolerance_' + media_type
        merge_tolerance = getattr(args, merge_tolerance_id)
        if merge_tolerance_id in query_specs:
            merge_tolerance = float(query_specs[merge_tolerance_id])

        filename_list = result[query_index]['match_filename_list']
        pts_list = result[query_index]['match_pts_list']
        score_list = result[query_index]['match_score_list']
        merged_filename_list, merged_pts_list, merged_score_list = merge_one_result(
            filename_list,
            pts_list,
            score_list,
            merge_tolerance)
        result[query_index]['match_filename_list'] = merged_filename_list
        result[query_index]['match_pts_list'] = merged_pts_list
        result[query_index]['match_score_list'] = merged_score_list
    return result

""" Merge two search results either from same modality (e.g. video) or from
    two different modalities (e.g. video and audio)

    Parameters
    ----------
    query_specs : a dictionary structured as shown in the example below
      {
        'query':[ "query-text-1", "query-text-2"],
        'media_type': [ "video", "audio"],
        'topk': 10
      }

    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] },
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] }
      ]


    args : command line arguments

    Returns
    -------
    merged_query_specs : a dictionary similar to query_specs but reflecting
        the result of merging process. For example, 'media_type' becomes
        'video and audio'

    merged_result : a list of length 1 and formatted as follows
      [ {
        'match_filename_list': [ ... ],
        'match_pts_list': [ ... ],
        'match_score_list': [ ... ],
        'search_time_sec': ...
      }]
"""
def merge1(query_specs, result, args):
    if len(result) != 2:
        print('merge1() can be only applied if result contains two entries')
        return
    N0 = len(result[0]['match_filename_list'])
    N1 = len(result[1]['match_filename_list'])
    merged_filename_list = []
    merged_score_list = []
    merged_pts_list = []
    for index_pair in itertools.product( range(0,N0), range(0,N1) ):
        index0 = index_pair[0]
        index1 = index_pair[1]
        filename0 = result[0]['match_filename_list'][index0]
        filename1 = result[1]['match_filename_list'][index1]
        score0 = result[0]['match_score_list'][index0]
        score1 = result[1]['match_score_list'][index1]
        if filename0 == filename1:
            merged_filename_list.append(filename0)
            if score0 > score1:
                merged_score = score0
            else:
                merged_score = score1
            merged_score_list.append(merged_score)
            pts0 = result[0]['match_pts_list'][index0]
            pts1 = result[1]['match_pts_list'][index1]
            if isinstance(pts0, list) and isinstance(pts1, list):
                merged_pts = pts0 + pts1
            else:
                merged_pts = [pts0, pts1]
            merged_pts.sort()
            if len(merged_pts) == 1:
                merged_pts_list.append([ merged_pts[0] ])
            else:
                merged_pts_list.append([ merged_pts[0], merged_pts[ len(merged_pts)-1 ] ])
    merged_result = [ {
        'match_filename_list': merged_filename_list,
        'match_pts_list': merged_pts_list,
        'match_score_list': merged_score_list,
        'search_time_sec': result[0]['search_time_sec'] + result[1]['search_time_sec']
    }]
    merged_query_specs = query_specs # to retain user provided arguments like --save-to-file
    merged_query_specs['query'] = [ ' and '.join(query_specs['query']) ]
    merged_query_specs['query_id'] = [ '-'.join(query_specs['query_id']) ]
    merged_query_specs['media_type'] = [ ' and '.join(query_specs['media_type']) ]
    return merged_query_specs, merged_result

""" Merge search results from one of the modalities (e.g. audio, video, etc.).
    The merge operation retains the rank of a search result and merges all lower
    ranking results with the same filename.

    Parameters
    ----------
    filename_list : a list of filenames ordered by the rank of search results

    pts_list : a list of presentation timestamp corresponding to audio
      and/or video filenames contained in filename_list

    pts_list : a list of similarity score between search query and the audio
      and/or video filenames contained in filename_list

    tolerance : entries with timestamp difference less than the tolerance
      gets merged

    Returns
    -------
    merged_filename_list : merged list of filenames
    merged_pts_list : presentation timestamp range corresponding to
      merged_filename_list
    merged_query_specs : a dictionary similar to query_specs but reflecting
        the result of merging process. For example, 'media_type' becomes
        'video and audio'

    merged_result : a list of length 1 and formatted as follows
      [ {
        'match_filename_list': [ ... ],
        'match_pts_list': [ ... ],
        'search_time_sec': ...
      }]

"""
def merge_one_result(filename_list, pts_list, score_list, tolerance):
    N = len(filename_list)
    merged_filename_list = []
    merged_pts_list = []
    merged_score_list = []
    skip_index_list = []
    for i in range(0, N):
        if i in skip_index_list:
            continue

        # for each unique filename, find all the pts
        filename_i = filename_list[i]
        pts_index_list = [i] # will contain all pts corresponding to filename_i
        for j in range(i+1, N):
            if j in skip_index_list:
                continue
            if filename_i == filename_list[j]:
                pts_index_list.append(j)
        merge_pts_index_list = set() # will contain all the pts index that can be merged
        merge_pts_index_list.add(pts_index_list[0])
        for pts_index_pair in itertools.combinations( range(0, len(pts_index_list)), 2):
            pts_index1 = pts_index_list[ pts_index_pair[0] ]
            pts_index2 = pts_index_list[ pts_index_pair[1] ]
            del_pts = math.fabs(pts_list[pts_index1] - pts_list[pts_index2])
            if del_pts <= tolerance:
                merge_pts_index_list.add(pts_index1)
                merge_pts_index_list.add(pts_index2)

        to_merge_pts_list = []
        for pts_index in merge_pts_index_list:
            to_merge_pts_list.append( pts_list[pts_index] )
            skip_index_list.append( pts_index )
        to_merge_pts_list.sort()
        if len(to_merge_pts_list) > 1:
            merged_pts_list.append( [ to_merge_pts_list[0], to_merge_pts_list[ len(to_merge_pts_list) - 1 ] ] )
        else:
            merged_pts_list.append( [ to_merge_pts_list[0] ] )
        merged_filename_list.append(filename_i)
        merged_score_list.append(score_list[i])
    return merged_filename_list, merged_pts_list, merged_score_list

""" Manages display of search results in console

    Parameters
    ----------
    query_specs : a dictionary structured as shown in the example below
      {
        'query':[ "query-text-1", "query-text-2"],
        'media_type': [ "video", "audio"],
        'topk': 10
      }

    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...] },
        { 'match_filename_list': [...], 'match_pts_list': [...] }
      ]


    args : command line arguments

    Returns
    -------
    no returns
"""
def show_result(query_specs, result, args):
    result_format = 'table'
    if hasattr(args, 'result_format') and args.result_format is not None:
        result_format = args.result_format
    if 'result-format' in query_specs:
        result_format = query_specs['result-format']
    if result_format == 'csv':
        show_result_as_csv(query_specs, result, args)
    else:
        show_result_as_table(query_specs, result, args)

def show_result_as_table(query_specs, result, args):
    out = sys.stdout
    writing_to_file = False
    if hasattr(args, 'save_to_file') and args.save_to_file is not None:
        out = io.open(args.save_to_file, 'a')
        writing_to_file = True
    elif 'save-to-file' in query_specs:
        out = io.open(query_specs['save-to-file'], 'a')
        writing_to_file = True

    console = Console(file=out, no_color=True)
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
        table.add_column('Score', justify='left', no_wrap=True)

        for rank in range(0, len(result[query_index]['match_filename_list'])):
            pts = result[query_index]['match_pts_list'][rank]
            if isinstance(pts, list):
                if len(pts) == 2:
                    pts_str = '%.1f - %.1f' % (pts[0], pts[1])
                else:
                    pts_str = '%.1f' % (pts[0])
            else:
                pts_str = '%.1f' % (pts)
            filename = result[query_index]['match_filename_list'][rank]
            score_str = '%.3f' % (result[query_index]['match_score_list'][rank])
            table.add_row(str(rank),
                          clamp_str(filename, getattr(args, 'max_filename_length')),
                          pts_str,
                          score_str)
        console.print(table)
        console.print('')
        total_search_time += result[query_index]['search_time_sec']
    if len(result) == 1:
        console.print('(search completed in %.3f sec.)' % (total_search_time))

def show_result_as_csv(query_specs, result, args):
    # Note: The CSV header is written by caller because the csv header needs
    # to be written only once irrespective of the number of times
    # show_result_as_csv() is executed
    out = sys.stdout
    writing_to_file = False
    if hasattr(args, 'save_to_file') and args.save_to_file is not None:
        out = io.open(args.save_to_file, 'a')
        writing_to_file = True
    elif 'save-to-file' in query_specs:
        out = io.open(query_specs['save-to-file'], 'a')
        writing_to_file = True

    for query_index in range(0, len(query_specs['query'])):
        query_text = query_specs['query'][query_index]
        media_type = query_specs['media_type'][query_index]
        query_id   = query_specs['query_id'][query_index]
        for rank in range(0, len(result[query_index]['match_filename_list'])):
            pts = result[query_index]['match_pts_list'][rank]
            if isinstance(pts, list):
                if len(pts) == 1:
                    pts_str = '%.1f,%.1f' % (pts[0], pts[0])
                else:
                    pts_str = '%.1f,%.1f' % (pts[0], pts[1])
            else:
                pts_str = '%.1f' % (result[query_index]['match_pts_list'][rank])
            filename = result[query_index]['match_filename_list'][rank]
            score_str = '%.3f' % (search_result[query_index]['match_score_list'][rank])
            out.write(f'{query_id},"{query_text}",{media_type},{rank},"{filename}",{pts_str},{score_str}\n')
    if writing_to_file:
        out.close()

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

    parser.add_argument('--no-merge',
                        action='store_true',
                        help='avoid merging of search results corresponding to adjacent audio-visual segments')

    parser.add_argument('--merge-tolerance-video',
                        required=False,
                        type=int,
                        default=4,
                        help='tolerance (in seconds) for merging video based search results')

    parser.add_argument('--merge-tolerance-audio',
                        required=False,
                        type=int,
                        default=8,
                        help='tolerance (in seconds) for merging audio based search results')

    parser.add_argument('--result-format',
                        required=False,
                        default='table',
                        choices=['table', 'csv'],
                        type=str,
                        help='show results in tabular format (default) or as comma separated values (csv)')

    parser.add_argument('--save-to-file',
                        required=False,
                        type=str,
                        help='save results to this file instead of showing it on console')

    parser.add_argument('--queries-from',
                        required=False,
                        type=str,
                        help='a CSV filename, must have a column header, each row must be [query_id, query_text]')

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

    # Print the CSV header only once.
    # If this task is delegated to the show_result_as_csv() method,
    # the CSV header may get printed multiple times.
    if hasattr(args, 'save_to_file') and args.save_to_file is not None:
        with open(args.save_to_file, 'w') as f:
            if args.result_format == 'csv':
                f.write(EXPORT_CSV_HEADER + '\n')
    else:
        if args.result_format == 'csv':
            print(f'{EXPORT_CSV_HEADER}')

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

    ## Based on command line arguments, we begin the search operation.
    ## The command line arguments can give rise to the following three possibilities.
    ## Case-1: Search query provided in the command line
    ## Case-2: Search queries contained in a CSV file
    ## Case-3: Search query not provided, start search console

    ## Case-1: Search query provided in the command line
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

        if args.result_format != 'csv' and not args.save_to_file:
            print(f'Searching {args.project_dir} for')
            for i in range(0, len(args.query)):
                print(f'  [{i}] "{args.query[i]}" in {args.media_type[i]}')
            print('\n')
        query_specs = {
            'query': args.query,
            'query_id': [ str(x) for x in range(0, len(args.query)) ],
            'media_type': args.media_type
        }
        search_result = process_query(search_index_list, query_specs, args)
        if args.no_merge:
            merge0_search_result = search_result
        else:
            merge0_search_result = merge0(query_specs, search_result, args)
        show_result(query_specs, merge0_search_result, args)
        if len(merge0_search_result) == 2 and args.queries_from is None:
            merge1_query_specs, merge1_search_result = merge1(query_specs, merge0_search_result, args)
            show_result(merge1_query_specs, merge1_search_result, args)

    ## Case-2: Search queries contained in a CSV file
    elif hasattr(args, 'queries_from') and args.queries_from is not None:
        with open(args.queries_from, 'r') as f:
            query_reader = csv.reader(f, delimiter=',', quotechar='"')
            header = next(query_reader)
            for row in query_reader:
                if len(row) != 2:
                    print(f'Skipping query: "{row}". Format each line as: query-id, query-text')
                    continue
                query_id = row[0]
                query_text = row[1]
                query_specs = {
                    'query': [query_text for _ in args.media_type], # repeat query for each media_type
                    'query_id': [ query_id for _ in args.media_type ],
                    'media_type': args.media_type
                }
                search_result = process_query(search_index_list, query_specs, args)
                if args.no_merge:
                    merge0_search_result = search_result
                else:
                    merge0_search_result = merge0(query_specs, search_result, args)
                show_result(query_specs, merge0_search_result, args)
                if len(merge0_search_result) == 2 and args.queries_from is None:
                    merge1_query_specs, merge1_search_result = merge1(query_specs, merge0_search_result, args)
                    show_result(merge1_query_specs, merge1_search_result, args)

    ## Case-3: Search query not provided, start search console
    else:
        print('Starting WISE search console ...')
        print('Some examples queries (press Ctrl + D to exit):')
        print('  1. find cooking videos with music playing in background')
        print('     > --query "cooking" --in video --query music --in audio')
        print('  2. find videos showing train, show only top 3 results and export results to a file')
        print('     > --query train --in video --topk 3 --result-format csv --save-to-file train.csv')

        cmd_id = 0
        # Start the WISE search console Read-Evaluate-Print loop (REPL)
        while True:
            try:
                cmd = input('[%d] > ' % (cmd_id))
                query_specs = parse_user_input(cmd)
                query_specs['query_id'] = [ str(x) for x in range(0, len(query_specs['query'])) ]
                search_result = process_query(search_index_list, query_specs, args)
                print(args.no_merge)
                print(query_specs)
                if 'save-to-file' in query_specs:
                    with open(query_specs['save-to-file'], 'w') as f:
                        if 'result-format' in query_specs and query_specs['result-format'] == 'csv':
                            f.write(EXPORT_CSV_HEADER + '\n')
                else:
                    if 'result-format' in query_specs and query_specs['result-format'] == 'csv':
                        print(f'{EXPORT_CSV_HEADER}')

                if args.no_merge or ('no-merge' in query_specs):
                    merge0_search_result = search_result
                else:
                    merge0_search_result = merge0(query_specs, search_result, args)
                show_result(query_specs, merge0_search_result, args)
                if len(merge0_search_result) == 2:
                    merge1_query_specs, merge1_search_result = merge1(query_specs, merge0_search_result, args)
                    show_result(merge1_query_specs, merge1_search_result, args)
                cmd_id += 1
            except EOFError:
                print('\nBye')
                break
            except KeyboardInterrupt:
                print('\nBye')
                break
