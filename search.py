"""
Table of Contents
  A. Process search query and obtain search results
  B. Merge search results based on audiovisual time segment, rank, etc.
  C. Methods to display and export search results
  D. Command line interface (CLI) parser and handler
"""
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
import copy

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from src.dataloader import AVDataset
from src.wise_project import WiseProject
from src.index.search_index_factory import SearchIndexFactory

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

##
##  A. Process search query and obtain search results
##

""" Merge search results for each modality (i.e. audio, video, etc) separately.

    Parameters
    ----------
    search_index_list : a list containing instances of SearchIndex

    args : command line arguments containing "query", "media_type_list", "topk"
           and optionally "media_type_not_list".

    Returns
    -------
    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...], 'search_time_sec': ... },
        { 'match_filename_list': [...], 'match_pts_list': [...], 'search_time_sec': ... }
      ]
"""
def process_query(search_index_list, args):
    assert hasattr(args, 'query'), 'args must contain "query" parameter'
    assert hasattr(args, 'media_type_list'), 'args must contain "media_type_list" parameter'
    assert hasattr(args, 'topk'), 'args must contain "topk" parameter'

    topk = [int(k) for k in args.topk]
    if len(args.topk) == 1:
        topk = [ int(args.topk[0]) ] * len(args.query)

    # collect all results related to --not-in queries
    all_not_result = []
    not_elapsed_time = 0
    if args.media_type_not_list is not None:
        not_start_time = time.time()

        for not_index in range(len(args.media_type_list), len(args.query)):
            not_query = args.query[not_index]
            not_media_type = args.media_type_not_list[len(args.media_type_list) - not_index]
            not_result = process_text_query(search_index_list,
                                            not_query,
                                            not_media_type,
                                            topk[not_index])
            all_not_result.append(not_result)
        not_end_time = time.time()
        not_elapsed_time = not_end_time - not_start_time

    # collect all results related to --in queries
    all_search_result = []
    for query_index in range(0, len(args.media_type_list)):
        start_time = time.time()
        query_text = args.query[query_index]
        media_type = args.media_type_list[query_index]
        search_result = process_text_query(search_index_list,
                                           query_text,
                                           media_type,
                                           topk[query_index])
        search_result['query'] = [ query_text ]
        search_result['in'] = [ media_type ]
        search_result['not_in'] = []

        # remove all results appearing in --not-in query
        if args.media_type_not_list is not None:
            for not_index in range(len(args.media_type_list), len(args.query)):
                not_search_result = all_not_result[not_index - len(args.media_type_list)]
                search_result = apply_subtract(search_result, not_search_result)
                search_result['query'].append(not_query)
                search_result['not_in'].append(not_media_type)
        end_time = time.time()
        search_result['search_time_sec'] = (end_time - start_time) + not_elapsed_time
        if args.query_id:
            search_result['query_id'] = args.query_id
        else:
            search_result['query_id'] = query_index
        all_search_result.append(search_result)
    return all_search_result

def process_text_query(search_index_list, query_text, media_type, topk):
    if media_type == 'metadata':
        result = search_index_list[media_type].search(media_type,
                                                      query_text,
                                                      topk,
                                                      query_type='text')
    else:
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
                if vector_metadata.end_timestamp == None:
                    pts = vector_metadata.timestamp
                else:
                    pts = [vector_metadata.timestamp, vector_metadata.end_timestamp]
                match_filename_list.append(filename)
                match_pts_list.append(pts)
                match_score_list.append(float(dist[rank]))
        result = {
            'match_filename_list': match_filename_list,
            'match_pts_list': match_pts_list,
            'match_score_list': match_score_list,
        }
    return result

def apply_subtract(search_result, not_search_result):
    new_search_result = {
        'match_filename_list':[],
        'match_pts_list':[],
        'match_score_list':[],
        'query': search_result['query'],
        'query_id': search_result['query_id'],
        'in': search_result['in'],
        'not_in': search_result['not_in']
    }
    for result_index in range(0, len(search_result['match_filename_list'])):
        match_filename = search_result['match_filename_list'][result_index]
        match_pts = search_result['match_pts_list'][result_index]
        match_score = search_result['match_score_list'][result_index]
        if not result_exists(match_filename, match_pts, not_search_result):
            new_search_result['match_filename_list'].append(match_filename)
            new_search_result['match_pts_list'].append(match_pts)
            new_search_result['match_score_list'].append(match_score)
    return new_search_result

def result_exists(filename, pts, results):
    for result_index in range(0, len(results['match_filename_list'])):
        if filename == results['match_filename_list'][result_index]:
            pts2 = results['match_pts_list'][result_index]
            does_overlap = does_segment_overlap(pts, pts2)
            if does_segment_overlap(pts, pts2):
                return True
    return False

def does_segment_overlap(seg1, seg2):
    seg1_is_point = False
    seg2_is_point = False
    if isinstance(seg1, float):
        seg1 = [seg1, seg1]
        seg1_is_point = True
    if isinstance(seg2, float):
        seg2 = [seg2, seg2]
        seg2_is_point = True
    if len(seg1) == 1:
        seg1.append(seg1[0])
        seg1_is_point = True
    if len(seg2) == 1:
        seg2.append(seg2[0])
        seg2_is_point = True
    assert len(seg1) == 2, f'segment1 must be defined using a list of length 2; received {seg1}'
    assert len(seg2) == 2, f'segment2 must be defined using a list of length 2; received {seg2}'

    all_pts = seg1 + seg2
    seg1_union_seg2 = max(all_pts) - min(all_pts)
    if seg1_is_point or seg2_is_point:
        # check if a point lies within a temporal segment
        if seg1_is_point:
            if seg1[0] >= seg2[0] and seg1[0] <= seg2[1]:
                return True
            else:
                return False
        else:
            if seg2[0] >= seg1[0] and seg2[0] <= seg1[1]:
                return True
            else:
                return False
    else:
        # check if a segment has any overlap with another segment
        iou = (min(seg1[1], seg2[1]) - max(seg1[0], seg2[0])) / seg1_union_seg2
        if iou > 0.01:
            return True
        else:
            return False

##
##  B. Merge search results based on audiovisual time segment, rank, etc.
##

""" Merge search results for each modality (i.e. audio, video, etc) separately.

    Parameters
    ----------
    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] },
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] }
      ]

    args : command line arguments containing "query", "media_type_list", "topk"
           and optionally "media_type_not_list".

    Returns
    -------
    merged search result in the same format as the "result" argument
"""
def merge0(result, args):
    for query_index in range(0, len(result)):
        assert len(result[query_index]['in']) == 1, f'unexpected {result[query_index]["in"]}'

        media_type = result[query_index]['in'][0]
        merge_tolerance_id = 'merge_tolerance_' + media_type
        time_tolerance = getattr(args, merge_tolerance_id)
        rank_tolerance = getattr(args, 'merge_rank_tolerance')

        filename_list = result[query_index]['match_filename_list']
        pts_list = result[query_index]['match_pts_list']
        score_list = result[query_index]['match_score_list']
        merged_filename_list, merged_pts_list, merged_score_list, merged_rank_list = merge_a_ranked_result_list(
            filename_list,
            pts_list,
            score_list,
            time_tolerance,
            rank_tolerance)
        result[query_index]['match_filename_list'] = merged_filename_list
        result[query_index]['match_pts_list'] = merged_pts_list
        result[query_index]['match_score_list'] = merged_score_list
        result[query_index]['merged_rank_list'] = merged_rank_list
    return result

""" Merge entries in a search result if they correspond to same filename and
    overlapping time range. Search results [i] and [j] are merged only when the
    following constraints are satisfied.
      1. filename_list[i] == filename_list[j]
      2. | i - j | <= rank_tolerance
      3. | pts_list[i] - pts_list[j] | <= pts_tolerance

    Parameters
    ----------
    filename_list : a list of filenames; order is same as the ranked search results
      (e.g. ['abc.mp4', 'xyz.mp4']

    pts_list : a list of presentation timestamp (pts) corresponding to entries in
      filename_list; each entry is a timestamp or a time range;
      (e.g. [ 1.3, [4.8, 9.7] ]

    score_list : a list of similarity score between search query and entries
      in filename_list and pts_list

    pts_tolerance : merge only when | pts_list[i] - pts_list[j] <= pts_tolerance

    rank_tolerance : merge only when |i-j| <= rank_tolerance

    Returns
    -------
    merged_filename_list, merged_pts_list, merged_score_list
"""
def merge_a_ranked_result_list(filename_list, pts_list, score_list, pts_tolerance, rank_tolerance):
    N = len(filename_list)
    merged_filename_list = []
    merged_pts_list = []
    merged_score_list = []
    merged_rank_list = []
    skip_index_list = []
    for i in range(0, N):
        if i in skip_index_list:
            continue

        # for each unique filename, find all the pts values
        filename_i = filename_list[i]
        pts_index_list = [i]  # this list will eventually contain all pts corresponding to filename_i
        for j in range(i+1, N):
            if j in skip_index_list:
                continue
            if filename_i == filename_list[j]:
                pts_index_list.append(j)

        merge_pts_index_list = set() # this list will eventually contain all the pts index that can be merged
        merge_pts_index_list.add(pts_index_list[0])
        for pts_index_pair in itertools.combinations( range(0, len(pts_index_list)), 2):
            pts_index1 = pts_index_list[ pts_index_pair[0] ]
            pts_index2 = pts_index_list[ pts_index_pair[1] ]

            # compute a metric that corresponds to the distance between a pair of pts values
            if isinstance(pts_list[pts_index1], list) and isinstance(pts_list[pts_index2], list):
                # if pts is a range of timestamp, compute separation using their mid points
                mid1 = sum(pts_list[pts_index1]) / len(pts_list[pts_index1])
                mid2 = sum(pts_list[pts_index2]) / len(pts_list[pts_index2])
                del_pts = math.fabs(mid1 - mid2)
            else:
                # if pts is a timestamp, compute their separation using their difference 
                del_pts = math.fabs(pts_list[pts_index1] - pts_list[pts_index2])

            if del_pts <= pts_tolerance:
                if math.fabs(pts_index1 - pts_index2) <= rank_tolerance:
                    merge_pts_index_list.add(pts_index1)
                    merge_pts_index_list.add(pts_index2)
        to_merge_pts_list = []
        for pts_index in merge_pts_index_list:
            if isinstance(pts_list[pts_index], list):
                to_merge_pts_list += pts_list[pts_index]
            else:
                to_merge_pts_list.append( pts_list[pts_index] )
            skip_index_list.append( pts_index )
        to_merge_pts_list.sort()
        if len(to_merge_pts_list) > 1:
            merged_pts_list.append( [ to_merge_pts_list[0], to_merge_pts_list[ len(to_merge_pts_list) - 1 ] ] )
        else:
            merged_pts_list.append( [ to_merge_pts_list[0] ] )
        merged_filename_list.append(filename_i)
        merged_score_list.append(score_list[i])
        merged_rank_list.append(list(merge_pts_index_list))
    return merged_filename_list, merged_pts_list, merged_score_list, merged_rank_list

""" Merge two search results either from same modality (e.g. video) or from
    two different modalities (e.g. video and audio)

    Parameters
    ----------
    result : a list of length 2 and structured as follows
      [
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] },
        { 'match_filename_list': [...], 'match_pts_list': [...], 'match_score_list': [...] }
      ]

    args : command line arguments containing "query", "media_type_list", "topk"
           and optionally "media_type_not_list".

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
def merge1(result, args):
    if len(result) != 2:
        print('merge1() can be only applied if result contains two entries')
        return
    N0 = len(result[0]['match_filename_list'])
    N1 = len(result[1]['match_filename_list'])
    merged_filename_list = []
    merged_score_list = []
    merged_pts_list = []
    merged_rank_list = []
    for index_pair in itertools.product( range(0,N0), range(0,N1) ):
        index0 = index_pair[0]
        index1 = index_pair[1]
        filename0 = result[0]['match_filename_list'][index0]
        filename1 = result[1]['match_filename_list'][index1]
        score0 = result[0]['match_score_list'][index0]
        score1 = result[1]['match_score_list'][index1]
        pts0 = result[0]['match_pts_list'][index0]
        pts1 = result[1]['match_pts_list'][index1]

        if filename0 == filename1 and does_segment_overlap(pts0, pts1):
            merged_filename_list.append(filename0)
            merged_score = score0 + score1
            merged_score_list.append(merged_score)
            if isinstance(pts0, list) and isinstance(pts1, list):
                merged_pts = pts0 + pts1
            else:
                merged_pts = [pts0, pts1]
            merged_pts.sort()
            if len(merged_pts) == 1:
                merged_pts_list.append([ merged_pts[0] ])
            else:
                merged_pts_list.append([ merged_pts[0], merged_pts[ len(merged_pts)-1 ] ])
            merged_rank_list.append([index0, index1])

    # sort results based on merged scores
    sort_index = sorted( range(len(merged_score_list)), key=merged_score_list.__getitem__, reverse=True )
    sorted_merged_result = {
        'match_filename_list': [ merged_filename_list[i] for i in sort_index ],
        'match_pts_list': [ merged_pts_list[i] for i in sort_index ],
        'match_score_list': [ merged_score_list[i] for i in sort_index ],
        'merged_rank_list': [ merged_rank_list[i] for i in sort_index ],
        'search_time_sec': result[0]['search_time_sec'] + result[1]['search_time_sec'],
        'query': result[0]['query'] + result[1]['query'],
        'in': result[0]['in'] + result[1]['in'],
        'not_in':[]
    }
    if 'not_in' in result[0]:
        sorted_merged_result['not_in'] += result[0]['not_in']
    if 'not_in' in result[1]:
        sorted_merged_result['not_in'] += result[1]['not_in']

    return [ sorted_merged_result ]

def sort_result_by_score(result, score_label):
    sort_index = sorted( range(len(result[score_label])), key=result[score_label].__getitem__ )
    sorted_result = {}
    for key in merged_query_specs:
        sorted_result[key] = [ merged_query_specs[key][i] for i in sort_index ]
    return sorted_result

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

##
## C. Methods to display and export search results
##

EXPORT_CSV_HEADER = 'query,rank,filename,start_time,end_time,score'

def show_result(result, args):
    result_format = 'table'
    if hasattr(args, 'result_format') and args.result_format is not None:
        result_format = args.result_format
    if result_format == 'csv':
        show_result_as_csv(result, args)
    else:
        show_result_as_table(result, args)

def show_result_as_table(result, args):
    out = sys.stdout
    writing_to_file = False
    if hasattr(args, 'save_to_file') and args.save_to_file is not None:
        out = io.open(args.save_to_file, 'a')
        writing_to_file = True

    console = Console(file=out, no_color=True)
    total_search_time = 0
    for query_index in range(0, len(result)):
        title = search_result_title(result[query_index])
        if len(result[query_index]['match_filename_list']) == 0:
            print(f'No results found for {title}')
            continue
        table = Table(title='Search results for ' + title,
                      show_lines=False,
                      show_edge=False,
                      box=None,
                      safe_box=True)
        table.add_column('Rank', justify='right', no_wrap=True)
        table.add_column('Filename', justify='left', no_wrap=True)
        table.add_column('Time', justify='left', no_wrap=True)
        table.add_column('Score', justify='left', no_wrap=True)
        if not args.no_merge:
            table.add_column('Original Ranks', justify='left', no_wrap=True)

        for rank in range(0, len(result[query_index]['match_filename_list'])):
            pts = result[query_index]['match_pts_list'][rank]
            pts_str = format_timestamp(pts, args.human_readable)
            filename = result[query_index]['match_filename_list'][rank]
            score_str = '%.3f' % (result[query_index]['match_score_list'][rank])
            if args.no_merge:
                table.add_row(str(rank),
                              clamp_str(filename, getattr(args, 'max_filename_length')),
                              pts_str,
                              score_str)
            else:
                merged_ranks_str = format_merged_ranks(result[query_index]['merged_rank_list'][rank])
                table.add_row(str(rank),
                              clamp_str(filename, getattr(args, 'max_filename_length')),
                              pts_str,
                              score_str,
                              merged_ranks_str)
        console.print(table)
        console.print('')
        total_search_time += result[query_index]['search_time_sec']
    if len(result) == 1 and len(result[0]['match_filename_list']) != 0:
        console.print('(search completed in %.3f sec.)' % (total_search_time))

def show_result_as_csv(result, args):
    # Note: The CSV header is written by caller because the csv header needs
    # to be written only once irrespective of the number of times
    # show_result_as_csv() is executed
    out = sys.stdout
    writing_to_file = False
    if hasattr(args, 'save_to_file') and args.save_to_file is not None:
        out = io.open(args.save_to_file, 'a')
        writing_to_file = True

    for query_index in range(0, len(result)):
        if 'query_id' in result[query_index]:
            query_id = result[query_index]['query_id'][0]
        else:
            query_id = search_result_title(result[query_index]).replace('"', '""')
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
            score_str = '%.3f' % (result[query_index]['match_score_list'][rank])
            out.write(f'"{query_id}",{rank},"{filename}",{pts_str},{score_str}\n')
    if writing_to_file:
        out.close()

def search_result_title(result):
    title = []
    query_count = len(result['query'])
    in_count = len(result['in'])
    for i in range(0, in_count):
        title.append(f'"{result["query"][i]}" in {result["in"][i]}')
    for i in range(in_count, query_count):
        title.append(f'"{result["query"][i]}" not in {result["not_in"][i-in_count]}')
    return ' and '.join(title)

def to_hhmmss(sec):
    hh = int(sec / (60*60))
    ds = sec - hh*60*60
    mm = int(ds / 60)
    ds = ds - mm*60
    ss = int(ds)
    ms = int((ds - ss) * 100)
    return '%02d:%02d:%02d.%02d' % (hh, mm, ss, ms)

def clamp_str(text, MAX_CHARS):
    if len(text) > MAX_CHARS:
        text_short = '...' + text[ len(text)-MAX_CHARS : len(text) ]
        return text_short
    else:
        return text

def format_timestamp(pts, human_readable=False):
    if isinstance(pts, list):
        if len(pts) == 2:
            if human_readable:
                pts_str = '%s - %s' % (to_hhmmss(pts[0]), to_hhmmss(pts[1]))
            else:
                pts_str = '%.1f - %.1f' % (pts[0], pts[1])
        else:
            if human_readable:
                pts_str = '%s' % (to_hhmmss(pts[0]))
            else:
                pts_str = '%.1f' % (pts[0])
    else:
        if human_readable:
            pts_str = '%s' % (to_hhmmss(pts))
        else:
            pts_str = '%.1f' % (pts)
    return pts_str

def format_merged_ranks(merged_rank_list):
    merged_rank_list.sort()
    N = len(merged_rank_list)
    if(N > 4):
        subset = merged_rank_list[0:4]
        return '%d,%d,...,%d,%d (%d)' % (merged_rank_list[0],
                                         merged_rank_list[1],
                                         merged_rank_list[N-2],
                                         merged_rank_list[N-1],
                                         N)
    else:
        merged_rank_list.sort()
        merged_rank_list_str = ','.join(str(x) for x in merged_rank_list)
        return merged_rank_list_str

##
## D. Command line interface (CLI) parser and handlers
##


"""A parser for user input obtained from the WISE search console
"""
def parse_user_input(cmd, args):
    args_copy = copy.deepcopy(args)
    list_args = ['query', 'in', 'not_in', 'topk']
    list_arg_map = {
        'in': 'media_type_list',
        'not_in': 'media_type_not_list'
    }
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
            last_token_name = token_name.strip().replace('-', '_')
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
            if last_token_name in list_args:
                if hasattr(args_copy, last_token_name):
                    if getattr(args_copy, last_token_name) is None:
                        setattr(args_copy, last_token_name, list())
                else:
                    setattr(args_copy, last_token_name, list())
                last_token_values = getattr(args_copy, last_token_name)
                last_token_values.append(token_value.strip(' "')) # remove space and quotation
                setattr(args_copy, last_token_name, last_token_values)
            else:
                setattr(args_copy, last_token_name, token_value.strip())
            parse_token_name = False
            parse_token_value = False
        else:
            tok_index += 1
    if not parse_token_name and parse_token_value:
        setattr(args_copy, last_token_name, '')
    # map arguments under new name keeping the old values
    for argname in list_arg_map:
        if not hasattr(args_copy, argname):
            continue
        arg_newname = list_arg_map[argname]
        argname_value = getattr(args_copy, argname)
        setattr(args_copy, arg_newname, argname_value)
    return args_copy

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
                        dest='media_type_list', # since "in" is a reserved keyword
                        choices=['audio', 'video', 'metadata'],
                        help='apply the search query term to these features; query applied to all features if --in argument is missing')

    parser.add_argument('--not-in',
                        required=False,
                        action='append',
                        dest='media_type_not_list',
                        choices=['audio', 'video', 'metadata'],
                        help='remove the results from the preceeding query obtained in this media_type; Note: all --not-in flags must come after --in flags')

    parser.add_argument('--index-type',
                        required=False,
                        default='IndexFlatIP',
                        choices=['IndexFlatIP', 'IndexIVFFlat'],
                        type=str,
                        help='the type of faiss index to search')

    parser.add_argument('--topk',
                        required=False,
                        action='append',
                        default=[10],
                        help='show only the top K search results, query specific --topk value can also be provided')

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

    parser.add_argument('--merge-rank-tolerance',
                        required=False,
                        type=int,
                        default=20,
                        help='merge results only if their rank is within this range')

    parser.add_argument('--merge-tolerance-audio',
                        required=False,
                        type=int,
                        default=8,
                        help='tolerance (in seconds) for merging audio based search results')

    parser.add_argument('--merge-tolerance-metadata',
                        required=False,
                        type=int,
                        default=0,
                        help='tolerance (in seconds) for merging metadata based search results')

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

    parser.add_argument('--human-readable',
                        action='store_true',
                        help='show values in human readable format (e.g. show 00:01:36.50 instead of 96.5 sec)')

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

    # sanity check of some CLI arguments
    if args.media_type_list is None and args.media_type_not_list is not None:
        print(f'--not-in flag required previous definitions of --in flag')
        sys.exit(0)
    if len(args.topk) != 1:
        args.topk = args.topk[1:] # remove the default value
    if args.query is not None and (len(args.topk) != len(args.query)):
        print(f'All --query flags should be accompanied by a --topk value')
        sys.exit(0)

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

    if args.query is None and args.media_type_list is None:
        # load all search index as query is not decided yet (e.g. search console mode)
        unique_required_media_type = list(project_assets.keys())
    else:
        # load only the required search index
        required_media_type = list(args.media_type_list)
        if args.media_type_not_list:
            required_media_type += args.media_type_not_list
        unique_required_media_type = list(set(required_media_type))

    for media_type in unique_required_media_type:
        asset_id_list = list(project_assets[media_type].keys())
        asset_index = 0
        N = len(asset_id_list)
        if N == 0:
            continue
        if N == 1:
            asset_index = 0
        else:
            print(f'{media_type} can be searched using the following search index:')
            for asset_index in range(0, N):
                print(f'  {asset_index}. {asset_id_list[asset_index]}')
            selected_asset_index = -1
            while True:
                try:
                    input_index = input(f'Enter the index of desired search index [0-{N-1}] : ')
                    selected_asset_index = int(input_index)
                    if selected_asset_index >= 0 and selected_asset_index < N:
                        break
                except:
                    print(f'invalid input {input_index}')
            asset_index = selected_asset_index
        asset_id = asset_id_list[asset_index]
        asset = project_assets[media_type][asset_id]
        search_index_list[media_type] = SearchIndexFactory(media_type, asset_id, asset)
        if not search_index_list[media_type].load_index(args.index_type):
            print(f'failed to load {media_type} index: {asset_id}')
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
        media_type_count = len(args.media_type_list)
        if args.media_type_not_list:
            media_type_count += len(args.media_type_not_list)
        if len(args.query) > 1 and len(args.query) != media_type_count:
            print('Each --query argument must be followed by a --in argument. For example:')
            print('  $ python search.py --query people --in video --query shouting --in audio ...')
            sys.exit(0)

        ## if "--in" argments are missing, assume that the search query is
        ## to be applied on all possible media types
        if len(args.query) == 1 and args.media_type_list is None:
            setattr(args, 'media_type_list', ['audio', 'video', 'metadata'])
            only_query = args.query[0]
            setattr(args, 'query', [only_query, only_query])

        if args.result_format != 'csv' and not args.save_to_file:
            print(f'Searching {args.project_dir} for')
            for i in range(0, len(args.media_type_list)):
                print(f'  [{i}] "{args.query[i]}" in {args.media_type_list[i]}')
            if args.media_type_not_list:
                k = len(args.media_type_list)
                for i in range(k, len(args.query)):
                    print(f'  [{i}] "{args.query[i]}" not in {args.media_type_not_list[i-k]}')
            print('\n')
        search_result = process_query(search_index_list, args)
        if args.no_merge:
            show_result(search_result, args)
        else:
            merge0_search_result = merge0(search_result, args)
            show_result(merge0_search_result, args)
            if len(merge0_search_result) == 2 and args.queries_from is None:
                merge1_search_result = merge1(merge0_search_result, args)
                show_result(merge1_search_result, args)

    ## Case-2: Search queries contained in a CSV file
    elif hasattr(args, 'queries_from') and args.queries_from is not None:
        if hasattr(args, 'media_type_not_list') and args.media_type_not_list is not None:
            print(f'--queries-from flag does not support --not-in flag')
            sys.exit(0)
        start_time = time.time()
        print(f'Processing queries from {args.queries_from} ...')
        with open(args.queries_from, 'r') as f:
            query_reader = csv.reader(f, delimiter=',', quotechar='"')
            header = next(query_reader)
            query_count = 0
            for row in query_reader:
                print(f'[{query_count}] {row}')
                if len(row) != 2 and len(row) != 4:
                    print(f'Skipping query: "{row}".')
                    print(f'Each input line must be formatted as one of the follows:')
                    print(f'  - "query_id,query" (with --in flag to specify the search target')
                    print(f'  - "query1,in,query2,not_in"')
                    continue

                args_copy = copy.deepcopy(args)
                setattr(args_copy, 'query', [ row[0] ])

                if len(row) == 2:
                    if args.media_type_list is None:
                        print(f'The CSV file {args.queries_from} contains a row with two columns: {row}')
                        print(f'For such entries, the --queries-from must be followed by --in argument.')
                        sys.exit(0)
                    else:
                        setattr(args_copy, 'query_id', [ row[0] ])
                        setattr(args_copy, 'query', [ row[1] ])
                        setattr(args_copy, 'media_type_not_list', None)
                if len(row) == 4:
                    MEDIA_TYPE_LIST = ['audio', 'video', 'metadata']
                    if row[1] not in MEDIA_TYPE_LIST:
                        print(f'Skipping row with invalid "in" column: {row[1]}')
                        continue
                    if row[3] not in MEDIA_TYPE_LIST:
                        print(f'Skipping row with invalid "not_in" column: {row[3]}')
                        continue
                    setattr(args_copy, 'query_id', [ query_count ])
                    setattr(args_copy, 'query', [ row[0], row[2] ])
                    setattr(args_copy, 'media_type_list', [ row[1] ])
                    setattr(args_copy, 'media_type_not_list', [ row[3] ])
                search_result = process_query(search_index_list, args_copy)

                if args.no_merge:
                    show_result(search_result, args_copy)
                else:
                    merge0_search_result = merge0(search_result, args_copy)
                    show_result(merge0_search_result, args_copy)
                    if len(merge0_search_result) == 2 and args.queries_from is None:
                        merge1_search_result = merge1(merge0_search_result,
                                                      args_copy)
                        show_result(merge1_search_result, args_copy)
                query_count += 1
        end_time = time.time()
        elapsed = end_time - start_time
        print(f'Processed {query_count} queries in {elapsed:.2f} sec. or {elapsed/60:.2f} min.')

    ## Case-3: Search query not provided, start search console
    else:
        print('Starting WISE search console ...')
        print('Some examples queries (press Ctrl + D to exit):')
        print('  1. find cooking videos with music playing in background')
        print('     > --query "cooking" --in video --query "music" --in audio')
        print('  2. find videos showing train, show only top 3 results and export results to a file')
        print('     > --query "train" --in video --topk 3 --result-format csv --save-to-file train.csv')

        cmd_id = 0
        # Start the WISE search console Read-Evaluate-Print loop (REPL)
        while True:
            try:
                cmd = input('[%d] > ' % (cmd_id))
                args2 = parse_user_input(cmd, args)
                search_result = process_query(search_index_list, args2)
                if hasattr(args2, 'save_to_file') and getattr(args2, 'save_to_file') is not None:
                    with open(getattr(args2, 'save_to_file'), 'w') as f:
                        if hasattr(args2, 'result_format') and getattr(args2, 'result_format') == 'csv':
                            f.write(EXPORT_CSV_HEADER + '\n')
                            print(f'writing results to file {getattr(args2, "save_to_file")} ...')
                else:
                    if hasattr(args2, 'result_format') and getattr(args2, 'result_format') == 'csv':
                        print(f'{EXPORT_CSV_HEADER}')

                if args2.no_merge:
                    show_result(search_result, args2)
                else:
                    merge0_search_result = merge0(search_result, args2)
                    show_result(merge0_search_result, args2)
                    if len(merge0_search_result) == 2 and args2.queries_from is None:
                        merge1_search_result = merge1(merge0_search_result,
                                                      args2)
                        show_result(merge1_search_result, args2)
                cmd_id += 1
            except EOFError:
                print('\nBye')
                break
            except KeyboardInterrupt:
                print('\nBye')
                break
