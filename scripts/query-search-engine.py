# Export search results obtained from exhaustive search (i.e. faiss
# IndexFlatIP) by the WISE Image Search Engine (WISE).
#
# Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
# Date  : 2023-05-09

import json
import uuid
import time
from datetime import datetime
import argparse
import os
import csv

import hashlib
from urllib.parse import urlencode, quote_plus, quote
from urllib.request import urlopen
import requests
import http.client

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f') # save float with only 3 decimal places

def search_wise(base_url, query, result_count=50):
    query = {
        'start':0,
        'end':result_count,
        'q': query,
        'thumbs':0
    }
    query_url = base_url + 'search?' + urlencode(query, quote_via=quote_plus)
    http_response = requests.get(query_url)
    search_result = json.loads(http_response.text)
    search_result['elapsed_seconds'] = http_response.elapsed.total_seconds()
    return search_result

def fetch_wise_info(base_url):
    info_url = base_url + 'info'
    http_response = requests.get(info_url)
    info = json.loads(http_response.text)
    return info

def main():
    parser = argparse.ArgumentParser(description="Create manual annotation dataset to benchmark image retrieval performance")
    parser.add_argument("--search-queries-fn",
                        required=True,
                        type=str,
                        help="a text file containing all the search queries (one per line)")
    parser.add_argument("--search-queries-count",
                        required=False,
                        type=int,
                        default=None,
                        help="only use the first N search queries in the --search-queries-fn file; use all if missing")
    parser.add_argument("--wise-server-url",
                        required=True,
                        type=str,
                        help="URL of WISE Image Search Engine (WISE) server (e.g. https://meru.robots.ox.ac.uk/wikimedia/)")
    parser.add_argument("--wise-server-description",
                        required=True,
                        type=str,
                        help="description text that will be added to saved results (for future reference)")
    parser.add_argument("--save-results-to",
                        required=True,
                        type=str,
                        help="save results as a JSON file")
    parser.add_argument("--results-to-return",
                        required=False,
                        type=int,
                        default=100,
                        help="the number of search results to be retain")

    args = parser.parse_args()
    
    wise_server_info = fetch_wise_info(args.wise_server_url)
    now_timestamp = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z'))
    perf_result = {
        'wise_server': {
            'url': args.wise_server_url,
            'info': wise_server_info,
            'description': args.wise_server_description,
            'date': now_timestamp,
            'last_updated': now_timestamp
        },
        'search_queries': {},
        'response_time_in_seconds':{}
    }
    if os.path.exists(args.save_results_to):
        with open(args.save_results_to, 'r') as f:
            perf_result = json.load(f)

    if 'description' not in perf_result['wise_server']:
        perf_result['wise_server']['description'] = args.wise_server_description

    search_query_list = []
    search_query_count = 0
    with open(args.search_queries_fn, 'r') as f:
        for search_query in f.readlines():
            search_query_list.append(search_query.rstrip())
            search_query_count += 1
            if args.search_queries_count is not None and search_query_count == args.search_queries_count:
                break

    for qi in range(0, len(search_query_list)):
        search_query = search_query_list[qi]
        search_query_santized = search_query.replace(' ', '-')
        print('%s' % (search_query), end='')

        if search_query in perf_result['search_queries']:
            print(' [SKIPPED]')
            continue
        else:
            perf_result['search_queries'][search_query] = []

        ## WISE
        wise_server_response = search_wise(args.wise_server_url, search_query, args.results_to_return)
        wise_result = wise_server_response[search_query]
        for ri in range(0, len(wise_result)):
            img_link = wise_result[ri]['link'];
            img_link_tok = img_link.split('/');
            img_filename = img_link_tok[ len(img_link_tok) - 2 ];
            wikimedia_file_page = 'https://commons.wikimedia.org/wiki/File:' + img_filename
            perf_result['search_queries'][search_query].append( {
                'filename':img_filename,
                'distance':wise_result[ri]['distance'],
            })

        perf_result['response_time_in_seconds'][search_query] = wise_server_response['elapsed_seconds']
        perf_result['wise_server']['last_updated'] = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z'))
        with open(args.save_results_to, 'w') as f:
            json.dump(perf_result, f)
        print(' [completed in %.3f sec.]' % (wise_server_response['elapsed_seconds']))
if __name__ == '__main__':
    main()

