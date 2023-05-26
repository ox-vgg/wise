# Evaluate the performance of a search engine
#
# Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
# Date  : 2023-05-24
## Compute image retrieval performance metrics (e.g. Precision@K) based on 
## search results obtained from a reference search engine (e.g. based on naive 
## or exhaustive search) and similar search results obtained from another
## search engine which uses search index based on approximate nearest neighbour search.
##
## Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date  : 2023-05-26

import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Create manual annotation dataset to benchmark image retrieval performance")
    parser.add_argument("--perf-data-dir",
                        required=True,
                        type=str,
                        help="reference search engine performance (e.g. obtained from naive approach to nearest neighbour search)")
    parser.add_argument("--ref-perf-filename",
                        required=False,
                        type=str,
                        default=None,
                        help="search results from search engine whose performance has to be evaluated (e.g. HNSW search index)")
    parser.add_argument("--out-plot-fn",
                        required=False,
                        type=str,
                        help="save precision and recall curve to this file")
    args = parser.parse_args()
    assert os.path.isdir(args.perf_data_dir)

    ref_perf_fn = os.path.join(args.perf_data_dir, args.ref_perf_filename)
    assert os.path.isfile(ref_perf_fn)

    with open( ref_perf_fn, 'r') as f:
        ref_perf = json.load(f)

    perf_fn_list = os.listdir(args.perf_data_dir)
    for perf_fn in perf_fn_list:
        if perf_fn == args.ref_perf_filename:
            continue
        with open(os.path.join(args.perf_data_dir, perf_fn), 'r') as f:
            eval_perf = json.load(f)

        ## sanity check
        for search_query in ref_perf['search_queries']:
            if search_query not in eval_perf['search_queries']:
                print('Warning: the eval and perf files mis-match in their search queries')

        # see https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
        MAX_K = 100
        mean_average_precision = 0.0
        average_search_time = 0.0
        for search_query in ref_perf['search_queries']:
            average_precision = 0
            for k in range(1, MAX_K+1):
                eval_filename_list = set()
                ref_filename_list = set()
                for i in range(0, k):
                    eval_filename_list.add(eval_perf['search_queries'][search_query][i]['filename'])
                    ref_filename_list.add(ref_perf['search_queries'][search_query][i]['filename'])
                tp = len( set.intersection(eval_filename_list, ref_filename_list) )
                Pk = tp / k
                eval_filename_k = eval_perf['search_queries'][search_query][k-1]['filename']
                ref_filename_k = ref_perf['search_queries'][search_query][k-1]['filename']

                if eval_filename_k == ref_filename_k:
                    average_precision += (tp / k)
            average_precision = average_precision / MAX_K
            mean_average_precision += average_precision
            average_search_time += eval_perf['response_time_in_seconds'][search_query]
        mean_average_precision = mean_average_precision / len(ref_perf['search_queries'])
        average_search_time = average_search_time / len(ref_perf['search_queries'])
        print('%s : MAP=%.3f, average search time=%.3f sec' % (eval_perf['wise_server']['description'], mean_average_precision, average_search_time))

    ## Compute the average search time of reference search index
    average_search_time = 0.0
    for search_query in ref_perf['search_queries']:
        average_search_time += ref_perf['response_time_in_seconds'][search_query]
    average_search_time = average_search_time / len(ref_perf['search_queries'])
    print('%s : MAP=1.0, average search time=%.3f sec' % (ref_perf['wise_server']['description'], average_search_time))

if __name__ == '__main__':
    main()

