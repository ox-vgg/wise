## Compute image retrieval performance metrics (e.g. Recall@K) based on 
## search results obtained from a reference search engine (e.g. based on naive 
## or exhaustive search) and similar search results obtained from another
## search engine which uses search index based on approximate nearest neighbour search.
##
## Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date  : 2023-11-03

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

        # see Jegou H, Douze M, Schmid C. Product quantization for nearest neighbor search. (2010)
        # for the definition of retrieval performance metric Recall@K
        MAX_K = 100
        k_list = []
        recall_at_k_list = []
        for k in range(1, MAX_K+1):
            queries_in_which_nn_in_first_k = 0
            average_search_time = 0
            for search_query in ref_perf['search_queries']:
                ref_filename_list = set()
                for i in range(0, MAX_K):
                    ref_filename_list.add(ref_perf['search_queries'][search_query][i]['filename'])

                eval_filename_list = set()
                for i in range(0, k):
                    eval_filename_list.add(eval_perf['search_queries'][search_query][i]['filename'])
                tp = len( set.intersection(eval_filename_list, ref_filename_list) )
                if tp:
                    queries_in_which_nn_in_first_k += 1
                average_search_time += eval_perf['response_time_in_seconds'][search_query]
            recall_at_k_list.append( queries_in_which_nn_in_first_k / len(ref_perf['search_queries']) )
            k_list.append(k)
        average_search_time = average_search_time / len(ref_perf['search_queries'])
        index_desc = eval_perf['wise_server']['description']
        first_dash_index = index_desc.find('-', 0)
        index_name = index_desc[0:first_dash_index]
        index_param = index_desc[first_dash_index+1:]
        index_param_tok = index_param.split('-')
        if len(index_param_tok) == 3:
            m = index_param_tok[0].split('m')[1]
            nbits = index_param_tok[1].split('nbits')[1]
            nlist = index_param_tok[2].split('nlist')[1]
            print('| %s | {%s, %s, %s} | ? | %.3f | %.3f | %.3f |' % (index_name, m, nbits, nlist, recall_at_k_list[99], recall_at_k_list[19], average_search_time))
        else:
            print('| %s | %s | ? | %.3f | %.3f | %.3f sec |' % (index_name, index_param, recall_at_k_list[99], recall_at_k_list[19], average_search_time))

    ## Compute the average search time of reference search index
    ref_average_search_time = 0.0
    for search_query in ref_perf['search_queries']:
        ref_average_search_time += ref_perf['response_time_in_seconds'][search_query]
    ref_average_search_time = ref_average_search_time / len(ref_perf['search_queries'])
    print('| Naive | IndexFlatIP | 158 GB | 1.0 | 1.0 | %.1f |' % (ref_average_search_time))

if __name__ == '__main__':
    main()

