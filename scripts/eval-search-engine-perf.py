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
    parser.add_argument("--index-data-dir",
                        required=False,
                        type=str,
                        help="search index file size is obtained from this folder")
    args = parser.parse_args()
    assert os.path.isdir(args.perf_data_dir)

    ref_perf_fn = os.path.join(args.perf_data_dir, args.ref_perf_filename)
    assert os.path.isfile(ref_perf_fn)

    assert os.path.isdir(args.index_data_dir)

    with open( ref_perf_fn, 'r') as f:
        ref_perf = json.load(f)

    perf_fn_list = os.listdir(args.perf_data_dir)
    for perf_fn in perf_fn_list:
        if perf_fn == args.ref_perf_filename:
            continue
        '''
        if perf_fn != 'IndexIVFPQ-m256-nbits8-nlist65536.json':
            continue
        '''

        index_size = 0
        index_size_str = '?'
        index_fn = os.path.join(args.index_data_dir, perf_fn.replace('.json', '.faiss'))
        if os.path.exists(index_fn):
            index_size_mb = os.path.getsize(index_fn) / (1024*1024)
            if(index_size_mb < 1024):
                index_size_str = '%dM' % (index_size_mb)
            else:
                index_size_str = '%.1fG' % (index_size_mb/1024)
        else:
            index_size_str = '?'
        with open(os.path.join(args.perf_data_dir, perf_fn), 'r') as f:
            eval_perf = json.load(f)

        ## sanity check
        for search_query in ref_perf['search_queries']:
            if search_query not in eval_perf['search_queries']:
                print('Warning: the eval and perf files mis-match in their search queries')

        # We compute two types of recall values (Recall0 and Recall1) defined as follows
        #
        # Recall0@K   = ( ref[0:K-1] INTERSECT ann[0:K-1] ) / K
        # Recall1@N,K = ( ref[0:N-1] INTERSECT ann[0:K-1] ) / N
        #
        # where,
        # ref[0:N-1] : denote the top-N results from exhaustive search (i.e. compare against all)
        # ann[0:K-1] : denote the top-K results from approximate nearest neighbour (ANN) search
        # N          : size of result set obtained from exhaustive search
        # K          : size of result set obtained from ANN search
        #
        # Note:
        # - Recall0 : is a more stricter measure of recall (lower bound of recall value)
        # - Recall1 : is a more lenient measure of recall (upper bound of recall value)
        # - The paper [1] uses Recall1@N=1,K metric
        #
        # References:
        # [1] Jegou H, Douze M, Schmid C. Product quantization for nearest neighbor search. (2010)
        N_value_list = [20, 100]
        K_value_list = [30, 100]

        recall0 = {}
        recall1 = {}
        search_query_count = len(ref_perf['search_queries'])
        for N in N_value_list:
            if N not in recall0:
                recall0[N] = {}
            if N not in recall1:
                recall1[N] = {}
            for K in K_value_list:
                recall0[N][K] = 0.0
                if K >= N:
                    recall1[N][K] = 0.0
                for search_query in ref_perf['search_queries']:
                    ref_N = set()
                    for i in range(0, N):
                        ref_N.add(ref_perf['search_queries'][search_query][i]['filename'])
                    ref_K = set()
                    ann_K = set()
                    for i in range(0, K):
                        ref_K.add(ref_perf['search_queries'][search_query][i]['filename'])
                        ann_K.add(eval_perf['search_queries'][search_query][i]['filename'])
                    recall0[N][K] += len(set.intersection(ref_K, ann_K)) / K
                    if K >= N:
                        recall1[N][K] += len(set.intersection(ref_N, ann_K)) / N
                recall0[N][K] = recall0[N][K] / search_query_count
                if K >= N:
                    recall1[N][K] = recall1[N][K] / search_query_count
        average_search_time = 0.0
        for search_query in ref_perf['search_queries']:
            average_search_time += eval_perf['response_time_in_seconds'][search_query]
        average_search_time = average_search_time / search_query_count

        index_desc = eval_perf['wise_server']['description']
        first_dash_index = index_desc.find('-', 0)
        index_name = index_desc[0:first_dash_index]
        if index_name == 'IndexIVFPQ':
            index_name = 'IVF+PQ'
        if index_name == 'IndexIVFFlat':
            index_name = 'IVF'

        index_param = index_desc[first_dash_index+1:]
        index_param_tok = index_param.split('-')
        if len(index_param_tok) == 3:
            m = index_param_tok[0].split('m')[1]
            nbits = index_param_tok[1].split('nbits')[1]
            nlist = index_param_tok[2].split('nlist')[1]
            N = 20
            print('| %s | {%s, %s, %s} | %s | %.3f | %.3f | %.3f | %.3f | %.3f |' % (index_name, m, nbits, nlist, index_size_str, recall0[N][30], recall0[N][100], recall1[N][30], recall1[N][100], average_search_time))
        else:
            N = 20
            print('| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |' % (index_name, index_param, index_size_str, recall0[N][30], recall0[N][100], recall1[N][30], recall1[N][100], average_search_time))

    ## Compute the average search time of reference search index
    ref_average_search_time = 0.0
    for search_query in ref_perf['search_queries']:
        ref_average_search_time += ref_perf['response_time_in_seconds'][search_query]
    ref_average_search_time = ref_average_search_time / len(ref_perf['search_queries'])
    print('| Naive | IndexFlatIP | 158G | 1.0 | 1.0 | 1.0 | 1.0 | %.1f |' % (ref_average_search_time))

if __name__ == '__main__':
    main()

