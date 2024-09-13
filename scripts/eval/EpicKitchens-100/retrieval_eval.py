# Compute similarity score between 3843 query sentences and 9668 videos
# from the EpicKitchens-100 dataset. The query sentences and it corresponding
# ground truth is defined in the multi-instance retrieval challenge.
# For more details, see
# https://epic-kitchens.github.io/2024#challenge-action-retrieval
#
import csv
import argparse
import numpy as np
import time
import sys
import pandas as pd

from pathlib import Path

# 00:00:12.30 -> 12.30
def hhmmss_to_sec(hhmmss):
    tok = hhmmss.split(':')
    assert len(tok) == 3
    hh = int(tok[0])
    mm = int(tok[1])
    stok = tok[2].split('.')
    ss = int(stok[0])
    ms = int(stok[1])

    return hh*60*60 + mm*60 + ss + (ms/1000)

# source: https://github.com/mwray/Joint-Part-of-Speech-Embeddings/blob/main/src/evaluation/mAP.py
def calculate_mAP(sim_mat, relevancy_matrix):
    """
    Computes the mean average precision according to the following formula of
    average precision:
    \frac{\sum_{k=1}^n p(k) x rel(k)}{num_rel_docs}

    where p(k) is the precision at k, rel(k) is an indicator function
    determining whether the kth returned item is relevant or not and
    num_rel_docs is the number of relevant items to find within the search.

    The mean average precision is the mean of the average precision for each
    query item (i.e row in the matrix)

    This function takes in two parameters:
        - sim_mat: a NxM matrix which represents the similarity between two
        modalities (with modality 1 being of size N and modality 2 of size M).
        - relevancy_matrix: an NxM matrix which represents the relevancy between two
        modalities of items (with modality 1 being of size N and modality 2 of
        size M).
    """
    #Find the order of the items in modality 2 according to modality 1
    ranked_order = (-sim_mat).argsort()
    ranked_sim_mat = sim_mat[np.arange(sim_mat.shape[0])[:, None], ranked_order]
    #re-order the relevancy matrix to accommodate the proposals
    ranked_rel_mat = relevancy_matrix[np.arange(relevancy_matrix.shape[0])[:, None], ranked_order]

    #find the number of relevant items found at each k
    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    #Mask this ensuring that it is non zero if the kth term is 1 (rel(k) above)
    cumulative_rel_mat[ranked_rel_mat != 1] = 0
    #find the divisor for p(k)
    divisor = np.arange(ranked_rel_mat.shape[1]) + 1

    #find the number of relevant docs per query item
    number_rel_docs = np.sum(ranked_rel_mat==1, axis=1)

    #find the average precision per query, within np.sum finds p(k) * rel(k)
    avg_precision = np.sum(cumulative_rel_mat / divisor, axis=1) / number_rel_docs
    mAP = np.mean(avg_precision)
    return mAP

def does_segment_overlap(seg1, seg2, iou_threshold):
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
        if iou > iou_threshold:
            return True
        else:
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="retrieval_eval",
        description="Compute evaluation performance metrics defined in EpicKitchens-100 Multi-Instance Action Retrieval."
    )
    parser.add_argument(
        "--epic-video-segments",
        required=True,
        type=str,
        help="e.g. EPIC_100_retrieval_test.csv",
    )

    parser.add_argument(
        "--relevancy-mat-file",
        required=True,
        type=str,
        help="a pkl file containing the relevancy matrix",
    )

    parser.add_argument(
        "--wise-query",
        required=True,
        type=str,
        help="a CSV file containing the list of WISE search queries (e.g. EPIC_100_retrieval_test.csv)",
    )

    parser.add_argument(
        "--wise-search-results",
        required=True,
        type=str,
        help="a CSV file containing search results returned by WISE for query sentences",
    )

    parser.add_argument(
        "--iou-threshold",
        required=True,
        type=float,
        help="a CSV file containing search results returned by WISE for query sentences",
    )

    args = parser.parse_args()

    # 1. Load all search queries
    all_query_text = []
    all_query_id = []
    with open(args.wise_query, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # discard header: narration_id,narration
        for row in reader:
            query_text_id = row[0]
            query_text = row[1]
            all_query_text.append(query_text)
            all_query_id.append(query_text_id)
    print(f'loaded {len(all_query_text)} queries')

    # 2. Load all video segments and their manually annotated narration text (i.e. ground truth)
    video_segments = {}
    video_index = 0
    with open(args.epic_video_segments, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # discard header: narration_id,participant_id,video_id,narration_timestamp,start_timestamp,stop_timestamp,start_frame,stop_frame,narration, ...
        for row in reader:
            video_id = row[2]
            starttime = row[4]
            stoptime = row[5]
            video_metadata = row[8]
            if video_id not in video_segments:
                video_segments[video_id] = []
            video_segments[video_id].append({
                'video_index': video_index,
                'video_metadata': video_metadata,
                'starttime': hhmmss_to_sec(starttime),
                'stoptime': hhmmss_to_sec(stoptime)
            })
            video_index += 1
    N_video = video_index
    print(f'loaded {N_video} video segments')

    # 3. Parse WISE search result and create similarity matrix
    N_text = len(all_query_id)
    sim_mat = np.zeros((N_video, N_text), dtype=np.float32)
    print(f'Computing {N_video}x{N_text} similarity matrix')
    with open(args.wise_search_results, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader) # discard header: query,rank,filename,start_time,end_time,score
        for row in reader:
            query_id = row[0]
            query_index = all_query_id.index(query_id)
            video_filename = Path(row[2]) # P01/videos/P01_12.MP4
            video_id = str(video_filename.stem)
            starttime = float(row[3])
            stoptime = float(row[4])
            result_segment = [starttime, stoptime]
            score = row[5]
            # find the video segment that matches this retrieval result
            if video_id not in video_segments:
                continue
            for i in range(0, len(video_segments[video_id])):
                gnd_segment = [video_segments[video_id][i]['starttime'],
                               video_segments[video_id][i]['stoptime']]

                if does_segment_overlap(result_segment, gnd_segment, args.iou_threshold):
                    video_index = video_segments[video_id][i]['video_index']
                    sim_mat[video_index][query_index] = float(score)

    # 4. Compute performance metrics (e.g. mAP)

    # Note: The relevancy matrix can be generated using
    # https://github.com/mwray/Semantic-Video-Retrieval/blob/main/src/scripts/create_relevancy_matrix.py
    print(f'Loading {N_video}x{N_text} relevancy matrix')
    rel_mat = pd.read_pickle(args.relevancy_mat_file)

    # The calculate_mAP function expects each row to correspond to a query
    # see "... query item (i.e row in the matrix)" in calculate_mAP()
    map = calculate_mAP(np.transpose(sim_mat), np.transpose(rel_mat))
    print(f'mAP = {map:.3f}')
