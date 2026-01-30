#!/usr/bin/env python3

# Compute similarity score between 3843 query sentences and 9668 videos
# from the EpicKitchens-100 dataset.

import argparse
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
import json
import os
import sys
import faiss
from tqdm import tqdm
from PIL import Image
import io
import shutil

# TODO: update when src is available as a module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(project_root, "src"))
from wise_project import WiseProject

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

def get_video_vector_ids(project_dir, video_ids, start_times, stop_times, delta_time):
    internal_db_path = Path(project_dir) / "metadata" / "internal.db"
    internal_db = sqlite3.connect(internal_db_path)
    cursor = internal_db.cursor()

    video_vector_id_list = []
    media_ids = get_media_ids(project_dir, video_ids)

    # create an index to speed up search
    index_name = 'idx_vectors_covering'
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    )
    index_exists = cursor.fetchone() is not None
    if not index_exists:
        cursor.execute(
            f"CREATE INDEX {index_name} ON vectors (media_id, timestamp, modality, id)"
        )    

    # Prepare final results
    for i, video_id in enumerate(tqdm(video_ids)):
        media_id = media_ids[i]

        if media_id is not None:
            start_time = max(start_times[i] - delta_time, 0)
            stop_time = stop_times[i] + delta_time
            
            cursor.execute(
                "SELECT id FROM vectors WHERE media_id = ? AND timestamp BETWEEN ? AND ? AND modality = ?",
                (media_id, start_time, stop_time, "VIDEO"),
            )

            vector_id_list = [row[0] for row in cursor.fetchall()]
            video_vector_id_list.append(vector_id_list)
            
            #print(f"[{i}/{len(video_ids)}] video_id: {video_id}, start_time: {start_time}, stop_time: {stop_time}, vector_id_list: {video_vector_id_list[i]}")
        else:
            video_vector_id_list.append([]) # an indicator of missing video
            #print(f'missing vector for video segment {video_id} at index {i}')

    # Close DB connection
    internal_db.close()
    return video_vector_id_list

class VECTOR_MERGE_TYPE:
    MEAN = "mean"
    MEDIAN = "median"

def get_video_embeddings(project_dir, video_vector_ids, vector_merge_type):
    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()
    media_type = 'video'
    index_type = 'IndexFlatIP'
    asset_id = list(project_assets[media_type].keys())[0]
    index_dir = Path(project_assets[media_type][asset_id]['index_dir'])
    
    index_fn = index_dir / (media_type + '-' + index_type + '.faiss')
    index = faiss.read_index(index_fn.as_posix(), faiss.IO_FLAG_READ_ONLY)
    #print(f'loaded index with {index.ntotal} features of {index.d} dimensions')

    video_embeddings = np.zeros((len(video_vector_ids), index.d), dtype=np.float32)
    for video_index in range(len(video_vector_ids)):
        vector_ids = video_vector_ids[video_index]
        features = np.zeros((len(vector_ids), index.d), dtype=np.float32)
        if len(vector_ids) == 0:
            # handle missing videos
            video_embeddings[video_index] = np.zeros((1,index.d))
            continue
        for i in range(len(vector_ids)):
            index.reconstruct(vector_ids[i], features[i])
        if vector_merge_type == VECTOR_MERGE_TYPE.MEAN:
            video_embeddings[video_index] = np.mean(features, axis=0)
        else:
            if vector_merge_type == VECTOR_MERGE_TYPE.MEDIAN:
                # use feature of center frame
                if len(vector_ids) == 1:
                    video_embeddings[video_index] = features[0]
                else:
                    median_index = int((len(vector_ids) + 1) / 2)
                    video_embeddings[video_index] = features[median_index]
    
    return video_embeddings

def compute_text_embedding(project_dir, text_queries):
    from feature.feature_extractor_factory import FeatureExtractorFactory

    project = WiseProject(args.project_dir)
    project_assets = project.discover_assets()
    media_type = 'video'
    query_prefix = "This is a photo of a"
    feature_extractor_id = list(project_assets[media_type].keys())[0]

    print(f'Initialising feature extractor for {feature_extractor_id}')
    feature_extractor = FeatureExtractorFactory(feature_extractor_id)
    feature_dim = feature_extractor.output_dim
    text_embeddings = np.zeros((len(text_queries), feature_dim), dtype=np.float32)
    for i in tqdm(range(0, len(text_queries))):
        prefixed_query = f"{query_prefix} {text_queries[i].strip()}".strip()
        text_embeddings[i] = feature_extractor.extract_text_features([ prefixed_query ])
    return text_embeddings

# source: https://github.com/adrianofragomeni/MI-MM/blob/main/src/evaluation/mAP.py
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

    # for missing videos, we set relevancy_matrix entry to 0
    # therefore, some queries may end up with 0 relevant video segments
    # hence we use np.nanmean() to discard such nan entries
    mAP = np.nanmean(avg_precision)
    return avg_precision, mAP

def get_media_ids(project_dir, video_ids):
    # Database Connection
    internal_db_path = Path(project_dir) / "metadata" / "internal.db"
    internal_db = sqlite3.connect(internal_db_path)
    cursor = internal_db.cursor()

    # Prepare media paths
    media_paths = [f"{video_id.split('_')[0]}/videos/{video_id}.MP4" for video_id in video_ids] # P01/videos/P01_14.MP4

    # Fetch all media IDs in one query
    cursor.execute(
        f"SELECT id, path FROM media WHERE path IN ({','.join(['?']*len(media_paths))})", 
        media_paths
    )
    media_id_map = {path: media_id for media_id, path in cursor.fetchall()}

    # Prepare final results
    media_ids = []
    for i, video_id in enumerate(video_ids):
        media_path = media_paths[i]
        media_id = media_id_map.get(media_path)
        media_ids.append(media_id)
    internal_db.close()
    return media_ids

def visualise_retrieval_results(args, 
                                text_queries, 
                                video_ids, 
                                start_times, 
                                stop_times, 
                                video_vector_ids, 
                                similarity_scores, 
                                relevancy_matrix):
    average_precision, mAP = calculate_mAP(similarity_scores, relevancy_matrix)

    text_query_index_list = [ i for i in range(0, len(text_queries)) ]
    sorted_pairs = sorted(zip(average_precision, text_query_index_list), reverse=True)

    ranked_order = (-similarity_scores).argsort()
    ranked_sim_mat = similarity_scores[np.arange(similarity_scores.shape[0])[:, None], ranked_order]
    ranked_rel_mat = relevancy_matrix[np.arange(relevancy_matrix.shape[0])[:, None], ranked_order]
    number_rel_docs = np.sum(ranked_rel_mat==1, axis=1)
    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    cumulative_rel_mat[ranked_rel_mat != 1] = 0

    # save all thumbnails to outdir
    thumbdir_name = 'thumbs'
    thumb_dir = os.path.join(args.out_dir, thumbdir_name)
    thumbnail_filename_list = []
    for i in range(len(video_ids)):
        start_time_str = f"{start_times[i]:.3f}".replace('.', '-')
        stop_time_str = f"{stop_times[i]:.3f}".replace('.', '-')
        thumb_filename = f"{i:05d}_{video_ids[i]}_{start_time_str}_{stop_time_str}.jpg"
        thumb_path = os.path.join(thumb_dir, thumb_filename)
        thumbnail_filename_list.append(thumb_filename)
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
        # retrieve thumbnail for each video segment
        delta = 0.5 # for video segments with less than 0.5 sec duration
        video_thumbnails= get_video_thumbnails(args.project_dir, video_ids, start_times, stop_times, delta)
        print(f'retrieved {len(video_thumbnails)} thumbnails')
    
        for i in range(len(video_thumbnails)):
            if len(video_thumbnails[i]) == 0:
                continue
            thumb_image = Image.open(io.BytesIO(video_thumbnails[i]))
            thumb_image.save(thumbnail_filename_list[i])
        print(f'saved thumbnails to {thumb_dir}')
    else:
        print(f'using existing thumbnails from {thumb_dir}')
    
    media_ids = get_media_ids(args.project_dir, video_ids)
    payload = {
        'video_ids': video_ids,
        'media_ids': media_ids,
        'start_times': start_times,
        'stop_times': stop_times,
        'video_narration': video_narration,
        'text_queries': text_queries,
        'number_rel_docs': number_rel_docs,
        'mAP': mAP,
        'sorted_by_ap':[]
    }
    
    MAX_RESULT = 20
    for ap, qi in sorted_pairs:
        query_results = {
            'qindex': qi,
            'ap': f"{ap:.2f}",
            'nrel':number_rel_docs[qi],
            'results': []
        }
        relevant_found_sofar = 0
        for k in range(0, len(video_ids)):
            vi = ranked_order[qi,k]
            if ranked_rel_mat[qi,k] == 1.0:
                relevant_found_sofar += 1
            query_results['results'].append({
                'vi': vi,
                'd': f"{ranked_sim_mat[qi,k]:.2f}",
                'rel': f"{ranked_rel_mat[qi,k]:.1f}",
            })
            if relevant_found_sofar == number_rel_docs[qi]:
                break
            if k >= MAX_RESULT:
                break
        
        payload['sorted_by_ap'].append(query_results)
    if args.binary_relevancy:
        suffix = 'binary_relevancy'
    else:
        suffix = 'noun_verb_relevancy'
    payload_filename = os.path.join(args.out_dir, 'retrieval-data-' + suffix + '.js')
    with open(payload_filename, 'w') as f:
        json_str = json.dumps(payload, default=custom_serializer, separators=(',', ':'), ensure_ascii=False)
        f.write(f'var payload_{suffix}={json_str};')

    # copy html based visualisation files
    html_filename = os.path.join(args.out_dir, 'index.html')
    shutil.copyfile('./index.html', html_filename)
    print(f'Open {html_filename} in a browser to visualise the retrieval results')

def custom_serializer(obj):
    if isinstance(obj, np.integer):  # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):  # Convert numpy float to Python float
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Convert numpy arrays to lists
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} is not serializable")

def get_video_thumbnails(project_dir, video_ids, start_times, stop_times, delta_time):
    # Database Connection
    internal_db_path = Path(project_dir) / "metadata" / "internal.db"
    internal_db = sqlite3.connect(internal_db_path)
    cursor = internal_db.cursor()

    # Prepare media paths
    media_paths = [f"{video_id.split('_')[0]}/videos/{video_id}.MP4" for video_id in video_ids] # P01/videos/P01_14.MP4
    # Fetch all media IDs in one query
    cursor.execute(
        f"SELECT id, path FROM media WHERE path IN ({','.join(['?']*len(media_paths))})", 
        media_paths
    )
    media_id_map = {path: media_id for media_id, path in cursor.fetchall()}
    internal_db.close()

    thumbs_db_path = Path(project_dir) / "thumbs.db"
    thumbs_db = sqlite3.connect(thumbs_db_path)
    cursor = thumbs_db.cursor()
    video_thumbnails = []
    for i, video_id in enumerate(tqdm(video_ids)):
        media_path = media_paths[i]
        media_id = media_id_map.get(media_path)

        if media_id is not None:
            start_time = max(start_times[i] - delta_time, 0)
            stop_time = stop_times[i] + delta_time

            cursor.execute(
                "SELECT content FROM thumbnails WHERE media_id = ? AND timestamp BETWEEN ? AND ?",
                (media_id, start_time, stop_time),
            )

            thumbnail_list = [row[0] for row in cursor.fetchall()]
            if len(thumbnail_list) == 1:
                video_thumbnails.append(thumbnail_list[0])
            else:
                mid_thumb_index = int((len(thumbnail_list) + 1) / 2)
                video_thumbnails.append(thumbnail_list[mid_thumb_index])
        else:
            video_thumbnails.append('') # an indicator of missing video
    thumbs_db.close()
    return video_thumbnails

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="text-to-video-retrieval-eval",
        description="Evaluate text-to-video retrieval performance of WISE using the EpicKitchens-100 video dataset"
    )
    parser.add_argument(
        "--path-epic-annotations",
        required=True,
        type=str,
        help="folder containing EPIC_100_retrieval_test.csv and EPIC_100_retrieval_test_sentence.csv",
    )

    parser.add_argument(
        "--project-dir",
        required=True,
        type=str,
        help="Path to the WISE project directory",
    )

    parser.add_argument(
        "--relevancy-matrix",
        required=True,
        type=str,
        help="Path to the relevancy matrix",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Store computed data in this folder",
    )

    parser.add_argument(
        "--binary-relevancy",
        action='store_true',
        required=False,
        help="relevancy is either 1 (i.e. relevant) or 0 (irrelevant)",
    )

    parser.add_argument(
        "--export-visualisation",
        action='store_true',
        required=False,
        help="relevancy is either 1 (i.e. relevant) or 0 (irrelevant)",
    )

    parser.add_argument(
        "--show-query-ap",
        action='store_true',
        required=False,
        help="relevancy is either 1 (i.e. relevant) or 0 (irrelevant)",
    )

    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # CSV format: narration_id,participant_id,video_id,narration_timestamp,start_timestamp,stop_timestamp,start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
    video_segments = pd.read_csv( Path(args.path_epic_annotations) / "EPIC_100_retrieval_test.csv")

    video_narration_id = video_segments.values[:,0]
    video_ids = video_segments.values[:, 2] # P01_10
    start_times = [hhmmss_to_sec(t) for t in video_segments.values[:, 4]]
    stop_times = [hhmmss_to_sec(t) for t in video_segments.values[:, 5]]
    video_narration = video_segments.values[:,8]

    # The following four videos are missing in WISE project as ffmpeg
    # cannot convert or read these videos in the EpicKitchens-100 dataset
    # P29_01.MP4, P29_05.MP4, P30_05.MP4, P30_08.MP4
    #missing_video_id = ['P29_01', 'P29_05', 'P30_05', 'P30_08']
    
    # create a 8773x768 matrix containing feature vectors for 8773 video segments
    video_vector_ids_filename = os.path.join(args.out_dir, 'video_vector_ids.json')
    if not os.path.exists(video_vector_ids_filename):
        # Some video segments are very short. For example:
        # P28_24, start=46.034, stop=46.091
        # P28_25, start=138.011, stop=138.07
        # Therefore, we add a delta to the start/stop time
        # to capture the video frames sampled at 2fps
        delta_time = 0.5 # in sec
        video_vector_ids = get_video_vector_ids(args.project_dir, video_ids, start_times, stop_times, delta_time)
        with open(video_vector_ids_filename, 'w') as f:
            json.dump(video_vector_ids, f)
    
    with open(video_vector_ids_filename, 'r') as f:
        video_vector_ids = json.load(f)
    
    assert len(video_ids) == len(video_vector_ids)
    print(f'loaded vector_ids for {len(video_ids)} segments')

    # find index of videos with missing features
    missing_video_index = []
    for i in range(len(video_vector_ids)):
        if len(video_vector_ids[i]) == 0:
            missing_video_index.append(i)
    print(f'found {len(missing_video_index)} missing video segments')

    vector_merge_type = VECTOR_MERGE_TYPE.MEAN
    #vector_merge_type = VECTOR_MERGE_TYPE.MEDIAN
    video_embedding_filename = os.path.join(args.out_dir, 'video_embedding_' + vector_merge_type + '.npy')
    if not os.path.exists(video_embedding_filename):
        video_embeddings = get_video_embeddings(args.project_dir, video_vector_ids, vector_merge_type)
        with open(video_embedding_filename, 'wb') as f:
            np.save(video_embedding_filename, video_embeddings)
    else:
        with open(video_embedding_filename, 'rb') as f:
            video_embeddings = np.load(video_embedding_filename)
    print(f'loaded video embeddings of shape {video_embeddings.shape}')
    
    # CSV format: narration_id,narration
    query_data = pd.read_csv( Path(args.path_epic_annotations) / "EPIC_100_retrieval_test_sentence.csv")
    text_query_ids = query_data.values[:, 0]
    text_queries = query_data.values[:, 1]
    
    # compute text embeddings
    text_embedding_filename = os.path.join(args.out_dir, 'text_embedding.npy')
    if not os.path.exists(text_embedding_filename):
        text_embeddings = compute_text_embedding(args.project_dir, text_queries)
        with open(text_embedding_filename, 'wb') as f:
            np.save(text_embedding_filename, text_embeddings)
    else:
        with open(text_embedding_filename, 'rb') as f:
            text_embeddings = np.load(text_embedding_filename)
    print(f'loaded text embeddings of shape {text_embeddings.shape}')
    
    # compute similarity score between 3842 query sentences and 9668 videos segments
    # video_embeddings = 9668 x 1024
    # text_embeddings  = 3842 x 1024
    # similarity_scores is a 3842x9668 matrix (S) with scores between 0 and 1 
    # with S[i][j] representing the similarity between the ith video and the jth caption.
    # For references, see:
    # - https://github.com/adrianofragomeni/MI-MM/blob/main/src/testing.py
    # - https://github.com/adrianofragomeni/MI-MM/blob/main/src/evaluation/create_submission.py
    # - https://github.com/adrianofragomeni/MI-MM/blob/main/src/evaluation/mAP.py
    similarity_scores = np.matmul(text_embeddings, video_embeddings.T)
    print(f'computed similarity score of shape {similarity_scores.shape}')
    
    # load relevancy matrix created using
    # https://github.com/mwray/Joint-Part-of-Speech-Embeddings/blob/main/src/scripts/create_relevancy_files.py
    relevancy_matrix = pd.read_pickle(args.relevancy_matrix)
    relevancy_matrix = relevancy_matrix.T
    print(f'loaded relevancy matrix of shape {relevancy_matrix.shape}')
    
    # update the relevancy matrix to take into account the missing video segments
    print(f'removing {len(missing_video_index)} missing video segments from relevancy matrix')
    for video_index in missing_video_index:
        relevancy_matrix[:, video_index] = 0
    
    if args.binary_relevancy:
        relevancy_matrix[relevancy_matrix != 1 ] = 0
        print(f'forcing relevancy matrix to be a binary indicator')

    #average_precision, mAP = calculate_mAP(similarity_scores.T, relevancy_matrix.T) # for video-to-text retrieval performance
    average_precision, mAP = calculate_mAP(similarity_scores, relevancy_matrix)  # for text-to-video retrieval performance
    print(f'mAP = {mAP}')
    text_query_index_list = [ i for i in range(0, len(text_queries)) ]
    sorted_pairs = sorted(zip(average_precision, text_query_index_list), reverse=True)
    if args.show_query_ap:
        for ap, qi in sorted_pairs:
            text_query = text_queries[qi]
            print(f'{text_query} : {ap:.2f}')

    if args.export_visualisation:
        visualise_retrieval_results(args, 
                                    text_queries, 
                                    video_ids, 
                                    start_times, 
                                    stop_times, 
                                    video_vector_ids, 
                                    similarity_scores, 
                                    relevancy_matrix)
