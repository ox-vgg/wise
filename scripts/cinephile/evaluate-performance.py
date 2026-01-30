#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import argparse
import json
import requests
import os
import time

def evaluate_performance(wise_url, queries, strategy_func, verbose=False):
    """
    Evaluates performance for a given query modification strategy.
    Returns the overall recall. If verbose is True, prints detailed logs and a recall-based summary.
    """
    total_queries = len(queries)
    overall_correct = 0
    overall_retrieved = 0
    overall_expected = 0

    query_recall_data = []

    if verbose:
        print("\n--- Detailed Log for Best Performing Strategy ---")

    for i, item in enumerate(queries):
        prompt = item['prompt']
        expected_files = set(item['expected_files'])
        
        text_query = strategy_func(prompt)

        search_params = {
            'start': 0,
            'end': 1000,
            'thumbs': 0,
            'search_in': 'video',
            'feature_extractor_id': 'mlfoundations/open_clip/ViT-L-16-SigLIP2-512/webli',
            'text_queries': text_query,
            'add_prefix': 0
        }

        search_url = f"{wise_url}/search"
        
        try:
            if verbose:
                print(f"Query {i+1}/{total_queries}: {text_query}")
                req = requests.Request('POST', search_url, params=search_params)
                prepared_req = req.prepare()
                print(f"  Search URL: {prepared_req.url}")

            response = requests.post(search_url, params=search_params)
            response.raise_for_status()
            search_results = response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if verbose:
                print(f"  Error during API call: {e}")
            continue

        # Get ordered list of all retrieved filenames and their timestamps to determine rank
        retrieved_files_ordered = []
        if 'video_results' in search_results and search_results['video_results'] and 'merged_windows' in search_results['video_results']:
            videos_map = search_results['video_results'].get('videos', {})
            for window in search_results['video_results']['merged_windows']:
                media_id = str(window.get('media_id'))
                start_timestamp = window.get('ts')
                end_timestamp = window.get('te')
                if media_id and media_id in videos_map:
                    filename = videos_map[media_id].get('filename')
                    if filename:
                        retrieved_files_ordered.append(
                            {'filename': os.path.basename(filename), 'start_timestamp': start_timestamp, 'end_timestamp': end_timestamp}
                        )

        # Get unique retrieved files for metrics
        retrieved_files_unique = set(item['filename'] for item in retrieved_files_ordered)
        correct_retrieved_files = expected_files.intersection(retrieved_files_unique)

        num_correct = len(correct_retrieved_files)
        num_retrieved = len(retrieved_files_unique)
        num_expected = len(expected_files)

        recall_for_query = num_correct / num_expected if num_expected > 0 else 0

        if verbose:
            precision = num_correct / num_retrieved if num_retrieved > 0 else 0
            print(f"  Retrieved: {num_retrieved}, Expected: {num_expected}, Correct: {num_correct}")
            print(f"  Precision: {precision:.2f}, Recall: {recall_for_query:.2f}")

            if num_correct > 0:
                correct_files_with_details = {}
                for rank, item in enumerate(retrieved_files_ordered):
                    filename = item['filename']
                    if filename in correct_retrieved_files and filename not in correct_files_with_details:
                        correct_files_with_details[filename] = {
                            'rank': rank + 1,
                            'start_timestamp': item['start_timestamp'],
                            'end_timestamp': item['end_timestamp']
                        }

                sorted_correct_files = sorted(correct_files_with_details.items(), key=lambda item: item[1]['rank'])
                found_files_list = [f"[{details['rank']}] {fname} ({details['start_timestamp']:.2f}s - {details['end_timestamp']:.2f}s)" for fname, details in sorted_correct_files]
                found_files_str = "\n    - ".join(found_files_list)
                print(f"  Found correct files:\n    - {found_files_str}")

            missed_files = expected_files - retrieved_files_unique
            if missed_files:
                print(f"  Missed files: {sorted(list(missed_files))}")

        query_recall_data.append({'query': text_query, 'recall': recall_for_query})

        overall_correct += num_correct
        overall_retrieved += num_retrieved
        overall_expected += num_expected

    if verbose:
        print("\n--- Queries Grouped by Recall ---")
        recall_groups = {
            "Recall == 1.0": [],
            "0.7 <= Recall < 1.0": [],
            "0.0 <= Recall < 0.7": [],
            "Recall == 0.0": [],
        }

        for item in query_recall_data:
            r = item['recall']
            q = item['query']
            if r == 1.0:
                recall_groups["Recall == 1.0"].append(q)
            elif r >= 0.7:
                recall_groups["0.7 <= Recall < 1.0"].append(q)
            elif r >= 0.0:
                recall_groups["0.0 <= Recall < 0.7"].append(q)
            else:
                recall_groups["Recall == 0.0"].append(q)

        for group_name, queries_in_group in recall_groups.items():
            print(f"\n{group_name} ({len(queries_in_group)} queries):")
            if queries_in_group:
                for q in sorted(queries_in_group):
                    print(f"  - {q}")
            else:
                print("  (None)")

    overall_recall = overall_correct / overall_expected if overall_expected > 0 else 0
    return overall_recall

def main():
    parser = argparse.ArgumentParser(description="Evaluate WISE search performance with multiple strategies.")
    parser.add_argument("--wise-url", required=True, help="WISE search engine URL")
    parser.add_argument("--ground-truth-url", required=True, help="Ground truth JSON file URL")
    args = parser.parse_args()

    # 1. Fetch ground truth
    print(f"Fetching ground truth from {args.ground_truth_url}")
    try:
        response = requests.get(args.ground_truth_url)
        response.raise_for_status()
        ground_truth_data = response.json()
        queries = ground_truth_data.get('queries')
        if queries is None:
            print("Error: 'queries' key not found in ground truth JSON.")
            return
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error fetching or parsing ground truth: {e}")
        return

    # 2. Define strategies
    strategies = {
        "Find videos -> Old photos": lambda p: p.replace("Find videos", "Old photos").rstrip('.'),
        "Find videos -> Photo": lambda p: p.replace("Find videos", "Photo").rstrip('.'),
        'Find videos -> ""': lambda p: p.replace("Find videos", "").strip().rstrip('.'),
        "Find videos -> German and Dutch archival photos": lambda p: p.replace("Find videos", "German and Dutch archival photos").rstrip('.'),
        "None (i.e. original query was used as it is)": lambda p: p.rstrip('.'),
        "videos -> photos": lambda p: p.replace("videos", "photos").rstrip('.'),
        "Find videos -> Old video frames": lambda p: p.replace("Find videos", "Old video frames").rstrip('.'),
        "videos -> images": lambda p: p.replace("videos", "images").rstrip('.'),
    }

    results = []
    
    # 3. Evaluate each strategy silently
    print("\nEvaluating strategies...")
    for desc, func in strategies.items():
        start_time = time.time()
        recall = evaluate_performance(args.wise_url, queries, func, verbose=False)
        end_time = time.time()
        duration = end_time - start_time
        print(f"  - Running strategy: \"{desc}\" (Recall: {recall:.2f}, took {duration:.2f}s)")
        results.append({"top-k": 1000, "description": desc, "recall": recall, "func": func})

    # 4. Sort results to find the best one
    results.sort(key=lambda x: x["recall"], reverse=True)
    
    # 5. Run the best strategy again with verbose logging
    if results:
        best_strategy = results[0]
        print(f"\nBest strategy is \"{best_strategy['description']}\" with recall {best_strategy['recall']:.2f}. Rerunning with detailed logs.")
        evaluate_performance(args.wise_url, queries, best_strategy['func'], verbose=True)

    # 6. Print the final summary table
    print("|-------+-------------------------------------------------+--------|")
    print("| top-k | Update to the original query                    | Recall |")
    print("|-------+-------------------------------------------------+--------|")
    for res in results:
        print(f"| {res['top-k']:<5} | {res['description']:<47} | {res['recall']:.2f}   |")
    print("|-------+-------------------------------------------------+--------|")

if __name__ == "__main__":
    main()
