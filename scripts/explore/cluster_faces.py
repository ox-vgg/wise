import argparse
from pathlib import Path
import logging
import numpy as np
import faiss
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import sort_graph_by_row_values
import collections

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.wise_project import WiseProject
from src.data_models import ModalityType
from scripts.explore.db import init_explore_db
from scripts.explore.models import Facet, Cluster, Assignment, ClusterStatus, KnownFaceCluster
from scripts.explore.config import MIN_FACE_SIZE
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cluster_faces(project_dir: Path, feature_extractor_id: str, k_neighbors: int, fallback_similarity_threshold: float):
    logger.info(f"Loading WISE project from {project_dir}")
    project = WiseProject(project_dir)
    
    logger.info("Initializing explore.db")
    engine, SessionLocal = init_explore_db(project_dir)
    
    with SessionLocal() as session:
        facet = session.query(Facet).filter_by(name="Face", feature_extractor_id=feature_extractor_id).first()
        if not facet:
            facet = Facet(name="Face", feature_extractor_id=feature_extractor_id)
            session.add(facet)
            session.commit()
            session.refresh(facet)
        facet_id = facet.id
        
        logger.info("Clearing old draft clusters...")
        draft_cluster_ids = session.query(Cluster.id).filter_by(facet_id=facet_id, status=ClusterStatus.draft).all()
        if draft_cluster_ids:
            session.query(Assignment).filter(Assignment.cluster_id.in_([c[0] for c in draft_cluster_ids])).delete(synchronize_session=False)
            session.query(Cluster).filter(Cluster.id.in_([c[0] for c in draft_cluster_ids])).delete(synchronize_session=False)
            session.commit()

        logger.info("Loading known face clusters from database...")
        known_clusters_data = session.query(KnownFaceCluster).all()
        
    known_vector_to_cluster = {kc.vector_id: kc.cluster_id for kc in known_clusters_data}
    cluster_to_known_vectors = collections.defaultdict(list)
    for kc in known_clusters_data:
        cluster_to_known_vectors[kc.cluster_id].append(kc.vector_id)

    logger.info(f"Loaded {len(known_vector_to_cluster)} vectors from {len(cluster_to_known_vectors)} reviewed clusters.")

    logger.info("Filtering ALL valid vectors by minimum face size...")
    vector_metadata = {}
    all_valid_vector_ids = set()
    with project.db_engine.connect() as conn:
        query = sa.text('''
            SELECT v.id, v.media_id, v.timestamp, (vmi.bbox_w * m.width), (vmi.bbox_h * m.height) 
            FROM vectors v
            JOIN media m ON v.media_id = m.id
            JOIN vector_metadata_insightface vmi ON v.id = vmi.vector_id
            WHERE v.feature_extractor_id = :fe_id
        ''')
        rows = conn.execute(query, {"fe_id": feature_extractor_id}).fetchall()
        for row in rows:
            vid, mid, ts, w, h = row[0], row[1], row[2], row[3], row[4]
            vector_metadata[vid] = (mid, ts)
            if w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1]:
                all_valid_vector_ids.add(vid)

    # Ensure all known vectors are included
    all_vector_ids = list(all_valid_vector_ids.union(set(known_vector_to_cluster.keys())))
    logger.info(f"Total vectors to cluster (known + unknown): {len(all_vector_ids)}")

    if not all_vector_ids:
        logger.info("No vectors to cluster. Exiting.")
        return

    index_path = project.index_dir(feature_extractor_id) / f"{ModalityType.VIDEO.value}-IndexFlatIP.faiss"
    if not index_path.exists():
        logger.error(f"FAISS index not found at {index_path}")
        return

    logger.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))

    if not hasattr(index, 'reconstruct'):
        logger.error("The existing FAISS index does not support vector reconstruction.")
        return

    logger.info("Reconstructing all valid vectors from FAISS index...")
    # Doing this one-by-one is slow but safe for 1.3M if memory is a concern.
    # With 300GB RAM, we could theoretically do index.reconstruct_n(0, index.ntotal) and slice it, 
    # but let's stick to the robust method that matches the exact IDs we validated.
    all_embeddings = np.array([index.reconstruct(int(vid)) for vid in all_vector_ids]).astype(np.float32)
    faiss.normalize_L2(all_embeddings)

    # --- 1. Automated Threshold Calibration ---
    calibrated_eps = 1.0 - fallback_similarity_threshold
    if len(cluster_to_known_vectors) > 0:
        logger.info("Calibrating DBSCAN epsilon based on intra-cluster distances of reviewed clusters...")
        intra_distances = []
        local_vid_to_idx = {vid: idx for idx, vid in enumerate(all_vector_ids)}
        
        for cid, vids in cluster_to_known_vectors.items():
            if len(vids) > 1:
                indices = [local_vid_to_idx[vid] for vid in vids if vid in local_vid_to_idx]
                if len(indices) > 1:
                    cluster_embs = all_embeddings[indices]
                    sim_matrix = np.dot(cluster_embs, cluster_embs.T)
                    dists = 1.0 - sim_matrix[np.triu_indices(len(indices), k=1)]
                    intra_distances.extend(dists)

        if intra_distances:
            # 90th percentile ensures most valid intra-person variance is captured, while discarding extreme outliers
            calibrated_eps = float(np.percentile(intra_distances, 90))
            calibrated_eps = max(0.1, min(0.5, calibrated_eps))
            logger.info(f"Calibration complete. Derived eps: {calibrated_eps:.4f} (Equivalent similarity: {1.0 - calibrated_eps:.4f})")
        else:
            logger.info("Not enough intra-cluster pairs to calibrate. Using fallback threshold.")

    # --- 2. Graph Construction ---
    logger.info("Building FAISS index for all vectors to compute KNN graph...")
    dim = all_embeddings.shape[1]
    index_all = faiss.IndexFlatIP(dim)
    index_all.add(all_embeddings)
    
    k_nn = min(k_neighbors * 2, len(all_embeddings))
    logger.info(f"Searching for {k_nn} nearest neighbors...")
    similarities, knn_indices = index_all.search(all_embeddings, k_nn)
    
    logger.info("Constructing constraint-aware sparse distance matrix...")
    n_samples = len(all_embeddings)
    distances = 1.0 - np.clip(similarities, -1.0, 1.0)
    
    # Vectorized edge extraction
    rows_arr = np.repeat(np.arange(n_samples), k_nn)
    cols_arr = knn_indices.flatten()
    data_arr = distances.flatten()

    # Filter out invalid FAISS edges (-1) and self-loops
    valid_mask = (cols_arr != -1) & (cols_arr != rows_arr)
    rows_arr = rows_arr[valid_mask]
    cols_arr = cols_arr[valid_mask]
    data_arr = data_arr[valid_mask]

    # --- 3. Apply Constraints (Must-Link & Cannot-Link) ---
    logger.info("Injecting Must-Link and Cannot-Link constraints into the graph...")

    # Map array index to cluster_id (-1 for unknown)
    idx_to_cid_arr = np.full(n_samples, -1, dtype=np.int32)
    local_vid_to_idx = {vid: idx for idx, vid in enumerate(all_vector_ids)}

    for vid, cid in known_vector_to_cluster.items():
        if vid in local_vid_to_idx:
            idx_to_cid_arr[local_vid_to_idx[vid]] = cid

    cid_rows = idx_to_cid_arr[rows_arr]
    cid_cols = idx_to_cid_arr[cols_arr]

    # Cannot-Link Mask: KEEP edge if either node is unknown (-1) OR they belong to the SAME known cluster
    keep_mask = (cid_rows == -1) | (cid_cols == -1) | (cid_rows == cid_cols)

    rows = rows_arr[keep_mask].tolist()
    cols = cols_arr[keep_mask].tolist()
    data = data_arr[keep_mask].tolist()

    # Must-Link: Connect all vectors in the same known cluster as a chain (requires only N edges, not N^2)
    for cid, vids in cluster_to_known_vectors.items():
        indices = [local_vid_to_idx[vid] for vid in vids if vid in local_vid_to_idx]
        for k in range(len(indices) - 1):
            idx_a = indices[k]
            idx_b = indices[k+1]
            rows.extend([idx_a, idx_b])
            cols.extend([idx_b, idx_a])
            data.extend([1e-6, 1e-6])

    sparse_graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    sparse_graph = sort_graph_by_row_values(sparse_graph)

    # --- 4. Clustering ---
    logger.info(f"Running DBSCAN on constrained graph (eps={calibrated_eps:.4f})...")
    dbscan = DBSCAN(eps=calibrated_eps, min_samples=2, metric='precomputed', n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(sparse_graph)

    # --- 5. Assignment & Label Resolution ---
    logger.info("Resolving cluster assignments...")
    final_assignments = {}
    label_to_vectors = collections.defaultdict(list)
    for i, label in enumerate(dbscan_labels):
        label_to_vectors[label].append(all_vector_ids[i])

    absorbed_count = 0

    # Track how known clusters are distributed across DBSCAN labels
    known_cid_to_dbscan_labels = collections.defaultdict(set)
    dbscan_label_to_known_cids = collections.defaultdict(set)

    for label, vids in label_to_vectors.items():
        if label == -1:
            for vid in vids:
                if vid not in known_vector_to_cluster:
                    final_assignments[vid] = "draft_-1"
            continue

        known_cids_in_cluster = {known_vector_to_cluster[vid] for vid in vids if vid in known_vector_to_cluster}

        for cid in known_cids_in_cluster:
            known_cid_to_dbscan_labels[cid].add(label)
            dbscan_label_to_known_cids[label].add(cid)

        if len(known_cids_in_cluster) == 0:
            draft_label = f"draft_{label}"
            for vid in vids:
                final_assignments[vid] = draft_label
        elif len(known_cids_in_cluster) == 1:
            target_cid = list(known_cids_in_cluster)[0]
            for vid in vids:
                if vid not in known_vector_to_cluster:
                    final_assignments[vid] = target_cid
                    absorbed_count += 1
        else:
            logger.warning(f"DBSCAN label {label} merged multiple known identities: {known_cids_in_cluster}. Falling back.")
            sorted_cids = sorted(list(known_cids_in_cluster))
            boundary_label = f"uncertain_boundary_" + "_".join(map(str, sorted_cids))
            for vid in vids:
                if vid not in known_vector_to_cluster:
                     final_assignments[vid] = boundary_label

    # Analyze Fragmentation and Merges
    fragmented_known_clusters = {cid: labels for cid, labels in known_cid_to_dbscan_labels.items() if len(labels) > 1}

    # Identify unreviewed clusters that were merged because they share a DBSCAN label with a known cluster
    # This logic is already handled implicitly by the absorption, but we want to quantify it.
    # To quantify it perfectly against Iteration 1 is hard without storing Iteration 1's state.
    # However, we CAN report how many distinct draft clusters the absorbed vectors WOULD have formed.
    # Since we can't easily do that retroactively, let's report the structural changes we CAN see.

    logger.info("Saving new cluster assignments to explore.db...")

    cluster_to_final_vids = collections.defaultdict(list)
    for vid, assigned_label in final_assignments.items():
        cluster_to_final_vids[assigned_label].append(vid)

    with SessionLocal() as session:
        new_draft_labels = {val for val in final_assignments.values() if isinstance(val, str) and val.startswith(('draft_', 'uncertain_boundary_'))}
        label_to_cluster_id = {}
        for label in new_draft_labels:
            feedbacks = []
            if label.startswith('uncertain_boundary_'):
                feedbacks.append("Uncertain Boundary")

            # Calculate unique_media_count
            media_ids = set()
            for vid in cluster_to_final_vids[label]:
                if vid in vector_metadata:
                    mid, _ = vector_metadata[vid]
                    media_ids.add(mid)

            unique_media_count = len(media_ids)

            machine_feedback_str = ", ".join(feedbacks) if feedbacks else None

            if label.startswith('draft_'):
                is_noise = (label == "draft_-1")
                cluster = Cluster(facet_id=facet_id, cluster_label="Noise" if is_noise else "", status=ClusterStatus.draft, machine_feedback=machine_feedback_str, unique_media_count=unique_media_count)
            elif label.startswith('uncertain_boundary_'):
                cluster = Cluster(facet_id=facet_id, cluster_label=label, status=ClusterStatus.draft, machine_feedback=machine_feedback_str, unique_media_count=unique_media_count)

            session.add(cluster)
            session.flush()
            label_to_cluster_id[label] = cluster.id

        # Update feedback on existing known clusters
        for cid in known_cid_to_dbscan_labels.keys():
            feedbacks = []
            if cid in fragmented_known_clusters:
                feedbacks.append("Fragmented Ground Truth")

            # Calculate unique_media_count for known clusters
            media_ids = set()
            all_c_vids = cluster_to_known_vectors.get(cid, []) + cluster_to_final_vids.get(cid, [])
            for vid in all_c_vids:
                if vid in vector_metadata:
                    mid, _ = vector_metadata[vid]
                    media_ids.add(mid)

            unique_media_count = len(media_ids)

            machine_feedback_str = ", ".join(feedbacks) if feedbacks else None
            session.query(Cluster).filter_by(id=cid).update({
                "machine_feedback": machine_feedback_str,
                "unique_media_count": unique_media_count
            })
            

        # Build the set of IDs of known clusters that received new assignments
        known_cids_receiving_vectors = {
            assigned_label for assigned_label in final_assignments.values()
            if isinstance(assigned_label, int)
        }

        # Only query existing assignments for known clusters involved in this iteration
        existing_assignments = set()
        if known_cids_receiving_vectors:
            existing_pairs = session.query(Assignment.vector_id, Assignment.cluster_id).filter(
                Assignment.cluster_id.in_(list(known_cids_receiving_vectors))
            ).all()
            existing_assignments = set(existing_pairs)

        assignments_to_insert = []
        for vec_id, assigned_label in final_assignments.items():
            cluster_id = label_to_cluster_id[assigned_label] if isinstance(assigned_label, str) else assigned_label
            # Only insert if this exact assignment doesn't already exist
            if (int(vec_id), int(cluster_id)) not in existing_assignments:
                assignments_to_insert.append(Assignment(vector_id=int(vec_id), cluster_id=cluster_id))
            
        session.bulk_save_objects(assignments_to_insert)
        session.commit()

    all_assigned_cluster_ids = set(label_to_cluster_id.values())
    for assigned_label in final_assignments.values():
        if isinstance(assigned_label, int):
            all_assigned_cluster_ids.add(assigned_label)

    logger.info(f"Summary:")
    logger.info(f" - Unknown vectors absorbed into reviewed clusters: {absorbed_count}")
    logger.info(f" - Total unique draft clusters created: {len(label_to_cluster_id)}")

    with SessionLocal() as session:
        all_clusters = session.query(Cluster).filter(Cluster.id.in_(list(known_cid_to_dbscan_labels.keys()))).all()
        cid_to_label = {c.id: c.cluster_label for c in all_clusters}

    logger.info(f" - Reviewed Cluster Survival & Fragmentation:")
    survived_intact = 0
    fragmented = 0
    for cid, labels in known_cid_to_dbscan_labels.items():
        label_name = cid_to_label.get(cid, f"Cluster {cid}")
        if -1 in labels: labels.remove(-1) # Ignore noise fragments

        if len(labels) <= 1:
            survived_intact += 1
        else:
            fragmented += 1
            logger.info(f"     * '{label_name}' (ID {cid}) was fragmented across {len(labels)} internal DBSCAN groups.")

    logger.info(f"     * {survived_intact} reviewed clusters formed cohesive, unbroken groups.")
    if fragmented > 0:
        logger.warning(f"     * {fragmented} reviewed clusters fragmented. Consider loosening --similarity-threshold or adding more examples.")

    logger.info("Semi-supervised iterative clustering completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform semi-supervised face clustering using existing reviewed clusters as constraints.")
    parser.add_argument("--project-dir", type=str, required=True, help="WISE project directory")
    parser.add_argument("--feature-extractor-id", type=str, default="deepinsight/insightface/buffalo_l/_unknown", help="Feature extractor ID for face embeddings.")
    parser.add_argument("--k-neighbors", type=int, default=50, help="Number of nearest neighbors to build the sparse graph for DBSCAN.")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Fallback cosine similarity threshold if automatic calibration fails.")
    args = parser.parse_args()
    
    cluster_faces(Path(args.project_dir), args.feature_extractor_id, args.k_neighbors, args.similarity_threshold)
