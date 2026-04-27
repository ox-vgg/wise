import argparse
from pathlib import Path
import logging
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import sort_graph_by_row_values

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

def cluster_faces(project_dir: Path, feature_extractor_id: str, k_neighbors: int, similarity_threshold: float):
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
        
    known_vector_ids = {kc.vector_id for kc in known_clusters_data}
    known_centroids = {}
    if known_clusters_data:
        for kc in known_clusters_data:
            if kc.cluster_id not in known_centroids:
                known_centroids[kc.cluster_id] = kc.centroid
        logger.info(f"Loaded {len(known_vector_ids)} vectors from {len(known_centroids)} known clusters.")

    logger.info("Filtering vectors by minimum face size...")
    with project.db_engine.connect() as conn:
        query = sa.text('''
            SELECT v.id FROM vectors v
            JOIN media m ON v.media_id = m.id
            JOIN vector_metadata_insightface vmi ON v.id = vmi.vector_id
            WHERE (vmi.bbox_w * m.width) >= :min_w AND (vmi.bbox_h * m.height) >= :min_h
              AND v.feature_extractor_id = :fe_id
        ''')
        valid_vector_ids = set(conn.execute(query, {"min_w": MIN_FACE_SIZE[0], "min_h": MIN_FACE_SIZE[1], "fe_id": feature_extractor_id}).scalars().all())

    unknown_vector_ids = valid_vector_ids - known_vector_ids
    logger.info(f"Found {len(unknown_vector_ids)} unknown vectors to cluster.")

    if not unknown_vector_ids:
        logger.info("No new vectors to cluster. Exiting.")
        return

    index_path = project.index_dir(feature_extractor_id) / f"{ModalityType.VIDEO.value}-IndexFlatIP.faiss"
    if not index_path.exists():
        logger.error(f"FAISS index not found at {index_path}")
        logger.error("Please run the create-index.py script first.")
        return

    logger.info(f"Loading existing FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))

    if not hasattr(index, 'reconstruct'):
        logger.error("The existing FAISS index does not support vector reconstruction. Please rebuild it with a compatible index type (e.g., IndexIDMap2 or by adding vectors with their IDs).")
        return

    logger.info("Reconstructing unknown vectors from FAISS index...")
    loaded_unknown_ids = list(unknown_vector_ids)
    unknown_embeddings = np.array([index.reconstruct(int(vid)) for vid in loaded_unknown_ids]).astype(np.float32)

    if len(unknown_embeddings) == 0:
        logger.warning("No features found for unknown vectors.")
        return
        
    faiss.normalize_L2(unknown_embeddings)

    logger.info("Building FAISS index for unknown vectors...")
    dim = unknown_embeddings.shape[1]
    index_unknowns = faiss.IndexFlatIP(dim)
    index_unknowns.add(unknown_embeddings)
    
    k_nn = min(k_neighbors, len(unknown_embeddings))
    logger.info(f"Searching for {k_nn} nearest neighbors among unknown vectors...")
    similarities, indices = index_unknowns.search(unknown_embeddings, k_nn)
    
    logger.info("Constructing sparse distance matrix for unknowns...")
    distances = 1.0 - np.clip(similarities, -1.0, 1.0)
    rows = np.repeat(np.arange(len(unknown_embeddings)), k_nn)
    cols = indices.flatten()
    mask = cols != -1
    sparse_dist_matrix = csr_matrix((distances.flatten()[mask], (rows[mask], cols[mask])), shape=(len(unknown_embeddings), len(unknown_embeddings)))
    sparse_dist_matrix = sort_graph_by_row_values(sparse_dist_matrix)
    
    epsilon = 1.0 - similarity_threshold
    logger.info(f"Clustering unknowns with DBSCAN (eps={epsilon:.3f})...")
    dbscan = DBSCAN(eps=epsilon, min_samples=2, metric='precomputed', n_jobs=-1)
    draft_labels = dbscan.fit_predict(sparse_dist_matrix)

    final_assignments = {}
    if known_centroids:
        logger.info("Building FAISS index for known cluster centroids...")
        centroid_matrix = np.vstack(list(known_centroids.values())).astype(np.float32)
        faiss.normalize_L2(centroid_matrix)
        index_centroids = faiss.IndexFlatIP(dim)
        index_centroids.add(centroid_matrix)
        centroid_cluster_ids = list(known_centroids.keys())

        logger.info("Searching for closest known cluster for each unknown vector...")
        sims_to_known, closest_centroid_indices = index_centroids.search(unknown_embeddings, 1)

        for i, unknown_vec_id in enumerate(loaded_unknown_ids):
            closest_cluster_id = centroid_cluster_ids[closest_centroid_indices[i][0]]
            similarity = sims_to_known[i][0]
            if similarity >= similarity_threshold:
                final_assignments[unknown_vec_id] = closest_cluster_id
            else:
                final_assignments[unknown_vec_id] = f"draft_{draft_labels[i]}"
    else:
        for i, unknown_vec_id in enumerate(loaded_unknown_ids):
            final_assignments[unknown_vec_id] = f"draft_{draft_labels[i]}"

    logger.info("Saving new cluster assignments to explore.db...")
    with SessionLocal() as session:
        new_draft_labels = {val for val in final_assignments.values() if isinstance(val, str) and val.startswith('draft_')}
        label_to_cluster_id = {}
        for label in new_draft_labels:
            is_noise = (label == "draft_-1")
            cluster = Cluster(facet_id=facet_id, cluster_label="Noise" if is_noise else "", status=ClusterStatus.draft)
            session.add(cluster)
            session.flush()
            label_to_cluster_id[label] = cluster.id
            
        assignments_to_insert = []
        for vec_id, assigned_label in final_assignments.items():
            cluster_id = label_to_cluster_id[assigned_label] if isinstance(assigned_label, str) else assigned_label
            assignments_to_insert.append(Assignment(vector_id=int(vec_id), cluster_id=cluster_id))
            
        session.bulk_save_objects(assignments_to_insert)
        session.commit()

    all_assigned_cluster_ids = set(label_to_cluster_id.values())
    for assigned_label in final_assignments.values():
        if isinstance(assigned_label, int):
            all_assigned_cluster_ids.add(assigned_label)
    logger.info(f"Total unique clusters involved in this iteration: {len(all_assigned_cluster_ids)}")
    logger.info("Iterative clustering completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform iterative face clustering using existing reviewed clusters as seeds.")
    parser.add_argument("--project-dir", type=str, required=True, help="WISE project directory")
    parser.add_argument("--feature-extractor-id", type=str, default="deepinsight/insightface/buffalo_l/_unknown", help="Feature extractor ID for face embeddings.")
    parser.add_argument("--k-neighbors", type=int, default=50, help="Number of nearest neighbors to build the sparse graph for DBSCAN.")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Cosine similarity threshold for a vector to be absorbed into a known cluster or for DBSCAN's epsilon.")
    args = parser.parse_args()
    
    cluster_faces(Path(args.project_dir), args.feature_extractor_id, args.k_neighbors, args.similarity_threshold)
