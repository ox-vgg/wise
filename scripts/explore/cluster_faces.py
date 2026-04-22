import argparse
from pathlib import Path
import logging
import numpy as np

from sklearn.metrics import pairwise_distances
import hdbscan

import sys
import os
# Add root to sys.path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.wise_project import WiseProject
from src.feature.store import FeatureStoreFactory
from src.data_models import ModalityType
from scripts.explore.db import init_explore_db
from scripts.explore.models import Facet, Cluster, Assignment, ClusterStatus
from scripts.explore.config import MIN_FACE_SIZE, FACE_SIMILARITY_THRESHOLD
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cluster_faces(project_dir: Path, feature_extractor_id: str):
    logger.info(f"Loading WISE project from {project_dir}")
    project = WiseProject(project_dir)
    
    logger.info("Initializing explore.db")
    engine, SessionLocal = init_explore_db(project_dir)
    
    # Check if facet already exists
    with SessionLocal() as session:
        facet = session.query(Facet).filter_by(name="Face").first()
        if not facet:
            facet = Facet(name="Face", feature_extractor_id=feature_extractor_id)
            session.add(facet)
            session.commit()
            session.refresh(facet)
            
        facet_id = facet.id
        
        logger.info("Clearing old draft clusters")
        session.query(Assignment).filter(Assignment.cluster_id.in_(
            session.query(Cluster.id).filter_by(facet_id=facet_id, status=ClusterStatus.draft)
        )).delete(synchronize_session=False)
        session.query(Cluster).filter_by(facet_id=facet_id, status=ClusterStatus.draft).delete()
        session.commit()
        
    store = FeatureStoreFactory.load_store(
        ModalityType.VIDEO, 
        str(project.features_dir(feature_extractor_id)) + "/"
    )
    
    # Filter by min face size
    logger.info("Filtering vectors by minimum face size...")
    with project.db_engine.connect() as conn:
        query = sa.text("""
            SELECT v.id 
            FROM vectors v
            JOIN media m ON v.media_id = m.id
            JOIN vector_metadata_insightface vmi ON v.id = vmi.vector_id
            WHERE (vmi.bbox_w * m.width) >= :min_w 
              AND (vmi.bbox_h * m.height) >= :min_h
              AND v.feature_extractor_id = :fe_id
        """)
        valid_vector_ids = set(conn.execute(query, {
            "min_w": MIN_FACE_SIZE[0], 
            "min_h": MIN_FACE_SIZE[1], 
            "fe_id": feature_extractor_id
        }).scalars().all())

    # Find already reviewed/published vectors to exclude them
    with SessionLocal() as session:
        reviewed_vector_ids = set(
            session.query(Assignment.vector_id)
            .join(Cluster)
            .filter(
                Cluster.facet_id == facet_id,
                Cluster.status.in_([ClusterStatus.reviewed, ClusterStatus.published])
            ).all()
        )
        # SQLAlchemy returns a list of tuples like (id,), so we extract the first element
        reviewed_vector_ids = {vid[0] for vid in reviewed_vector_ids}
        
    valid_vector_ids = valid_vector_ids - reviewed_vector_ids
    logger.info(f"Excluded {len(reviewed_vector_ids)} already reviewed vectors.")
    logger.info(f"Found {len(valid_vector_ids)} valid unreviewed vectors matching size criteria.")
    
    vector_ids = []
    embeddings = []
    
    logger.info("Loading vectors from feature store...")
    batch_count = 0
    for batch_ids, batch_features in store.iter_batch(batch_size=1024):
        mask = [vid in valid_vector_ids for vid in batch_ids]
        if any(mask):
            vector_ids.extend(batch_ids[mask])
            embeddings.append(batch_features[mask])
        batch_count += 1
        if batch_count % 10 == 0:
            logger.info(f"Processed {batch_count * 1024} vectors...")
        
    if not embeddings:
        logger.warning("No features found.")
        return
        
    embeddings = np.vstack(embeddings)
    
    logger.info(f"Loaded {len(vector_ids)} vectors.")
    
    logger.info("Computing pairwise cosine distance matrix using all available CPU cores...")
    # n_jobs=-1 safely uses all CPU cores for distance calculation in scikit-learn
    distance_matrix = pairwise_distances(embeddings, metric='cosine', n_jobs=-1)
    
    # Ensure no negative distances due to floating point inaccuracies
    distance_matrix = np.clip(distance_matrix, 0, None).astype(np.float64)
    logger.info("Distance matrix computed.")
    
    # Cosine distance = 1 - cosine_similarity
    epsilon = float(1.0 - FACE_SIMILARITY_THRESHOLD)
    
    logger.info(f"Clustering using HDBSCAN with precomputed cosine distance (epsilon={epsilon:.3f})...")
    logger.info("Building the cluster hierarchy... (core_dist_n_jobs=1 to prevent deadlocks)")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2, 
        metric='precomputed', 
        cluster_selection_epsilon=epsilon,
        core_dist_n_jobs=1  # Setting to 1 completely bypasses joblib and prevents deadlocks
    )
    
    cluster_labels = clusterer.fit_predict(distance_matrix)
    logger.info("HDBSCAN fit_predict completed.")
    
    unique_labels = set(cluster_labels)
    logger.info(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters + noise")
    
    logger.info("Saving clusters to explore.db")
    with SessionLocal() as session:
        cluster_map = {}
        for label in unique_labels:
            if label == -1:
                cluster = Cluster(facet_id=facet_id, cluster_label="Noise", status=ClusterStatus.draft)
            else:
                cluster = Cluster(facet_id=facet_id, cluster_label="", status=ClusterStatus.draft)
            session.add(cluster)
            session.flush() # get ID
            cluster_map[label] = cluster.id
            
        assignments = []
        for v_id, label in zip(vector_ids, cluster_labels):
            assignments.append(Assignment(
                vector_id=int(v_id),
                cluster_id=cluster_map[label]
            ))
            
        session.bulk_save_objects(assignments)
        session.commit()
        
    logger.info("Clustering completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=str, required=True, help="WISE project directory")
    parser.add_argument("--feature-extractor-id", type=str, default="deepinsight/insightface/buffalo_l/_unknown", help="Feature extractor to use")
    args = parser.parse_args()
    
    cluster_faces(Path(args.project_dir), args.feature_extractor_id)
