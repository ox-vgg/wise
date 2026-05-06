import argparse
from pathlib import Path
import logging
import numpy as np
import faiss
import collections

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.wise_project import WiseProject
from src.data_models import ModalityType
from scripts.explore.db import init_explore_db
from scripts.explore.models import Cluster, Assignment, KnownFaceCluster
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_faces(unknown_project_dir: Path, known_project_dir: Path, feature_extractor_id: str, similarity_threshold: float, metadata_table: str, name_field: str, status_filter: list, dry_run: bool, is_threshold_explicit: bool, force_proposal: bool):
    logger.info(f"Loading UNKNOWN project from {unknown_project_dir}")
    unknown_project = WiseProject(unknown_project_dir)
    
    logger.info(f"Loading KNOWN project from {known_project_dir}")
    known_project = WiseProject(known_project_dir)
    
    # 1. Load the KNOWN Data
    logger.info(f"Loading metadata from KNOWN project (table: {metadata_table})...")
    known_meta_map = {}
    with known_project.db_engine.connect() as conn:
        try:
            # Query the user-specified table.
            # We join with the `vectors` table to ensure we always have a valid vector_id,
            # even if the metadata was imported at the `media` level and the metadata table's vector_id is NULL.
            # This assumes there is exactly one primary face vector per profile image in the KNOWN project.
            query = sa.text(f'''
                WITH RankedVectors AS (
                    SELECT 
                        v.id as vector_id,
                        v.media_id,
                        ROW_NUMBER() OVER(
                            PARTITION BY v.media_id 
                            ORDER BY ((vmi.bbox_w * vmi.bbox_h) * vmi.detection_score) DESC
                        ) as rn
                    FROM vectors v
                    JOIN vector_metadata_insightface vmi ON v.id = vmi.vector_id
                )
                SELECT m.*, rv.vector_id as true_vector_id
                FROM "{metadata_table}" m
                JOIN RankedVectors rv ON m.media_id = rv.media_id
                WHERE rv.rn = 1
            ''')
            res = conn.execute(query)
            
            columns = res.keys()
            if name_field not in columns:
                logger.error(f"The specified name field '{name_field}' does not exist in table '{metadata_table}'.")
                return

            for row in res:
                row_dict = {col: getattr(row, col) for col in columns if col != 'true_vector_id'}
                # Use the true vector_id from the join
                known_meta_map[row.true_vector_id] = row_dict
                
            logger.info(f"Loaded {len(known_meta_map)} known profiles.")
        except sa.exc.OperationalError as e:
            logger.error(f"Database error querying '{metadata_table}': {e}")
            return

    known_index_path = known_project.index_dir(feature_extractor_id) / f"{ModalityType.IMAGE.value}-IndexFlatIP.faiss"
    if not known_index_path.exists():
        logger.error(f"FAISS index not found for KNOWN project at {known_index_path}")
        return
        
    logger.info(f"Loading KNOWN FAISS index from {known_index_path}...")
    known_index = faiss.read_index(str(known_index_path))

    # 2. Load the UNKNOWN Data
    logger.info("Initializing explore.db connection for UNKNOWN project...")
    engine, SessionLocal = init_explore_db(unknown_project_dir)
    
    unknown_index_path = unknown_project.index_dir(feature_extractor_id) / f"{ModalityType.VIDEO.value}-IndexFlatIP.faiss"
    if not unknown_index_path.exists():
        logger.error(f"FAISS index not found for UNKNOWN project at {unknown_index_path}")
        return
        
    logger.info(f"Loading UNKNOWN FAISS index from {unknown_index_path}...")
    unknown_index = faiss.read_index(str(unknown_index_path))
    
    if not hasattr(unknown_index, 'reconstruct'):
        logger.error("The UNKNOWN FAISS index does not support vector reconstruction.")
        return

    # 3. Automated Threshold Calibration
    calibrated_sim = similarity_threshold # Fallback
    
    if not is_threshold_explicit:
        logger.info("Checking for reviewed clusters to calibrate matching threshold...")
        with SessionLocal() as session:
            known_clusters_data = session.query(KnownFaceCluster).all()
            cluster_to_known_vectors = collections.defaultdict(list)
            for kc in known_clusters_data:
                cluster_to_known_vectors[kc.cluster_id].append(kc.vector_id)
                
            if len(cluster_to_known_vectors) > 0:
                known_vids = [vid for vids in cluster_to_known_vectors.values() for vid in vids]
                try:
                    known_embeddings = np.array([unknown_index.reconstruct(int(vid)) for vid in known_vids]).astype(np.float32)
                    faiss.normalize_L2(known_embeddings)
                    
                    intra_distances = []
                    local_vid_to_idx = {vid: idx for idx, vid in enumerate(known_vids)}
                    
                    for cid, vids in cluster_to_known_vectors.items():
                        if len(vids) > 1:
                            indices = [local_vid_to_idx[vid] for vid in vids if vid in local_vid_to_idx]
                            if len(indices) > 1:
                                cluster_embs = known_embeddings[indices]
                                sim_matrix = np.dot(cluster_embs, cluster_embs.T)
                                dists = 1.0 - sim_matrix[np.triu_indices(len(indices), k=1)]
                                intra_distances.extend(dists)
                                
                    if intra_distances:
                        calibrated_eps = float(np.percentile(intra_distances, 90))
                        calibrated_eps = max(0.1, min(0.5, calibrated_eps))
                        calibrated_sim = 1.0 - calibrated_eps
                        logger.info(f"Calibrated matching threshold from {len(cluster_to_known_vectors)} reviewed clusters: {calibrated_sim:.4f}")
                except RuntimeError:
                    logger.warning("Could not reconstruct known vectors for calibration. Using user-provided threshold.")
            else:
                logger.info(f"No reviewed clusters found. Using user-provided threshold: {calibrated_sim:.4f}")
    else:
        logger.info(f"Calibration disabled. Using user-provided threshold: {calibrated_sim:.4f}")

    # 4. Process Clusters and Generate Proposals
    logger.info("Processing UNKNOWN clusters to generate identity proposals...")
    proposals_made = 0
    proposed_names = []
    
    with SessionLocal() as session:
        # Only check clusters matching the status filter and lacking a proposed identity
        from scripts.explore.models import ClusterStatus
        allowed_statuses = [ClusterStatus(s) for s in status_filter]
        
        clusters = session.query(Cluster).filter(Cluster.status.in_(allowed_statuses)).all()
        
        for i, cluster in enumerate(clusters):
            # Skip if it already has a proposal to save time
            if cluster.metadata_json and "_proposed_identity" in cluster.metadata_json:
                continue
                
            assignments = session.query(Assignment).filter_by(cluster_id=cluster.id).all()
            vector_ids = [a.vector_id for a in assignments]
            
            if not vector_ids:
                continue
                
            # Reconstruct embeddings and calculate centroid
            try:
                embeddings = np.array([unknown_index.reconstruct(int(vid)) for vid in vector_ids]).astype(np.float32)
            except RuntimeError:
                continue # Skip if vector is missing from index
                
            if len(embeddings) > 0:
                centroid = np.mean(embeddings, axis=0, keepdims=True)
                faiss.normalize_L2(centroid)
                
                # Query the KNOWN index
                similarities, indices = known_index.search(centroid, 1)
                best_sim = similarities[0][0]
                best_vid = indices[0][0]
                
                # Use the CALIBRATED threshold here!
                if best_sim >= calibrated_sim and best_vid != -1:
                    if best_vid in known_meta_map:
                        proposed_meta = known_meta_map[best_vid]
                        proposed_name = proposed_meta[name_field]
                        
                        # Check if the user has already manually labeled this cluster
                        if cluster.cluster_label and cluster.cluster_label.strip() != "" and not cluster.cluster_label.startswith(("Noise", "draft_", "uncertain_boundary_")):
                            import re
                            is_default_face_label = re.match(r"^Face \d+$", cluster.cluster_label) is not None
                            
                            if not (is_default_face_label and force_proposal):
                                logger.info(f"SKIPPED Cluster ({cluster.id}) '{cluster.cluster_label}' (Proposal: '{proposed_name}', score={best_sim:.4f})")
                                continue

                        # Remove vector_id, media_id etc from the proposal payload as they are KNOWN project specific
                        clean_proposed_meta = {k: v for k, v in proposed_meta.items() if k not in ['id', 'vector_id', 'media_id', 'timestamp', 'end_timestamp']}
                        # Ensure the UI standard 'name' key is present regardless of the source table's column name
                        clean_proposed_meta['name'] = proposed_name

                        # 4. Save Proposal to Database
                        new_meta = dict(cluster.metadata_json) if cluster.metadata_json else {}
                        new_meta["_proposed_identity"] = {
                            "similarity": float(best_sim),
                            **clean_proposed_meta
                        }
                        
                        feedbacks = []
                        if cluster.machine_feedback:
                            feedbacks = cluster.machine_feedback.split(', ')
                        if "Identity Proposed" not in feedbacks:
                            feedbacks.append("Identity Proposed")
                            
                        cluster.metadata_json = new_meta
                        cluster.machine_feedback = ", ".join(feedbacks)
                        proposals_made += 1
                        proposed_names.append(proposed_name)
                        
                        prefix = "[DRY RUN] " if dry_run else ""
                        logger.info(f"{prefix}PROPOSAL: Cluster ({cluster.id}) '{proposed_name}' (score={best_sim:.4f})")
                        
            if (i + 1) % 100 == 0:
                if not dry_run:
                    session.commit()
                logger.info(f"Processed {i + 1}/{len(clusters)} clusters...")
                
        if not dry_run:
            session.commit()
        else:
            logger.info("DRY RUN COMPLETE: No changes were saved to the database.")
            
    logger.info(f"Successfully generated identity proposals for {proposals_made} clusters.")
    if proposed_names:
        unique_names = list(set(proposed_names))
        logger.info(f"Proposed {len(unique_names)} unique identities: {', '.join(unique_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-reference unknown clusters against a known knowledge-base project to propose identities.")
    parser.add_argument("--unknown-project-dir", type=str, required=True, help="Path to the WISE project you want to label.")
    parser.add_argument("--known-project-dir", type=str, required=True, help="Path to the WISE project containing known profiles.")
    parser.add_argument("--metadata-table", type=str, required=True, help="Name of the metadata table in the KNOWN project (e.g. 'metadata-profiles').")
    parser.add_argument("--name-field", type=str, required=True, help="The column name in the metadata table that contains the identity's name (e.g. 'name', 'person_name').")
    parser.add_argument("--feature-extractor-id", type=str, default="deepinsight/insightface/buffalo_l/_unknown", help="Feature extractor ID to use for matching.")
    parser.add_argument("--similarity-threshold", type=float, default=0.6, help="Fallback cosine similarity threshold required to generate a proposal if calibration fails.")
    parser.add_argument("--status-filter", type=str, nargs='+', choices=['draft', 'reviewed', 'published'], default=['draft', 'reviewed', 'published'], help="Only generate proposals for clusters with these statuses (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="Run the script and print proposals to the console without saving them to the database.")
    parser.add_argument("--force-proposal", action="store_true", help="Overwrite existing manually provided cluster_label if it matches the pattern 'Face N'.")
    args = parser.parse_args()
    
    is_threshold_explicit = "--similarity-threshold" in sys.argv
    
    identify_faces(Path(args.unknown_project_dir), Path(args.known_project_dir), args.feature_extractor_id, args.similarity_threshold, args.metadata_table, args.name_field, args.status_filter, args.dry_run, is_threshold_explicit, args.force_proposal)