import argparse
from pathlib import Path
import logging
import collections

from wise.wise_project import WiseProject
from scripts.explore.db import init_explore_db
from scripts.explore.models import Cluster, Assignment
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recompute_unique_media_count(project_dir: Path):
    logger.info(f"Loading WISE project from {project_dir}")
    project = WiseProject(project_dir)
    
    logger.info("Initializing explore.db connection...")
    engine, SessionLocal = init_explore_db(project_dir)
    
    with engine.connect() as conn:
        # 1. Check if the column exists
        has_column = False
        result = conn.execute(sa.text("PRAGMA table_info(clusters)")).fetchall()
        for col in result:
            if col[1] == 'unique_media_count':
                has_column = True
                break
                
        if not has_column:
            logger.info("Column 'unique_media_count' not found. Adding it to the clusters table...")
            conn.execute(sa.text("ALTER TABLE clusters ADD COLUMN unique_media_count INTEGER NOT NULL DEFAULT 0;"))
            conn.commit()
            logger.info("Column unique_media_count added successfully.")

        has_size_column = False
        for col in result:
            if col[1] == 'size':
                has_size_column = True
                break
        if not has_size_column:
            logger.info("Column 'size' not found. Adding it to the clusters table...")
            conn.execute(sa.text("ALTER TABLE clusters ADD COLUMN size INTEGER NOT NULL DEFAULT 0;"))
            conn.commit()
            logger.info("Column size added successfully.")
            
    logger.info("Recomputing unique media counts and sizes...")
    
    with SessionLocal() as session:
        # 1. Fetch all assignments
        logger.info("Fetching all cluster assignments...")
        all_assignments = session.query(Assignment.cluster_id, Assignment.vector_id).all()
        
        if not all_assignments:
            logger.info("No assignments found. Nothing to recompute.")
            return

        cluster_to_vids = collections.defaultdict(list)
        all_unique_vids = set()
        
        for cluster_id, vector_id in all_assignments:
            cluster_to_vids[cluster_id].append(vector_id)
            all_unique_vids.add(vector_id)
            
        all_unique_vids_list = list(all_unique_vids)
        logger.info(f"Found {len(all_unique_vids_list)} unique vectors across {len(cluster_to_vids)} clusters.")

        # 2. Fetch metadata in safe chunks
        logger.info("Fetching media metadata for all vectors (chunked)...")
        vid_to_media_id = {}
        chunk_size = 900
        for i in range(0, len(all_unique_vids_list), chunk_size):
            chunk = all_unique_vids_list[i:i + chunk_size]
            metadata_chunk = project.get_vector_media_metadata_for_ids(chunk)
            for meta in metadata_chunk:
                vid_to_media_id[meta.id] = meta.media_id
            
            if (i > 0) and (i % 90000 == 0):
                logger.info(f"  Fetched metadata for {i}/{len(all_unique_vids_list)} vectors...")

        # 3. Calculate unique counts and sizes in memory
        logger.info("Calculating unique media counts and sizes in memory...")
        updates = []
        for cid, vids in cluster_to_vids.items():
            unique_media_ids = {vid_to_media_id[vid] for vid in vids if vid in vid_to_media_id}
            count = len(unique_media_ids)
            size = len(vids)
            updates.append({"id": cid, "unique_media_count": count, "size": size})

        # 4. Perform bulk update
        logger.info(f"Executing bulk update for {len(updates)} clusters...")
        # We process bulk updates in chunks to avoid blowing out memory on massive datasets
        update_chunk_size = 5000
        for i in range(0, len(updates), update_chunk_size):
            session.bulk_update_mappings(Cluster, updates[i:i+update_chunk_size])
            session.commit()
            
        logger.info("Successfully recomputed unique media counts for all clusters.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forcefully recompute the unique_media_count for all clusters.")
    parser.add_argument("--project-dir", type=str, required=True, help="WISE project directory")
    args = parser.parse_args()
    
    recompute_unique_media_count(Path(args.project_dir))
