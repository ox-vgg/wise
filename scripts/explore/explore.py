import argparse
from pathlib import Path
import logging
from typing import Optional, List

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import json

import sys
import os
from typing import BinaryIO

# Add root to sys.path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.wise_project import WiseProject
from src.data_models import MediaType, SourceCollectionType
from scripts.explore.db import init_explore_db
from scripts.explore.models import Facet, Cluster, Assignment, ClusterStatus, FacetMetadataSchema
import src.db.tables.facets as wise_tables
import sqlalchemy as sa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WISE Explore API")

# Helper functions for video streaming
def send_bytes_range_requests(file_obj: BinaryIO, start: int, end: int, chunk_size: int = 10_000):
    with file_obj as f:
        f.seek(start)
        while (pos := f.tell()) <= end:
            read_size = min(chunk_size, end + 1 - pos)
            yield f.read(read_size)

def _get_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    def _invalid_range():
        return HTTPException(
            status_code=416,
            detail=f"Invalid request range (Range:{range_header!r})",
        )
    try:
        h = range_header.replace("bytes=", "").split("-")
        start = int(h[0]) if h[0] != "" else 0
        end = int(h[1]) if h[1] != "" else file_size - 1
    except ValueError:
        raise _invalid_range()
    if start > end or start < 0 or end > file_size - 1:
        raise _invalid_range()
    return start, end

# Setup templates and static files once project is parsed
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Assume frontend built assets will be in 'frontend/dist'
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")

def get_vite_assets():
    manifest_path = os.path.join(frontend_dist, ".vite", "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            # The entry point is src/main.tsx
            if "src/main.tsx" in manifest:
                return f"/{manifest['src/main.tsx']['file']}"
    return None

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

class AppState:
    project_dir: Path
    engine = None
    SessionLocal = None
    project = None

app_state = AppState()

def get_db():
    db = app_state.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for API
class SchemaCreate(BaseModel):
    key_name: str
    data_type: str

class SchemaUpdate(BaseModel):
    key_name: Optional[str] = None
    data_type: Optional[str] = None

class BatchMetadataUpdate(BaseModel):
    metadata_json: dict

class ClusterUpdate(BaseModel):
    cluster_label: Optional[str] = None
    metadata_json: Optional[dict] = None
    status: Optional[str] = None

class AssignmentUpdate(BaseModel):
    vector_ids: List[int]
    new_cluster_id: int

@app.get("/{project_name}/explore/", response_class=HTMLResponse)
async def view_index(request: Request, project_name: str, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    facets = db.query(Facet).all()
    initial_state = {
        "view": "index",
        "project_name": project_name,
        "facets": [{"id": f.id, "name": f.name, "feature_extractor_id": f.feature_extractor_id} for f in facets]
    }
    return templates.TemplateResponse("index.html", {
        "request": request,
        "initial_state": json.dumps(initial_state),
        "main_js": get_vite_assets()
    })

@app.get("/{project_name}/explore/{facet_name}/{feature_extractor_slug:path}/", response_class=HTMLResponse)
async def view_facet(request: Request, project_name: str, facet_name: str, feature_extractor_slug: str, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    facet = db.query(Facet).filter(Facet.name.ilike(facet_name), Facet.feature_extractor_id.contains(feature_extractor_slug)).first()
    if not facet:
        raise HTTPException(status_code=404, detail="Facet not found")
        
    total_clusters = db.query(Cluster).filter_by(facet_id=facet.id).count()
    
    # State injection for React
    initial_state = {
        "view": "facet",
        "project_name": project_name,
        "facet": {"id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id},
        "total_clusters": total_clusters
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "initial_state": json.dumps(initial_state),
        "main_js": get_vite_assets()
    })

@app.get("/{project_name}/explore/{facet_name}/{feature_extractor_slug:path}/metadata", response_class=HTMLResponse)
async def view_metadata_settings(request: Request, project_name: str, facet_name: str, feature_extractor_slug: str, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    facet = db.query(Facet).filter(Facet.name.ilike(facet_name), Facet.feature_extractor_id.contains(feature_extractor_slug)).first()
    if not facet:
        raise HTTPException(status_code=404, detail="Facet not found")
        
    initial_state = {
        "view": "metadata",
        "project_name": project_name,
        "facet": {"id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id}
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "initial_state": json.dumps(initial_state),
        "main_js": get_vite_assets()
    })

@app.get("/{project_name}/explore/{facet_name}/{feature_extractor_slug:path}/cluster/{cluster_id}", response_class=HTMLResponse)
async def view_cluster(request: Request, project_name: str, facet_name: str, feature_extractor_slug: str, cluster_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    facet = db.query(Facet).filter(Facet.name.ilike(facet_name), Facet.feature_extractor_id.contains(feature_extractor_slug)).first()
    if not facet:
        raise HTTPException(status_code=404, detail="Facet not found")
    cluster = db.query(Cluster).filter_by(id=cluster_id, facet_id=facet.id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    cluster_label = cluster.cluster_label if cluster.cluster_label else f"{facet.name} {cluster.id}"
    cluster_size = db.query(Assignment).filter_by(cluster_id=cluster.id).count()
    
    initial_state = {
        "view": "cluster",
        "project_name": project_name,
        "facet": {"id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id},
        "cluster": {"id": cluster.id, "cluster_label": cluster_label, "status": cluster.status.value, "metadata": cluster.metadata_json, "size": cluster_size}
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "initial_state": json.dumps(initial_state),
        "main_js": get_vite_assets()
    })

from fastapi.responses import Response
@app.get("/{project_name}/api/thumbnail")
def get_thumbnail(project_name: str, media_id: int, timestamp: float, high_res: bool = False):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        thumbnail = app_state.project.thumbnail(media_id, timestamp, highres=high_res)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return Response(content=thumbnail, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/{project_name}/api/media/{media_id}")
def get_media_file(request: Request, project_name: str, media_id: int):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        metadata = app_state.project.metadata(media_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Media not found")
        
        file_path = metadata.full_path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
            
        if metadata.media_type in {MediaType.VIDEO, MediaType.AV, MediaType.AUDIO}:
            file_size = file_path.stat().st_size
            range_header = request.headers.get("range")
            
            content_type = f"{metadata.media_type.value}/{metadata.format}" if metadata.media_type == MediaType.AUDIO else f"video/mp4"
            headers = {
                "content-type": content_type,
                "accept-ranges": "bytes",
                "content-length": str(file_size),
                "access-control-expose-headers": "content-type, accept-ranges, content-length, content-range",
            }
            start = 0
            end = file_size - 1
            status_code = 200
            
            if range_header is not None:
                start, end = _get_range_header(range_header, file_size)
                size = end - start + 1
                headers["content-length"] = str(size)
                headers["content-range"] = f"bytes {start}-{end}/{file_size}"
                status_code = 206
                
            return StreamingResponse(
                send_bytes_range_requests(open(file_path, mode="rb"), start, end),
                headers=headers,
                status_code=status_code,
            )
        else:
            return FileResponse(file_path, media_type=f"image/{metadata.format.lower()}")
            
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/{project_name}/api/facet/{facet_id}/schema")
def get_facet_schema(project_name: str, facet_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    schema_entries = db.query(FacetMetadataSchema).filter_by(facet_id=facet_id).all()
    return [{"id": s.id, "key_name": s.key_name, "data_type": s.data_type} for s in schema_entries]

@app.post("/{project_name}/api/facet/{facet_id}/schema")
def create_facet_schema(project_name: str, facet_id: int, schema_create: SchemaCreate, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if key already exists
    existing = db.query(FacetMetadataSchema).filter_by(facet_id=facet_id, key_name=schema_create.key_name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Key already exists in schema")
        
    new_schema = FacetMetadataSchema(
        facet_id=facet_id,
        key_name=schema_create.key_name,
        data_type=schema_create.data_type
    )
    db.add(new_schema)
    db.commit()
    db.refresh(new_schema)
    return {"id": new_schema.id, "key_name": new_schema.key_name, "data_type": new_schema.data_type}

@app.put("/{project_name}/api/facet/{facet_id}/schema/{schema_id}")
def update_facet_schema(project_name: str, facet_id: int, schema_id: int, schema_update: SchemaUpdate, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    
    schema_entry = db.query(FacetMetadataSchema).filter_by(id=schema_id, facet_id=facet_id).first()
    if not schema_entry:
        raise HTTPException(status_code=404, detail="Schema entry not found")
        
    old_key = schema_entry.key_name
    new_key = schema_update.key_name if schema_update.key_name else old_key
    
    if new_key != old_key:
        existing = db.query(FacetMetadataSchema).filter_by(facet_id=facet_id, key_name=new_key).first()
        if existing:
            raise HTTPException(status_code=400, detail="Key already exists in schema")
            
        clusters = db.query(Cluster).filter_by(facet_id=facet_id).all()
        for cluster in clusters:
            if old_key in cluster.metadata_json:
                cluster.metadata_json[new_key] = cluster.metadata_json.pop(old_key)
                cluster.metadata_json = dict(cluster.metadata_json)
                
    if schema_update.key_name is not None:
        schema_entry.key_name = schema_update.key_name
    if schema_update.data_type is not None:
        schema_entry.data_type = schema_update.data_type
        
    db.commit()
    return {"id": schema_entry.id, "key_name": schema_entry.key_name, "data_type": schema_entry.data_type}

@app.delete("/{project_name}/api/facet/{facet_id}/schema/{schema_id}")
def delete_facet_schema(project_name: str, facet_id: int, schema_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
        
    schema_entry = db.query(FacetMetadataSchema).filter_by(id=schema_id, facet_id=facet_id).first()
    if not schema_entry:
        raise HTTPException(status_code=404, detail="Schema entry not found")
        
    key_to_delete = schema_entry.key_name
    clusters = db.query(Cluster).filter_by(facet_id=facet_id).all()
    for cluster in clusters:
        if key_to_delete in cluster.metadata_json:
            del cluster.metadata_json[key_to_delete]
            cluster.metadata_json = dict(cluster.metadata_json)
            
    db.delete(schema_entry)
    db.commit()
    return {"status": "success"}

@app.delete("/{project_name}/api/facet/{facet_id}/stars")
def clear_starred_clusters(project_name: str, facet_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
        
    db.query(Cluster).filter_by(facet_id=facet_id).update({"is_starred": False})
    db.commit()
    return {"status": "success"}

@app.post("/{project_name}/api/cluster/{cluster_id}/star")
def toggle_star_cluster(project_name: str, cluster_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
        
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
        
    cluster.is_starred = not cluster.is_starred
    db.commit()
    
    return {"status": "success", "starred": cluster.is_starred}

@app.post("/{project_name}/api/facet/{facet_id}/metadata/batch")
def batch_update_metadata(project_name: str, facet_id: int, batch_update: BatchMetadataUpdate, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
        
    starred_clusters = db.query(Cluster).filter(Cluster.facet_id == facet_id, Cluster.is_starred == True).all()
    
    for cluster in starred_clusters:
        new_meta = dict(cluster.metadata_json)
        new_meta.update(batch_update.metadata_json)
        cluster.metadata_json = new_meta
        cluster.status = ClusterStatus.reviewed
        
    db.commit()
    return {"status": "success", "updated_count": len(starred_clusters)}

@app.get("/{project_name}/api/facets")
def get_facets(project_name: str, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    facets = db.query(Facet).all()
    return [{"id": f.id, "name": f.name, "feature_extractor_id": f.feature_extractor_id} for f in facets]

@app.get("/{project_name}/api/facet/{facet_id}/clusters")
def get_clusters(project_name: str, facet_id: int, page: int = 1, page_size: int = 10, status_filter: str = "All", machine_feedback: str = "All", db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
        
    total_query = db.query(Cluster).filter(Cluster.facet_id == facet_id)
    if status_filter == "starred":
        total_query = total_query.filter(Cluster.is_starred == True)
    elif status_filter != "All":
        total_query = total_query.filter(Cluster.status == ClusterStatus(status_filter))

    if machine_feedback != "All":
        total_query = total_query.filter(Cluster.machine_feedback.contains(machine_feedback))

    total_count = total_query.count()
        
    # Get paginated clusters sorted by pre-calculated size (descending)
    query = db.query(Cluster).filter(Cluster.facet_id == facet_id)
    
    if status_filter == "starred":
        query = query.filter(Cluster.is_starred == True)
    elif status_filter != "All":
        query = query.filter(Cluster.status == ClusterStatus(status_filter))
        
    if machine_feedback != "All":
        query = query.filter(Cluster.machine_feedback.contains(machine_feedback))

    clusters = (
        query.order_by(Cluster.size.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    
    if not clusters:
        return {"clusters": [], "total": total_count}
        
    cluster_ids = [c.id for c in clusters]

    # Fetch assignments for each cluster efficiently by limiting to 100 per cluster natively in SQL
    cluster_ids_str = ",".join(map(str, cluster_ids))
    sampled_query = sa.text(f"""
        SELECT cluster_id, vector_id
        FROM (
            SELECT cluster_id, vector_id, 
                   ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY RANDOM()) as rn
            FROM assignments
            WHERE cluster_id IN ({cluster_ids_str})
        )
        WHERE rn <= 100
    """)
    result = db.execute(sampled_query).fetchall()
    
    from collections import defaultdict
    
    cluster_to_sampled_vectors = defaultdict(list)
    all_sampled_vector_ids = []
    for row in result:
        cluster_to_sampled_vectors[row.cluster_id].append(row.vector_id)
        all_sampled_vector_ids.append(row.vector_id)
    
    vector_info = {}
    if all_sampled_vector_ids:
        metadata_list = app_state.project.get_vector_media_metadata_for_ids(all_sampled_vector_ids)
        facet = db.query(Facet).filter_by(id=facet_id).first()
        ext_metadata_list = app_state.project.get_vector_ext_metadata_for_ids(facet.feature_extractor_id, all_sampled_vector_ids)
        for m, ext in zip(metadata_list, ext_metadata_list):
            area = (ext.bbox.w * m.width) * (ext.bbox.h * m.height) if hasattr(ext, 'bbox') else 0
            vector_info[m.id] = {
                "media_id": m.media_id,
                "timestamp": m.timestamp,
                "bbox": {"x": ext.bbox.x, "y": ext.bbox.y, "w": ext.bbox.w, "h": ext.bbox.h} if hasattr(ext, 'bbox') else None,
                "area": area
            }

    clusters_data = []
    for c in clusters:
        cluster_label = c.cluster_label if c.cluster_label else f"{facet.name} {c.id}"
        
        sampled_vids = cluster_to_sampled_vectors.get(c.id, [])
        unique_media_count = c.unique_media_count # READ FROM DB directly!

        vids_info = [vector_info[vid] for vid in sampled_vids if vid in vector_info]
        vids_info.sort(key=lambda x: x["area"], reverse=True)
        
        reps = []
        seen_media_ids = set()
        for info in vids_info:
            if info["media_id"] not in seen_media_ids:
                reps.append(info)
                seen_media_ids.add(info["media_id"])
                if len(reps) == 9:
                    break
                    
        # If we couldn't find 9 from distinct media_ids, fill the rest with largest remaining faces
        if len(reps) < 9:
            for info in vids_info:
                if info not in reps:
                    reps.append(info)
                    if len(reps) == 9:
                        break
                        
        clusters_data.append({
            "id": c.id, 
            "cluster_label": cluster_label, 
            "status": c.status.value,
            "machine_feedback": c.machine_feedback,
            "size": c.size,
            "starred": c.is_starred,
            "unique_media_count": unique_media_count,
            "representative_faces": reps
        })
        
    return {"clusters": clusters_data, "total": total_count}

@app.get("/{project_name}/api/cluster/{cluster_id}/faces")
def get_cluster_faces(project_name: str, cluster_id: int, page: int = 1, page_size: int = 50, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
        
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        return []
    facet = db.query(Facet).filter_by(id=cluster.facet_id).first()
    
    assignments = db.query(Assignment).filter_by(cluster_id=cluster_id).offset((page - 1) * page_size).limit(page_size).all()
    vector_ids = [a.vector_id for a in assignments]
    
    if not vector_ids:
        return []
        
    # Get metadata from internal.db
    metadata = app_state.project.get_vector_media_metadata_for_ids(vector_ids)
    ext_metadata = app_state.project.get_vector_ext_metadata_for_ids(facet.feature_extractor_id, vector_ids)
    
    results = []
    for m, ext in zip(metadata, ext_metadata):
        results.append({
            "vector_id": m.id,
            "media_id": m.media_id,
            "filename": Path(m.path).name,
            "timestamp": m.timestamp,
            "bbox": {"x": ext.bbox.x, "y": ext.bbox.y, "w": ext.bbox.w, "h": ext.bbox.h} if hasattr(ext, 'bbox') else None
        })
    return results

@app.get("/{project_name}/api/cluster/{cluster_id}/faces_by_media")
def get_cluster_faces_by_media(project_name: str, cluster_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")

    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    facet = db.query(Facet).filter_by(id=cluster.facet_id).first()
    if not facet:
        raise HTTPException(status_code=500, detail="Parent facet not found for cluster.")

    assignments = db.query(Assignment).filter_by(cluster_id=cluster_id).all()
    vector_ids = [a.vector_id for a in assignments]

    if not vector_ids:
        return {}

    # Get metadata from internal.db
    metadata = app_state.project.get_vector_media_metadata_for_ids(vector_ids)
    ext_metadata = app_state.project.get_vector_ext_metadata_for_ids(facet.feature_extractor_id, vector_ids)

    # Group by media_id
    grouped_faces = {}
    for m, ext in zip(metadata, ext_metadata):
        if m.media_id not in grouped_faces:
            grouped_faces[m.media_id] = {
                "filename": Path(m.path).name,
                "faces": []
            }

        grouped_faces[m.media_id]["faces"].append({
            "vector_id": m.id,
            "timestamp": m.timestamp,
            "bbox": {"x": ext.bbox.x, "y": ext.bbox.y, "w": ext.bbox.w, "h": ext.bbox.h} if hasattr(ext, 'bbox') else None
        })

    # Sort faces within each group by timestamp
    for media_id in grouped_faces:
        grouped_faces[media_id]["faces"].sort(key=lambda x: x["timestamp"])

    return grouped_faces

@app.put("/{project_name}/api/cluster/{cluster_id}")
def update_cluster(project_name: str, cluster_id: int, update: ClusterUpdate, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
        
    if update.cluster_label is not None:
        cluster.cluster_label = update.cluster_label
    if update.metadata_json is not None:
        cluster.metadata_json = update.metadata_json
    if update.status is not None:
        cluster.status = ClusterStatus(update.status)
        
        if cluster.status == ClusterStatus.reviewed:
            # Populate the known_face_clusters table
            logger.info(f"Populating known-face-clusters for cluster {cluster_id}")
            from scripts.explore.models import KnownFaceCluster
            import faiss
            import numpy as np
            from src.data_models import ModalityType

            # Get all vectors for this cluster
            assignments = db.query(Assignment).filter_by(cluster_id=cluster_id).all()
            vector_ids = [a.vector_id for a in assignments]

            if vector_ids:
                # First, get the facet to find the feature_extractor_id
                facet = db.query(Facet).filter_by(id=cluster.facet_id).first()
                if not facet:
                    logger.error(f"Could not find parent facet for cluster {cluster_id}")
                    raise HTTPException(status_code=500, detail="Parent facet not found for cluster.")

                try:
                    # Load the main FAISS index to reconstruct vectors
                    logger.info("Loading FAISS index to reconstruct vectors for centroid calculation...")
                    index_path = app_state.project.index_dir(facet.feature_extractor_id) / f"{ModalityType.VIDEO.value}-IndexFlatIP.faiss"
                    if not index_path.exists():
                        logger.error(f"Cannot populate known clusters: FAISS index not found at {index_path}")
                        raise HTTPException(status_code=500, detail="FAISS index not found.")

                    index = faiss.read_index(str(index_path))
                    if not hasattr(index, 'reconstruct'):
                        logger.error("Cannot populate known clusters: FAISS index does not support vector reconstruction.")
                        raise HTTPException(status_code=500, detail="FAISS index does not support reconstruction.")

                    logger.info(f"Reconstructing {len(vector_ids)} vectors for cluster {cluster_id}...")
                    embeddings = np.array([index.reconstruct(int(vid)) for vid in vector_ids]).astype(np.float32)
                    logger.info("Vector reconstruction successful.")

                    if len(embeddings) > 0:
                        logger.info("Calculating centroid...")
                        centroid = np.mean(embeddings, axis=0)
                        logger.info("Centroid calculation successful.")

                        # Clear old entries and insert new ones
                        logger.info(f"Updating known_face_clusters table for cluster {cluster_id}...")
                        db.query(KnownFaceCluster).filter_by(cluster_id=cluster_id).delete()
                        db.flush()

                        new_known_clusters = []
                        for vid in vector_ids:
                            new_known_clusters.append(KnownFaceCluster(
                                cluster_id=cluster_id,
                                vector_id=vid,
                                centroid=centroid
                            ))
                        db.bulk_save_objects(new_known_clusters)
                        logger.info("Successfully updated known_face_clusters table.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing cluster {cluster_id} for review: {e}", exc_info=True)
                    # Raise an HTTPException to provide feedback to the frontend
                    raise HTTPException(status_code=500, detail=f"Failed to process reviewed cluster: {e}")

        elif cluster.status == ClusterStatus.draft:
            # If user reverts a cluster back to draft, remove it from known clusters
            from scripts.explore.models import KnownFaceCluster
            db.query(KnownFaceCluster).filter_by(cluster_id=cluster_id).delete()

    db.commit()
    return {"status": "success"}

@app.put("/{project_name}/api/assignments")
def update_assignments(project_name: str, update: AssignmentUpdate, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    assignments = db.query(Assignment).filter(Assignment.vector_id.in_(update.vector_ids)).all()
    for a in assignments:
        a.cluster_id = update.new_cluster_id
        a.is_manual_override = True
        
    db.commit()
    return {"status": "success"}

class MergeClustersRequest(BaseModel):
    primary_cluster_id: int
    secondary_cluster_ids: List[int]
    new_cluster_label: str

@app.post("/{project_name}/api/clusters/merge")
def merge_clusters(project_name: str, request: MergeClustersRequest, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")

    # Verify primary cluster exists
    primary_cluster = db.query(Cluster).filter_by(id=request.primary_cluster_id).first()
    if not primary_cluster:
        raise HTTPException(status_code=404, detail=f"Primary cluster with id {request.primary_cluster_id} not found.")

    # Update assignments from secondary clusters to the primary one
    db.query(Assignment).filter(Assignment.cluster_id.in_(request.secondary_cluster_ids)).update({
        'cluster_id': request.primary_cluster_id
    }, synchronize_session=False)

    # Delete the now-empty secondary clusters
    db.query(Cluster).filter(Cluster.id.in_(request.secondary_cluster_ids)).delete(synchronize_session=False)

    # Update the primary cluster's label
    primary_cluster.cluster_label = request.new_cluster_label

    # Recalculate unique_media_count for the newly merged primary cluster
    try:
        logger.info(f"Recalculating unique_media_count for merged cluster {primary_cluster.id}")
        all_assignments = db.query(Assignment).filter_by(cluster_id=primary_cluster.id).all()
        all_vector_ids = [a.vector_id for a in all_assignments]
        if all_vector_ids:
            chunk_size = 900
            all_metadata = []
            for i in range(0, len(all_vector_ids), chunk_size):
                chunk = all_vector_ids[i:i + chunk_size]
                all_metadata.extend(app_state.project.get_vector_media_metadata_for_ids(chunk))

            unique_media_ids = {m.media_id for m in all_metadata}
            primary_cluster.unique_media_count = len(unique_media_ids)
            primary_cluster.size = len(all_vector_ids)
    except Exception as e:
        logger.error(f"Failed to recalculate unique_media_count after merge for cluster {primary_cluster.id}: {e}", exc_info=True)

    # After merging, the primary cluster is implicitly reviewed, so we should update its known_cluster entry
    try:
        logger.info(f"Updating known-face-clusters for merged cluster {primary_cluster.id}")
        assignments = db.query(Assignment).filter_by(cluster_id=primary_cluster.id).all()
        vector_ids = [a.vector_id for a in assignments]
        if vector_ids:
            facet = db.query(Facet).filter_by(id=primary_cluster.facet_id).first()
            if facet:
                index_path = app_state.project.index_dir(facet.feature_extractor_id) / f"{ModalityType.VIDEO.value}-IndexFlatIP.faiss"
                if index_path.exists():
                    index = faiss.read_index(str(index_path))
                    if hasattr(index, 'reconstruct'):
                        embeddings = np.array([index.reconstruct(int(vid)) for vid in vector_ids]).astype(np.float32)
                        if len(embeddings) > 0:
                            centroid = np.mean(embeddings, axis=0)
                            db.query(KnownFaceCluster).filter_by(cluster_id=primary_cluster.id).delete()
                            db.flush()
                            new_known_clusters = [KnownFaceCluster(cluster_id=primary_cluster.id, vector_id=vid, centroid=centroid) for vid in vector_ids]
                            db.bulk_save_objects(new_known_clusters)
    except Exception as e:
        logger.error(f"Failed to update known clusters after merge for cluster {primary_cluster.id}: {e}", exc_info=True)

    db.commit()

    return {"status": "success", "merged_into": request.primary_cluster_id}

@app.post("/{project_name}/api/facet/{facet_id}/publish")
def publish_facet(project_name: str, facet_id: int, db = Depends(get_db)):
    if project_name != app_state.project.name:
        raise HTTPException(status_code=404, detail="Project not found")
    """Publish reviewed clusters to internal.db ExploreMetadata"""
    facet = db.query(Facet).filter_by(id=facet_id).first()
    if not facet:
        raise HTTPException(status_code=404, detail="Facet not found")
        
    reviewed_clusters = db.query(Cluster).filter_by(facet_id=facet_id, status=ClusterStatus.reviewed).all()
    
    if not reviewed_clusters:
        return {"status": "no reviewed clusters to publish"}
        
    internal_engine = app_state.project.db_engine

    # Ensure facet tables are created in internal.db before publishing
    from src.db.base import facets_metadata_obj
    facets_metadata_obj.create_all(internal_engine)

    with internal_engine.begin() as conn:
        # Sync facet to internal.db first
        existing_facet = conn.execute(
            sa.select(wise_tables.facets_table).where(wise_tables.facets_table.c.id == facet.id)
        ).first()
        
        if not existing_facet:
            conn.execute(wise_tables.facets_table.insert().values([{
                "id": facet.id,
                "name": facet.name,
                "feature_extractor_id": facet.feature_extractor_id
            }]))

        for cluster in reviewed_clusters:
            assignments = db.query(Assignment).filter_by(cluster_id=cluster.id).all()
            if not assignments:
                continue
                
            # Delete old records for this cluster to support updates
            conn.execute(
                wise_tables.facet_metadata_table.delete().where(
                    wise_tables.facet_metadata_table.c.cluster_id == cluster.id
                )
            )
            conn.execute(
                wise_tables.cluster_metadata_table.delete().where(
                    wise_tables.cluster_metadata_table.c.cluster_id == cluster.id
                )
            )
                
            publish_label = cluster.cluster_label if cluster.cluster_label else f"{facet.name} {cluster.id}"
            
            # Insert into cluster_metadata_table
            conn.execute(wise_tables.cluster_metadata_table.insert().values([{
                "facet_id": facet.id,
                "cluster_id": cluster.id,
                "cluster_label": publish_label,
                "metadata_json": cluster.metadata_json
            }]))

            # Insert into facet_metadata_table
            insert_data = []
            for a in assignments:
                insert_data.append({
                    "vector_id": a.vector_id,
                    "cluster_id": cluster.id
                })
            
            if insert_data:
                conn.execute(wise_tables.facet_metadata_table.insert().values(insert_data))
                
            # Mark as published
            cluster.status = ClusterStatus.published
            
        db.commit()
        
    return {"status": "success", "published_clusters": len(reviewed_clusters)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=str, required=True, help="WISE project directory")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    
    project_dir = Path(args.project_dir)
    app_state.project_dir = project_dir
    app_state.project = WiseProject(project_dir)
    app_state.engine, app_state.SessionLocal = init_explore_db(project_dir)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
