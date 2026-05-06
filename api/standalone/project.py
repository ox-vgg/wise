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

import logging
import io
import json
import sqlalchemy as sa
from pathlib import Path
from typing import BinaryIO

from .. import common
from ..services.project import (
    MediaNotFoundException, ThumbnailNotFoundException, LocalWiseProjectService, WiseProjectService
)

from src.data_models import MediaMetadata, MediaType, SourceCollectionType
from fastapi import HTTPException, status, APIRouter, Request, Depends
from fastapi.responses import (
    Response,
    FileResponse,
    PlainTextResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
    HTMLResponse,
)
from ..dependencies import ConfigDep, ProjectServiceDep, ProjectInfoDep

logger = logging.getLogger(__name__)


def send_bytes_range_requests(
    file_obj: BinaryIO, start: int, end: int, chunk_size: int = 10_000
):
    """Send a file in chunks using Range Requests specification RFC7233

    `start` and `end` parameters are inclusive due to specification
    """
    with file_obj as f:
        f.seek(start)
        while (pos := f.tell()) <= end:
            read_size = min(chunk_size, end + 1 - pos)
            yield f.read(read_size)


def _get_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    def _invalid_range():
        return HTTPException(
            status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
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


"""
Returns a router with API routes for reading the project data

Provides
- /media/{_media_id} -> Access the original image/video/audio file from URL / disk
- /thumbs/{_id} -> Read the thumbnail as bytes from dataset
- /storyboard/{_media_id} -> Get a storyboard (a set of thumbnails used for the timeline hover previews in the video player UI)
- /metadata/{_media_id} -> Read the metadata associated with a media file
- /info -> Read the project level metadata
"""

router = APIRouter()

@router.get("/info")
def get_info(project_info: ProjectInfoDep):
    return project_info.normalized().model_dump(by_alias=True)


@router.api_route(
    "/media/{media_id}",
    responses={404: {"content": "text/plain"}, 302: {}},
    methods=["GET", "HEAD"],
)
def get_media_file(media_id: int, request: Request, config: ConfigDep, project_service: ProjectServiceDep):
    """
    Returns a media file given the media_id.
    If the requested file is an image, a FileResponse is returned.
    If the requested file is a video or audio file, a StreamingResponse is returned using Range Requests of a given file
    See: https://github.com/tiangolo/fastapi/discussions/7718#discussioncomment-5143493
    """
    try:
        metadata = project_service.metadata(media_id)

        # TODO (WISE 2) get source URI from imported_metadata table
        # # Send the source_uri if present, or try to read from source
        # # we read from
        # # Maybe do a HEAD request to check existence before redirect
        # # so that we can try to serve the file from disk if present?
        # if metadata.source_uri and is_valid_uri(metadata.source_uri):
        #     return RedirectResponse(metadata.source_uri, status_code=302)
        file_path = metadata.full_path
        if config.redirect_media_url_by_path:
            num_components = min(config.redirect_media_url_num_components, len(file_path.parts) - 1)
            parts = (config.redirect_media_url_prefix,) + file_path.parts[-num_components:]
            return RedirectResponse(f"{'/'.join(parts)}", status_code=302)

        if metadata.media_type in {MediaType.VIDEO, MediaType.AV, MediaType.AUDIO}:
            file_size = file_path.stat().st_size
            range_header = request.headers.get("range")

            content_type = f"{metadata.media_type.value}/{metadata.format}" if metadata.media_type == MediaType.AUDIO else f"video/mp4"
            headers = {
                "content-type": content_type,
                "accept-ranges": "bytes",
                "content-length": str(file_size),
                "access-control-expose-headers": (
                    "content-type, accept-ranges, content-length, "
                    "content-range"
                ),
            }
            start = 0
            end = file_size - 1
            status_code = status.HTTP_200_OK

            if range_header is not None:
                start, end = _get_range_header(range_header, file_size)
                size = end - start + 1
                headers["content-length"] = str(size)
                headers["content-range"] = f"bytes {start}-{end}/{file_size}"
                status_code = status.HTTP_206_PARTIAL_CONTENT

            return StreamingResponse(
                send_bytes_range_requests(open(file_path, mode="rb"), start, end),
                headers=headers,
                status_code=status_code,
            )

        else:
            # Image files

            # Look up the source_collections table and find the location and type
            # Handle case where we read the media file from disk, but it may not be there
            if metadata.source_collection.type == SourceCollectionType.DIR:
                # metadata.source_uri will be None, so we have to search for it on disk
                if file_path.is_file():
                    return FileResponse(
                        file_path, media_type=f"image/{metadata.format.lower()}"
                    )
            return PlainTextResponse(
                status_code=404, content=f"{media_id} not found!"
            )

    except MediaNotFoundException:
        return PlainTextResponse(
            status_code=404, content=f"{media_id} not found!"
        )


@router.get(
    "/thumbnail",
    responses={200: {"content": "image/jpeg"}, 404: {"content": "text/plain"}},
)
def get_thumbnail(config: ConfigDep, project_service: ProjectServiceDep, media_id: int, timestamp: float, high_res: bool = False):
    # Get a thumbnail given a thumbnail id
    try:
        if high_res:
            thumbnail = project_service.thumbnail(media_id, timestamp, highres=True)
        else:
            try:
                thumbnail = project_service.thumbnail(media_id=media_id, timestamp=timestamp)
            except ThumbnailNotFoundException as e:
                if config.use_shots:
                    # If no thumbnail found, try to get a high-res thumbnail from the original video
                    thumbnail = project_service.thumbnail(media_id=media_id, timestamp=timestamp, highres=True)
                else:
                    raise e
        return Response(
            content=thumbnail,
            media_type="image/jpeg",
            status_code=200,
        )
    except ThumbnailNotFoundException:
        raise HTTPException(
            status_code=404,
            detail=f"Thumbnail for media_id {media_id} and timestamp {timestamp} not found!",
        )


@router.get(
    "/storyboard/{_video_media_id}/{_partition}.jpg",
    responses={
        200: {"content": "image/jpeg"},
        404: {"content": "application/json"},
    },
)
def get_storyboard_image(_video_media_id: int, _partition: int, project_service: ProjectServiceDep, project_info: ProjectInfoDep):
    # TODO config
    # For videos longer than 30 minutes, reduce the frequency of thumbnails to 1 every 4 seconds
    num_seconds_per_image = 2 if project_info.num_thumbnails < (2 * 30 * 60) else 4
    try:

        storyboard = project_service.get_thumbnail_spritesheet(
            _video_media_id, num_seconds_per_image, _partition
        )
        buffered = io.BytesIO()
        storyboard.save(buffered, format="JPEG", quality=70)
        return Response(
            content=buffered.getvalue(),
            media_type="image/jpeg",
            status_code=200,
            headers={
                "Cache-Control": "public, max-age=86400",
            },
        )
    except ThumbnailNotFoundException:
        raise HTTPException(
            status_code=404,
            detail=f"Thumbnails not found for media_id={_video_media_id} partition: {_partition}!",
        )


@router.get(
    "/storyboard/{_video_media_id}.vtt",
    responses={
        200: {"content": "text/vtt"},
        404: {"content": "application/json"},
    },
)
def get_storyboard(_video_media_id: int, project_service: ProjectServiceDep, project_info: ProjectInfoDep):
    """
    Generate JSON storyboard for a given video (as per this documentation: https://www.vidstack.io/docs/player/core-concepts/loading?styling=default-theme#json).
    A storyboard image (like this example: https://media-files.vidstack.io/storyboard.jpg)
    is generated based on the existing thumbnails of the video, and is included in the response.
    This is used for the preview thumbnails in the frontend UI when hovering over the timeline in the video player.
    """
    # For videos longer than 30 minutes, reduce the frequency of thumbnails to 1 every 4 seconds
    num_seconds_per_image = 2 if project_info.num_thumbnails < (2 * 30 * 60) else 4
    try:
        vtt_content = project_service.get_webvtt_spritesheet(
            _video_media_id, num_seconds_per_image
        )
        return Response(
            content=vtt_content,
            status_code=200,
            media_type="text/vtt",
            headers={
                "Cache-Control": "public, max-age=86400",
            },
        )
    except ThumbnailNotFoundException:
        raise HTTPException(
            status_code=404,
            detail=f"Thumbnails not found for media_id={_video_media_id}!",
        )

@router.get(
    "/metadata/{_id}",
    response_model=MediaMetadata,
    response_model_exclude=set(["id", "source_collection_id", "size_in_bytes", "date_modified"]),
    responses={200: {"content": "application/json"}},
)
def get_metadata(_id: int, project_service: ProjectServiceDep):
    try:
        media_metadata = project_service.metadata(_id)
        return MediaMetadata(**media_metadata.model_dump(exclude={'source_collection'}))
    except MediaNotFoundException:
        raise HTTPException(status_code=404, detail=f"Metadata not found!")

@router.get(
    "/related-vectors/{_vector_id}",
    response_model=list[common.VectorInfo],
    responses={200: {"content": "application/json"}},
)
def get_related_vectors(_vector_id: int, project_service: ProjectServiceDep):
    related_rows = project_service.related_vectors(_vector_id)

    if not related_rows:
        vectors_ext_metadata = []
    else:
        vectors_ext_metadata = project_service.get_vector_ext_metadata_for_ids(
            related_rows[0].feature_extractor_id, [v.id for v in related_rows]
        )

    vectors_info = []
    for row, extm in zip(related_rows, vectors_ext_metadata):
        vectors_info.append(
            common.VectorInfo(
                vector_id=str(row.id),
                media_id=str(row.media_id),
                link=f"media/{row.media_id}#t={row.timestamp},{row.end_timestamp}",
                thumbnail=f"thumbnail?media_id={row.media_id}&timestamp={row.timestamp}",
                bbox=extm.bbox,
            )
        )
    return vectors_info

def _get_facets_with_previews(project_service: LocalWiseProjectService):
    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facets_table, cluster_metadata_table, facet_metadata_table
        facets = conn.execute(sa.select(facets_table)).fetchall()

        result = []
        for f in facets:
            # Get up to 4 clusters for this facet
            clusters = conn.execute(
                sa.select(cluster_metadata_table.c.cluster_id)
                .where(cluster_metadata_table.c.facet_id == f.id)
                .order_by(sa.func.random())
                .limit(4)
            ).fetchall()

            reps = []
            if clusters:
                cluster_ids = [c[0] for c in clusters]
                # For each cluster, get its assignments
                assignments = conn.execute(
                    sa.select(facet_metadata_table)
                    .where(facet_metadata_table.c.cluster_id.in_(cluster_ids))
                ).fetchall()

                from collections import defaultdict
                import random

                cluster_to_vectors = defaultdict(list)
                for a in assignments:
                    cluster_to_vectors[a.cluster_id].append(a.vector_id)

                all_vector_ids = []
                for c_id, vids in cluster_to_vectors.items():
                    if len(vids) > 20:
                        vids = random.sample(vids, 20)
                    all_vector_ids.extend(vids)

                vector_info = {}
                if all_vector_ids:
                    metadata_list = project_service.wise_project.get_vector_media_metadata_for_ids(all_vector_ids)
                    ext_metadata_list = project_service.wise_project.get_vector_ext_metadata_for_ids(f.feature_extractor_id, all_vector_ids)
                    for m, ext in zip(metadata_list, ext_metadata_list):
                        area = (ext.bbox.w * m.width) * (ext.bbox.h * m.height) if hasattr(ext, 'bbox') else 0
                        vector_info[m.id] = {
                            "media_id": m.media_id,
                            "timestamp": m.timestamp,
                            "bbox": {"x": ext.bbox.x, "y": ext.bbox.y, "w": ext.bbox.w, "h": ext.bbox.h} if hasattr(ext, 'bbox') else None,
                            "area": area
                        }

                for c_id in cluster_ids:
                    vids = cluster_to_vectors[c_id]
                    vids_info = [vector_info[vid] for vid in vids if vid in vector_info]
                    if vids_info:
                        # Get the largest face region
                        largest_face = max(vids_info, key=lambda x: x["area"])
                        reps.append(largest_face)

            result.append({
                "id": f.id,
                "name": f.name,
                "feature_extractor_id": f.feature_extractor_id,
                "preview_faces": reps
            })

    return result

@router.get("/facets/", response_class=HTMLResponse)
def get_facets_index(config: ConfigDep, project_info: ProjectInfoDep, project_service: ProjectServiceDep):
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        return HTMLResponse("<h1>Facets Unavailable</h1><p>The Facets interface is not currently supported while WISE is running in Aggregator (Multi-Shard) Mode. Please run WISE in Standalone mode to access this feature.</p>", status_code=400)

    facets_html_path = config.project_dir.parent.parent.parent / "frontend" / "dist" / "facets.html"
    # Alternative path if running installed vs dev
    if not facets_html_path.exists():
        facets_html_path = Path(__file__).parent.parent.parent / "frontend" / "dist" / "facets.html"

    try:
        with open(facets_html_path, "r") as f:
            html = f.read()
    except FileNotFoundError:
        return HTMLResponse("Facets UI not built. Please run npm run build in frontend.", status_code=500)

    facets_data = _get_facets_with_previews(project_service)

    initial_state = {
        "view": "index",
        "project_name": project_info.name,
        "facets": facets_data
    }

    html = html.replace("<!-- INITIAL_STATE_PLACEHOLDER -->", f"<script>window.__INITIAL_STATE__ = {json.dumps(initial_state)};</script>")
    html = html.replace('href="./', f'href="/{project_info.name}/')
    html = html.replace('src="./', f'src="/{project_info.name}/')
    return HTMLResponse(html)

@router.get("/facets/{facet_name}/{feature_extractor_slug:path}/cluster/{cluster_id}/", response_class=HTMLResponse)
def get_facets_cluster_detail(config: ConfigDep, project_info: ProjectInfoDep, project_service: ProjectServiceDep, facet_name: str, feature_extractor_slug: str, cluster_id: int):
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        return HTMLResponse("<h1>Facets Unavailable</h1><p>The Facets interface is not currently supported while WISE is running in Aggregator (Multi-Shard) Mode. Please run WISE in Standalone mode to access this feature.</p>", status_code=400)

    facets_html_path = config.project_dir.parent.parent.parent / "frontend" / "dist" / "facets.html"
    if not facets_html_path.exists():
        facets_html_path = Path(__file__).parent.parent.parent / "frontend" / "dist" / "facets.html"

    try:
        with open(facets_html_path, "r") as f:
            html = f.read()
    except FileNotFoundError:
        return HTMLResponse("Facets UI not built. Please run npm run build in frontend.", status_code=500)

    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facets_table, cluster_metadata_table, facet_metadata_table
        facet = conn.execute(
            sa.select(facets_table).where(
                facets_table.c.name.ilike(facet_name),
                facets_table.c.feature_extractor_id.contains(feature_extractor_slug)
            )
        ).first()

        if not facet:
            raise HTTPException(status_code=404, detail="Facet not found")

        cluster_info = conn.execute(
            sa.select(cluster_metadata_table).where(
                cluster_metadata_table.c.cluster_id == cluster_id,
                cluster_metadata_table.c.facet_id == facet.id
            )
        ).first()

        if not cluster_info:
            raise HTTPException(status_code=404, detail="Cluster not found")

        cluster_size = conn.execute(
            sa.select(sa.func.count()).select_from(facet_metadata_table).where(facet_metadata_table.c.cluster_id == cluster_info.cluster_id)
        ).scalar()

    initial_state = {
        "view": "cluster",
        "project_name": project_info.name,
        "facet": {"id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id},
        "cluster": {"id": cluster_info.cluster_id, "cluster_label": cluster_info.cluster_label, "metadata": cluster_info.metadata_json, "size": cluster_size}
    }

    html = html.replace("<!-- INITIAL_STATE_PLACEHOLDER -->", f"<script>window.__INITIAL_STATE__ = {json.dumps(initial_state)};</script>")
    html = html.replace('href="./', f'href="/{project_info.name}/')
    html = html.replace('src="./', f'src="/{project_info.name}/')
    return HTMLResponse(html)

@router.get("/facets/{facet_name}/{feature_extractor_slug:path}/", response_class=HTMLResponse)
def get_facets_cluster_overview(config: ConfigDep, project_info: ProjectInfoDep, project_service: ProjectServiceDep, facet_name: str, feature_extractor_slug: str):
    if not isinstance(project_service, LocalWiseProjectService):
        raise HTTPException(status_code=400, detail="Facets only supported on local projects")

    facets_html_path = config.project_dir.parent.parent.parent / "frontend" / "dist" / "facets.html"
    if not facets_html_path.exists():
        facets_html_path = Path(__file__).parent.parent.parent / "frontend" / "dist" / "facets.html"

    try:
        with open(facets_html_path, "r") as f:
            html = f.read()
    except FileNotFoundError:
        return HTMLResponse("Facets UI not built. Please run npm run build in frontend.", status_code=500)

    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facets_table, cluster_metadata_table
        facet = conn.execute(
            sa.select(facets_table).where(
                facets_table.c.name.ilike(facet_name),
                facets_table.c.feature_extractor_id.contains(feature_extractor_slug)
            )
        ).first()

        if not facet:
            raise HTTPException(status_code=404, detail="Facet not found")

        total_clusters = conn.execute(
            sa.select(sa.func.count()).select_from(cluster_metadata_table).where(cluster_metadata_table.c.facet_id == facet.id)
        ).scalar()

    initial_state = {
        "view": "facet",
        "project_name": project_info.name,
        "facet": {"id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id},
        "total_clusters": total_clusters
    }

    html = html.replace("<!-- INITIAL_STATE_PLACEHOLDER -->", f"<script>window.__INITIAL_STATE__ = {json.dumps(initial_state)};</script>")
    html = html.replace('href="./', f'href="/{project_info.name}/')
    html = html.replace('src="./', f'src="/{project_info.name}/')
    return HTMLResponse(html)

@router.get("/api/facets")
def get_facets_list_api(config: ConfigDep, project_service: ProjectServiceDep):
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        raise HTTPException(status_code=400, detail="Facets only supported on local projects")
    return _get_facets_with_previews(project_service)

@router.get("/api/facets/{facet_name}/{feature_extractor_slug:path}/cluster/{cluster_id}/info")
def get_facet_cluster_info_api(facet_name: str, feature_extractor_slug: str, cluster_id: int, project_service: ProjectServiceDep):
    if not isinstance(project_service, LocalWiseProjectService):
        return JSONResponse({"error": "Facets are not supported in Aggregator Mode."}, status_code=400)
    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facets_table, cluster_metadata_table, facet_metadata_table
        facet = conn.execute(
            sa.select(facets_table).where(
                facets_table.c.name.ilike(facet_name),
                facets_table.c.feature_extractor_id.contains(feature_extractor_slug)
            )
        ).first()
        if not facet:
            raise HTTPException(status_code=404, detail="Facet not found")
        cluster_info = conn.execute(
            sa.select(cluster_metadata_table).where(
                cluster_metadata_table.c.cluster_id == cluster_id,
                cluster_metadata_table.c.facet_id == facet.id
            )
        ).first()
        if not cluster_info:
            raise HTTPException(status_code=404, detail="Cluster not found")
        cluster_size = conn.execute(
            sa.select(sa.func.count()).select_from(facet_metadata_table).where(facet_metadata_table.c.cluster_id == cluster_info.cluster_id)
        ).scalar()

    return {
        "id": cluster_info.cluster_id,
        "cluster_label": cluster_info.cluster_label,
        "metadata": cluster_info.metadata_json,
        "size": cluster_size,
        "facet": {"id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id}
    }

@router.get("/api/facets/{facet_name}/{feature_extractor_slug:path}/info")
def get_facet_info_api(config: ConfigDep, facet_name: str, feature_extractor_slug: str, project_service: ProjectServiceDep):
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        raise HTTPException(status_code=400, detail="Facets only supported on local projects")
    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facets_table, cluster_metadata_table
        facet = conn.execute(
            sa.select(facets_table).where(
                facets_table.c.name.ilike(facet_name),
                facets_table.c.feature_extractor_id.contains(feature_extractor_slug)
            )
        ).first()
        if not facet:
            raise HTTPException(status_code=404, detail="Facet not found")
        total_clusters = conn.execute(
            sa.select(sa.func.count()).select_from(cluster_metadata_table).where(cluster_metadata_table.c.facet_id == facet.id)
        ).scalar()
    return {
        "id": facet.id, "name": facet.name, "feature_extractor_id": facet.feature_extractor_id,
        "total_clusters": total_clusters
    }

@router.get("/api/facets/{facet_id}/clusters")
def get_published_clusters(config: ConfigDep, facet_id: int, project_service: ProjectServiceDep, page: int = 1, page_size: int = 10):
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        return JSONResponse({"error": "Facets are not supported in Aggregator Mode."}, status_code=400)

    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import cluster_metadata_table, facet_metadata_table
        from src.db.tables import vectors_table

        # Paginated clusters
        query = (
            sa.select(
                cluster_metadata_table.c.id,
                cluster_metadata_table.c.cluster_id,
                cluster_metadata_table.c.cluster_label,
                cluster_metadata_table.c.metadata_json,
                sa.func.count(facet_metadata_table.c.vector_id).label('size'),
                sa.func.count(sa.distinct(vectors_table.c.media_id)).label('unique_media_count')
            )
            .select_from(
                cluster_metadata_table.outerjoin(
                    facet_metadata_table,
                    cluster_metadata_table.c.cluster_id == facet_metadata_table.c.cluster_id
                ).outerjoin(
                    vectors_table,
                    facet_metadata_table.c.vector_id == vectors_table.c.id
                )
            )
            .where(cluster_metadata_table.c.facet_id == facet_id)
            .group_by(
                cluster_metadata_table.c.id,
                cluster_metadata_table.c.cluster_id,
                cluster_metadata_table.c.cluster_label
            )
            .order_by(sa.desc('size'))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        clusters = conn.execute(query).fetchall()

        total_clusters = conn.execute(
            sa.select(sa.func.count()).select_from(cluster_metadata_table).where(cluster_metadata_table.c.facet_id == facet_id)
        ).scalar()

        if not clusters:
            return {"clusters": [], "total": total_clusters}

        cluster_ids = [c.cluster_id for c in clusters]

        # We need a map of size and unique counts since we modified the query return structure
        cluster_stats = {c.cluster_id: {"size": c.size, "unique_media_count": c.unique_media_count, "cluster_label": c.cluster_label, "metadata_json": c.metadata_json} for c in clusters}

        # Fetch vector assignments using window function for efficiency
        cluster_ids_str = ",".join(map(str, cluster_ids))
        sampled_query = sa.text(f"""
            SELECT cluster_id, vector_id
            FROM (
                SELECT cluster_id, vector_id,
                       ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY RANDOM()) as rn
                FROM facet_metadata
                WHERE cluster_id IN ({cluster_ids_str})
            )
            WHERE rn <= 100
        """)
        assignments = conn.execute(sampled_query).fetchall()

    from collections import defaultdict
    cluster_to_vectors = defaultdict(list)
    for a in assignments:
        cluster_to_vectors[a.cluster_id].append(a.vector_id)

    all_vector_ids = [vid for vids in cluster_to_vectors.values() for vid in vids]

    vector_info = {}
    if all_vector_ids:
        metadata_list = project_service.wise_project.get_vector_media_metadata_for_ids(all_vector_ids)
        with project_service.wise_project.db_engine.connect() as conn:
            from src.db.tables.facets import facets_table
            facet = conn.execute(sa.select(facets_table).where(facets_table.c.id == facet_id)).first()
        ext_metadata_list = project_service.wise_project.get_vector_ext_metadata_for_ids(facet.feature_extractor_id, all_vector_ids)

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
        stats = cluster_stats[c.cluster_id]
        vids = cluster_to_vectors[c.cluster_id]
        vids_info = [vector_info[vid] for vid in vids if vid in vector_info]
        vids_info.sort(key=lambda x: x["area"], reverse=True)

        reps = []
        seen_media_ids = set()
        for info in vids_info:
            if info["media_id"] not in seen_media_ids:
                reps.append(info)
                seen_media_ids.add(info["media_id"])
                if len(reps) == 9:
                    break

        if len(reps) < 9:
            for info in vids_info:
                if info not in reps:
                    reps.append(info)
                    if len(reps) == 9:
                        break

        clusters_data.append({
            "id": c.cluster_id,
            "cluster_label": stats["cluster_label"],
            "metadata_json": stats["metadata_json"],
            "size": stats["size"],
            "unique_media_count": stats["unique_media_count"],
            "representative_faces": reps
        })

    return {"clusters": clusters_data, "total": total_clusters}

@router.get("/api/facets/cluster/{cluster_id}/faces")
def get_facets_cluster_faces(config: ConfigDep, cluster_id: int, project_service: ProjectServiceDep, page: int = 1, page_size: int = 50):
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        return JSONResponse({"error": "Facets are not supported in Aggregator Mode."}, status_code=400)

    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facet_metadata_table, cluster_metadata_table, facets_table

        assignments = conn.execute(
            sa.select(facet_metadata_table).where(facet_metadata_table.c.cluster_id == cluster_id).offset((page - 1) * page_size).limit(page_size)
        ).fetchall()
        vector_ids = [a.vector_id for a in assignments]
        if not vector_ids:
            return []

        cluster_row = conn.execute(sa.select(cluster_metadata_table).where(cluster_metadata_table.c.cluster_id == cluster_id)).first()
        facet = conn.execute(sa.select(facets_table).where(facets_table.c.id == cluster_row.facet_id)).first()

    # Get metadata from internal.db
    metadata = project_service.wise_project.get_vector_media_metadata_for_ids(vector_ids)
    ext_metadata = project_service.wise_project.get_vector_ext_metadata_for_ids(facet.feature_extractor_id, vector_ids)

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

@router.get("/api/facets/cluster/{cluster_id}/faces_by_media")
async def get_published_cluster_faces_by_media(cluster_id: int, request: Request, project_service: ProjectServiceDep):
    config = project_service.config
    if not config.enable_facets:
        raise HTTPException(status_code=404, detail="Facets feature is disabled.")
    if not isinstance(project_service, LocalWiseProjectService):
        return JSONResponse({"error": "Facets are not supported in Aggregator Mode."}, status_code=400)

    with project_service.wise_project.db_engine.connect() as conn:
        from src.db.tables.facets import facet_metadata_table, cluster_metadata_table, facets_table

        # 1. Verify cluster exists and get its facet
        cluster_row = conn.execute(
            sa.select(cluster_metadata_table).where(cluster_metadata_table.c.cluster_id == cluster_id)
        ).first()

        if not cluster_row:
            return {}

        facet = conn.execute(sa.select(facets_table).where(facets_table.c.id == cluster_row.facet_id)).first()

        # 2. Get all assigned vectors
        assignments = conn.execute(
            sa.select(facet_metadata_table.c.vector_id)
            .where(facet_metadata_table.c.cluster_id == cluster_id)
        ).fetchall()

        vector_ids = [a.vector_id for a in assignments]

        if not vector_ids:
            return {}

    # 3. Get metadata from the main DB (Chunked for performance on massive clusters)
    metadata = []
    ext_metadata = []
    chunk_size = 900
    for i in range(0, len(vector_ids), chunk_size):
        chunk = vector_ids[i:i + chunk_size]
        metadata.extend(project_service.wise_project.get_vector_media_metadata_for_ids(chunk))
        ext_metadata.extend(project_service.wise_project.get_vector_ext_metadata_for_ids(facet.feature_extractor_id, chunk))

    # 4. Group by media_id
    grouped_faces = {}
    from pathlib import Path
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
