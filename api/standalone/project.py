import logging
import io
from typing import BinaryIO

from .. import common
from ..services.project import (
    MediaNotFoundException, ThumbnailNotFoundException,
)

from src.data_models import MediaMetadata, MediaType, SourceCollectionType
from fastapi import HTTPException, status, APIRouter, Request
from fastapi.responses import (
    Response,
    FileResponse,
    PlainTextResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
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

@router.get("/info")
def get_info(project_info: ProjectInfoDep):
    return project_info.model_dump(by_alias=True)
