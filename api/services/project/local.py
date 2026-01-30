#!/usr/bin/env python3

import datetime
import io
import math
import logging
from typing import Any

from config import APIConfig
from .exceptions import MediaNotFoundException, ThumbnailNotFoundException
from .base import WiseProjectService, ProjectInfo
from src.wise_project import WiseProject
from src.data_models import VectorAndMediaMetadata, MediaType
from src.repository import get_featured_images

from src.utils import convert_uint8array_to_base64

from PIL import Image
from webvtt import WebVTT, Caption

logger = logging.getLogger(__name__)

# TODO config
NUM_THUMBNAILS_PER_PARTITION = 300


def timedelta_to_vtt_timestamp(dt: datetime.timedelta):
    """
    Convert the timedelta python object to a HH:MM:SS.sss string
    Required by the VTT Captions
    """
    _seconds = dt.seconds

    hours = (dt.days * 24) + (_seconds // 3600)
    _seconds = _seconds % 3600

    minutes = _seconds // 60
    seconds = _seconds % 60

    milliseconds = dt.microseconds // 1000

    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

class LocalWiseProjectService(WiseProjectService):
    def __init__(self, wise_project: WiseProject, config: APIConfig):
        # Initialize other necessary attributes here
        self.wise_project = wise_project
        self.config = config

        # cache the current project assets
        self.project_assets = self.wise_project.discover_assets()
        self.search_indices = self.wise_project.load_search_indices()

        # Shot property (e.g. shot_scale, camera_motion, etc.) based filters
        shot_based_filters = None
        if config.use_shots:
            if self.wise_project.num_shots == 0:
                logger.warning('use_shots is set to True, but shots table is empty! Please make sure to populate the shots table before using this feature.')
            
            shot_scales = self.wise_project.shot_scales()
            if shot_scales:
                logger.info("shot_scale filter enabled with values =%s", shot_scales)
                shot_based_filters = {}
                shot_based_filters["shot_scale"] = {
                    "name": "Shot Scale",
                    "description": "Filter by the scale (or size) of the shot in edited videos.",
                    "options": shot_scales,
                }
            else:
                logger.warning("No shot_scale values found in shots table!")

        self.shot_based_filters = shot_based_filters
    
    @property
    def name(self) -> str:
        return self.wise_project.name

    def info(self) -> ProjectInfo:
        # Implement logic to retrieve project info from local files
        models = {
            media_type: [
                feature_extractor_id for feature_extractor_id in self.project_assets[media_type]
            ] for media_type in self.project_assets if media_type in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO]
        }
        search_target_order = getattr(self.config, "search_target_order", None)
        return ProjectInfo(
            project_name=self.name,
            num_vectors=self.wise_project.num_vectors,
            num_media_files=self.wise_project.num_media,
            num_thumbnails=self.wise_project.num_thumbnails,
            num_shots=self.wise_project.num_shots,
            media_file_counts=self.wise_project.media_file_counts,
            total_duration=self.wise_project.total_duration,
            models=models,
            search_targets=self.get_active_search_targets(search_target_order),
            shot_based_filters=self.shot_based_filters,
        )
    
    def metadata(self, media_id: str):
        # Implement logic to retrieve metadata for a given media_id from local files
        metadata = self.wise_project.metadata(media_id)
        if metadata is None:
            raise MediaNotFoundException(f"Media with ID {media_id} not found")
        
        return metadata
    
    def thumbnail(self, media_id: str, timestamp: float, get_id_only:bool = False, highres: bool = False) -> bytes | int:
        # Implement logic to retrieve thumbnail for a given media_id and timestamp from local files
        thumbnail = self.wise_project.thumbnail(media_id, timestamp, get_id_only=get_id_only, highres=highres)
        if thumbnail is None:
            raise ThumbnailNotFoundException(f"Thumbnail for media ID {media_id} at timestamp {timestamp} not found")
        return thumbnail
    
    def get_thumbnail_reader(self, thumbnails_to_send: int = 0):
        project = self.wise_project
        def _thumbnail_url(_m: VectorAndMediaMetadata):
            return f"thumbnail?media_id={_m.media_id}&timestamp={_m.timestamp}"

        def _thumbnail(_m: VectorAndMediaMetadata):
            thumbnail = project.thumbnail(media_id=_m.id, timestamp=_m.timestamp)
            return convert_uint8array_to_base64(thumbnail)

        def inner(vector_and_media_metadata_list: list[VectorAndMediaMetadata]):
            thumbs = [
                (
                    _thumbnail(vector_and_media_metadata)
                    if i < thumbnails_to_send
                    else _thumbnail_url(vector_and_media_metadata)
                )
                for i, vector_and_media_metadata in enumerate(
                    vector_and_media_metadata_list
                )
            ]
            return thumbs

        return inner
    
    def related_vectors(self, vector_id: int) -> list:
        return self.wise_project.related_vectors(vector_id)
    
    def featured_vectors_for_targets(self) -> dict[str, dict[str, list[int]]]:
        project_engine = self.wise_project.db_engine
        # Generate a list of random featured images for each modality and feature extractor
        ids: dict[str: dict[str: list[int]]] = {}
        search_targets = self.search_indices
        with project_engine.connect() as conn:
            for modality in search_targets:
                ids[modality] = {}
                for feature_extractor_id in search_targets[modality]:
                    if feature_extractor_id == 'wise/metadata':
                        continue
                    
                    this_ids = get_featured_images(
                        conn,
                        modality,
                        feature_extractor_id,
                    )
                    this_ids = list(filter(None, this_ids))
                    ids[modality][feature_extractor_id] = this_ids
        return ids

    def get_thumbnail_spritesheet(
        self,
        media_id: int,
        num_seconds_per_image: int,
        partition_id: int,
    ):
        all_thumbs = list(
            self.wise_project.get_thumbnails(
                media_id, num_seconds_per_image, partition_id, NUM_THUMBNAILS_PER_PARTITION
            )
        )
        num_thumbs = len(all_thumbs)
        if num_thumbs == 0:
            raise ThumbnailNotFoundException(
                f"No thumbnails found for media {media_id} and parition {partition_id}"
            )
        # Get thumbnails
        with Image.open(io.BytesIO(all_thumbs[0].content)) as im:
            w, h = im.size  # Assumes all thumbnails have the same size

        # Create storyboard
        num_columns = 10
        num_rows = math.ceil(num_thumbs / num_columns)
        storyboard = Image.new("RGB", (w * num_columns, h * num_rows))
        for idx, thumb in enumerate(all_thumbs):
            x = (idx % num_columns) * w
            y = (idx // num_columns) * h
            with Image.open(io.BytesIO(thumb.content)) as _thumb:
                storyboard.paste(_thumb, (x, y))

        return storyboard

    def get_webvtt_spritesheet(
        self, media_id: int, num_seconds_per_image: int = 2
    ):
        all_thumbs = list(
            self.wise_project.get_thumbnails(media_id, num_seconds_per_image)
        )
        num_thumbs = len(all_thumbs)
        if num_thumbs == 0:
            raise ThumbnailNotFoundException(f"No thumbnails found for media {media_id}!")

        w, h = self.wise_project.thumbnail_size_for_media_id(media_id)
        
        # Create storyboard
        num_columns = 10
        vtt = WebVTT()
        next_timestamp = datetime.timedelta(seconds=0)
        for thumb_idx, thumb in enumerate(all_thumbs):
            partition_id = thumb_idx // NUM_THUMBNAILS_PER_PARTITION
            idx = thumb_idx % NUM_THUMBNAILS_PER_PARTITION
            x = (idx % num_columns) * w
            y = (idx // num_columns) * h

            current_timestamp = datetime.timedelta(seconds=thumb.timestamp)
            next_timestamp = current_timestamp + datetime.timedelta(
                seconds=num_seconds_per_image
            )
            vtt.captions.append(
                Caption(
                    timedelta_to_vtt_timestamp(current_timestamp),
                    timedelta_to_vtt_timestamp(next_timestamp),
                    f"storyboard/{media_id}/{partition_id}.jpg#xywh={x},{y},{w},{h}",
                )
            )

        return vtt.content
    
    def get_active_search_targets(
       self, search_target_order: list[str] | None = None
    ):

        active_search_targets: dict[str, list[str]] = {
            k: list(self.search_indices[k].keys()) for k in self.search_indices
        }

        default_search_target_order = ["open_clip", "insightface", "owlv2", "clap", "wise/metadata"]
        _search_target_order = search_target_order or default_search_target_order

        # sort active search targets based on user defined order in config.search_target_order
        for media_type in active_search_targets:

            def sort_key(x):
                for i, partial in enumerate(_search_target_order):
                    if partial in x:
                        return i
                return len(_search_target_order)

            active_search_targets[media_type].sort(key=sort_key)

        preferred_order = [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO]
        active_search_targets = {
            x: active_search_targets[x]
            for x in sorted(
                active_search_targets.keys(), key=lambda x: preferred_order.index(x)
            )
        }

        return active_search_targets

    def get_vector_and_media_metadata_for_ids(
        self, ids: list[int], external_metadata_tables: list[str] | None = None
    ) -> list[VectorAndMediaMetadata]:
        _external_metadata_tables = external_metadata_tables
        if _external_metadata_tables is None:
            _external_metadata_tables = self.wise_project.external_metadata_tables()
        return self.wise_project.get_vector_media_metadata_for_ids(
            ids, _external_metadata_tables
        )
    
    def get_vector_ext_metadata_for_ids(
        self, feature_extractor_id: str, ids: list[int]
    ) -> list[Any]:
        return self.wise_project.get_vector_ext_metadata_for_ids(feature_extractor_id, ids)
