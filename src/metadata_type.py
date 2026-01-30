#!/usr/bin/env python3

from enum import Enum

class MetadataType(Enum):
    """
    Metadata corresponds to textual description of a media file (e.g. image,
    audio or video). These descriptions are often created manually but sometimes
    they can be generated automatically, for example, by a computer vision model.

    Media   : description of an image, audio or video file (e.g. caption)
    Segment : description of a temporal segment (e.g. 2.5s to 11.6s) in an audio or a video
    Frame   : description of a video frame (e.g. at time 6.43s)
    Region  : description of a spatial region (e.g. rectangle) defined in an image or video frame
    """
    MEDIA   = 1
    SEGMENT = 2
    FRAME   = 3
    REGION  = 4
