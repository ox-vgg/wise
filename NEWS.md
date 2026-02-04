Summary of important user-visible changes for WISE
==================================================

Next version (unreleased)
-------------------------

* Added face search (refer to `InsightFaceFeatureExtractor` and
  `InsightFaceAverageFeatureExtractor`), and object search (refer to
  `TransformersOWLv2Model`).

* Removed support for feature stores in webdataset and numpy formats,
  only the faiss feature store is now supported.  The extract-features
  `--feature-store` option now only supports the `faiss` value.

* Removed the query blocklist feature because it was not effective, it
  was far too easy to bypass.

* The new `/search2` API uses a different query structure. Consider switching to
this new endpoint if you develop your own frontend. The older `/search` endpoint will be removed in the next
major release (and `/search2` will become `/search`)


2.1.0 (2025-02-04)
------------------

New features:

  * WISE CLI can now handle images.  WISE projects can now contain
    image, videos and audios.

  * WISE UI can display and search on images, videos and audio tracks
    in videos.

  * Internal search for images (searching using an exemplar from the
    collection) is now supported in WISE UI.

Breaking changes:

  * The `--media-dir` CLI option in the `extract-features.py` script
    has been changed to a positional argument (in commit
    a25a3346). This means instead of running the script like this:

        python extract-features.py \
            --media-dir /path/to/media/dir \
            --project-dir /path/to/project/dir

    Users should instead run the script like this (passing the media
    dir directly, without the `--media-dir`):

        python extract-features.py \
            /path/to/media/dir \
            --project-dir /path/to/project/dir

  * Internal metadata database change:

    Previously, the format column of the media table in the internal
    metadata database was used to store the codec (e.g. h264 or opus)
    of each video file.  From commit f54d3575 onwards, the format
    column is now used store the actual file format of each media file
    (e.g. mp4) based on its MIME type, rather than the media codec.
    For now, this probably won't cause any breaking changes, but it
    should be noted that future releases of WISE might not work with
    older projects (where the internal metadata database has saved the
    codec rather than file format of each media file).

Miscellaneous:

  * Add Colab notebook.

  * Docker - Dockerfile, Docker Compose Files, Multi arch Gitlab CI
    builds and Usage docs added. See `docs/Docker.md`

  * Various bug fixes.


2.0.1 (2024-06-11)
------------------

  * Clarify installation instructions.

  * Fix indexing performance (and show progress).

  * UI - Fix calculation of total duration of videos searched over.

  * UI - Minor layout changes.

  * Add script to import metadata and search with CLI.


1.2.0 (2024-03-13)
------------------

Features:

  * new search bar UI (cd321e05)

  * show visually similar images in the image pop-up view (a155c98f)

Enhancements:

  * add OPQ transform (7b7176d0)

  * add development mode to backend (67fc63ac)

  * add support for internal search based on reconstructed features
    (4e33eb06)

Bug fixes:

  * fix bug in wise.py index command and add option to save trained
    index to save time when adding more data (507d7ed2)

  * fix issue #21 (closed) (unable to process non-RGB images)

  * fix issue #22 (closed) ('IndexFlat' object has no attribute
    'direct_map')

  * fix bug where the ImageDetailsModal initially shows the previous
    image briefly when the modal is opened for a new image (6579cf39)

  * fix issue where ReportImageModal appears behind ImageDetailsModal
    (71b18fde)


1.1.0 (2023-11-18)
------------------

Some minor changes including:

  * Improved code for featured images (59ff3447)

  * Make text queries weighting and negative queries weighting
    configurable parameters in config.py (4bcbdc2c)

  * Added support for IVF+PQ based search index (f9b922f2)

  * Update internal search code to use feature vectors from h5 dataset
    (c8c12839)


1.0.0 (2023-09-12)
------------------

Various changes and feature additions including:

  * New CLI with commands such as python wise.py init, python wise.py
    index, python wise.py serve, etc.

  * New frontend UI

    * New UI design and logo.

    * Image search: users can drag and drop images, or enter the URL
      of an image.

    * Compound multimodal search: users can combine multiple image and
      text queries together.

    * Internal image queries.

    * Negative image queries.

    * Server-side pagination with client-side caching.

    * WISE Overview card on main page.

    * Image pop-up view with some metadata displayed if available.

    * Display a semi-random sample of featured images from the user's
      project, instead of using featured images from Wikimedia
      Commons.

  * Safety features:

    * Query blacklist/blocklist.
    * Report images that are inappropriate/offensive/etc.
    * Allow sensitive/NSFW images to be hidden behind a warning.


  * Updated REST API to support the above features including
    multimodal (image+text) search queries, improved pagination,
    safety features, etc.

  * Use OpenCLIP library instead of OpenAI CLIP library.

  * Use PyTorch Dataloader to improve image loading speed using
    multiple workers.

  * Add support for loading images from WebDatasets.

  * Implemented benchmarking code to compare performance between
    exhaustive nearest neighbour search and approximate nearest
    neighbour search.

  * Various improvements to faiss indexing.


0.1.0 (2023-03-16)
------------------

Initial release.

  * a basic web based user interface showing grid of images and a
    navigation panel to browser these images;
  * uses OpenAI CLIP model to extract image features and Facebook's
    faiss library to perform nearest neighbour search.
