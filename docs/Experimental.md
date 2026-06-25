# Experimental Features

> **Note:** The features described in this document are still being finalized and may change. They are not yet stable.

## Updating an Existing Project

There are three main ways to update a WISE project:

- **Add New Features:**
  Run `extract-features` on your project without specifying a folder path. Provide the desired feature extractor to process all media files in the project.
  Example (adding face search features):
  ```
  python -m wise extract-features \
    --media-include "*.mp4" \
    --video-feature-id "deepinsight/insightface/buffalo_l/_unknown" \
    --project-dir /data/wise/my-project/
  ```

- **Add New Media:**
  Supply a folder containing new media files to `extract-features` to add them to your project.
  ```
  python -m wise extract-features \
    "/data/videos/new-set/" \
    --media-include "*.mp4" \
    --project-dir /data/wise/my-project/
  ```

- **Merge Projects:**
  Combine multiple WISE projects (with the same feature extraction setup but different media) into a new project. This is useful for handling large datasets in smaller chunks. The merge method copies features, thumbnails, media, and shots from source projects, and supports a dry run for review. Metadata merging is not yet automated and must be handled manually via SQLite dumps.
  ```
  python3 -m wise.wise_project merge --into DEST_PROJECT PROJECT_1 PROJECT_2 PROJECT_3 ...
  ```

## Aggregator
The aggregator feature of WISE allows to present search results from multiple standalone WISE projects.
The `tests/test-aggregator.sh` script shows an example of how to use this
feature on 3 sample projects based on the [aggregator-3](https://thor.robots.ox.ac.uk/wise/assets/test/aggregator-3.zip)
dataset. Here is an example of how this feature can be used in general.

```
# We assume that standalone WISE projects are hosted at the following URLs:
#   * http://localhost:10001/1/
#   * http://localhost:10002/2/
#   * http://localhost:10003/3/

REMOTE_PROJECTS='["http://localhost:10001/1/","http://localhost:10002/2/","http://localhost:10003/3/"]' \
  PORT=10000 \
  wise serve --project-dir tmp/123/
```

Users visiting the URL `http://localhost:10000/123/` will see WISE search interface that allows audiovisual
search on the contents of all the three standalone projects. This feature is useful for serving large scale
projects where a large dataset of media files are split across multiple projects. Each sub-project is hosted
on a separate machine with compute and memory resources that are only sufficient for hosting the sub-project.
