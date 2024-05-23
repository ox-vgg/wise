# WISE Project Thumbnails

To help serve wise projects quickly, thumbnails (height: 192px) are obtained from the input media files at **0.5s** intervals and stored in the `thumbs.db` database in the [extract-features](../extract-features.py) script

Each frame is then encoded as jpeg (with quality set as 80) and stored in the `thumbnails` table in the `thumbs.db` database, under the `content` column, along with the timestamp and media_id

See [shot-detection](../src/repository/__init__.py#L59) code for an example of querying by `id` and `timestamp` interval

See [dataloader](../src/dataloader/dataset.py#L247) for the configuration of the thumbnail stream

See [Internal Metadata Database documentation](../src/db/README.md) for more details on how the thumbnails database (and internal metadata database) can be accessed.