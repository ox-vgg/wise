# WISE Project Thumbnails

To help serve wise projects quickly, thumbnails (height: 192px) are obtained from the input media files at **0.5s** intervals and stored in the metadata database (described [here](./MetadataStore.md)) in the [extract-features](../extract-features.py) script

Each frame is then encoded as jpeg (with quality set as 80) and stored in the `thumbnails` table in the metadata database, under the `content` column, along with the timestamp and media_id

TODO:
Add index to timestamp column and media_id for faster access.

See [shot-detection](https://gitlab.com/vgg/wise/shot-detection/-/blob/main/shot_detection/repository/__init__.py?ref_type=heads#L16) code for an example of querying by `id` and `timestamp` interval

See [dataloader](../src/dataloader/dataset.py#L242) for the configuration of the thumbnail stream
