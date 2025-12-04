# Metadata
WISE2 aims to support the following four types of metadata.

| Type of Metadata | Reserved Column Names in Metadata Table | Description |
|------------------|-----------------------------------------|-------------|
| Media Metadata   | media_id, NULL, NULL, NULL              | metadata associated with an image, video or audio file (e.g. file caption, author, description, etc) |
| Frame Metadata   | media_id, timestamp, NULL, NULL         | metadata associated with a video frame or audio sample |
| Segment Metadata | media_id, timestamp, end_timestamp, NULL| metadata associated with a video or audio temporal segment (e.g. automatic speech recognition data, etc) |
| Region Metadata  | media_id, timestamp, end_timestamp, vector_id | metadata associated with an image or frame region (e.g. face, object, etc) |

For each type of metadata, we write scripts that will populate the `metadata/internal.db` SQLite database with a new table that must have all the columns (i.e. reserved column names) described above. Illustrative examples of each type of metadata is shown below.

## Media Metadata
The script [`media-metadata.py`](../../tree/media-metadata/media-metadata.py) allows import of metadata associated with each image, video or audio file. Here is an example based on Kinetics-6c dataset which is a set of 30 videos taken from the [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) dataset.

The [Install](Install.md) guide describes the process of installing WISE. We assume that the WISE software has already been installed in the `wise` folder.

```bash
## 1. Download the Kinetics-6c dataset
mkdir -p wise-data/
curl -sLO "https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6c.tar.gz"
tar -zxvf Kinetics-6c.tar.gz -C wise-data/
```

Next, we create a WISE project based on these videos.

```bash
## 2. Extract audiovisual features
mkdir -p wise-projects/
python3 extract-features.py \
  wise-data/Kinetics-6c/ \
  --project-dir wise-projects/Kinetics-6c/
```

The Kinetics-6 dataset comes with a sample metadata as shown below.

```bash
cat wise-data/Kinetics-6c/metadata.csv

media_path,media_category,media_description
coughing/6XvsLPDioVA_000000_000010.mp4,"coughing","A person coughing while driving a car"
coughing/7XXXwvatW1U_000051_000061.mp4,"coughing","A girl coughs while talking"
coughing/ADHjOYdb450_000002_000012.mp4,"coughing","A baby coughts while opening a book" 
coughing/AFRoHj8B8DM_000116_000126.mp4,"coughing","Hillary Clinton coughts while speaking on stage"
...
```

This metadata can be imported into the existing WISE project using the [media-metadata.py](media-metadata.py) script as follows.

```bash
python3 media-metadata.py import \
  --metadata-id "Kinetics-6c" \
  --from-csv wise-data/Kinetics-6c/metadata.csv \
  --metadata-type "media" \
  --project-dir wise-projects/Kinetics-6c/

Loading metadata from CSV file wise-data/Kinetics-6c/metadata.csv ...
inserted 30 rows into table metadata-Kinetics-6c
```

**TODO**: show how this metadata appears in the web based search user interface of WISE

## Segment Metadata

**TODO**: Show an example based on the Automatic Speech Recognition (ASR) model applied to audio channel of videos.

# Importing Metadata Stored in VISE
The metadata stored in a project created using the VGG Image Search Engine
([VISE](https://www.robots.ox.ac.uk/~vgg/software/vise/)) software can be
imported into a WISE project as follows.

```bash
# VISE project folder: /data/vise/1516ci/
# WISE project folder: /data/wise/projects/1516ci/

python3 scripts/metadata/export-vise-metadata-as-csv.py\
  --vise-metadata-db /data/vise/1516ci/metadata_db.sqlite\
  --vise-join-colname filename\
  --project-dir /data/wise/projects/1516ci/\
  --out-csv-file /data/wise/dataset/1516ci/metadata_db.csv

  exported 15822 rows to /data/wise/dataset/1516ci/metadata_db.csv

# we manually rename column names "file_id" to "vise_file_id" and "filename"
# to "vise_filename" in the exported csv file so that these entries can be
# identified in the frontend

python3 media-metadata.py import \
  --metadata-id "1516ci-2024-11-25" \
  --from-csv /data/wise/dataset/1516ci/metadata_db.csv \
  --metadata-type "media" \
  --project-dir /data/wise/projects/1516ci/
```
