# Wikimedia Commons Media of the Day (MOTD)

These instructions relate to the [Wikimedia Commons MOTD](https://meru.robots.ox.ac.uk/motd/) online demo.
The media files contained in this demo correspond to media that appeared as "Media of the Day" in the [Wikimedia Commons homepage](https://commons.wikimedia.org/wiki/Main_Page).

## Importing Metadata

```
# 1. Download all the metadata as JSON from Wikimedia Commons
cd scripts/wikimedia/motd
python3 fetch_json_metadata_from_wikimedia.py

# 2. Convert JSON metadata to CSV format
python3 export-json-metadata-to-csv.py \
    --project-dir ... \
    --metadata-dir ... \
    --out-csv-file /tmp/wise-motd-metadata.csv \
    --overrides-csv ... \
    --allow-missing

# 3. Import metadata into WISE project
cd ../../
python3 media-metadata.py import \
    --project-dir ... \
    --metadata-id wikimedia-motd \
    --metadata-type media \
    --from-csv /tmp/wise-motd-metadata.csv  

# 4. Create Full Text Search (FTS) index
export FTS_CONFIG='{"metadata-wikimedia-motd": ["media_id","image_description","date_time_original","artist","usage_terms","license_url","credit","restrictions"]}'
echo $FTS_CONFIG > /tmp/wise-fts-config.json
python3 create-index.py \
  --media-type metadata \
  --fts-config /tmp/wise-fts-config.json \
  --project-dir ...
```

