# Metadata

WISE supports metadata defined over temporal segments of videos. In
the future, the WISE team will add support to various other types of
metadata. Here is an example of metadata import and search based on
the Kinetics-6 dataset which is a set of 30 videos taken from the
[Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset).

The [Install](Install.md) guide describes the process of installing
WISE. We assume that the WISE software is installed in the `wise` folder.

First, we download the Kinetics-6 dataset as follows.

```
## 1. Download the Kinetics-6 dataset
mkdir -p wise-data/Kinetics-6
curl -sLO "https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6b.tar.gz"
tar -zxvf Kinetics-6b.tar.gz -C wise-data/Kinetics-6
```

Next, we extract visual and audio features and create a search index
that will allow us to perform audio and visual search on the video
collection.

```
## 2. Extract audiovisual features
mkdir -p wise-projects/
python3 extract-features.py \
  wise-data/Kinetics-6/ \
  --project-dir wise-projects/Kinetics-6/
```

The Kinetics-6 dataset comes with a sample metadata as shown below.
```
cat wise-data/Kinetics-6/metadata.csv

metadata_id,filename,starttime,stoptime,metadata
0,6XvsLPDioVA_000000_000010.mp4,0,10,coughing
1,7XXXwvatW1U_000051_000061.mp4,0,10,coughing
2,ADHjOYdb450_000002_000012.mp4,0,10,coughing
3,AFRoHj8B8DM_000116_000126.mp4,0,10,coughing
4,alcbLCIrT-s_000208_000218.mp4,0,10,coughing
5,5E20wCGF6Ig_000122_000132.mp4,0,10,frying-vegetables
6,hxK9mej0_zw_000086_000096.mp4,0,10,frying-vegetables
...
28,Pp45zkGEEp4_000019_000029.mp4,0,10,whistling
29,tzgEoLzwRDo_000005_000015.mp4,0,10,whistling
```

We import this metadata into WISE as follows.

```
python3 metadata.py import \
  --from-csv wise-data/Kinetics-6/metadata.csv \
  --metadata-id "Kinetics/6b/video_categories" \
  --col-metadata-id metadata_id \
  --col-filename "{metadata}/{filename}" \
  --col-starttime starttime \
  --col-stoptime stoptime \
  --col-metadata metadata \
  --project-dir wise-projects/Kinetics-6/
```

This creates a `video_categories` table in a sqlite database named
`wise-projects/Kinetics-6/metadata/Kinetics/6b.sqlite`. The database
and table name for storing the metadata is extracted from
`--metadata-id` flag. The video files in Kinetics-6 dataset are stored
as `jogging/xyz.mp4`, `singing/abc.mp4`, etc. Each row in the
`metadata.csv` file needs to be matched with existing video filenames
in the WISE project. Therefore, we use `--col-filname
"{metadata}/{filename}"` to generate a filename for each row in
`metadata.csv` such that it matches with the filenames stored in the
WISE project.

Next, we create an index of audiovisual features as well as text
metadata.

```
python3 create-index.py \
  --project-dir wise-projects/Kinetics-6/
```

Now, we are ready to search. First, let us verify that the metadata
has been imported successfuly by searching for videos tagged as `singing`.

```
python search.py \
  --query "singing" --in metadata \
  --project-dir wise-projects/Kinetics-6/

Searching wise-projects/Kinetics-6/ for
  [0] "singing" in metadata


                    Search results for "singing" in metadata                     
 Rank  Filename                               Time        Score   Original Ranks 
    0  singing/GO5DhmRmHco_000112_000122.mp4  0.0 - 10.0  -1.629  0              
    1  singing/I6NDj1EcP6w_000073_000083.mp4  0.0 - 10.0  -1.629  1              
    2  singing/WKSxT9T-P_U_000157_000167.mp4  0.0 - 10.0  -1.629  2              
    3  singing/arBpk6QCVFs_000064_000074.mp4  0.0 - 10.0  -1.629  3              
    4  singing/vdnskiY-DRc_000023_000033.mp4  0.0 - 10.0  -1.629  4              

(search completed in 0.001 sec.)
```

This shows that all the 5 videos inside `singing` folder have been
correctly tagged using the `singing` text metadata. We want to find
all the videos that contains music but is not tagged with the
`singing` metadata. We can search for `music` in the audio stream
(because the video stream is not ideal for identifying music) and
remove all the results that are tagged with `singing` metadata as follows.

```
python search.py \
  --query "music" --in audio \
  --query "singing" --not-in metadata \
  --project-dir wise-projects/Kinetics-6/

Searching wise-projects/Kinetics-6/ for
  [0] "music" in audio
  [1] "singing" not in metadata


            Search results for "music" in audio and "singing" not in metadata            
 Rank  Filename                                         Time       Score  Original Ranks 
    0  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 4.0  0.256  0              
    1  jogging/OmWoDAQM1kk_000000_000010.mp4            0.0 - 8.0  0.237  1,2            

(search completed in 0.263 sec.)
```

You can download the [Kinetics-6 dataset](https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6b.tar.gz)
(50MB) and manually verify that these two videos files does indeed contain
music.