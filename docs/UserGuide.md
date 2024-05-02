# WISE User Guide

The [Installation](Install.md) page describes the process of installing the
WISE software. This tutorial assumes that the WISE software has already been
installed.

First, we download a set of sample videos which can be used to test the
audio and visual search capabilities of WISE.

```
# We assume that the current directory contains
# the WISE software source tree.
mkdir -p wise-data/Kinectics-6
curl -sLO "https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/Kinectics-6.tar.gz"
tar -zxvf Kinectics-6.tar.gz -C wise-data/Kinectics-6
```

Next, we extract visual and audio features and create a search index that will allows
us to perform audio and visual search on the video collection.

```
mkdir -p wise-projects/
python3 extract-features.py \
  --media-dir wise-data/Kinectics-6/ \
  --project-dir wise-projects/Kinectics-6/

python3 create-index.py \
  --project-dir wise-projects/Kinectics-6/
```

We can now search the video collection as follows.

```
python search.py \
  --query "cooking" --in video \
  --query "music" --in audio \
  --topk 20 \
  --project-dir wise-projects/Kinectics-6/
```

The search results, shown below, shows that this search query is able to find the video that shows
someone cooking food with music playing in the background.

```
Searching /data/beegfs/ultrafast/home/adutta/temp/wise/Kinectics-6/ for
  [0] "cooking" in video
  [1] "music" in audio

              Search results for "cooking" in video               
 Rank  Filename                                         Time      
    0  frying-vegetables/mwkOrWZxvrU_000006_000016.mp4  0.5 - 1.5 
    1  frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4  0.5 - 9.0 
    2  frying-vegetables/lUyXiF6KfgU_000296_000306.mp4  5.0 - 8.5 
    3  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 9.5

               Search results for "music" in audio                
 Rank  Filename                                         Time      
    0  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 4.0 
    1  jogging/OmWoDAQM1kk_000000_000010.mp4            0.0 - 4.0 
    2  singing/vdnskiY-DRc_000023_000033.mp4            0.0 - 4.0 
    3  singing/GO5DhmRmHco_000112_000122.mp4            0.0 - 4.0 
    4  singing/arBpk6QCVFs_000064_000074.mp4            0.0 - 4.0 
    5  singing/WKSxT9T-P_U_000157_000167.mp4            0.0 - 4.0 
    6  shouting/9NdaqLe2gIs_000022_000032.mp4           0.0 - 4.0 
    7  singing/I6NDj1EcP6w_000073_000083.mp4            4.0       
    8  jogging/UQsA-W-q3oA_000002_000012.mp4            4.0       
    9  frying-vegetables/5E20wCGF6Ig_000122_000132.mp4  0.0 - 4.0 
   10  jogging/QY8RJBxbLnA_000116_000126.mp4            0.0 - 4.0 
(search completed in 0.261 sec.)


    Search results for "cooking and music" in video and audio     
 Rank  Filename                                         Time      
    0  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 9.5 
```

The range value shown in the `Time` column (e.g. `0.0 - 9.5`) is obtained
by merging two or more results from the initial ranked list. The ranked
results that do not get merged are reported as a single timestamp
(e.g. `4.0`) which is same as it was in the original unmerged ranked list of
nearest neighbours.

If you want to try a large number of search queries, you can
try the WISE search console which is much faster as it needs to
load all the required assets (e.g. models) only once. Here is an
example of the same search query run in the search console.

```
$ python search.py \
  --project-dir wise-projects/Kinectics-6/

Starting WISE search console ...
Some examples queries (press Ctrl + D to exit):
  1. find cooking videos with music playing in background
     > --query "cooking" --in video --query music --in audio
  2. find videos showing train, show only top 3 results and export results to a file
     > --query train --in video --topk 3 --export-csv train.csv

[0] > --query "cooking" --in video --query music --in audio --topk 20
             Search results for ""cooking"" in video              
 Rank  Filename                                         Time      
    0  frying-vegetables/mwkOrWZxvrU_000006_000016.mp4  0.0 - 1.5 
    1  frying-vegetables/lUyXiF6KfgU_000296_000306.mp4  5.0 - 9.5 
    2  frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4  0.5 - 9.0 
    3  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 9.5 
               Search results for "music" in audio                
 Rank  Filename                                         Time      
    0  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 4.0 
    1  jogging/OmWoDAQM1kk_000000_000010.mp4            0.0 - 4.0 
    2  singing/vdnskiY-DRc_000023_000033.mp4            0.0 - 4.0 
    3  singing/GO5DhmRmHco_000112_000122.mp4            0.0 - 4.0 
    4  singing/arBpk6QCVFs_000064_000074.mp4            0.0 - 4.0 
    5  singing/WKSxT9T-P_U_000157_000167.mp4            0.0 - 4.0 
    6  shouting/9NdaqLe2gIs_000022_000032.mp4           0.0 - 4.0 
    7  singing/I6NDj1EcP6w_000073_000083.mp4            4.0       
    8  jogging/UQsA-W-q3oA_000002_000012.mp4            4.0       
    9  frying-vegetables/5E20wCGF6Ig_000122_000132.mp4  0.0 - 4.0 
   10  jogging/QY8RJBxbLnA_000116_000126.mp4            0.0 - 4.0 
(search completed in 0.273 sec.)


   Search results for ""cooking" and music" in video and audio    
 Rank  Filename                                         Time      
    0  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 9.5 
```

To save your results to a CSV text file, you can add the `--export-csv`
flag in the search console as shown below.

```
[1] > --query "cooking" --in video --query music --in audio --topk 20 --export-csv cooking-music.csv
...
...
saved results to cooking-music.csv

[2] > 
(press Ctrl + D to exit)
Bye

$ cat cooking-music.csv
query_text,media_type,rank,filename,timestamp
""cooking"",video,0,frying-vegetables/mwkOrWZxvrU_000006_000016.mp4,0.0 - 1.5
""cooking"",video,1,frying-vegetables/lUyXiF6KfgU_000296_000306.mp4,5.0 - 9.5
""cooking"",video,2,frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4,0.5 - 9.0
""cooking"",video,3,frying-vegetables/hxK9mej0_zw_000086_000096.mp4,0.0 - 9.5
"music",audio,0,frying-vegetables/hxK9mej0_zw_000086_000096.mp4,0.0 - 4.0
"music",audio,1,jogging/OmWoDAQM1kk_000000_000010.mp4,0.0 - 4.0
"music",audio,2,singing/vdnskiY-DRc_000023_000033.mp4,0.0 - 4.0
"music",audio,3,singing/GO5DhmRmHco_000112_000122.mp4,0.0 - 4.0
"music",audio,4,singing/arBpk6QCVFs_000064_000074.mp4,0.0 - 4.0
"music",audio,5,singing/WKSxT9T-P_U_000157_000167.mp4,0.0 - 4.0
"music",audio,6,shouting/9NdaqLe2gIs_000022_000032.mp4,0.0 - 4.0
"music",audio,7,singing/I6NDj1EcP6w_000073_000083.mp4,4.0
"music",audio,8,jogging/UQsA-W-q3oA_000002_000012.mp4,4.0
"music",audio,9,frying-vegetables/5E20wCGF6Ig_000122_000132.mp4,0.0 - 4.0
"music",audio,10,jogging/QY8RJBxbLnA_000116_000126.mp4,0.0 - 4.0
```

All the subsequent queries in the search console gets completed almost
instantly because the required assets (e.g. model files) have already
been loaded into the computer memory.