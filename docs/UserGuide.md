# WISE User Guide

The [Installation](Install.md) page describes the process of installing the
WISE software. This tutorial assumes that the WISE software has already been
installed.

First, we download a set of sample videos which can be used to test the
audio and visual search capabilities of WISE.

```
# We assume that the current directory contains
# the WISE software source tree.
mkdir -p wise-data/
curl -sLO "https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/Kinectics-6.tar.gz"
tar -zxvf Kinectics-6.tar.gz -C wise-data/
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
python3 search.py \
  --query cooking --in video \
  --query "music playing in background --in audio \
  --topk 10 \
  --project-dir wise-projects/Kinectics-6/
```

The search results, shown below, shows that this search query is able to find the video that shows
someone cooking food with music playing in the background.

```
Searching wise-projects/Kinectics-6/ for
  [0] "cooking" in video
  [1] "music playing in background" in audio


            Search results for "cooking" in video             
 Rank  Filename                                         Time  
    0  frying-vegetables/mwkOrWZxvrU_000006_000016.mp4  1.500 
    1  frying-vegetables/mwkOrWZxvrU_000006_000016.mp4  0.500 
    2  frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4  4.000 
    3  frying-vegetables/lUyXiF6KfgU_000296_000306.mp4  8.500 
    4  frying-vegetables/mwkOrWZxvrU_000006_000016.mp4  1.000 
    5  frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4  0.500 
    6  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.000 
    7  frying-vegetables/lUyXiF6KfgU_000296_000306.mp4  5.000 
    8  frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4  9.000 
    9  frying-vegetables/mT7vy1-KP_Q_000398_000408.mp4  7.000 


  Search results for "music playing in background" in audio   
 Rank  Filename                                         Time  
    0  singing/vdnskiY-DRc_000023_000033.mp4            4.000 
    1  singing/vdnskiY-DRc_000023_000033.mp4            0.000 
    2  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.000 
    3  frying-vegetables/hxK9mej0_zw_000086_000096.mp4  4.000 
    4  shouting/9NdaqLe2gIs_000022_000032.mp4           4.000 
    5  jogging/OmWoDAQM1kk_000000_000010.mp4            0.000 
    6  jogging/OmWoDAQM1kk_000000_000010.mp4            4.000 
    7  whistling/tzgEoLzwRDo_000005_000015.mp4          4.000 
    8  jogging/QY8RJBxbLnA_000116_000126.mp4            4.000 
    9  jogging/QY8RJBxbLnA_000116_000126.mp4            0.000 


 Search results for "cooking" in video and "music playing in 
                    background" in audio                     
 Filename                                         Time Range 
 frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 0.0  
 frying-vegetables/hxK9mej0_zw_000086_000096.mp4  0.0 - 4.0  


Search completed in 35 sec.
```
