# WISE User Guide

```
## 1. Get the code
git clone -b wise2-integration https://gitlab.com/vgg/wise/wise.git
cd wise

## 2. Install software dependencies
## There are two ways to install software dependencies.
## 2.1 Using Conda
## 2.2 Using Python venv (requires Python >= 3.10)

## 2.1 Installing software dependencies using Conda
conda env create -f environment.yml
conda activate wise
pip install --no-deps msclap==1.3.3  # avoids installing conflicting version of torch

## 2.2 Alternatively, you can also use Python's venv module which
# supports creating lightweight “virtual environments”. The lines
# below are commented out to avoid confusion.
# python3 --version                  # must be >= 3.10
# sudo apt install ffmpeg            # ffmpeg is required to load videos
# python3 -m venv wise-dep/          # create virtual environment
# source wise-dep/bin/activate
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# pip install --no-deps msclap==1.3.3
# pip install -r torch-faiss-requirements.txt

## 3. Download some sample videos
mkdir -p wise-data/
curl -sLO "https://www.robots.ox.ac.uk/~vgg/software/wise/data/test/Kinectics-6.tar.gz"
tar -zxvf Kinectics-6.tar.gz -C wise-data/

## 4. Extract features
mkdir -p wise-projects/
python3 extract-features.py \
  --media-dir wise-data/Kinectics-6/ \
  --project-dir wise-projects/Kinectics-6/

## 5. Create search index
python3 create-index.py \
  --project-dir wise-projects/Kinectics-6/

## 6. Search
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
