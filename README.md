# WISE

WISE - WISE Image Search Engine

WISE is an open-source image search engine for efficiently searching through large
collections of images. WISE uses various algorithms and models developed in the field
of computer vision and search & retrieval to allow the user to select the best
method for their purpose.

## Pre-requisites

(Tested on Debian 11 distro, but should work on all platforms)

- conda (You can install miniconda - [instructions](https://docs.conda.io/en/latest/miniconda.html))

## Setup

Set up a conda environment and activate it

```bash
conda env create -f environment.yml
conda activate clip-faiss
```

## Steps to search a collection of images using text queries

To search a collection of images using text, we use the OpenAI CLIP model.

To perform a search we must first extract the features from the dataset by processing the images
and then provide the search query.

### Extract features

Image features are extracted once using the CLIP visual model.
The features are written to an HDF5 file. This file is described here (TODO)

To extract features from the image collection, we use the following command

```bash
python3 app.py extract-features IMAGE_DIR DATASET.h5
```

(Optional Arguments - TODO)

### Search

Once the features are extracted, we can re-use them for every search query.
Query word embeddings are obtained from CLIP text model and then searched against
the extracted features.

```bash
python3 app.py search --dataset DATASET.h5 QUERY1 QUERY2 [...]
```

(Optional Arguments, Output description - TODO)

### Web Interface

A web interface is also provided to make queries against the collection.

To load the interface,

```bash
ln -s IMAGES_DIR public/images
# Optionally, you can also create thumbnails and set it accordingly or re-use image dir
ln -s IMAGES_DIR public/thumbs

DATASET="DATASET.h5" python3 app.py serve
```

You can now open http://localhost:8000/public/index.html to view the demo.
