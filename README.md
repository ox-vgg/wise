# WISE

## Overview

WISE (WISE Image Search Engine) is an open-source image search engine for efficiently searching through large
collections of images. WISE uses various algorithms and models developed in the field
of computer vision and search & retrieval to allow the user to perform searches in different ways:
- natural language (text) queries (find images that match a given keyword like "bus" or a detailed description like "green double-decker bus at intersection")
- visual similarity search (find images that look like a given query image)
- image classification queries (find images with content similar to a set of query images)

## Pre-requisites

(Tested on Debian 11 distro, but should work on all platforms)

- conda (You can install miniconda - [instructions](https://docs.conda.io/en/latest/miniconda.html))

## Setup

Set up a conda environment and activate it

```bash
conda env create -f environment.yml
conda activate clip-faiss
```

## Steps to search a collection of images

As mentioned above, WISE can search a collection of images using natural language (text) queries, visual similarity, or image classification queries.

To perform a search we must first extract the features from the dataset by processing the images
and then provide the search query.

### Extract features

Image features are extracted once using the vision model from OpenAI CLIP.
The features are written to an HDF5 file. This file is described here (TODO)

To extract features from the image collection, we use the following command

```bash
python3 app.py extract-features IMAGE_DIR DATASET.h5
```

(Optional Arguments - TODO)

### Search

Once the features are extracted, we can re-use them for every search query.

#### Natural language (text) query
Query word embeddings are obtained from CLIP language model and then searched against
the extracted features.

```bash
python3 app.py search --dataset DATASET.h5 QUERY1 QUERY2 [...]
```

(Optional Arguments, Output description - TODO)

#### Visual similarity search
You can search for images that are similar to a given query image by passing the path to the query image as follows:
```bash
python3 app.py search --dataset DATASET.h5 /path/to/query/image.jpg [...]
```

(Optional Arguments, Output description - TODO)

#### Image classification query
To perform an image classification query, simply pass the path to a directory containing a set of query images. This trains a binary classification model on the images in the directory provided, and uses it to retrieve the images from the dataset that have the highest score from the classifier.
```bash
python3 app.py search --dataset DATASET.h5 /path/to/query/directory [...]
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
