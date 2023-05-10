<h1 align="center">WISE Image Search Engine (WISE)</h1>

<p align="center">
  <img src="docs/assets/wise_logo.svg" alt="wise-logo" width="160px" height="70px"/>
  <br>
  WISE is an AI-based image search engine for efficiently searching through large collections of images.
</p>

> The code and documentation contained in this repository is not stable yet for production usage. We are working on making it production ready.

<hr>

## Features

* **Natural language search** - use natural language to describe what you want to search for

  <img src="docs/assets/natural_language_search.png" width="600px">

  WISE uses a language model to understand the meaning behind your query, allowing you to flexibly describe what you are looking for. Moreover, WISE uses a vision model to understand what's being depicted in an image (i.e. it searches by image content rather than metadata such as keywords, tags, or descriptions), so the images do not need to be manually tagged or labelled with text captions.

* **Visual similarity search** - upload an image or paste an image link to find similar images

  <img src="docs/assets/visual_similarity_search.png" width="600px">

* **Multi-modal search** - combine images and text in your query. For example, if you upload a picture of a golden retriever and enter the text "in snow", WISE will find images of golden retrievers in snow.

  <img src="docs/assets/multimodal_search.png" width="600px">

* Searches can be performed via:
  * CLI
  * REST API
  * Web frontend

  (Note: currently the search functionality in the CLI may be missing some features.)

* Safety features
  * Specify a list of search terms that users should be blocked from searching
  * 'Report image' button allows users to report inappropriate/offensive/etc images



## How it works
WISE uses vision-language models such as OpenAI [CLIP](https://openai.com/research/clip) (specifically [OpenCLIP](https://github.com/mlfoundations/open_clip), which is an open-source implementation of CLIP trained on the [LAION](https://laion.ai/blog/laion-5b/) dataset).

Vision-language models are able to map both images and text onto the same feature space. Images and/or text that have similar semantics (meanings) are placed closer together in this feature space. This means users can search a collection of images, using natural language or using another image.

<img src="docs/assets/clip_diagram.png" width="600px">

The [Faiss](https://github.com/facebookresearch/faiss) library is used to perform approximate nearest neighbour search.

## Installation

### Setup virtual environment and install dependencies
(You will need to have Python 3 and conda installed beforehand)
```bash
git clone git@gitlab.com:vgg/wise/wise.git
cd wise
conda env create -f environment.yml
conda activate wise
```

## Usage
For more details on the commands available, run
```bash
python3 wise.py --help
```
### Initialise project with a collection of images

The `init` command creates a new project. For each data source provided, features, thumbnails and metadata are extracted.

```bash
python3 wise.py init your-project-name \
  --batch-size 16 --model "ViT-L-14:laion2b_s32b_b82k" \
  --store-in /path/to/some/folder \
  --source /path/to/a/folder/of/images \
  --source /you/can/specify/multiple/sources \
  --source /path/to/a/webdataset{000..999}.tar
```

Parameters:
* `--source`: you can pass in a folder of images, or a [WebDataset](https://webdataset.github.io/webdataset/). You can also provide more than one `--source` as shown above
* `--store-in`: folder where you would like the extracted features, indices, thumbnails, and metadata to be stored. (Make sure this is on a disk with sufficient space.) If unspecified, these files will be stored within the `~/.wise/projects/project-name` folder (in your home directory)
* `--batch-size`: number of images in each batch to pass to the model. Default value: 1
* `--model`: specify an OpenCLIP model to use for extracting features (for a full list of models available, run `python3 wise.py init --help`). Default value: `ViT-B-32:openai`.
* For more details, run `python3 wise.py init --help`

### Add more images to an existing project
```bash
python3 wise.py update your-project-name \
  --batch-size 128 \
  --source "/path/to/folder/or/webdataset"
```
* For more details, run `python3 wise.py init --help`

### Create a search index based on approximate nearest neighbour search
```bash
python3 wise.py index your-project-name --index-type IndexIVFFlat
# (for exhaustive search, use --index-type IndexFlatIP)
```
* For more details, run `python3 wise.py index --help`

### Serve the web interface for the search engine
```bash
python3 wise.py serve your-project-name \
  --index-type IndexIVFFlat \
  --theme-asset-dir www/dynamic/
```
* For now you will need to replace the `<base href="/wikimedia/">` in `frontend/dist/index.html` with your project name, e.g. `<base href="/your-project-name-here/">`. This will be done automatically later on.
* Once the server has been started, go to http://localhost:9670/your-project-name in your browser
* You can optionally provide a query blocklist (i.e. a list of queries that users should be blocked from searching) using `--query-blocklist /path/to/blocklist.txt`
* For more details, run `python3 wise.py serve --help`

## Frontend

WISE currently has two frontends, `imgrid` and `dynamic`. When running `python3 wise.py serve`, you can either pass in `--theme-asset-dir www/imgrid` or `--theme-asset-dir www/dynamic`.

* `imgrid` is a simple frontend written in vanilla JavaScript and its source code is located in `www/imgrid`
* `dynamic` is built using React and TypeScript and contains additional features. The source code for this frontend is located in the `frontend` folder. The production build is located in the `frontend/dist` folder and is also symlinked in `www/dynamic`.

You can also develop your own frontend that interacts with the WISE backend. The backend API endpoints are defined in `api/routes.py`.

## Test
WISE contains some automated tests to verify the software's
functionality.

```
conda activate wise
python -m pytest -s tests
```

## Contact
For any queries or feedback related to the WISE software, contact [Prasanna Sridhar](mailto:prasanna@robots.ox.ac.uk), [Horace Lee](mailto:horacelee@robots.ox.ac.uk) or [Abhishek Dutta](mailto:adutta@robots.ox.ac.uk).

## Acknowledgements
Development and maintenance of WISE software has been supported by the following grant: Visual AI: An Open World Interpretable Visual Transformer (UKRI Grant [EP/T028572/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/T028572/1))
