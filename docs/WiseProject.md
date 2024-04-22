# WiseProject

All the assets (e.g. features, index, thumbnails, metadata, etc.) related to
a WISE project is stored in a folder structure as shown below.

```
user@temp:/data/wise/Kinectics-7$ tree
.
├── Kinectics-7.db
└── store
    ├── microsoft
    │   └── clap
    │       └── 2023
    │           └── four-datasets
    │               ├── features
    │               │   └── audio-000000.tar
    │               └── index
    │                   └── audio-IndexFlatIP.faiss
    └── mlfoundations
        └── open_clip
            └── xlm-roberta-large-ViT-H-14
                └── frozen_laion5b_s13b_b90k
                    ├── features
                    │   └── video-000000.tar
                    └── index
                        └── video-IndexFlatIP.faiss
```

The `store` folder contains all the extracted features and their corresponding
search index. The sub-folders of `store` represent the namespace of each feature
extractor. The [WiseProject](src/wise_project.py) module manages this folder
and provides the full path of specific folders like `features`, or `index` as shown
below.

```
from src.wise_project import WiseProject
...

project_dir = '/data/wise/Kinectics-7/'
feature_extractor_id = 'microsoft/clap/2023/four-datasets'
project = WiseProject(args.project_dir)
project_assets = project.discover_assets()
project.create_index_dir(feature_extractor_id) # created the folder

feature_dir = project_assets[media_type][feature_extractor_id]['features_dir']
# feature_dir contains "/data/wise/Kinectics-7/store/microsoft/clap/2023/four-datasets/features/"

index_dir = project_assets[media_type][feature_extractor_id]['index_dir']
# index_dir contains "/data/wise/Kinectics-7/store/microsoft/clap/2023/four-datasets/index/"
```