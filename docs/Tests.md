# Integration Test

We use the Kinetics-6 dataset for quickly testing various
functionalities of the WISE software. This dataset contains only 30
videos and therefore the full test completes in less than 6
minutes. This test can be executed as follows.

```
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd wise/tests
./test-kinetics-6.sh /tmp/wise-test/

Updating WISE2 code in /tmp/wise-test/wise-code/
Ensuring dependencies are installed...
Downloading Kinetics-6 dataset to /tmp/wise-test/wise-data ...
Extracting features from videos (takes about 3 min.) ...
Extracting features from 30 files
Feature extraction completed in 78 sec (1.30 min)
Importing metadata
Adding 30 rows of metadata (discarded 0 rows)
added 30 rows of metadata to table video_categories
Creating index (takes about 1 min.) ...
saved index to /tmp/wise-test/wise-projects/Kinetics-6/store/mlfoundations/open_clip/ViT-L-16-SigLIP-384/webli/index/video-IndexFlatIP.faiss
saved index to /tmp/wise-test/wise-projects/Kinetics-6/store/microsoft/clap/2023/four-datasets/index/audio-IndexFlatIP.faiss
Created metadata index for "Kinetics/6b/video_categories" with 30 entries
...
...
Test 1 PASSED (completed in 164 sec.)
Test 2 PASSED (completed in 190 sec.)
```

# Unit Tests

Individual tests can be executed as follows.

```bash
python -m unittest src/feature/test_feature_extractor.py
python -m unittest src/feature/store/test_feature_store.py
```

All tests can be discovered and run as follows.

```bash
python -m unittest discover -s src/
```
