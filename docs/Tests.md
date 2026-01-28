# Integration Test

We use the [wikimedia-commons-25](https://thor.robots.ox.ac.uk/wise/assets/test/wikimedia-commons-25.zip) dataset for quickly testing various
functionalities of the WISE software. This dataset contains only 25 videos sourced from [Wikimedia Commons]() repository and therefore the 
full test completes in less than 1 minute. This test can be executed as follows.

```bash
# 1. Clone WISE code repository
cd $HOME
git clone https://gitlab.com/vgg/wise/wise.git

# 2. Activate virtual environment with all required python dependencies
cd $HOME/wise/
python3 -m venv venv
source venv/bin/activate
pip install ...  # see Install.md

# 3. Run the tests based on videos
bash test-wikimedia-commons-25.sh  $HOME/temp/

# 4. Run tests based on images
bash tests/test-wikimedia-commons-images-25.sh $HOME/temp/

# 5. Run tests based on edited videos with shots
bash tests/test-wikimedia-commons-edited-videos.sh $HOME/temp/

# 6. Run tests for WISE aggregator
bash tests/test-aggregator.sh $HOME/temp/
```

Here is a sample output obtained by executing the test based on videos.
```bash
...
Extracting features from videos (takes about 3 min.) ...
...
inserted 25 rows into table metadata-wikimedia-commons-25
...
Test 3.1 PASSED
Test 3.2 PASSED
Creating index (takes about 1 min.) ...
Test 4.1 PASSED
Test 4.2 PASSED
Starting WISE2 server on 0.0.0.0:10001 (takes about 1 min.) ...
Waiting for 5 sec. before checking again (1/15) ...
...
Waiting for 5 sec. before checking again (12/15) ...
Checking if server is running at http://0.0.0.0:10001/wikimedia-commons-25/info
Server started successfully.
Test 5.1 PASSED
Test 5.2 PASSED
Test 5.3 PASSED
Test 5.4 PASSED
*** All tests for wikimedia-commons-25 completed in 66 sec. ***
```

Here is a sample output obtained by executing the test based on images.
```bash
Starting tests for wikimedia-commons-images-25 ...
Skipping test dataset download
Test 3.1 PASSED
Test 3.2 PASSED
Test 4.1 PASSED
Test 4.2 PASSED
Starting WISE2 server on 0.0.0.0:10001 (takes about 1 min.) ...
Waiting for 5 sec. before checking again (1/15) ...
...
Waiting for 5 sec. before checking again (8/15) ...
Checking if server is running at http://0.0.0.0:10001/wikimedia-commons-images-25/info
Server started successfully.
Test 5.1 PASSED
Test 5.2 PASSED
Test 5.3 PASSED
*** All tests for wikimedia-commons-images-25 completed in 46 sec. ***
```

Here is a sample output obtained by executing the test based on edited videos (with shots).
```bash
Starting tests for wikimedia-commons-edited-videos ...
...
Test 7.1 PASSED
Test 7.2 PASSED
Test 8.1 PASSED
Test 8.2 PASSED
Test 8.3 PASSED

*** All tests for wikimedia-commons-edited-videos completed in 60 sec. ***
```

To use the triton inference server, run tests with the `FEATURE_EXTRACTOR_CONFIG` 
environment variable defined as follows:
```bash
export FEATURE_EXTRACTOR_CONFIG="{\"default\": {\"url\": \"localhost:8801\"}}" 
bash test-wikimedia-commons-25.sh  $HOME/temp/
bash tests/test-wikimedia-commons-images-25.sh $HOME/temp/
bash tests/test-wikimedia-commons-edited-videos.sh $HOME/temp/
```

The aggregator feature enables WISE to aggregate search response from multiple standalone 
WISE projects. The aggregator feature can be tested as follows:

```bash
export FEATURE_EXTRACTOR_CONFIG='{
    "default": {
        "url": "localhost:8801"
    },
    "transformers/owlv2/google/owlv2-large-patch14-ensemble": {
        "objectness_threshold": 0.11
    }
}'
bash tests/test-aggregator.sh $HOME/temp/
```

This test takes around 30 minutes to generate the four test projects named `1`, `2`, `3`, and `123`.
Here is a sample output obtained by executing the test based on the [aggregator-3](https://thor.robots.ox.ac.uk/wise/assets/test/aggregator-3.zip) dataset.
```bash
Starting tests for aggregator-3 ...
...
Started server for 1 on port 10001 with PID 1098030
Started server for 2 on port 10002 with PID 1098031
Started server for 3 on port 10003 with PID 1098032
REMOTE_PROJECTS=["http://localhost:10001/1/","http://localhost:10002/2/","http://localhost:10003/3/"]
...
Server for project 123 on port 10004 is running.
All servers are up and running.
...
------ Test Summary ------
Test 6.1 PASSED: identical values in /info endpoints
Test 6.2 PASSED: identical results for video search query 'panda'
Test 6.3 PASSED: identical results for audio search query 'gunshot'
Test 6.4 PASSED: identical results for object search query 'boat'
Test 6.5 PASSED: identical results for face search query
Test 6.6 PASSED: identical results for metadata search query
--------------------------
All tests passed.
```

# Unit Tests

Individual tests can be executed as follows.

```bash
python -m unittest api/test_api.py
python -m unittest src/feature/test_feature_extractor.py
python -m unittest src/feature/store/test_feature_store.py
python -m unittest src/search/test_query_parser.py
```
