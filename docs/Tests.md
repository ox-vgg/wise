# Integration Test

We use the [wikimedia-commons-25](https://thor.robots.ox.ac.uk/wise/assets/test/wikimedia-commons-25.zip) dataset for quickly testing various
functionalities of the WISE software. This dataset contains only 25 videos sourced from [Wikimedia Commons]() repository and therefore the 
full test completes in less than 1 minute. This test can be executed as follows.

```
cd $HOME
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd $HOME/wise/tests
./test-wikimedia-commons-25.sh $HOME/wise/ $HOME/wise-test-data/

...
...
Test 4.2.1 PASSED
Test 4.2.2 PASSED
Test 4.3.1 PASSED
Test 4.3.2 PASSED
Starting WISE2 server (takes about 1 min.) ...
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

# Unit Tests

Individual tests can be executed as follows.

```bash
python -m unittest src/feature/test_feature_extractor.py
python -m unittest src/feature/store/test_feature_store.py
python -m unittest src/search/test_query_parser.py
```

All tests can be discovered and run as follows.

```bash
python -m unittest discover -s src/
```
