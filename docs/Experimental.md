# Experimental Features

> **Note:** The features described in this document are still being finalized and may change. They are not yet stable.

## Aggregator
The aggregator feature of WISE allows to present search results from multiple standalone WISE projects.
The [tests/test-aggregator.sh](../tests/test-aggregator.sh) script shows an example of how to use this
feature on 3 sample projects based on the [aggregator-3](https://thor.robots.ox.ac.uk/wise/assets/test/aggregator-3.zip)
dataset. Here is an example of how this feature can be used in general.

```
# We assume that standalone WISE projects are hosted at the following URLs:
#   * http://localhost:10001/1/
#   * http://localhost:10002/2/
#   * http://localhost:10003/3/

REMOTE_PROJECTS==["http://localhost:10001/1/","http://localhost:10002/2/","http://localhost:10003/3/"] \
  PORT=10000 \
  python3 serve.py --project-dir tmp/123/ &
```

Users visiting the URL `http://localhost:10000/123/` will see WISE search interface that allows audiovisual
search on the contents of all the three standalone projects. This feature is useful for serving large scale
projects where a large dataset of media files are split across multiple projects. Each sub-project is hosted
on a separate machine with compute and memory resources that are only sufficient for hosting the sub-project.
