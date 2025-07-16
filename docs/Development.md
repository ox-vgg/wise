# Development Notes

Here are some notes for developers:

- Ensure that you have the latest version of the frontend assets by running `npm install && npm run build` inside the `frontend` folder. This command generates the latest frontend assets in the `frontend/dist` folder. See [frontend/README.md](../frontend/README.md) for more details.

- Ensure that the `tests/test-wikimedia-commons-25.sh` test script runs successfully before committing your changes to the WISE repository. See [Tests.md](Tests.md) for more details.

### Profiling the API

[PyInstrument](https://pyinstrument.readthedocs.io/en/latest/home.html) is used to help profile the API requests in development mode. It is added as a middleware.

#### Setup
```
pip install pyinstrument
```

#### Usage

Serve the project with the flag `MODE=development` and `ENABLE_PROFILING=1` and Add `profile=1` to the API request as query param

Example

```
curl -XPOST 'http://API_BASE_URL/PROJECT/search?start=0&end=500&thumbs=1&search_in=video&feature_extractor_id=mlfoundations%2Fopen_clip%2FViT-L-16-SigLIP2-512%2Fwebli&text_queries=making+tea&profile=1'
```

This will save `profile.html` in the current directory