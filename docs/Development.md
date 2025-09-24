# Development Notes

Here are some notes for developers:

- Ensure that you have the latest version of the frontend assets by running `npm install && npm run build` inside the `frontend` folder. This command generates the latest frontend assets in the `frontend/dist` folder. See [frontend/README.md](../frontend/README.md) for more details.

- Ensure that the `tests/test-wikimedia-commons-25.sh` test script runs successfully before committing your changes to the WISE repository. See [Tests.md](Tests.md) for more details.

## Best practices
- Use dataclasses / pydantic models to give the object you pass around meaningful names, types and validation. This will help catch mistakes at development time.
- Depend on an interface rather than the implementation for a sufficiently complex feature - this allows future optimisation and extensions
- If there is a dependency, Keep initilisation (factory, constructor, etc) separate from the dependent class / function which uses it. This enables dependency injection from top level based on config passed by user.
- Add a test that shows how to initialise and use a feature. Tests will complement the documentation and code is likely to be more up-to-date than text strings


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