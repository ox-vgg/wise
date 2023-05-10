# WISE "Dynamic" Frontend
WISE currently has two frontends, `imgrid` and `dynamic`. This folder contains the source code for the `dynamic` frontend, built using React and TypeScript. The production build of this frontend is located in the `dist` subfolder and is symlinked in `www/dynamic`. (The `imgrid` frontend only uses vanilla JavaScript so its source code is located in `www/imgrid`.)

## Key Features
- Fluid and interactive user experience
- Support for visual queries (upload an image via drag and drop, or paste an image link)
- Support for compound multi-modal queries (combine multiple image and text queries together)
- Pagination for search results, with some caching
- Custom error message for blocked queries
- Allows users to report inappropriate/offensive images

## Built with
This frontend was built with [React](https://react.dev), [Ant Design](https://ant.design), and TypeScript + SASS + HTML. The [Vite.js](https://vitejs.dev) development tool was also used.

## Usage
If you would like to use this frontend without modifying the source code, simply run `python3 app.py serve your-project-name --theme-asset-dir www/dynamic` from the root directory of this repository. See the [main README](../README.md) for more details.

Also, for now you will need to replace the `<base href="/wikimedia/">` in `dist/index.html` with your project name, e.g. `<base href="/your-project-name-here/">`. This will be done automatically later on.

## Installation / development setup
1. Make sure you have `npm` installed beforehand
2. `cd` into this directory and then run `npm install` to install the project dependencies
3. Start the development server using `npm run dev`

Note: You will need to separately run the API server using `python3 app.py serve` from the root directory of this repository. You might also need to modify some of the configurations in `src/config.ts` such as `API_BASE_URL` depending on the URL of the the API server.

### Production build
To build the project, simply run `npm run build`.
