// Load config from public/config.js

type ConfigType = {
  API_BASE_URL: string;
  MAX_SEARCH_RESULTS: number;
  PAGE_SIZE: number;
  NUM_PAGES_PER_REQUEST: number;
  FETCH_THUMBS: number;
  FETCH_TIMEOUT: number;
  REPO_URL: string;
  EXAMPLE_QUERIES: string[];
}

declare global {
  var wiseConfig: {
    devConfig: ConfigType,
    productionConfig: ConfigType
  }
};
let config = (import.meta.env.DEV) ? wiseConfig.devConfig : wiseConfig.productionConfig;

export default config;
