// Copyright 2026 University of Oxford
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Load config from public/config.js

type ConfigType = {
  MAX_SEARCH_RESULTS: number;
  PAGE_SIZE: number;
  NUM_PAGES_PER_REQUEST: number;
  FETCH_THUMBS: number;
  FETCH_TIMEOUT: number;
  WISE_OVERVIEW_CARD: {
    OVERVIEW?: string;
    ABOUT: string;
    DISCLAIMER: string;
  };
  EXAMPLE_QUERIES: string[] | Record<string, string[]>;  
  MULTIMODAL_EXAMPLE_QUERIES: Record<string, {
    url?: string;
    text?: string;
    displayText?: string;
  }[]>;
  
  ENABLE_REPORT_MEDIA: boolean;
  SHOT_SCALE_FILTER_LABEL: { [key: number]: string };
  METADATA_TABLE_COLUMNS?: string[];
  METADATA_FILTER_PLACEHOLDER?: string;
  METADATA_FILTER_HELP?: string;
  PREFERRED_SEARCH_TARGETS_NAME: Record<string, string>;
};

declare global {
  var wiseConfig: {
    devConfig: ConfigType,
    productionConfig: ConfigType
  }
};
let config = (import.meta.env.DEV) ? wiseConfig.devConfig : wiseConfig.productionConfig;

export default config;
