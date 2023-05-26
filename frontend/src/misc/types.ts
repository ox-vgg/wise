import { MutableRefObject } from "react";
import { UploadFile } from "antd";

export type Query = {
  id: string;
  type: 'TEXT' | 'URL';
  displayText?: string;
  value: string;
} | {
  id: string;
  type: 'FILE';
  displayText: string;
  value: UploadFile;
}

export type FeaturedImagesJSONObject = {
  row_num: number;
  img_title: string;
  orig_width: number;
  orig_height: number;
  original_download_url: string;
};

export type SearchResponseJSONObject = {
  link: string;
  thumbnail: string;
  distance: number;
  info: {
    filename: string;
    width: number;
    height: number;
    title: string;
    caption: string;
    copyright: string;
  }
};
export type SearchResponse = Record<string, SearchResponseJSONObject[]>;

export interface ProcessedSearchResult {
  id: string;
  link: string;
  thumbnail: string;
  distance?: number;
  info: {
    // filename: string;
    width: number;
    height: number;
    title: string;
    caption?: string;
    copyright?: string;
  }
};

export interface DataServiceOutput {
  searchResults: ProcessedSearchResult[];
  isSearching: boolean;
  searchLatency: number;
  totalResults: number;
  pageNum: number;
  changePageNum: (x: number) => void;
  performNewSearch: (queries: Query[]) => Promise<void>;
  fetchAndTransformFeaturedImages: () => Promise<void>;
  reportImage: (imageId: string, reasons: string[]) => Promise<string>;
}

interface RefsForTour {
  searchBar: MutableRefObject<any>;
  visualSearchButton: MutableRefObject<any>;
  multimodalSearchButton: MutableRefObject<any>;
  paginationControls: MutableRefObject<any>;
  reportImageButton: MutableRefObject<any>;
}


/* ------ Component props ------ */
export interface CompoundSearchPopoverProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  handleTextInputChange?: (x: React.ChangeEvent<HTMLInputElement>) => void;
  submitSearch: () => void;
  onlyVisualSearch?: boolean;
}
export interface WiseHeaderProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  submitSearch: () => void;
  refsForTour: RefsForTour;
  isHomePage?: boolean;
  isSearching?: boolean;
};
export interface WiseOverviewCardProps {
  handleExampleQueryClick: (exampleQuery: string) => void;
  projectInfo: any;
  refsForTour: RefsForTour;
};
export interface SearchResultsProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  projectInfo: any;
};
export interface ReportImageModalProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  selectedImageId?: string;
  setSelectedImageId: (imageId?: string) => void;
  setDropdownImageId: (imageId?: string) => void;
}