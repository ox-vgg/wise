import { MutableRefObject } from "react";
import { UploadFile } from "antd";

export type Query = {
  id: string;
  type: 'TEXT' | 'URL';
  displayText?: string;
  value: string;
  isNegative?: boolean;
} | {
  id: string;
  type: 'FILE';
  displayText: string;
  value: UploadFile;
  isNegative?: boolean;
} | {
  id: string;
  type: 'INTERNAL_IMAGE';
  displayText: string;
  value: string; // this value represents the internal image id
  isNegative?: boolean;
}

export type SearchResponseJSONObject = {
  link: string;
  thumbnail: string;
  distance: number;
  info: {
    id: string;
    filename: string;
    width: number;
    height: number;
    title: string;
    author?: string;
    caption: string;
    copyright: string;
    is_nsfw?: boolean;
  }
};
export type SearchResponse = Record<string, SearchResponseJSONObject[]>;

export interface ProcessedSearchResult {
  link: string;
  thumbnail: string;
  distance?: number;
  info: {
    id: string;
    // filename: string;
    width: number;
    height: number;
    title: string;
    author?: string;
    caption?: string;
    copyright?: string;
    is_nsfw?: boolean;
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
  setSearchText: (x: string) => void;
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  submitSearch: () => void;
};
export interface ReportImageModalProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  selectedImageId?: string;
  setSelectedImageId: (imageId?: string) => void;
}