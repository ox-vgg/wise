import { MutableRefObject } from "react";
import { UploadFile } from "antd";

export type Query = {
  id: string;
  type: 'TEXT' | 'IMAGE_URL' | 'AUDIO_URL';
  displayText?: string;
  value: string;
  isNegative?: boolean;
} | {
  id: string;
  type: 'IMAGE_FILE' | 'AUDIO_FILE';
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
    


type MediaMetadata = {
  id: string;
  filename: string;
  width: number;
  height: number;
  media_type: string;
  format: string;
  duration: number;
  title: string;
  caption: string;
  copyright: string;
}
// A search result containing the metadata fields from MediaMetadata, as well as additional fields like `thumbnail` and `distance`
type MediaInfo = MediaMetadata & {
  link: string;
  thumbnail: string;
  distance?: number;
}
export type ImageInfo = MediaInfo & {}
export type AudioInfo = MediaInfo & {}
export type VideoInfo = MediaInfo & {
  timeline_hover_thumbnails: string;
}

type MediaSegment = {
  vector_id: string;
  media_id: string;
  ts: number;
  te: number;
  link: string;
  distance: number;
}
export type AudioSegment = MediaSegment & {}
export type VideoSegment = MediaSegment & {
  thumbnail: string;
  thumbnail_score: number;
}

export type AudioResults = {
  total: number;
  unmerged_windows: AudioSegment[];
  audios: Record<string, AudioInfo>;
}
export type VideoAudioResults = {
  total: number;
  unmerged_windows: VideoSegment[];
  merged_windows: VideoSegment[];
  videos: Record<string, VideoInfo>;
}
export type VideoResults = {
  total: number;
  unmerged_windows: VideoSegment[];
  merged_windows: VideoSegment[];
  videos: Record<string, VideoInfo>;
}
export type ImageResults = {
  total: number;
  results: ImageInfo[];
}
export type SearchResponse = {
  time: number;
  audio_results?: AudioResults;
  video_audio_results?: VideoAudioResults;
  video_results?: VideoResults;
  image_results?: ImageResults;

  // Delete later:
  // unmerged_segments: VideoSegment[];
  // shots: VideoSegment[];
  // videos: Record<string, VideoInfo>;
  // audio_segments?: VideoSegment[];
  // audio_shots?: VideoSegment[];
  // audio_videos?: Record<string, VideoInfo>;
  // visual_segments?: VideoSegment[];
  // visual_shots?: VideoSegment[];
  // visual_videos?: Record<string, VideoInfo>;
}

// TODO update everything below
export type ProcessedVideoSegment = VideoSegment & {
  videoInfo: ProcessedVideoInfo;
}
export type ProcessedVideoInfo = VideoInfo & {
  shots: VideoSegment[] | ProcessedVideoSegment[];
  title: string;
}
export type ProcessedSearchResults = {
  Video: {
    unmerged_windows: ProcessedVideoSegment[];
    merged_windows: ProcessedVideoSegment[];
    videos: Map<string, ProcessedVideoInfo>;
  };
  VideoAudio: {
    unmerged_windows: ProcessedVideoSegment[];
    merged_windows: ProcessedVideoSegment[];
    videos: Map<string, ProcessedVideoInfo>;
  };
  Audio: {
    unmerged_windows: ProcessedVideoSegment[];
    merged_windows: ProcessedVideoSegment[];
    videos: Map<string, ProcessedVideoInfo>;
  };
}

export interface DataServiceOutput {
  searchResults: ProcessedSearchResults;
  isSearching: boolean;
  searchLatency: number;
  totalResults: number;
  // pageNum: number;
  // changePageNum: (x: number) => void;
  performNewSearch: (queries: Query[], viewModality: string) => Promise<void>;
  fetchFeaturedImagesAndSetState: () => Promise<void>;
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
export interface TextSearchFormProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  handleTextInputChange?: (x: React.ChangeEvent<HTMLInputElement>) => void;
  submitSearch: () => void;
}
export interface MediaSearchFormProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  submitSearch: () => void;
  modality: string;
}
export interface SearchExamplesProps {
  setMultimodalQueries: (x: Query[]) => void;
  setSearchText: (x: string) => void;
  submitSearch: () => void;
}
export interface SearchDropdownProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  handleTextInputChange?: (x: React.ChangeEvent<HTMLInputElement>) => void;
  viewModality: string;
  submitSearch: () => void;
  clearSearchBar: () => void;
  isHomePage?: boolean;
}
export interface WiseHeaderProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  viewModality: string;
  setViewModality: (x: string) => void;
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
  viewModality: string;
  submitSearch: () => void;
};

export interface ImageDetailsModalProps {
  imageDetails?: ProcessedVideoSegment;
  setImageDetails: (x?: ProcessedVideoSegment) => void;
  setSelectedImageId: (imageId?: string) => void;
};

export interface ReportImageModalProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  selectedImageId?: string;
  setSelectedImageId: (imageId?: string) => void;
}

export interface VideoOccurrencesViewProps {
  videoInfo: ProcessedVideoInfo;
  handleClickOccurrence: (videoSegment: ProcessedVideoSegment) => void;
  customHeaderSingular?: string;
  customHeaderPlural?: string;
}
