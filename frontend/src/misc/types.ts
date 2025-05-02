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
  value: ProcessedImageVector | ProcessedVideoSegment;
  isNegative?: boolean;
};



type MediaMetadata = {
  id: string;
  filename: string;
  width: number;
  height: number;
  media_type: string;
  format: string;
  duration: number;
  title: string;
  external_metadata: Record<string, any>;
};
// A search result containing the metadata fields from MediaMetadata, as well as additional fields like `thumbnail` and `distance`
type MediaInfo = MediaMetadata & {
  link: string;
  thumbnail: string;
  distance?: number;
};
export type ImageInfo = MediaInfo & {};
export type AudioInfo = MediaInfo & {};
export type VideoInfo = MediaInfo & {
  timeline_hover_thumbnails: string;
};

// Bounding Box in (x0, y0, w, h) format and in range [0, 1].  They
// are relative to the size of the image.
type BBoxXYWH = {
  x: number;
  y: number;
  w: number;
  h: number;
};

type VectorResult = {
  vector_id: string;
  media_id: string;
  link: string;
  distance: number;
  bbox?: BBoxXYWH;
};
type ImageVector = VectorResult & {
  thumbnail: string;
  thumbnail_score: number;
};
type MediaSegment = VectorResult & {
  ts: number;
  te: number;
};
export type AudioSegment = MediaSegment & {};
export type VideoSegment = MediaSegment & {
  thumbnail: string;
  thumbnail_score: number;
};

export type AudioResults = {
  total: number;
  unmerged_windows: AudioSegment[];
  audios: Record<string, AudioInfo>;
};
export type VideoAudioResults = {
  total: number;
  unmerged_windows: VideoSegment[];
  merged_windows: VideoSegment[];
  videos: Record<string, VideoInfo>;
};
export type VideoResults = {
  total: number;
  unmerged_windows: VideoSegment[];
  merged_windows: VideoSegment[];
  videos: Record<string, VideoInfo>;
};
export type ImageResults = {
  total: number;
  vectors: ImageVector[];
  images: Record<string, ImageInfo>;
};
export type SearchResponse = {
  time: number;
  audio_results?: AudioResults;
  video_audio_results?: VideoAudioResults;
  video_results?: VideoResults;
  image_results?: ImageResults;
};

// TODO update everything below
export type ProcessedImageVector = ImageVector & {
  mediaType: 'IMAGE';
  mediaInfo: ProcessedImageInfo;
};
export type ProcessedImageInfo = ImageInfo & {
  vectors: ImageVector[] | ProcessedImageVector[];
};
export type ProcessedVideoSegment = VideoSegment & {
  mediaType: 'VIDEO';
  mediaInfo: ProcessedVideoInfo;
};
export type ProcessedVideoInfo = VideoInfo & {
  shots: VideoSegment[] | ProcessedVideoSegment[];
  title: string;
};
export type ProcessedSearchResults = {
  Image: {
    vectors: ProcessedImageVector[];
    mediaInfo: Map<string, ProcessedImageInfo>;
  };
  Video: {
    unmerged_windows: ProcessedVideoSegment[];
    merged_windows: ProcessedVideoSegment[];
    mediaInfo: Map<string, ProcessedVideoInfo>;
  };
  VideoAudio: {
    unmerged_windows: ProcessedVideoSegment[];
    merged_windows: ProcessedVideoSegment[];
    mediaInfo: Map<string, ProcessedVideoInfo>;
  };
  Audio: {
    unmerged_windows: ProcessedVideoSegment[];
    merged_windows: ProcessedVideoSegment[];
    mediaInfo: Map<string, ProcessedVideoInfo>;
  };
};
export type ProcessedSearchResponse = {
  processedSearchResults: ProcessedSearchResults;
  time: number;
};

export interface DataServiceOutput {
  searchResults: ProcessedSearchResults;
  isLoadingNewSearch: boolean;
  searchLatency: number;
  totalResults: number;
  // pageNum: number;
  // changePageNum: (x: number) => void;
  performNewSearch: (queries: Query[], viewModality: keyof ProcessedSearchResults) => Promise<void>;
  fetchFeaturedImagesAndSetState: () => Promise<void>;
  reportImage: (imageId: string, reasons: string[]) => Promise<string>;
};

export interface ProjectInfo {
  project_name?: string;
  models?: {
    image?: string[],
    video?: string[],
    audio?: string[],
  },
  search_modalities?: ('image' | 'video' | 'audio')[],
  num_vectors?: number;
  num_media_files?: number;
  media_file_counts?: {
    image?: number;
    video?: number;
    audio?: number;
  };
  total_duration?: number;
};

interface RefsForTour {
  searchBar: MutableRefObject<any>;
  visualSearchButton: MutableRefObject<any>;
  multimodalSearchButton: MutableRefObject<any>;
  paginationControls: MutableRefObject<any>;
  reportImageButton: MutableRefObject<any>;
};


/* ------ Component props ------ */
export interface TextSearchFormProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  handleTextInputChange?: (x: React.ChangeEvent<HTMLInputElement>) => void;
  submitSearch: () => void;
};
export interface MediaSearchFormProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  submitSearch: () => void;
  modality: string;
};
export interface SearchExamplesProps {
  setMultimodalQueries: (x: Query[]) => void;
  setSearchText: (x: string) => void;
  submitSearch: () => void;
};
export interface SearchDropdownProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  handleTextInputChange?: (x: React.ChangeEvent<HTMLInputElement>) => void;
  viewModality: keyof ProcessedSearchResults;
  submitSearch: () => void;
  clearSearchBar: () => void;
  isHomePage?: boolean;
};
export interface WiseHeaderProps {
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  searchText: string;
  setSearchText: (x: string) => void;
  viewModality: keyof ProcessedSearchResults;
  setViewModality: (x: keyof ProcessedSearchResults) => void;
  submitSearch: () => void;
  refsForTour: RefsForTour;
  projectInfo: ProjectInfo;
  isHomePage?: boolean;
  isLoadingNewSearch?: boolean;
};
export interface WiseOverviewCardProps {
  handleExampleQueryClick: (exampleQuery: string) => void;
  projectInfo: ProjectInfo;
  refsForTour: RefsForTour;
};
export interface SearchResultsProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  projectInfo: ProjectInfo;
  setSearchText: (x: string) => void;
  multimodalQueries: Query[];
  setMultimodalQueries: (x: Query[]) => void;
  viewModality: keyof ProcessedSearchResults;
  submitSearch: () => void;
};

export interface ImageDetailsModalProps {
  isHomePage: boolean;
  imageDetails: ProcessedImageVector | ProcessedVideoSegment;
  setImageDetails: (x?: ProcessedImageVector | ProcessedVideoSegment) => void;
  setSelectedImageId: (imageId?: string) => void;
  handleInternalSearchButtonClick: (vector_id: string) => void;
};

export interface ReportImageModalProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  selectedImageId?: string;
  setSelectedImageId: (imageId?: string) => void;
};

export interface VideoOccurrencesViewProps {
  videoInfo: ProcessedVideoInfo;
  handleClickOccurrence: (videoSegment: ProcessedVideoSegment) => void;
  customHeaderSingular?: string;
  customHeaderPlural?: string;
};

export interface StillImageViewProps {
  imageDetails: ProcessedImageVector;
  isModalView: boolean;
  handleInternalSearchButtonClick: (vector_id: string) => void;
};
