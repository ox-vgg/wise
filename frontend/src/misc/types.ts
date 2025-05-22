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
  value: ProcessedVectorInfo;
  isNegative?: boolean;
};


type MediaInfo = {
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
export type ImageInfo = MediaInfo & {};
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

export type VectorInfo = {
  vector_id: string;
  media_id: string;
  link: string;
  thumbnail: string;
  bbox?: BBoxXYWH;
  related_vectors?: ProcessedVectorInfo[];  // fetched by request only
};

// The backend VectorInfo does not come with the related MediaInfo, it
// is added when the backend response is processed.  mediaInfo will
// actually then be one of the Processed*Info types, all of which are
// derived from MediaInfo (see the Processed*Info types below).
export type ProcessedVectorInfo = VectorInfo & {
  mediaInfo: MediaInfo;
};

type VectorResult = VectorInfo & {
  distance: number;
};

type ImageVector = VectorResult;
export type VideoSegment = VectorResult & {
  ts: number;
  te: number;
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
  video_audio_results?: VideoAudioResults;
  video_results?: VideoResults;
  image_results?: ImageResults;
};

// The Processed* vectors types have the corresponding Processed*Info
// as an attribute which has the related Processed* vectors/shots as
// an attribute themselves.  This introduces a circular reference but
// enables jumping from vector to media easily which is only used for
// the "Videos" view.
//
// ProcessedImageInfo.vectors and ProcessedSearchResults.Image.vectors
// attributes are unused but they provide a nice symmetry to the video
// shots.
export type ProcessedImageVector = ImageVector & {
  mediaType: 'IMAGE';
  mediaInfo: ProcessedImageInfo;
};
export type ProcessedImageInfo = ImageInfo & {
  vectors: ProcessedImageVector[];
};
export type ProcessedVideoSegment = VideoSegment & {
  mediaType: 'VIDEO';
  mediaInfo: ProcessedVideoInfo;
};
export type ProcessedVideoInfo = VideoInfo & {
  shots: ProcessedVideoSegment[];
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
  performNewSearch: (queries: Query[], viewModality: keyof ProcessedSearchResults, featureExtractorId: string) => Promise<void>;
  fetchFeaturedImagesAndSetState: () => Promise<void>;
  reportImage: (imageId: string, reasons: string[]) => Promise<string>;
  fillRelatedVectors: (imageDetails: ProcessedImageVector) => Promise<ProcessedImageVector>;
};

export interface ProjectInfo {
  project_name?: string;
  models?: {
    image?: string[],
    video?: string[],
    audio?: string[],
  };
  search_targets?: {
    image?: string[],
    video?: string[],
    audio?: string[],
  };
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
  featureExtractorId: string;
  setFeatureExtractorId: (x: string) => void;
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
  handleInternalSearchButtonClick: (vector: ProcessedVectorInfo) => void;
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
  imageDetails: ProcessedVectorInfo;
  isModalView: boolean;
  // If handleInternalSearchButtonClick is missing, the "Find Similar"
  // button is omitted.
  handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
};
