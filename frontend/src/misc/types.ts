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
} | {
  id: string;
  type: 'METADATA';
  displayText?: string;
  value: string;
  isNegative?: false;
};

export type ASRSegment = {
  start: number;
  end: number;
  text: string;
}

export type MediaInfo = {
  id: string;
  filename: string;
  width: number;
  height: number;
  media_type: string;
  format: string;
  duration: number;
  title: string;
  external_metadata: {
    asr_segments?: ASRSegment[];
    title?: string;
  }
};
export type ImageInfo = MediaInfo & {};
export type VideoInfo = MediaInfo & {
  timeline_hover_thumbnails: string;
};

// Bounding Box in (x0, y0, w, h) format and in range [0, 1].  They
// are relative to the size of the image.
export type BBoxXYWH = {
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
};

// The backend VectorInfo does not come with the related MediaInfo, it
// is added when the backend response is processed.  mediaInfo will
// actually then be one of the Processed*Info types, all of which are
// derived from MediaInfo (see the Processed*Info types below).  In
// addition, the frontend will also add related_vectors when
// inspecting the vector in the Image details modal dialog (a second
// "processing" step).
type ProcessedVectorMixin = {
  mediaType: 'IMAGE' | 'VIDEO';
  mediaInfo: MediaInfo;
  related_vectors?: ProcessedVectorInfo[];
};

type VectorResultMixin = {
  distance: number;
};

export type ProcessedVectorInfo = VectorInfo & ProcessedVectorMixin;

export type VectorResult = VectorInfo & VectorResultMixin;


type ImageVector = VectorResult;
export type VideoSegment = VectorResult & {
  ts: number;
  te: number;
  thumbnail_ts: number;
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
export type ProcessedImageVector = ImageVector & ProcessedVectorMixin & {
  mediaType: 'IMAGE';
  mediaInfo: ProcessedImageInfo;
};
export type ProcessedImageInfo = ImageInfo & {
  vectors: ProcessedImageVector[];
};
export type ProcessedVideoSegment = VideoSegment & ProcessedVectorMixin & {
  mediaType: 'VIDEO';
  mediaInfo: ProcessedVideoInfo;
};
export type ProcessedVideoInfo = VideoInfo & {
  vectors: ProcessedVideoSegment[];
  shots: ProcessedVideoSegment[];
  asrSegments?: ASRSegment[];
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
export type ViewModality = keyof ProcessedSearchResults;
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
  performNewSearch: (queries: Query[], viewModality: ViewModality, featureExtractorId: string, shotScaleFilter: number[]) => Promise<void>;
  fetchFeaturedImagesAndSetState: (viewModality: ViewModality, featureExtractorId: string) => Promise<void>;
  reportImage: (imageId: string, reasons: string[]) => Promise<string>;
  fillRelatedVectors: (imageDetails: ProcessedImageVector | ProcessedVideoSegment) => Promise<ProcessedImageVector | ProcessedVideoSegment>;
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
  shot_based_filters?: {
    shot_scale?: {
      description: string;
      options: number[];
    };
  };
  num_vectors?: number;
  max_search_results?: number;
  num_media_files?: number;
  media_file_counts?: {
    image?: number;
    video?: number;
    audio?: number;
  };
  total_duration?: number;
  is_metadata_supported?: boolean;
  enable_facets?: boolean;
};

export interface TourVariables {
  isSearchDropdownOpenForTour: boolean;
  setIsSearchDropdownOpenForTour: (x: boolean) => void;
  searchBar: MutableRefObject<any>;
  imageUploadButton: MutableRefObject<any>;
  multimodalSearchArea: MutableRefObject<any>;
  paginationControls: MutableRefObject<any>;
  reportImageButton: MutableRefObject<any>;
};


/* ------ Component props ------ */
export interface TextSearchFormProps {
  placeholder?: string;
  buttonText?: string;
  queryType?: 'TEXT' | 'METADATA';
  multimodalQueries: Query[];
  setMultimodalQueries: React.Dispatch<React.SetStateAction<Query[]>>
  searchText?: string;
  handleTextInputChange?: (x: string) => void;
  submitSearch: (q?: Query[]) => void;
};
export interface MediaSearchFormProps {
  multimodalQueries: Query[];
  setMultimodalQueries: React.Dispatch<React.SetStateAction<Query[]>>
  submitSearch: (q?: Query[]) => void;
  modality: string;
  featureExtractorId: string;
};
export interface SearchExamplesProps {
  setMultimodalQueries: React.Dispatch<React.SetStateAction<Query[]>>
  setSearchText: React.Dispatch<React.SetStateAction<string>>;
  submitSearch: (q?: Query[]) => void;
  viewModality: ViewModality;
  featureExtractorId: string;
};
export interface SearchDropdownProps {
  multimodalQueries: Query[];
  setMultimodalQueries: React.Dispatch<React.SetStateAction<Query[]>>
  searchText: string;
  setSearchText: React.Dispatch<React.SetStateAction<string>>;
  handleTextInputChange?: (x: string) => void;
  viewModality: ViewModality;
  featureExtractorId: string;
  shotScaleFilter: number[];
  setShotScaleFilter: (x: number[]) => void;
  submitSearch: (q?: Query[]) => void;
  clearSearchBar: () => void;
  tourVariables: TourVariables;
  projectInfo: ProjectInfo;
  isHomePage?: boolean;
};
export interface WiseHeaderProps {
  multimodalQueries: Query[];
  setMultimodalQueries: React.Dispatch<React.SetStateAction<Query[]>>
  searchText: string;
  setSearchText: React.Dispatch<React.SetStateAction<string>>;
  viewModality: ViewModality;
  setViewModality: React.Dispatch<React.SetStateAction<ViewModality>>;
  featureExtractorId: string;
  setFeatureExtractorId: (x: string) => void;
  shotScaleFilter: number[];
  setShotScaleFilter: (x: number[]) => void;
  submitSearch: (q?: Query[]) => void;
  tourVariables: TourVariables;
  projectInfo: ProjectInfo;
  isHomePage?: boolean;
  isLoadingNewSearch?: boolean;
};
export interface WiseOverviewCardProps {
  handleExampleQueryClick: (exampleQuery: string, viewModality?: ViewModality, featureExtractorId?: string) => void;
  projectInfo: ProjectInfo;
  tourVariables: TourVariables;
};
export interface SearchResultsProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  projectInfo: ProjectInfo;
  setSearchText: React.Dispatch<React.SetStateAction<string>>;
  multimodalQueries: Query[];
  setMultimodalQueries: React.Dispatch<React.SetStateAction<Query[]>>
  viewModality: ViewModality;
  featureExtractorId: string;
  submitSearch: (q?: Query[]) => void;
};

export interface ImageDetailsModalProps {
  imageDetails: ProcessedImageVector | ProcessedVideoSegment;
  setImageDetails: (x?: ProcessedImageVector | ProcessedVideoSegment) => void;
  setSelectedImageId: (imageId?: string) => void;
  isHomePage: boolean;
  featureExtractorId: string;
  handleInternalSearchButtonClick: (vector: ProcessedVectorInfo) => void;
};

export interface ReportImageModalProps {
  dataService: DataServiceOutput;
  isHomePage: boolean;
  selectedImageId?: string;
  setSelectedImageId: (imageId?: string) => void;
};

export interface VideoOccurrencesViewProps {
  featureExtractorId: string;
  shots: ProcessedVideoSegment[];
  handleClickOccurrence: (videoSegment: ProcessedVideoSegment) => void;
  customHeaderSingular?: string;
  customHeaderPlural?: string;
};

export type ProcessedVectorInfoWithBBox = ProcessedVectorInfo & {
    bbox: NonNullable<ProcessedVectorInfo['bbox']>
};

export const isVideoSegment = (
    vector: ProcessedVectorInfo
): vector is ProcessedVideoSegment => {
    return Boolean(vector.mediaType === "VIDEO");
};

export const isWithBBox = (
    vector_info: ProcessedVectorInfo
): vector_info is ProcessedVectorInfoWithBBox => {
    return Boolean(vector_info.bbox);
}

export const isWithVectors = (
    mediaInfo: MediaInfo
): mediaInfo is ProcessedImageInfo | ProcessedVideoInfo => {
    return Boolean("vectors" in mediaInfo);
};

export const isResult = (
    vector: ProcessedVectorInfo
): vector is ProcessedImageVector => {
    return Boolean("distance" in vector);
}

export type ConfigType = {
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
    EXAMPLE_QUERIES: string[];
    MULTIMODAL_EXAMPLE_QUERIES: {
        url: string;
        text: string;
    }[];
    ENABLE_REPORT_MEDIA: boolean;
    SHOT_SCALE_FILTER_LABEL: Record<number, string>;
    METADATA_TABLE_COLUMNS?: string[];
    METADATA_FILTER_PLACEHOLDER?: string;
    METADATA_FILTER_HELP?: string;
};
