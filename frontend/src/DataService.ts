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

import { useState } from 'react';
import { 
  DataServiceOutput,
  ProcessedSearchResults,
  ProcessedVideoSegment,
  ProcessedVideoInfo,
  Query,
  SearchResponse,
  VideoSegment,
  VideoInfo,
  ProcessedSearchResponse,
  ProcessedImageInfo,
  ProcessedImageVector,
  VectorInfo,
  ASRSegment,
  ViewModality, 
} from './misc/types.ts';
import config from './config.ts';
import { fetchWithTimeout /*, chunk, getArrayOfEmptyArrays */ } from './misc/utils.ts';

// const NUM_PAGES = Math.ceil(config.MAX_SEARCH_RESULTS / config.PAGE_SIZE);

const MAX_FEATURED_IMAGES = 1000; // TODO set this based on actual number of featured images
const FEATURED_IMAGES_RANDOM_SEED = Math.floor(Math.random()*100); // Generate a random number between 0-100 to be used as the random seed when fetching the featured images

const processVideos = (videos: Record<string, VideoInfo>, shots: VideoSegment[]) => {
  return new Map(
    Object.entries(videos).map(([mediaId, videoInfo]) => {
      // const title = videoInfo.filename;
      const { external_metadata } = videoInfo;
      let asr_segments: ASRSegment[] = [];
      if (external_metadata.asr_segments) {
        asr_segments = external_metadata.asr_segments.slice();
      }
      const title = videoInfo.title || external_metadata.title || videoInfo.filename;
      return [
        mediaId,
        {
          ...videoInfo,
          shots: shots.filter(shot => shot.media_id === mediaId), // populate shots
          title: title,
          asrSegments: asr_segments,
        }
      ] as [string, ProcessedVideoInfo]
    })
    // Sort videos
    .sort(([, videoInfoA], [, videoInfoB]) => {
      const distanceA = Math.max(...videoInfoA.shots.map(shot => shot.distance));
      const distanceB = Math.max(...videoInfoB.shots.map(shot => shot.distance));
      return distanceB - distanceA;
    })
  );
}

const processUnmergedSegments = (unmergedSegments: VideoSegment[], processedVideos: Map<string, ProcessedVideoInfo>): ProcessedVideoSegment[] => {
  // Populate video info
  return unmergedSegments.map(segment => {
    return {
      ...segment,
      mediaType: 'VIDEO',
      mediaInfo: processedVideos.get(segment.media_id)!
    }
  });
}

const processShots = (shots: VideoSegment[], processedVideos: Map<string, ProcessedVideoInfo>): ProcessedVideoSegment[] => {
  // Populate video info
  return shots.map(shot => {

    return {
      ...shot,
      mediaType: 'VIDEO',
      mediaInfo: processedVideos.get(shot.media_id)!
    }
  });
}

const processSearchResults = (results: SearchResponse, isFeaturedImages: boolean = false): ProcessedSearchResponse => {
  console.log('Search response', results);
  
  let processedSearchResults = {
    Image: {
      vectors: [],
      mediaInfo: new Map(),
    },
    Video: {
      unmerged_windows: [],
      merged_windows: [],
      mediaInfo: new Map(),
    },
    VideoAudio: {
      unmerged_windows: [],
      merged_windows: [],
      mediaInfo: new Map(),
    },
  } as ProcessedSearchResults;

  if (!(results.image_results || results.video_results || results.video_audio_results)) {
    return {
      processedSearchResults,
      time: results.time,
    } as ProcessedSearchResponse;
  }

  if (results.image_results) {
    processedSearchResults.Image.mediaInfo = new Map(
      Object.entries(results.image_results.images)
        .map(([mediaId, imageInfo]) => {
          // Populate title field with filename if it doesn't exist
          imageInfo.title = imageInfo.title || imageInfo.external_metadata.title || imageInfo.filename;
          return [
            mediaId,
            {
              ...imageInfo,
              vectors: results.image_results!.vectors.filter(vector => vector.media_id === mediaId), // populate vectors array
            }
          ] as [string, ProcessedImageInfo]
        })
        // Sort images
        .sort(([, imageInfoA], [, imageInfoB]) => {
          const distanceA = Math.max(...imageInfoA.vectors.map(vector => vector.distance));
          const distanceB = Math.max(...imageInfoB.vectors.map(vector => vector.distance));
          return distanceB - distanceA;
        })
    );
    processedSearchResults.Image.vectors = results.image_results.vectors.map(vector => {
      return {
        ...vector,
        mediaType: 'IMAGE',
        mediaInfo: processedSearchResults.Image.mediaInfo.get(vector.media_id)!
      };
    });
    for (let [mediaId, processedImageInfo] of processedSearchResults.Image.mediaInfo) {
      processedImageInfo.vectors = processedSearchResults.Image.vectors.filter(vector => vector.media_id === mediaId)
    }
  }
  if (results.video_results) {
    processedSearchResults.Video.mediaInfo = processVideos(results.video_results.videos, results.video_results.merged_windows);
    processedSearchResults.Video.unmerged_windows = processUnmergedSegments(results.video_results.unmerged_windows, processedSearchResults.Video.mediaInfo);
    processedSearchResults.Video.merged_windows = processShots(results.video_results.merged_windows, processedSearchResults.Video.mediaInfo);
    for (let [mediaId, processedVideo] of processedSearchResults.Video.mediaInfo) {
      processedVideo.vectors = processedSearchResults.Video.unmerged_windows.filter(segment => segment.media_id === mediaId);
      processedVideo.shots = processedSearchResults.Video.merged_windows.filter(shot => shot.media_id === mediaId)
    }

    // This ensures that when the user changes the 'viewModality' selection on the home page from 'Visual' to 'Audio', they still see the same set of featured videos
    if (isFeaturedImages) {
      processedSearchResults.VideoAudio = processedSearchResults.Video;
    }
  }
  if (results.video_audio_results) {
    processedSearchResults.VideoAudio.mediaInfo = processVideos(results.video_audio_results.videos, results.video_audio_results.merged_windows);
    processedSearchResults.VideoAudio.unmerged_windows = processUnmergedSegments(results.video_audio_results.unmerged_windows, processedSearchResults.VideoAudio.mediaInfo);
    processedSearchResults.VideoAudio.merged_windows = processShots(results.video_audio_results.merged_windows, processedSearchResults.VideoAudio.mediaInfo);
    for (let [mediaId, processedVideo] of processedSearchResults.VideoAudio.mediaInfo) {
      processedVideo.vectors = processedSearchResults.Video.unmerged_windows.filter(segment => segment.media_id === mediaId);
      processedVideo.shots = processedSearchResults.VideoAudio.merged_windows.filter(shot => shot.media_id === mediaId)
    }
  }
  
  return {
    processedSearchResults,
    time: results.time,
  } as ProcessedSearchResponse;
};

const viewModalityToSearchInType = {
  'Image': 'image',
  'Video': 'video',
  'VideoAudio': 'av',
  'Audio': 'audio',
};

const fetchFeaturedImages = (
  viewModality: ViewModality,
  featureExtractorId: string,
  pageStart: number,
  pageEnd: number
): Promise<ProcessedSearchResponse> => {
  const start = pageStart*config.PAGE_SIZE;
  const end = Math.min(MAX_FEATURED_IMAGES, pageEnd*config.PAGE_SIZE);

  const urlParams = new URLSearchParams([
    ['featured_in', viewModalityToSearchInType[viewModality]],
    ['feature_extractor_id', featureExtractorId],
    ['start', start.toString()],
    ['end', end.toString()],
    ['thumbs', config.FETCH_THUMBS.toString()],
    ['random_seed', FEATURED_IMAGES_RANDOM_SEED.toString()]
  ]);

  return fetchWithTimeout(`featured?${urlParams.toString()}`, config.FETCH_TIMEOUT, {
    method: 'GET'
  }).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Failed to fetch featured images. ${response.status} - ${response.statusText}`);
    }
    return response.json() as Promise<SearchResponse>;
  }).then((results) => processSearchResults(results, true));
  // .then((results: SearchResponseJSONObject[]) => {
  //   // Populate title field with filename if it doesn't exist
  //   results.forEach(result => {
  //     if (!result.info.title) {
  //       result.info.title = result.info.filename;
  //     }
  //   });
  //   return results;
  // });
}

const convertQueriesToFormData = (queries: Query[]) => {
  let formData = new FormData();
  for (const q of queries) {
    const qterm: any = {
      term_id: q.id,
      is_negative: q.isNegative ?? false,
    };
    if (q.type === 'TEXT') {
      qterm.txt = q.value;
    } else if (q.type === 'INTERNAL_IMAGE') {
      qterm.vector_id = `${q.value.media_id}/${q.value.vector_id}`;
    } else if (q.type === 'IMAGE_FILE' || q.type === 'IMAGE_URL'
               || q.type === 'AUDIO_FILE' || q.type === 'AUDIO_URL') {
      // Files go in a separate part with a filename matching term_id.
      qterm.src = q.type.endsWith('_URL') ? q.value : null;
      qterm.qtype = q.type.startsWith('IMAGE_') ? 'visual' : 'audio';
    } else {
      throw new Error('Invalid query type');
    }
    formData.append('query_term', JSON.stringify(qterm));
    if (q.type === 'IMAGE_FILE' || q.type === 'AUDIO_FILE') {
      // filename must match the qterm.term_id
      formData.append('query_file', (q.value as unknown) as File, q.id);
    }
  }
  return formData;
}


const fetchRelatedVectors = async (vector_id: string, media_id: string): Promise<VectorInfo[]> => {
  const params = new URLSearchParams();
  params.append("media_id", media_id);

  const r = await fetchWithTimeout(
    `related-vectors/${vector_id}?${params}`,
    config.FETCH_TIMEOUT,
    { method: "GET" });
  return await r.json();
}


const fetchSearchResults = (queries: Query[], viewModality: ViewModality, featureExtractorId: string, pageStart: number, pageEnd: number, shotScaleFilter: number[]): Promise<ProcessedSearchResponse> => {
  console.log('Fetching queries', queries);
  const start = pageStart*config.PAGE_SIZE;
  const end = Math.min(config.MAX_SEARCH_RESULTS, pageEnd*config.PAGE_SIZE);
  const metadataFilterQueries = queries.filter(q => q.type === "METADATA");
  
  const formData = convertQueriesToFormData(queries.filter(q => q.type !== "METADATA"));
  const urlParamsArray = [
    ['start', start.toString()],
    ['end', end.toString()],
    ['thumbs', config.FETCH_THUMBS.toString()],
    ['search_in', viewModalityToSearchInType[viewModality]],
    ['feature_extractor_id', featureExtractorId],
    ...shotScaleFilter.map(s => ['shot_scale', s.toString()]),
    ...metadataFilterQueries.map(q => ['metadata_filter', q.value as string])
  ];
  const urlParams = new URLSearchParams(urlParamsArray);
  const endpoint = `search2?${urlParams.toString()}`;
  
  return fetchWithTimeout(endpoint, config.FETCH_TIMEOUT, {
    method: 'POST',
    body: formData
  }).then(async (response) => {
    if (!response.ok) {
      const contentType = response.headers.get('content-type');
      let message;
      if (contentType && contentType.includes('application/json')) {
        const responseJSON = await response.json();
        if (responseJSON['detail'] && responseJSON['detail']['message']) {
          message = responseJSON['detail']['message'];
        } else {
          message = JSON.stringify(responseJSON);
        }
      } else {
        message = await response.text();
      }
      throw {
        summary: `Failed to fetch search results. ${response.status} - ${response.statusText}`,
        status: response.status,
        message: message
      }
    }
    return response.json() as Promise<SearchResponse>;
  }).then(processSearchResults);
  // .then((results: SearchResponseJSONObject[]) => {
  //   // Populate title field with filename if it doesn't exist
  //   results.forEach(result => {
  //     if (!result.info.title) {
  //       result.info.title = result.info.filename;
  //     }
  //   });
  //   return results;
  // });
}


export const useDataService = (): DataServiceOutput => {
  const [ searchingState, setSearchingState ] = useState({
    queries: [] as Query[],
    isFeaturedImages: false,
    isLoadingNewSearch: false,
    searchLatency: NaN,
    totalResults: NaN
  });
  // // pagedResults will be an array of arrays (each sub-array represents the results in a given page)
  // const [ pagedResults, setPagedResults ] = useState<any[][]>(getArrayOfEmptyArrays(NUM_PAGES));
  // const [ pageNum, setPageNum ] = useState(0);
  const [ searchResponse, setSearchResponse ] = useState<ProcessedSearchResults>({
    Image: { vectors: [], mediaInfo: new Map() },
    Video: { unmerged_windows: [], merged_windows: [], mediaInfo: new Map() },
    VideoAudio: { unmerged_windows: [], merged_windows: [], mediaInfo: new Map() },
  });

  // Get featured images to display on home page
  const fetchFeaturedImagesAndSetState = (
    viewModality: ViewModality, featureExtractorId: string
  ) => {
    return fetchFeaturedImages(
      viewModality, featureExtractorId, 0, config.NUM_PAGES_PER_REQUEST
    ).then((_searchResponse: ProcessedSearchResponse) => {
      setSearchingState({
        queries: [],
        isFeaturedImages: true,
        isLoadingNewSearch: false,
        searchLatency: _searchResponse.time,
        totalResults: MAX_FEATURED_IMAGES
      });
      setSearchResponse(_searchResponse.processedSearchResults);
      // setPageNum(0);

      // // Page slicing
      // const _pagedResults = getArrayOfEmptyArrays(NUM_PAGES);
      // const resultPages = chunk(images, config.PAGE_SIZE);
      // _pagedResults.splice(0, resultPages.length, ...resultPages);

      // setPagedResults(_pagedResults);
      // return;
    });
  };

  // // Navigate to a different page for the current query
  // const changePageNum = async (page: number) => {
  //   setPageNum(page);
    
  //   // Fetch page if the page hasn't been fetched yet (multiple pages are fetched at once based on config.NUM_PAGES_PER_REQUEST)
  //   if (pagedResults[page].length === 0) {
  //     const fetchStartPageNum =
  //       Math.floor(page / config.NUM_PAGES_PER_REQUEST) * config.NUM_PAGES_PER_REQUEST;
  //     const fetchEndPageNum = fetchStartPageNum + config.NUM_PAGES_PER_REQUEST;

  //     let searchResponseJSON: SearchResponseJSONObject[];
  //     if (searchingState.isFeaturedImages) {
  //       searchResponseJSON = await fetchFeaturedImages(fetchStartPageNum, fetchEndPageNum);
  //     } else {
  //       searchResponseJSON = await fetchSearchResults(searchingState.queries, fetchStartPageNum, fetchEndPageNum);
  //     }
  
  //     // Page slicing
  //     setPagedResults(_pagedResults => {
  //       _pagedResults = [..._pagedResults];
  //       const resultPages = chunk(searchResponseJSON, config.PAGE_SIZE);
  //       _pagedResults.splice(fetchStartPageNum, config.NUM_PAGES_PER_REQUEST, ...resultPages);
  //       return _pagedResults;
  //     });
  //   }
  // }


  const fillRelatedVectors = async (imageDetails: ProcessedImageVector | ProcessedVideoSegment) => {
    const vectors = await fetchRelatedVectors(imageDetails.vector_id, imageDetails.media_id);
    // This mediaInfo is used to get the image width/height which is
    // used in a bunch of places to draw the thumbnail in a manner
    // compatible with the compact/justified image grid in the search
    // results.
    imageDetails.related_vectors = vectors.map(
      v => {return {...v, mediaType: imageDetails.mediaType, mediaInfo: imageDetails.mediaInfo}}
    );
    return imageDetails;
  };


  // Get results for a new search query
  const performNewSearch = async (queries: Query[], viewModality: ViewModality, featureExtractorId: string, shotScaleFilter: number[]) => {
    setSearchingState((_searchingState) => ({
      ..._searchingState,
      isLoadingNewSearch: true
    }));
    let searchResponseJSON: ProcessedSearchResponse;
    try {
      searchResponseJSON = await fetchSearchResults(queries, viewModality, featureExtractorId, 0, config.NUM_PAGES_PER_REQUEST, shotScaleFilter);
    } catch (e) {
      setSearchingState((_searchingState) => ({
        ..._searchingState,
        isLoadingNewSearch: false
      }));
      throw e;
    }
    setSearchingState({
      queries: queries,
      isFeaturedImages: false,
      isLoadingNewSearch: false,
      searchLatency: searchResponseJSON.time,
      totalResults: config.MAX_SEARCH_RESULTS
    });
    setSearchResponse(searchResponseJSON.processedSearchResults);
    // setPageNum(0);

    // // Page slicing
    // const _pagedResults = getArrayOfEmptyArrays(NUM_PAGES);
    // const resultPages = chunk(searchResponseJSON, config.PAGE_SIZE);
    // _pagedResults.splice(0, resultPages.length, ...resultPages);

    // setPagedResults(_pagedResults);
    // return;
  };

  // Report an image
  const reportImage = async (imageId: string, reasons: string[]) => {
    const formData = convertQueriesToFormData(searchingState.queries);
    formData.append('sourceURI', imageId);
    for (let reason of reasons) {
      formData.append('reasons', reason);
    }
    return fetchWithTimeout('report', 40000, {
      method: 'POST',
      body: formData
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`Fetch failed. ${response.status} - ${response.statusText}`);
      }
      return response.text();
    });
  }


  return {
    searchResults: searchResponse,
    isLoadingNewSearch: searchingState.isLoadingNewSearch,
    searchLatency: searchingState.searchLatency,
    totalResults: searchingState.totalResults,
    // pageNum,
    // changePageNum,
    performNewSearch,
    fetchFeaturedImagesAndSetState,
    reportImage,
    fillRelatedVectors,
  }
}
