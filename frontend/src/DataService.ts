import { useState } from 'react';
import { DataServiceOutput, ProcessedSearchResults, ProcessedVideoSegment, ProcessedVideoInfo, Query, SearchResponse, VideoSegment, VideoInfo, ProcessedSearchResponse } from './misc/types.ts';
import config from './config.ts';
import { fetchWithTimeout /*, chunk, getArrayOfEmptyArrays */ } from './misc/utils.ts';

// const NUM_PAGES = Math.ceil(config.MAX_SEARCH_RESULTS / config.PAGE_SIZE);

const MAX_FEATURED_IMAGES = 1000; // TODO set this based on actual number of featured images
const FEATURED_IMAGES_RANDOM_SEED = Math.floor(Math.random()*100); // Generate a random number between 0-100 to be used as the random seed when fetching the featured images

const processVideos = (videos: Record<string, VideoInfo>, shots: VideoSegment[]) => {
  return new Map(
    Object.entries(videos).map(([mediaId, videoInfo]) => {
      if (!videoInfo.link.startsWith('http')) {
        videoInfo.link = config.API_BASE_URL + videoInfo.link; // Fixes video URLs for dev mode
      }

      // const title = videoInfo.filename;
      if (!videoInfo.timeline_hover_thumbnails.startsWith('http')) {
        videoInfo.timeline_hover_thumbnails = config.API_BASE_URL + videoInfo.timeline_hover_thumbnails; // Fixes URLs for dev mode
      }
      if (!videoInfo.thumbnail.startsWith('http') && !videoInfo.thumbnail.startsWith('data:')) {
        videoInfo.thumbnail = config.API_BASE_URL + videoInfo.thumbnail; // Fixes URLs for dev mode
      }

      return [
        mediaId,
        {
          ...videoInfo,
          shots: shots.filter(shot => shot.media_id === mediaId), // populate shots
          title: videoInfo.filename,
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
    if (segment.link && !segment.link.startsWith('http')) {
      segment.link = config.API_BASE_URL + segment.link; // Fixes video URLs for dev mode
    }
    if (!segment.thumbnail.startsWith('http') && !segment.thumbnail.startsWith('data:')) {
      segment.thumbnail = config.API_BASE_URL + segment.thumbnail; // Fixes URLs for dev mode
    }
    return {
      ...segment,
      videoInfo: processedVideos.get(segment.media_id)!
    }
  });
}

const processShots = (shots: VideoSegment[], processedVideos: Map<string, ProcessedVideoInfo>): ProcessedVideoSegment[] => {
  // Populate video info
  return shots.map(shot => {
    if (shot.link && !shot.link.startsWith('http')) {
      shot.link = config.API_BASE_URL + shot.link; // Fixes video URLs for dev mode
    }
    if (!shot.thumbnail.startsWith('http') && !shot.thumbnail.startsWith('data:')) {
      shot.thumbnail = config.API_BASE_URL + shot.thumbnail; // Fixes URLs for dev mode
    }

    return {
      ...shot,
      videoInfo: processedVideos.get(shot.media_id)!
    }
  });
}

const processSearchResults = (results: SearchResponse, isFeaturedImages: boolean = false): ProcessedSearchResponse => {
  console.log('Search response', results);
  if (!(results.video_results || results.video_audio_results)) throw new Error("Cannot process search results");
  
  let processedSearchResults = {
    Video: {
      unmerged_windows: [],
      merged_windows: [],
      videos: new Map(),
    },
    VideoAudio: {
      unmerged_windows: [],
      merged_windows: [],
      videos: new Map(),
    },
    Audio: {
      unmerged_windows: [],
      merged_windows: [],
      videos: new Map(),
    },
  } as ProcessedSearchResults;
  if (results.video_results) {
    processedSearchResults.Video.videos = processVideos(results.video_results.videos, results.video_results.merged_windows);
    processedSearchResults.Video.unmerged_windows = processUnmergedSegments(results.video_results.unmerged_windows, processedSearchResults.Video.videos);
    processedSearchResults.Video.merged_windows = processShots(results.video_results.merged_windows, processedSearchResults.Video.videos);
    for (let [mediaId, processedVideo] of processedSearchResults.Video.videos) {
      processedVideo.shots = processedSearchResults.Video.merged_windows.filter(shot => shot.media_id === mediaId)
    }

    // This ensures that when the user changes the 'viewModality' selection on the home page from 'Visual' to 'Audio', they still see the same set of featured videos
    if (isFeaturedImages) {
      processedSearchResults.VideoAudio = processedSearchResults.Video;
    }
  }
  if (results.video_audio_results) {
    processedSearchResults.VideoAudio.videos = processVideos(results.video_audio_results.videos, results.video_audio_results.merged_windows);
    processedSearchResults.VideoAudio.unmerged_windows = processUnmergedSegments(results.video_audio_results.unmerged_windows, processedSearchResults.VideoAudio.videos);
    processedSearchResults.VideoAudio.merged_windows = processShots(results.video_audio_results.merged_windows, processedSearchResults.VideoAudio.videos);
    for (let [mediaId, processedVideo] of processedSearchResults.VideoAudio.videos) {
      processedVideo.shots = processedSearchResults.VideoAudio.merged_windows.filter(shot => shot.media_id === mediaId)
    }
  }
  
  return {
    processedSearchResults,
    time: results.time,
  } as ProcessedSearchResponse;
};

const fetchFeaturedImages = (pageStart: number, pageEnd: number): Promise<ProcessedSearchResponse> => {
  const start = pageStart*config.PAGE_SIZE;
  const end = Math.min(MAX_FEATURED_IMAGES, pageEnd*config.PAGE_SIZE);

  const urlParams = new URLSearchParams([
    ['start', start.toString()],
    ['end', end.toString()],
    ['thumbs', config.FETCH_THUMBS.toString()],
    ['random_seed', FEATURED_IMAGES_RANDOM_SEED.toString()]
  ]);

  return fetchWithTimeout(config.API_BASE_URL + `featured?${urlParams.toString()}`, config.FETCH_TIMEOUT, {
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
    if (q.type === 'IMAGE_FILE') {
      let query_type = 'image_file_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, (q.value as unknown) as File);
    } else if (q.type === 'AUDIO_FILE') {
      let query_type = 'audio_file_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, (q.value as unknown) as File);
    } else if (q.type === 'IMAGE_URL') {
      let query_type = 'image_url_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, q.value);
    } else if (q.type === 'AUDIO_URL') {
      let query_type = 'audio_url_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, q.value);
    } else if (q.type === 'INTERNAL_IMAGE') {
      let query_type = 'internal_image_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, q.value);
    } else if (q.type === 'TEXT') {
      let query_type = 'text_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, q.value);
    } else {
      throw new Error('Invalid query type');
    }
  }
  return formData;
}

const fetchSearchResults = (queries: Query[], viewModality: string, pageStart: number, pageEnd: number): Promise<ProcessedSearchResponse> => {
  console.log('Fetching queries', queries);
  const start = pageStart*config.PAGE_SIZE;
  const end = Math.min(config.MAX_SEARCH_RESULTS, pageEnd*config.PAGE_SIZE);

  const textQueries = queries.filter(q => q.type === "TEXT");
  const internalImageQueries = queries.filter(q => q.type === "INTERNAL_IMAGE");
  const otherQueries = queries.filter(q => q.type !== 'TEXT' && q.type !== 'INTERNAL_IMAGE');
  let formData = undefined;
  if (otherQueries.length > 0) {
    formData = convertQueriesToFormData(otherQueries);
  }

  let searchIn = 'undefined';
  if (viewModality == 'Video') searchIn = 'video';
  if (viewModality == 'VideoAudio') searchIn = 'av';

  const urlParams = new URLSearchParams([
    ['start', start.toString()],
    ['end', end.toString()],
    ['thumbs', config.FETCH_THUMBS.toString()],
    ['search_in', searchIn],
    ...textQueries.map(q => [(q.isNegative ? 'negative_' : '') + 'text_queries', q.value as string]),
    ...internalImageQueries.map(q => [(q.isNegative ? 'negative_' : '') + 'internal_image_queries', q.value as string])
  ]);
  
  const endpoint = config.API_BASE_URL + `search?${urlParams.toString()}`;
  
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
    isSearching: false,
    searchLatency: NaN,
    totalResults: NaN
  });
  // // pagedResults will be an array of arrays (each sub-array represents the results in a given page)
  // const [ pagedResults, setPagedResults ] = useState<any[][]>(getArrayOfEmptyArrays(NUM_PAGES));
  // const [ pageNum, setPageNum ] = useState(0);
  const [ searchResponse, setSearchResponse ] = useState<ProcessedSearchResults>({
    Video: { unmerged_windows: [], merged_windows: [], videos: new Map() },
    VideoAudio: { unmerged_windows: [], merged_windows: [], videos: new Map() },
    Audio: { unmerged_windows: [], merged_windows: [], videos: new Map() },
  });

  // Get featured images to display on home page
  const fetchFeaturedImagesAndSetState = () => {
    return fetchFeaturedImages(0, config.NUM_PAGES_PER_REQUEST).then((_searchResponse: ProcessedSearchResponse) => {
      setSearchingState({
        queries: [],
        isFeaturedImages: true,
        isSearching: false,
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

  // Get results for a new search query
  const performNewSearch = async (queries: Query[], viewModality: string) => {
    setSearchingState((_searchingState) => ({
      ..._searchingState,
      isSearching: true
    }));
    let searchResponseJSON: ProcessedSearchResponse;
    try {
      searchResponseJSON = await fetchSearchResults(queries, viewModality, 0, config.NUM_PAGES_PER_REQUEST);
    } catch (e) {
      setSearchingState((_searchingState) => ({
        ..._searchingState,
        isSearching: false
      }));
      throw e;
    }
    setSearchingState({
      queries: queries,
      isFeaturedImages: false,
      isSearching: false,
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
    return fetchWithTimeout(config.API_BASE_URL+'report', 40000, {
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
    isSearching: searchingState.isSearching,
    searchLatency: searchingState.searchLatency,
    totalResults: searchingState.totalResults,
    // pageNum,
    // changePageNum,
    performNewSearch,
    fetchFeaturedImagesAndSetState,
    reportImage
  }
}