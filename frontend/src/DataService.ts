import { useState } from 'react';
import { DataServiceOutput, Query, SearchResponse, SearchResponseJSONObject } from './misc/types.ts';
import config from './config.ts';
import { fetchWithTimeout, chunk, getArrayOfEmptyArrays } from './misc/utils.ts';

const NUM_PAGES = Math.ceil(config.MAX_SEARCH_RESULTS / config.PAGE_SIZE);

const MAX_FEATURED_IMAGES = 1000; // TODO set this based on actual number of featured images
const FEATURED_IMAGES_RANDOM_SEED = Math.floor(Math.random()*100); // Generate a random number between 0-100 to be used as the random seed when fetching the featured images

const fetchFeaturedImages = (pageStart: number, pageEnd: number): Promise<SearchResponseJSONObject[]> => {
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
  }).then(
    (response: SearchResponse) => Object.values(response)[0] // Get value corresponding to first key
  ).then((results: SearchResponseJSONObject[]) => {
    // Populate title field with filename if it doesn't exist
    results.forEach(result => {
      if (!result.info.title) {
        result.info.title = result.info.filename;
      }
    });
    return results;
  });
}

const convertQueriesToFormData = (queries: Query[]) => {
  let formData = new FormData();
  for (const q of queries) {
    if (q.type === 'FILE') {
      let query_type = 'file_queries';
      if (q.isNegative) query_type = 'negative_' + query_type
      formData.append(query_type, (q.value as unknown) as File);
    } else if (q.type === 'URL') {
      let query_type = 'url_queries';
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

const fetchSearchResults = (queries: Query[], pageStart: number, pageEnd: number): Promise<SearchResponseJSONObject[]> => {
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

  const urlParams = new URLSearchParams([
    ['start', start.toString()],
    ['end', end.toString()],
    ['thumbs', config.FETCH_THUMBS.toString()],
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
  }).then(
    (response: SearchResponse) => Object.values(response)[0] // Get value corresponding to first key
  ).then((results: SearchResponseJSONObject[]) => {
    // Populate title field with filename if it doesn't exist
    results.forEach(result => {
      if (!result.info.title) {
        result.info.title = result.info.filename;
      }
    });
    return results;
  });
}


export const useDataService = (): DataServiceOutput => {
  const [ searchingState, setSearchingState ] = useState({
    queries: [] as Query[],
    isFeaturedImages: false,
    isSearching: false,
    searchLatency: NaN,
    totalResults: NaN
  });
  // pagedResults will be an array of arrays (each sub-array represents the results in a given page)
  const [ pagedResults, setPagedResults ] = useState<any[][]>(getArrayOfEmptyArrays(NUM_PAGES));
  const [ pageNum, setPageNum ] = useState(0);

  // Get featured images to display on home page
  const fetchFeaturedImagesAndSetState = () => {
    return fetchFeaturedImages(0, config.NUM_PAGES_PER_REQUEST).then((images: SearchResponseJSONObject[]) => {
      setSearchingState({
        queries: [],
        isFeaturedImages: true,
        isSearching: false,
        searchLatency: NaN,
        totalResults: MAX_FEATURED_IMAGES
      });
      setPageNum(0);

      // Page slicing
      const _pagedResults = getArrayOfEmptyArrays(NUM_PAGES);
      const resultPages = chunk(images, config.PAGE_SIZE);
      _pagedResults.splice(0, resultPages.length, ...resultPages);

      setPagedResults(_pagedResults);
      return;
    });
  };

  // Navigate to a different page for the current query
  const changePageNum = async (page: number) => {
    setPageNum(page);
    
    // Fetch page if the page hasn't been fetched yet (multiple pages are fetched at once based on config.NUM_PAGES_PER_REQUEST)
    if (pagedResults[page].length === 0) {
      const fetchStartPageNum =
        Math.floor(page / config.NUM_PAGES_PER_REQUEST) * config.NUM_PAGES_PER_REQUEST;
      const fetchEndPageNum = fetchStartPageNum + config.NUM_PAGES_PER_REQUEST;

      let searchResponseJSON: SearchResponseJSONObject[];
      if (searchingState.isFeaturedImages) {
        searchResponseJSON = await fetchFeaturedImages(fetchStartPageNum, fetchEndPageNum);
      } else {
        searchResponseJSON = await fetchSearchResults(searchingState.queries, fetchStartPageNum, fetchEndPageNum);
      }
  
      // Page slicing
      setPagedResults(_pagedResults => {
        _pagedResults = [..._pagedResults];
        const resultPages = chunk(searchResponseJSON, config.PAGE_SIZE);
        _pagedResults.splice(fetchStartPageNum, config.NUM_PAGES_PER_REQUEST, ...resultPages);
        return _pagedResults;
      });
    }
  }

  // Get results for a new search query
  const performNewSearch = async (queries: Query[]) => {
    setSearchingState((_searchingState) => ({
      ..._searchingState,
      isSearching: true
    }));
    const time0 = performance.now();
    let searchResponseJSON: SearchResponseJSONObject[];
    try {
      searchResponseJSON = await fetchSearchResults(queries, 0, config.NUM_PAGES_PER_REQUEST);
    } catch (e) {
      setSearchingState((_searchingState) => ({
        ..._searchingState,
        isSearching: false
      }));
      throw e;
    }
    const time1 = performance.now();
    setSearchingState({
      queries: queries,
      isFeaturedImages: false,
      isSearching: false,
      searchLatency: time1 - time0,
      totalResults: config.MAX_SEARCH_RESULTS
    });
    setPageNum(0);

    // Page slicing
    const _pagedResults = getArrayOfEmptyArrays(NUM_PAGES);
    const resultPages = chunk(searchResponseJSON, config.PAGE_SIZE);
    _pagedResults.splice(0, resultPages.length, ...resultPages);

    setPagedResults(_pagedResults);
    return;
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
    searchResults: pagedResults[pageNum],
    isSearching: searchingState.isSearching,
    searchLatency: searchingState.searchLatency,
    totalResults: searchingState.totalResults,
    pageNum,
    changePageNum,
    performNewSearch,
    fetchFeaturedImagesAndSetState,
    reportImage
  }
}