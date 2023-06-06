import { useState } from 'react';
import { DataServiceOutput, FeaturedImagesJSONObject, Query, SearchResponse, SearchResponseJSONObject } from './misc/types.ts';
import config from './config.ts';
import { fetchWithTimeout, chunk, getArrayOfEmptyArrays } from './misc/utils.ts';

const NUM_PAGES = Math.ceil(config.MAX_SEARCH_RESULTS / config.PAGE_SIZE);

const fetchFeaturedImages = (): Promise<FeaturedImagesJSONObject[]> => {
  return fetchWithTimeout("./featured_images.json", config.FETCH_TIMEOUT, {
    method: 'GET'
  }).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Failed to fetch featured images. ${response.status} - ${response.statusText}`);
    }
    return response.json() as Promise<FeaturedImagesJSONObject[]>;
  });
}

const convertQueriesToFormData = (queries: Query[]) => {
  let formData = new FormData();
  for (const q of queries) {
    if (q.type === 'FILE') {
      formData.append('file_queries', (q.value as unknown) as File);
    } else if (q.type === 'URL') {
      formData.append('url_queries', q.value);
    } else if (q.type === 'TEXT') {
      formData.append('text_queries', q.value);
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
  const nonTextQueries = queries.filter(q => q.type !== 'TEXT');
  let formData = undefined;
  if (nonTextQueries.length > 0) {
    formData = convertQueriesToFormData(nonTextQueries);
  }

  const urlParams = new URLSearchParams([
    ['start', start.toString()],
    ['end', end.toString()],
    ['thumbs', config.FETCH_THUMBS.toString()],
    ...textQueries.map(q => ['text_queries', q.value as string])
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
  const fetchAndTransformFeaturedImages = () => {
    return fetchFeaturedImages().then((featuredImagesJSON: FeaturedImagesJSONObject[]) => {
      // console.log(featuredImagesJSON);
  
      // Shuffle images
      const featuredImagesJSONShuffled = featuredImagesJSON
        .map((value) => ({ value, sort: Math.random() }))
        .sort((a, b) => a.sort - b.sort)
        .map(({ value }) => value);
  
      // Data transformation
      const images = featuredImagesJSONShuffled.map((featuredImagesJSONObject) => ({
        id: featuredImagesJSONObject.original_download_url,
        link: featuredImagesJSONObject.original_download_url,
        thumbnail: featuredImagesJSONObject.original_download_url,
        info: {
          width: featuredImagesJSONObject.orig_width,
          height: featuredImagesJSONObject.orig_height,
          title: featuredImagesJSONObject.img_title,
          author: featuredImagesJSONObject.Artist,
          caption: featuredImagesJSONObject.ImageDescription,
          copyright: featuredImagesJSONObject.LicenseShortName
        }
      }));

      setPagedResults(chunk(images, config.PAGE_SIZE));
      setSearchingState({
        queries: [],
        isFeaturedImages: true,
        isSearching: false,
        searchLatency: NaN,
        totalResults: images.length
      });
      setPageNum(0);
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

      const searchResponseJSON = await fetchSearchResults(searchingState.queries, fetchStartPageNum, fetchEndPageNum);
  
      // Data transformation
      const images = searchResponseJSON.map((searchResponseJSONObject) => ({
        ...searchResponseJSONObject,
        id: searchResponseJSONObject.link,
      }));
  
      setPagedResults(_pagedResults => {
        _pagedResults = [..._pagedResults];
        const resultPages = chunk(images, config.PAGE_SIZE);
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

    // Data transformation
    const images = searchResponseJSON.map((searchResponseJSONObject) => ({
      ...searchResponseJSONObject,
      id: searchResponseJSONObject.link,
    }));

    const _pagedResults = getArrayOfEmptyArrays(NUM_PAGES);
    const resultPages = chunk(images, config.PAGE_SIZE);
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
    fetchAndTransformFeaturedImages,
    reportImage
  }
}