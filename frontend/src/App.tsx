import React, { useEffect, useRef, useState } from 'react';
import { Layout, Modal } from 'antd';
const { Content } = Layout;
import { CloseCircleFilled, StopTwoTone } from '@ant-design/icons';
import { nanoid } from 'nanoid';

import './App.scss';
import SearchResults from './SearchResults.tsx';
import WiseHeader from './WiseHeader.tsx';
import WiseOverviewCard from './WiseOverviewCard.tsx';
import { ProcessedSearchResults, ProjectInfo, Query, TourVariables } from './misc/types.ts';
import { fetchWithTimeout } from './misc/utils.ts';
import { useDataService } from './DataService.ts';

export const App: React.FunctionComponent = () => {
  const [multimodalQueries, setMultimodalQueries] = useState<Query[]>([]); // Stores the file, URL, and text queries
  const [searchText, setSearchText] = useState(''); // Stores the main text query entered in the search bar
  const [viewModality, setViewModality] = useState<keyof ProcessedSearchResults>('Image');
  const [featureExtractorId, setFeatureExtractorId] = useState<string>('');

  const dataService = useDataService();
  const [isHomePage, setIsHomePage] = useState(true);
  const [projectInfo, setProjectInfo] = useState<ProjectInfo>({});

  const [isSearchDropdownOpenForTour, setIsSearchDropdownOpenForTour] = useState(false);
  const tourVariables: TourVariables = {
    searchBar: useRef(null),
    isSearchDropdownOpenForTour,
    setIsSearchDropdownOpenForTour,
    imageUploadButton: useRef(null),
    multimodalSearchArea: useRef(null),
    paginationControls: useRef(null),
    reportImageButton: useRef(null)
  }

  useEffect(() => {
    // Fetch project info
    fetchWithTimeout("info", 30000, { method: 'GET' })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to fetch project info. ${response.status} - ${response.statusText}`);
        }
        return response.json();
      })
      .then(setProjectInfo)
      .catch((err) => {
        Modal.error({
          title: 'Error: unable to load project info',
          content: 'Please try again later',
        });
        console.error(err);
      });
  }, []);

  useEffect(() => {
    // Set viewModality based on the first search modality listed in projectInfo.search_targets
    if (projectInfo.search_targets && Object.keys(projectInfo.search_targets).length > 0) {
      const media_types = Object.keys(projectInfo.search_targets) as Array<keyof typeof projectInfo.search_targets>;
      const default_media_type = media_types[0];
      const default_feature_extractor_id = projectInfo.search_targets[default_media_type]?.[0];
      const _viewModality = {
        'image': 'Image',
        'video': 'Video',
        'audio': 'VideoAudio'
      }[default_media_type];
      setViewModality(_viewModality as keyof ProcessedSearchResults)
      setFeatureExtractorId(default_feature_extractor_id ?? '');
    }
  }, [projectInfo]);

  useEffect(() => {
    // When WISE starts, these may be the empty string.  Wait until
    // they are set before fetching featured images.
    if (! viewModality || ! featureExtractorId)
      return;

    // Now that we have set the feature extractor and modality, we can
    // initialise home page with featured images
    dataService.fetchFeaturedImagesAndSetState(
      viewModality, featureExtractorId
    ).then(_ => {
      setIsHomePage(true); // TODO set setIsFeaturedImages based on the page route, rather than setting it here
    }).catch((err) => {
      Modal.error({
        title: 'Error: unable to load featured images',
        content: 'Please try again later',
      });
      console.error(err);
    });
  }, [viewModality, featureExtractorId]);

  const _submitSearch = (queries: Query[]) => {
    dataService.performNewSearch(queries, viewModality, featureExtractorId).then(_ => {
      setIsHomePage(false); // TODO set setIsFeaturedImages based on the page route, rather than setting it here
    }).catch((err) => {
      Modal.error({
        icon: (err.status === 403) ? <StopTwoTone twoToneColor="#ff4d4f" /> : <CloseCircleFilled />,
        title: 'Error: unable to load search results',
        content: (err.message) ? err.message : 'Please try again later',
      });
      console.error(err);
    });
  }

  const submitSearch = () => {
    let queries: Query[] = [...multimodalQueries];
    let searchTextTrimmed = searchText.trim();
    if (searchTextTrimmed) queries.push({
      id: nanoid(),
      type: "TEXT",
      value: searchTextTrimmed
    });
    if (queries.length === 0) return;
    else if (queries.length > 5) {
      Modal.error({
        title: 'The maximum number of queries is 5 queries',
        content: 'Please delete some of the queries',
      });
      return;
    }

    _submitSearch(queries);
  }

  const handleExampleQueryClick = (exampleQuery: string) => {
    setMultimodalQueries([]);
    setSearchText(exampleQuery);
    _submitSearch([{
      id: nanoid(),
      type: "TEXT",
      value: exampleQuery
    }]);
  }

  return <Layout>
    <WiseHeader multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                searchText={searchText} setSearchText={setSearchText}
                viewModality={viewModality} setViewModality={setViewModality}
                featureExtractorId={featureExtractorId} setFeatureExtractorId={setFeatureExtractorId}
                submitSearch={submitSearch}
                tourVariables={tourVariables}
                projectInfo={projectInfo}
                isHomePage={isHomePage} isLoadingNewSearch={dataService.isLoadingNewSearch}></WiseHeader>
    <Content className="wise-content">
      {isHomePage && // Only show if isHomePage is true
        <WiseOverviewCard handleExampleQueryClick={handleExampleQueryClick} projectInfo={projectInfo} tourVariables={tourVariables} />
      }
      <SearchResults dataService={dataService} isHomePage={isHomePage} projectInfo={projectInfo}
                      setSearchText={setSearchText} multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                      viewModality={viewModality}
                      featureExtractorId={featureExtractorId}
                      submitSearch={submitSearch} />
    </Content>
  </Layout>
};