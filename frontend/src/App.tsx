import React, { useEffect, useRef, useState } from 'react';
import { Layout, Modal } from 'antd';
const { Content } = Layout;
import { CloseCircleFilled, StopTwoTone } from '@ant-design/icons';

import './App.scss';
import SearchResults from './SearchResults.tsx';
import WiseHeader from './WiseHeader.tsx';
import WiseOverviewCard from './WiseOverviewCard.tsx';
import { Query } from './misc/types.ts';
import config from './config.ts';
import { fetchWithTimeout } from './misc/utils.ts';
import { useDataSerivce } from './DataService.ts';

export const App: React.FunctionComponent = () => {
  // const [isAboutModalOpen, setIsAboutModalOpen] = useState(false);
  // const showAboutModal = () => { setIsAboutModalOpen(true) };
  // const closeAboutModal = () => { setIsAboutModalOpen(false) };

  const [multimodalQueries, setMultimodalQueries] = useState<Query[]>([]); // Stores the file, URL, and text queries
  const [searchText, setSearchText] = useState(''); // Stores the main text query entered in the search bar

  const dataService = useDataSerivce();
  const [isHomePage, setIsHomePage] = useState(true);
  const [projectInfo, setProjectInfo] = useState<any>({});

  const refsForTour = {
    searchBar: useRef(null),
    visualSearchButton: useRef(null),
    multimodalSearchButton: useRef(null),
    paginationControls: useRef(null),
    reportImageButton: useRef(null)
  }

  useEffect(() => {
    // Initialise home page with featured images
    dataService.fetchAndTransformFeaturedImages().then(_ => {
      setIsHomePage(true); // TODO set setIsFeaturedImages based on the page route, rather than setting it here
    }).catch((err) => {
      Modal.error({
        title: 'Error: unable to load featured images',
        content: 'Please try again later',
      });
      console.error(err);
    });

    // Fetch project info
    fetchWithTimeout(config.API_BASE_URL+"info", 7000, { method: 'GET' })
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

  const _submitSearch = (queries: Query[]) => {
    dataService.performNewSearch(queries).then(_ => {
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
      id: '',
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
      id: '',
      type: "TEXT",
      value: exampleQuery
    }]);
  }  

  return <Layout style={(dataService.searchResults.length === 0) ? {background: 'transparent'} : {}}>
    <WiseHeader multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                searchText={searchText} setSearchText={setSearchText}
                submitSearch={submitSearch}
                refsForTour={refsForTour}
                isHomePage={isHomePage} isSearching={dataService.isSearching}></WiseHeader>
    <Content className="wise-content">
      {isHomePage && // Only show if isHomePage is true
        <WiseOverviewCard handleExampleQueryClick={handleExampleQueryClick} projectInfo={projectInfo} refsForTour={refsForTour} />
      }
      {/* <Modal title="About WISE"
              open={isAboutModalOpen}
              onCancel={closeAboutModal}
              footer={<Button type="primary" onClick={closeAboutModal}>Close</Button>}
      >
        <p>Explanation of what WISE does, and how to use WISE</p>
        <p>TODO add explanation...</p>
        <p>WISE is developed at Oxford VGG and the code is available open-source.</p>
      </Modal> */}
      <SearchResults dataService={dataService} isHomePage={isHomePage} projectInfo={projectInfo} />
    </Content>
  </Layout>
};