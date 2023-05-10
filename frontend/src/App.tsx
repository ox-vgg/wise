import React, { useEffect, useState } from 'react';
import { Button, Card, Layout, Modal, Tooltip } from 'antd';
const { Content } = Layout;
import { CloseCircleFilled, InfoCircleOutlined, StopTwoTone } from '@ant-design/icons';

import './App.scss';
import SearchResults from './SearchResults.tsx';
import WiseHeader from './WiseHeader.tsx';
import { Query } from './misc/types.ts';
import config from './config.ts';
import { fetchWithTimeout } from './misc/utils.ts';
import { useDataSerivce } from './DataService.ts';

let exampleQueries = config.EXAMPLE_QUERIES;
exampleQueries = exampleQueries.map(value => ({ value, sort: Math.random() }))
                              .sort((a, b) => a.sort - b.sort)
                              .map(({ value }) => value); // Shuffle array
exampleQueries = exampleQueries.slice(0,5);

export const App: React.FunctionComponent = () => {
  // const [isAboutModalOpen, setIsAboutModalOpen] = useState(false);
  // const showAboutModal = () => { setIsAboutModalOpen(true) };
  // const closeAboutModal = () => { setIsAboutModalOpen(false) };

  const [multimodalQueries, setMultimodalQueries] = useState<Query[]>([]); // Stores the file, URL, and text queries
  const [searchText, setSearchText] = useState(''); // Stores the main text query entered in the search bar

  const dataService = useDataSerivce();
  const [isHomePage, setIsHomePage] = useState(true);
  const [projectInfo, setProjectInfo] = useState<any>({});

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

  const aboutWiseTabContent: Record<string, React.ReactNode> = {
    'Overview': (
      <div className="wise-overview">
        <p>WISE is a smart search engine for images, using AI to understand the meaning behind your search query, to find the most relevant images that match what you're looking for.
          {/* <br />
          <a role="button" onClick={showAboutModal}>Learn more about WISE and how to use WISE</a> */}
        </p>
        <p>Here, you can search a subset of {Math.floor(projectInfo.num_images / 1000000)} million images
        <Tooltip title="This subset only includes JPEG and PNG images uploaded on/before 1 Jan 2023 with a minimum height and width of 224px. We plan on adding more images to this set over time.">
          <InfoCircleOutlined style={{marginLeft: '3px', marginRight: '5px'}} />
        </Tooltip>
        from Wikimedia Commons.</p>
        <p className="wise-example-queries">
          Example queries: {exampleQueries.map((x, i) => 
            <Button size="small" shape="round" type='primary' ghost onClick={() => handleExampleQueryClick(x)} key={i}>{x}</Button>
          )}
        </p>
      </div>
    ),
    'About WISE': <>
      <p>WISE Image Search Engine (WISE) is an open-source image search engine which leverages recent advances in machine learning and vision-language models that enable search based on image content using natural language. The expressive power of natural language allows the user to flexibly describe what they are looking for.</p>
      {/* TODO add more explanations */}
      <p>WISE is developed at the Visual Geometry Group, University of Oxford.</p>
      <p><a href="https://gitlab.com/vgg/wise/wise" target='_blank'>Code repository</a></p>
      <p><a href="https://www.robots.ox.ac.uk/~vgg/software/wise/" target='_blank'>Project webpage</a></p>
    </>,
    'How to use WISE': <p>TODO</p>,
    'Disclaimer': (
      <>
        <ul>
          <li>The images shown below are hosted on Wikimedia Commons and this website only provides search. The images belong to their respective authors and they are not the property of the University of Oxford.</li>
          <li>We currently do not use cookies on this website</li>
        </ul>
      </>
    )
  };
  const aboutWiseTabList = Object.keys(aboutWiseTabContent).map(x => ({key: x, tab: x}));
  const [aboutWiseActiveTabKey, setAboutWiseActiveTabKey] = useState('Overview');
  

  return <Layout style={(dataService.searchResults.length === 0) ? {background: 'transparent'} : {}}>
    <WiseHeader multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                searchText={searchText} setSearchText={setSearchText}
                isHomePage={isHomePage} isSearching={dataService.isSearching}
                submitSearch={submitSearch}></WiseHeader>
    <Content className="wise-content">
      {isHomePage && // Only show if isHomePage is true
        <Card
          id="wise-overview-card"
          style={{ width: '100%' }}
          size='small'
          tabList={aboutWiseTabList}
          activeTabKey={aboutWiseActiveTabKey}
          tabBarExtraContent={<></>} // <a href="#">Close</a>
          onTabChange={setAboutWiseActiveTabKey}
        >
          {aboutWiseTabContent[aboutWiseActiveTabKey]}
        </Card>
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