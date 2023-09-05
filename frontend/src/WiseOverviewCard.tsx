import { useState } from 'react';
import { InfoCircleOutlined } from '@ant-design/icons';
import { Button, Card, Tooltip, Tour, TourProps } from 'antd';
import sanitizeHtml from 'sanitize-html';

import './WiseOverviewCard.scss';
import config from './config';
import { WiseOverviewCardProps } from './misc/types';

let exampleQueries = config.EXAMPLE_QUERIES;
exampleQueries = exampleQueries.map(value => ({ value, sort: Math.random() }))
                              .sort((a, b) => a.sort - b.sort)
                              .map(({ value }) => value); // Shuffle array
exampleQueries = exampleQueries.slice(0,5);

const WiseOverviewCard: React.FunctionComponent<WiseOverviewCardProps> = ({handleExampleQueryClick, projectInfo, refsForTour}) => {
  const [isTourOpen, setIsTourOpen] = useState<boolean>(false);
  
  const tourSteps: TourProps['steps'] = [
    {
      title: 'Enter your search query here',
      description: <>
        You can enter a detailed description such as 
        <Button size="small" shape="round" type='primary' ghost onClick={() => handleExampleQueryClick('Person riding a horse jumping')}>Person riding a horse jumping</Button>
        <br />
        WISE uses a language model to understand the meaning behind your query, allowing you to flexibly describe what you are looking for. WISE then tries to find images whose visual contents match what you are trying to look for.
      </>,
      // cover: <img />
      target: () => refsForTour.searchBar.current.input.parentElement,
    },
    {
      title: 'Visual search',
      description: 'Upload an image or paste an image link to find similar images',
      target: () => refsForTour.visualSearchButton.current,
    },
    {
      title: 'Compound multi-modal search',
      description: 'Combine images and text in your query. For example, if you upload a picture of a golden retriever and enter the text "in snow", WISE will find images of golden retrievers in snow.',
      target: () => refsForTour.multimodalSearchButton.current,
    },
    {
      title: 'Pagination',
      description: 'Navigate to another page in the search results',
      target: () => document.querySelector("#search-results > .ant-pagination"),
    },
    {
      title: 'Report image',
      description: <>
        If you want to report an image that is inappropriate, offensive, irrelevant to the search query, etc, you can hover over the image and click on the <img src="more_icon.png" height="14px" /> button on the top right corner.
      </>,
      target: () => document.querySelector("#wise-image-grid > .wise-image-wrapper:nth-of-type(2)"),
    },
  ];

  const handleTourChange = (current?: number) => {
    // Make the 'three dots' icon (for reporting images) visible
    if (current === 4) {
      document.querySelector("#wise-image-grid > .wise-image-wrapper:nth-of-type(2)")?.classList.add('wise-image-dropdown-open');
    } else {
      document.querySelector("#wise-image-grid > .wise-image-wrapper:nth-of-type(2)")?.classList.remove('wise-image-dropdown-open');
    }
  }

  const handleTourClose = () => {
    setIsTourOpen(false);
    handleTourChange();
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
        <Button type="primary" onClick={() => { setIsTourOpen(true) }}>Show me how to use WISE</Button>
      </div>
    ),
    'About WISE': <div dangerouslySetInnerHTML={{__html: sanitizeHtml(config.WISE_OVERVIEW_CARD.ABOUT)}}></div>,
    'Disclaimer': <div dangerouslySetInnerHTML={{__html: sanitizeHtml(config.WISE_OVERVIEW_CARD.DISCLAIMER)}}></div>
  };
  const aboutWiseTabList = Object.keys(aboutWiseTabContent).map(x => ({key: x, tab: x}));
  const [aboutWiseActiveTabKey, setAboutWiseActiveTabKey] = useState('Overview');

  return (
    <Card
      id="wise-overview-card"
      size='small'
      tabList={aboutWiseTabList}
      activeTabKey={aboutWiseActiveTabKey}
      tabBarExtraContent={<></>} // <a href="#">Close</a>
      onTabChange={setAboutWiseActiveTabKey}
    >
      {aboutWiseTabContent[aboutWiseActiveTabKey]}
      <Tour open={isTourOpen} onClose={handleTourClose} steps={tourSteps} onChange={handleTourChange} />
    </Card>
  );
}

export default WiseOverviewCard;