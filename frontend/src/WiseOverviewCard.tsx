import { useEffect, useState } from 'react';
import { Button, Card, Tour, TourProps } from 'antd';
import sanitizeHtml from 'sanitize-html';

import './WiseOverviewCard.scss';
import config from './config';
import { ViewModality, WiseOverviewCardProps } from './misc/types';

// TODO: Is the type right?
const process_example_queries = (exampleQueries: string[] | Record<string, string[]>) => {
  if (Array.isArray(exampleQueries)) {
    return process_example_queries({ ':': exampleQueries });
  }
  const _shuffle = (arr: string[]) => {
    return arr.map(value => ({ value, sort: Math.random() }))
      .sort((a, b) => a.sort - b.sort)
      .map(({ value }) => value); // Shuffle array
  }
  return Object.fromEntries(
    Object.entries(exampleQueries).map(([key, value]) => {
      return [key, _shuffle(value).slice(0, 5)];
    })
  ) as Record<string, string[]>;
}

const exampleQueries = process_example_queries(config.EXAMPLE_QUERIES);
const WiseOverviewCard: React.FunctionComponent<WiseOverviewCardProps> = ({handleExampleQueryClick, projectInfo, tourVariables}) => {
  const [isTourOpen, setIsTourOpen] = useState<boolean>(false);
  
  const tourSteps: TourProps['steps'] = [
    {
      title: 'Enter your search query here',
      description: <>
        You can enter a detailed description such as 
        <Button size="small" shape="round" type='primary' ghost onClick={() => handleExampleQueryClick('hand holding a cup')}>Hand holding a cup</Button>
        <br />
        WISE uses a language model to understand the meaning behind your query, allowing you to flexibly describe what you are looking for. WISE then tries to find images whose visual contents match what you are trying to look for.
      </>,
      // cover: <img />
      target: () => tourVariables.searchBar.current.input.parentElement,
    },
    {
      title: 'Visual similarity search',
      description: 'Click on this button and then upload an image or paste an image link to find similar images',
      target: () => tourVariables.imageUploadButton.current,
    },
    {
      title: 'Compound multi-modal search',
      description: 'Search with a combination of images and text. For example, if you upload a picture of a golden retriever and then enter the text "in snow", WISE will find images of golden retrievers in snow.',
      target: () => tourVariables.multimodalSearchArea.current,
    },
    {
      title: 'Select model and media type',
      description: <>
        WISE supports searching across multiple media types (images, videos, etc.) using different models.
        <br />
        Using this dropdown menu, you can select the media type and model you want to use for your search.
      </>,
      target: () => document.querySelector(".wise-view-modality-select"),
    },
    {
      title: 'Pagination',
      description: 'Navigate to another page in the search results',
      target: () => document.querySelector("#search-results > .ant-pagination"),
    },
    ...config.ENABLE_REPORT_MEDIA ? [{
      title: 'Report image',
      description: <>
        If you want to report an image that is inappropriate, offensive, irrelevant to the search query, etc, you can hover over the image and click on the <img src="more_icon.png" height="14px" /> button on the top right corner.
      </>,
      target: () => document.querySelector("#wise-image-grid > .wise-image-wrapper:nth-of-type(2)"),
    }] : [],
  ];

  useEffect(() => {
    // Open the search dropdown for the tour (needed for the "Visual similarity search" and "Compound multi-modal search" to display correctly)
    if (isTourOpen) {
      tourVariables.setIsSearchDropdownOpenForTour(true);
    } else {
      tourVariables.setIsSearchDropdownOpenForTour(false);
    }
  }, [isTourOpen])

  const handleTourChange = (current?: number) => {
    // Make the 'three dots' icon (for reporting images) visible
    if (current === 5) {
      document.querySelector("#wise-image-grid > .wise-image-wrapper:nth-of-type(2)")?.classList.add('wise-image-dropdown-open');
    } else {
      document.querySelector("#wise-image-grid > .wise-image-wrapper:nth-of-type(2)")?.classList.remove('wise-image-dropdown-open');
    }
  }

  const handleTourClose = () => {
    setIsTourOpen(false);
    handleTourChange();
  }
  let exampleQueriesHTML = <></>
  if (projectInfo.search_targets && Object.keys(projectInfo.search_targets).length > 0) {
    const media_types = Object.keys(projectInfo.search_targets) as Array<keyof typeof projectInfo.search_targets>;

    exampleQueriesHTML = <>{
      Object.entries(exampleQueries).map(([key, value]) => {
        // No examples provided
        if (value.length == 0) {
          return <></>
        }
        // key is of type ViewModality:FeatureExtractor
        const [_viewModality, _featureExtractorId] = key.split(':', 2);
        let viewModality = (_viewModality === '') ? undefined : (_viewModality as ViewModality);
        let featureExtractorId = _featureExtractorId;
        if (viewModality) {
          let _media_type = viewModality.toLowerCase();
          _media_type = (_media_type === 'videoaudio') ? 'audio' : _media_type
          let media_type = (_media_type as keyof Required<NonNullable<typeof projectInfo.search_targets>>);
          if (!(media_types.includes(media_type))) {
            // View modality doesn't exist on project, but example was given in config
            return <></>
          }
          const featureIDS = projectInfo.search_targets?.[media_type] || [];
          const validFeatureId = featureIDS.find((fid) => fid.includes(featureExtractorId));
          if (!validFeatureId) {
            return <></>
          }
          featureExtractorId = validFeatureId;

        } else {
          featureExtractorId = ''
        }

        let keyName = config.PREFERRED_SEARCH_TARGETS_NAME[featureExtractorId] || (_viewModality === 'VideoAudio' ? 'Audio' : _viewModality)
        return (
          <p key={key} className="wise-example-queries">
            Sample {keyName} queries: &nbsp;
            {value.map((x, i) => {
              return <Button size="small" shape="round" type='primary' ghost onClick={() => handleExampleQueryClick(x, viewModality, featureExtractorId)} key={`${key}-${i}`}>{x}</Button>
            }
            )}
            <br />
          </p>

        );
      })
    }</>
  }

  const aboutWiseTabContent: Record<string, React.ReactNode> = {
    'Overview': (
      <div className="wise-overview">
        {config.WISE_OVERVIEW_CARD.OVERVIEW ? <div dangerouslySetInnerHTML={{ __html: sanitizeHtml(config.WISE_OVERVIEW_CARD.OVERVIEW) }}></div> : <></>}
        {projectInfo.num_vectors ? 
          <p>Here, you can search within a set of {projectInfo.num_media_files?.toLocaleString('en-us')} media files ({projectInfo.num_vectors?.toLocaleString('en-us')} vectors).</p>
          :
          <></>
        }
        {exampleQueriesHTML}
        <Button type="primary" onClick={() => { setIsTourOpen(true) }}>Show me how to use WISE</Button>
      </div>
    ),
    'About WISE': <div dangerouslySetInnerHTML={{ __html: sanitizeHtml(config.WISE_OVERVIEW_CARD.ABOUT) }}></div>,
    'Disclaimer': <div dangerouslySetInnerHTML={{
      __html: sanitizeHtml(config.WISE_OVERVIEW_CARD.DISCLAIMER, {
        allowedTags: sanitizeHtml.defaults.allowedTags.concat(['img']),
        allowedAttributes: { 'img': ['src', 'width', 'height'], 'a': ['href', 'target', 'rel'] },
        allowedSchemes: ['data', 'http', 'https']
      })
    }}></div>
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