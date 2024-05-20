import React, { useEffect, useState } from 'react';
import { Dropdown, Pagination, Segmented, Tooltip } from 'antd';
import { AppstoreOutlined, BarsOutlined, FlagFilled, LoadingOutlined, MinusCircleFilled, PictureOutlined, PlusCircleFilled } from '@ant-design/icons';
import { nanoid } from 'nanoid';

import './SearchResults.scss'
import { ProcessedSearchResults, ProcessedVideoSegment, SearchResultsProps } from './misc/types.ts';
import ReportImageModal from './misc/ReportImageModal.tsx';
// import SensitiveImageWarning from './misc/SensitiveImageWarning.tsx';
import ImageDetailsModal from './misc/ImageDetailsModal.tsx';
import VideoOccurrencesView from './misc/VideoOccurrencesView.tsx';
// import config from './config.ts';

const FRONTEND_PAGE_SIZE = 50;

const SearchResults: React.FunctionComponent<SearchResultsProps> = ({
  dataService, isHomePage, projectInfo, setSearchText,
  multimodalQueries, setMultimodalQueries, viewModality, submitSearch
}: SearchResultsProps) => {
  const { searchResults, isSearching, searchLatency /*, totalResults */ } = dataService;
  const [viewMode, setViewMode] = useState<string | number>('Segments');
  const [pageNum, changePageNum] = useState(0);
  useEffect(() => {
    // When searchResults changes (i.e. a new search is performed), reset page number
    changePageNum(0);
  }, [searchResults, viewMode]);
  const [selectedImageId, setSelectedImageId] = useState<string>();

  const [dropdownImageId, setDropdownImageId] = useState<string>();
  const handleOpenDropdownChange = (open: boolean, imageId: string) => {
    if (open) setDropdownImageId(imageId);
    else setDropdownImageId(undefined);
  }

  const [isSubmitSearch, setIsSubmitSearch] = useState(false);
  useEffect(() => {
    if (isSubmitSearch === true) {
      submitSearch();
      setIsSubmitSearch(false);
    }
  }, [isSubmitSearch]);
  const handleDropdownItemClick = ({key}: {key: string}) => {
    if (key.startsWith('report_')) {
      key = key.replace(/^report_/, '');
      setDropdownImageId(undefined);
      setSelectedImageId(key);
    } else if (key.startsWith('add_image_query_')) {
      key = key.replace(/^add_image_query_/, '');
      setDropdownImageId(undefined);
      setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'INTERNAL_IMAGE', displayText: 'Internal image', value: key }]);
      setIsSubmitSearch(true);
    } else if (key.startsWith('add_negative_image_query_')) {
      key = key.replace(/^add_negative_image_query_/, '');
      setDropdownImageId(undefined);
      setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'INTERNAL_IMAGE', displayText: 'Internal image', value: key, isNegative: true }]);
      setIsSubmitSearch(true);
    }
  }

  const handleInternalSearchButtonClick = (imageId: string) => {
    setSearchText('');
    setMultimodalQueries([{ id: nanoid(), type: 'INTERNAL_IMAGE', displayText: 'Internal image', value: imageId }]);
    setIsSubmitSearch(true);
  }

  const [imageDetails, setImageDetails] = useState<ProcessedVideoSegment | undefined>();
  
  let searchResultsHTML;
  let totalResultsCount;
  if (viewMode == 'UnmergedSegments' || viewMode == 'Segments') {
    // Unmerged Segments / Segments view mode
    let segments;
    if (viewMode == 'UnmergedSegments') {
      segments = searchResults[viewModality as keyof ProcessedSearchResults].unmerged_windows;
    } else {
      segments = searchResults[viewModality as keyof ProcessedSearchResults].merged_windows;
    }
    totalResultsCount = segments.length;

    searchResultsHTML = segments
      .slice(pageNum*FRONTEND_PAGE_SIZE,(pageNum+1)*FRONTEND_PAGE_SIZE)
      .map((searchResult: ProcessedVideoSegment) => {
        const { title, width, height } = searchResult.videoInfo;

        const dropdownItems = [
          {
            label: 'Report image',
            key: 'report_' + searchResult.vector_id,
            icon: <FlagFilled style={{color: '#d48806'}} />,
          },
          {
            label: 'Add this image as an additional query',
            key: 'add_image_query_' + searchResult.vector_id,
            icon: <PlusCircleFilled style={{color: '#389e0d'}} />,
          },
          {
            label: 'Add this image as a negative query',
            key: 'add_negative_image_query_' + searchResult.vector_id,
            icon: <MinusCircleFilled style={{color: '#cf1322'}} />,
          }
        ];

        return (
          <div key={searchResult.vector_id}
              style={{width: `${width*170/height}px`, flexGrow: width*170/height}}
              className={'wise-image-wrapper ' + ((dropdownImageId === searchResult.vector_id) ? 'wise-image-dropdown-open' : '')}
          >
            <div style={{display: 'none'}}>
              <Tooltip title="Find visually similar images">
                <img src="internal_search_icon.png" className="wise-internal-image-search-button"
                      onClick={() => handleInternalSearchButtonClick(searchResult.vector_id)} />
              </Tooltip>
              <Tooltip title="More options">
                <Dropdown menu={{
                  items: dropdownItems,
                  onClick: handleDropdownItemClick
                }}
                  onOpenChange={(open: boolean) => { handleOpenDropdownChange(open, searchResult.vector_id) }}
                  placement="bottomRight" trigger={['click']} arrow>
                  <img src="more_icon.png"
                        className="wise-image-more-button"
                        onClick={(e) => { e.stopPropagation(); e.preventDefault(); return false;}}
                        />
                </Dropdown>
              </Tooltip>
            </div>
            <i style={{paddingBottom: `${height/width*100}%`}}></i>
            <a onClick={() => setImageDetails(searchResult)}>
              {/* <img src={searchResult.thumbnail}
                  title={title + (searchResult.distance ? ` | Distance = ${searchResult.distance.toFixed(2)}` : '')}
                  className="wise-image"
              ></img> */}
              <video src={searchResult.link}
                  poster={searchResult.thumbnail}
                  // title={searchResult.distance ? `Distance = ${searchResult.distance.toFixed(2)}` : ''}
                  playsInline
                  muted
                  preload="none"
                  className="wise-video-preview"
                  onMouseEnter={(e) => e.currentTarget.play()}
                  onMouseLeave={(e) => e.currentTarget.load()}
              />
            </a>
            <div className="wise-image-hover-display">{title}</div>
            {/* <SensitiveImageWarning isSensitive={searchResult.is_nsfw || false} /> */}
          </div>
        )
      });
  } else {
    // Videos view mode
    let videos = searchResults[viewModality as keyof ProcessedSearchResults].videos;
    totalResultsCount = videos.size;
    searchResultsHTML = Array.from(videos).slice(pageNum*FRONTEND_PAGE_SIZE,(pageNum+1)*FRONTEND_PAGE_SIZE).map(([videoId, video]) => {
      if (video.shots.length == 0) {
        console.error('Occurrences length is 0 for video ' + videoId);
      }
  
      const topMatch = video.shots.reduce((maxScoreOccurrence, currentOccurrence) => {
        return (maxScoreOccurrence.distance > currentOccurrence.distance) ? maxScoreOccurrence : currentOccurrence;
      });
      const thumbnail = topMatch.thumbnail;
      const previewVideoLink = topMatch.link;
      // const distance = topMatch.distance;
  
      return (
        <div className="wise-video-wrapper" key={videoId}
            onClick={() => setImageDetails(topMatch as ProcessedVideoSegment)}
            onMouseEnter={(e) => e.currentTarget.querySelector('video')?.play()}
            onMouseLeave={(e) => e.currentTarget.querySelector('video')?.load()}
        >
          <div className="wise-video-result-background"></div>
          {/* <img className="wise-video-thumbnail" src={thumbnail} onClick={() => openImageDetails(idForImageDetailsModal)} /> */}
          <video src={previewVideoLink}
              poster={thumbnail}
              // title={distance ? `Distance = ${distance.toFixed(2)}` : ''}
              playsInline
              muted
              preload="none"
              className="wise-video-thumbnail"
          />
          <div className="wise-video-text-wrapper">
            <h1>{video.title}</h1>
            {/* <p>Some metadata here</p> */}
            <VideoOccurrencesView videoInfo={video} handleClickOccurrence={setImageDetails} />
          </div>
        </div>
      )
    });  
  }

  let showTotal;
  if (isHomePage) {
    showTotal = (total: number, [rangeStart, rangeEnd]: number[]) =>
                `${rangeStart}-${rangeEnd} of ${total.toLocaleString('en', { useGrouping: true })} featured segments/videos`;
  } else {
    showTotal = (total: number, [rangeStart, rangeEnd]: number[]) =>
                `${rangeStart}-${rangeEnd} of top ${total.toLocaleString('en', { useGrouping: true })} retrieved results`;
  }

  const numMediaFilesString: string = projectInfo.num_media_files?.toLocaleString('en', { useGrouping: true }) || '?';
  let loadingMessage = <></>;
  if (isSearching) {
    loadingMessage = <p className="wise-loading-message">Searching in {numMediaFilesString} videos <LoadingOutlined /></p>;
  } else if (!isHomePage && !isSearching) {
    loadingMessage = <p className="wise-loading-message">Search completed in {searchLatency.toFixed(2)} seconds of {numMediaFilesString} videos</p>;
  }

  const isLoadingFeaturedImages = (isHomePage && searchResults[viewModality as keyof ProcessedSearchResults].unmerged_windows.length === 0);
  
  let pagination = (<Pagination
    total={totalResultsCount}
    showTotal={showTotal}
    current={pageNum+1}
    // pageSize={config.PAGE_SIZE}
    pageSize={FRONTEND_PAGE_SIZE}
    showSizeChanger={false}
    onChange={(page) => { changePageNum(page-1) }}
  />);
  if (isLoadingFeaturedImages) pagination = <></>;

  return <>
    {loadingMessage}
    
    <Segmented
      options={[
        { label: 'Frames', value: 'UnmergedSegments', icon: <PictureOutlined /> },
        { label: 'Segments', value: 'Segments', icon: <AppstoreOutlined /> },
        { label: 'Videos', value: 'Videos', icon: <BarsOutlined /> },
      ]}
      value={viewMode} onChange={setViewMode}
      style={{marginBottom: '20px', marginTop: '20px', float: 'right'}}
    />

    <section id="search-results">
      {(searchResults[viewModality as keyof ProcessedSearchResults].unmerged_windows.length === 0) ? 
        <div className="wise-large-loading-screen"><LoadingOutlined /></div> : <></>
      }
      <div id="wise-image-grid" className="wise-image-grid">
        {searchResultsHTML}
      </div>
      {(searchResults[viewModality as keyof ProcessedSearchResults].unmerged_windows.length === 0) ? <></> : pagination}
    </section>
    <ReportImageModal dataService={dataService} isHomePage={isHomePage}
                      selectedImageId={selectedImageId} setSelectedImageId={setSelectedImageId} />
    <ImageDetailsModal imageDetails={imageDetails} setImageDetails={setImageDetails} setSelectedImageId={setSelectedImageId} />
  </>
};

export default SearchResults;
