import React, { useEffect, useState } from 'react';
import { Dropdown, Pagination, Tooltip } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import { nanoid } from 'nanoid';

import './SearchResults.scss'
import { SearchResultsProps } from './misc/types.ts';
import ReportImageModal from './misc/ReportImageModal.tsx';
import SensitiveImageWarning from './misc/SensitiveImageWarning.tsx';
import ImageDetailsModal from './misc/ImageDetailsModal.tsx';
import config from './config.ts';

const SearchResults: React.FunctionComponent<SearchResultsProps> = (
  {dataService, isHomePage, projectInfo, setSearchText, multimodalQueries, setMultimodalQueries, submitSearch}: SearchResultsProps
) => {
  const { searchResults, isSearching, searchLatency, totalResults, pageNum, changePageNum } = dataService;
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

  const [imageDetails, setImageDetails] = useState<any>({});
  const openImageDetails = (imageId: string) => {
    console.log(imageId)
    const openedImage = searchResults.find(x => x.info.id === imageId);
    setImageDetails(openedImage);
  }
      
  let searchResultsHTML = searchResults
    .map((searchResult) => {
      const title = searchResult.info.title;
      const width = searchResult.info.width;
      const height = searchResult.info.height;

      const dropdownItems = [
        {
          label: 'Report image',
          // TODO change this to link instead of id (also change it in ImageDetailsModal.tsx)
          key: 'report_' + searchResult.info.id
        },
        {
          label: 'Add this image as an additional query',
          key: 'add_image_query_' + searchResult.info.id
        },
        {
          label: 'Add this image as a negative query',
          key: 'add_negative_image_query_' + searchResult.info.id
        }
      ];

      return (
        <div key={searchResult.info.id}
            style={{width: `${width*170/height}px`, flexGrow: width*170/height}}
            className={'wise-image-wrapper ' + ((dropdownImageId === searchResult.info.id) ? 'wise-image-dropdown-open' : '')}
        >
          <Tooltip title="Find visually similar images">
            <img src="internal_search_icon.png" className="wise-internal-image-search-button"
                  onClick={() => handleInternalSearchButtonClick(searchResult.info.id)} />
          </Tooltip>
          <Tooltip title="More options">
            <Dropdown menu={{
              items: dropdownItems,
              onClick: handleDropdownItemClick
            }}
              onOpenChange={(open: boolean) => { handleOpenDropdownChange(open, searchResult.info.id) }}
              placement="bottomRight" trigger={['click']} arrow>
              <img src="more_icon.png"
                    className="wise-image-more-button"
                    onClick={(e) => { e.stopPropagation(); e.preventDefault(); return false;}}
                    />
            </Dropdown>
          </Tooltip>
          <i style={{paddingBottom: `${height/width*100}%`}}></i>
          <a onClick={() => openImageDetails(searchResult.info.id)}>
            <img src={searchResult.thumbnail}
                title={title + (searchResult.distance ? ` | Distance = ${searchResult.distance.toFixed(2)}` : '')}
                className="wise-image"
            ></img>
          </a>
          <div className="wise-image-hover-display">{title}</div>
          <SensitiveImageWarning isSensitive={searchResult.info.is_nsfw || false} />
        </div>
      )
    });

  let showTotal;
  if (isHomePage) {
    showTotal = (total: number, [rangeStart, rangeEnd]: number[]) =>
                `${rangeStart}-${rangeEnd} of ${total.toLocaleString('en', { useGrouping: true })} featured images`;
  } else {
    showTotal = (total: number, [rangeStart, rangeEnd]: number[]) =>
                `${rangeStart}-${rangeEnd} of top ${total.toLocaleString('en', { useGrouping: true })} retrieved images`;
  }

  const numImagesString: string = projectInfo.num_images?.toLocaleString('en', { useGrouping: true }) || '?';
  let loadingMessage = <></>;
  if (isSearching) {
    loadingMessage = <p className="wise-loading-message">Searching in {numImagesString} images <LoadingOutlined /></p>;
  } else if (!isHomePage && !isSearching) {
    loadingMessage = <p className="wise-loading-message">Search completed in {(searchLatency / 1000).toFixed(1)} seconds of {numImagesString} images</p>;
  }

  const isLoadingFeaturedImages = (isHomePage && searchResults.length === 0);
  
  let pagination = (<Pagination
    total={totalResults}
    showTotal={showTotal}
    current={pageNum+1}
    pageSize={config.PAGE_SIZE}
    showSizeChanger={false}
    onChange={(page) => { changePageNum(page-1) }}
  />);
  if (isLoadingFeaturedImages) pagination = <></>;

  return <>
    {loadingMessage}
    <section id="search-results">
      {pagination}
      {(searchResults.length === 0) ? 
        <div className="wise-large-loading-screen"><LoadingOutlined /></div> : <></>
      }
      <div id="wise-image-grid" className="wise-image-grid">
        {searchResultsHTML}
      </div>
      {(searchResults.length === 0) ? <></> : pagination}
    </section>
    <ReportImageModal dataService={dataService} isHomePage={isHomePage}
                      selectedImageId={selectedImageId} setSelectedImageId={setSelectedImageId} />
    <ImageDetailsModal imageDetails={imageDetails} setImageDetails={setImageDetails} setSelectedImageId={setSelectedImageId} />
  </>
};

export default SearchResults;
