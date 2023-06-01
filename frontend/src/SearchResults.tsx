import React, { useState } from 'react';
import { Dropdown, Pagination } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

import './SearchResults.scss'
import { SearchResultsProps } from './misc/types.ts';
import ReportImageModal from './misc/ReportImageModal.tsx';
import SensitiveImageWarning from './misc/SensitiveImageWarning.tsx';
import ImageDetailsModal from './misc/ImageDetailsModal.tsx';
import config from './config.ts';

const SearchResults: React.FunctionComponent<SearchResultsProps> = ({dataService, isHomePage, projectInfo}: SearchResultsProps) => {
  const { searchResults, isSearching, searchLatency, totalResults, pageNum, changePageNum } = dataService;
  const [selectedImageId, setSelectedImageId] = useState<string>();

  const [dropdownImageId, setDropdownImageId] = useState<string>();
  const handleOpenDropdownChange = (open: boolean, imageId: string) => {
    if (open) setDropdownImageId(imageId);
    else setDropdownImageId(undefined);
  }

  const [imageDetails, setImageDetails] = useState<any>({});
  const openImageDetails = (imageId: string) => {
    console.log(imageId)
    const openedImage = searchResults.find(x => x.id === imageId);
    setImageDetails(openedImage);
  }
      
  let searchResultsHTML = searchResults
    .map((searchResult) => {
      const title = searchResult.info.title;
      const width = searchResult.info.width;
      const height = searchResult.info.height;

      return (
        <div key={searchResult.id}
            style={{width: `${width*170/height}px`, flexGrow: width*170/height}}
            className={'wise-image-wrapper ' + ((dropdownImageId === searchResult.id) ? 'wise-image-dropdown-open' : '')}
        >
          <Dropdown menu={{
            items: [{
              label: 'Report image',
              key: searchResult.id
            }],
            onClick: ({key}) => { setSelectedImageId(key) }
          }}
            onOpenChange={(open: boolean) => { handleOpenDropdownChange(open, searchResult.id) }}
            placement="bottomRight" trigger={['click']} arrow>
            <img src="more_icon.png"
                  className="wise-image-more-button"
                  onClick={(e) => { e.stopPropagation(); e.preventDefault(); return false;}}
                  />
          </Dropdown>
          <i style={{paddingBottom: `${height/width*100}%`}}></i>
          <a onClick={() => openImageDetails(searchResult.id)}>
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
    loadingMessage = <p className="wise-loading-message">Searching in {numImagesString} Wikimedia images <LoadingOutlined /></p>;
  } else if (!isHomePage && !isSearching) {
    loadingMessage = <p className="wise-loading-message">Search completed in {(searchLatency / 1000).toFixed(1)} seconds of {numImagesString} Wikimedia images</p>;
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
                      selectedImageId={selectedImageId} setSelectedImageId={setSelectedImageId}
                      setDropdownImageId={setDropdownImageId}/>
    <ImageDetailsModal imageDetails={imageDetails} setImageDetails={setImageDetails} setSelectedImageId={setSelectedImageId} />
  </>
};

export default SearchResults;
