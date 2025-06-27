import React, { useCallback, useEffect, useState } from 'react';
import { Dropdown, Pagination, Row, Segmented, Tooltip } from 'antd';
import { AppstoreOutlined, BarsOutlined, FlagFilled, LoadingOutlined, MinusCircleFilled, PictureOutlined, PlusCircleFilled } from '@ant-design/icons';
import { nanoid } from 'nanoid';

import './SearchResults.scss'
import { ProcessedImageVector, ProcessedVectorInfo, ProcessedVideoSegment, SearchResultsProps } from './misc/types.ts';
import ReportImageModal from './misc/ReportImageModal.tsx';
// import SensitiveImageWarning from './misc/SensitiveImageWarning.tsx';
import ImageDetailsModal from './misc/ImageDetailsModal.tsx';
import StillImageView from "./misc/StillImageView.tsx";
import { excludeKey } from "./misc/utils.ts";
import VideoOccurrencesView from './misc/VideoOccurrencesView.tsx';
import { BoundingBoxes } from './misc/BoundingBox.tsx';
// import config from './config.ts';

const FRONTEND_PAGE_SIZE = 50;

const SearchResults: React.FunctionComponent<SearchResultsProps> = ({
  dataService, isHomePage, projectInfo, setSearchText,
  multimodalQueries, setMultimodalQueries,
  viewModality, featureExtractorId,
  submitSearch
}: SearchResultsProps) => {
  const { searchResults, isLoadingNewSearch, searchLatency /*, totalResults */ } = dataService;
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
      const vector = searchResults.Image.vectors.find(v => v.vector_id === key);
      if (!vector) {
        console.error(`Could not find vector with id ${key}`)
        return;
      }
      // If this was a search result, vector comes with the distance
      // property.  We remove the distance property so it is no longer
      // identified as a result and it doesn't show up on WiseHeader.
      const vectorInfo = excludeKey(vector, "distance");
      setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'INTERNAL_IMAGE', displayText: 'Internal image', value: vectorInfo }]);
      setIsSubmitSearch(true);
    } else if (key.startsWith('add_negative_image_query_')) {
      key = key.replace(/^add_negative_image_query_/, '');
      setDropdownImageId(undefined);
      const vector = searchResults.Image.vectors.find(v => v.vector_id === key);
      if (!vector) {
        console.error(`Could not find vector with id ${key}`)
        return;
      }
      // If this was a search result, vector comes with the distance
      // property.  We remove the distance property so it is no longer
      // identified as a result and it doesn't show up on WiseHeader.
      const vectorInfo = excludeKey(vector, "distance");
      setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'INTERNAL_IMAGE', displayText: 'Internal image', value: vectorInfo, isNegative: true }]);
      setIsSubmitSearch(true);
    }
  }

  const handleInternalSearchButtonClick = useCallback((vector: ProcessedVectorInfo) => {
    setSearchText('');
    // If this was a search result, vector comes with the distance
    // property.  We remove the distance property so it is no longer
    // identified as a result and it doesn't show up on WiseHeader.
    const vectorInfo = excludeKey(vector, "distance");
    setMultimodalQueries([{ id: nanoid(), type: 'INTERNAL_IMAGE', displayText: 'Internal image', value: vectorInfo }]);
    setIsSubmitSearch(true);
  }, []);

  // setImageDetails is called when the user selects an image/video
  // for view on the modal dialog.  Only at that point, do we fetch
  // info about related vectors (other vectors for the same image or
  // timestamp).
  const [imageDetails, _setImageDetails] = useState<ProcessedVideoSegment | ProcessedImageVector | undefined>();
  const setImageDetails = (d: ProcessedVideoSegment | ProcessedImageVector | undefined) => {
    if (featureExtractorId.includes('insightface') && d && !d.related_vectors && !isNaN(parseInt(d.vector_id)))
      dataService.fillRelatedVectors(d).then(x => _setImageDetails(x));
    else
      _setImageDetails(d);
  };

  let searchResultsHTML;
  let totalResultsCount;
  if (viewModality == 'Image' || viewMode == 'UnmergedSegments' || viewMode == 'Segments') {
    // Unmerged Segments / Segments view mode
    let _searchResults: ProcessedImageVector[] | ProcessedVideoSegment[] = [];
    if (viewModality == 'Image') {
      if (!featureExtractorId.includes('insightface')) {
        // show unique images instead of duplicated images, when there are multiple vectors per image
        _searchResults = Array.from(searchResults.Image.mediaInfo).map(([_, imageInfo]) => {
          return imageInfo.vectors[0] as ProcessedImageVector;
        });
      } else {
        // for insightface specifically, we show each vector (box) as an individual search result
        _searchResults = searchResults.Image.vectors;
      }
    } else if (viewModality == 'Video' || viewModality == 'VideoAudio' || viewModality == 'Audio') {
      if (viewMode == 'UnmergedSegments') {
        _searchResults = searchResults[viewModality].unmerged_windows;
        
        if (!featureExtractorId.includes('insightface')) {
          // show unique frames instead of duplicated frames, when there are multiple vectors/boxes per frame
          const uniqueFrames = new Map<string, ProcessedVideoSegment>();
          _searchResults.forEach((segment: ProcessedVideoSegment) => {
            const frameId = `${segment.media_id}_${segment.thumbnail_ts}`;
            if (!uniqueFrames.has(frameId)) {
              uniqueFrames.set(frameId, segment);
            }
          });
          _searchResults = Array.from(uniqueFrames.values());
        }
      } else {
        _searchResults = searchResults[viewModality].merged_windows;
      }
    } else {
      console.error('Unexpected value for viewModality:', viewModality)
    }
    totalResultsCount = _searchResults.length;

    searchResultsHTML = _searchResults
      .slice(pageNum*FRONTEND_PAGE_SIZE,(pageNum+1)*FRONTEND_PAGE_SIZE)
      .map((searchResult: ProcessedImageVector | ProcessedVideoSegment, index) => {
        const { title, width, height } = searchResult.mediaInfo;

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
        const isVector = (searchResult.vector_id != 'None');
        const renderKey = isVector ? searchResult.vector_id : `result-${pageNum*FRONTEND_PAGE_SIZE + index}`
        return (
          <div key={renderKey}
              style={{width: `${width*170/height}px`, flexGrow: width*170/height}}
              className={'wise-image-wrapper ' + ((dropdownImageId === searchResult.vector_id) ? 'wise-image-dropdown-open' : '')}
          >
            {
              /* Enable internal search only for images for now */
              (searchResult.mediaType === 'IMAGE' && isVector) &&
              <>
                <Tooltip title="Find visually similar images">
                  <img src="internal_search_icon.png" className="wise-internal-image-search-button"
                        onClick={() => handleInternalSearchButtonClick(searchResult)} />
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
              </>
            }
            <i style={{paddingBottom: `${height/width*100}%`}}></i>
            <a onClick={() => setImageDetails(searchResult)}>
              <StillImageView
                imageDetails={searchResult}
                isModalView={false}
                boundingBoxes={
                  <BoundingBoxes
                    imageDetails={searchResult}
                    isModalView={false}
                    featureExtractorId={featureExtractorId}
                    handleInternalSearchButtonClick={handleInternalSearchButtonClick}
                  />
                }
              />
            </a>
            <div className="wise-image-hover-display">{title}</div>
            {/* <SensitiveImageWarning isSensitive={searchResult.is_nsfw || false} /> */}
          </div>
        )
      });
  } else {
    // Videos view mode
    let videos = searchResults[viewModality].mediaInfo;
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
            onClick={() => setImageDetails(topMatch)}
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
            {
              !isHomePage &&
              <VideoOccurrencesView featureExtractorId={featureExtractorId} shots={video.shots} handleClickOccurrence={setImageDetails} />
            }
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

  let numMediaFilesString;
  if (viewModality == 'Image') {
    const numImagesString = projectInfo.media_file_counts?.image?.toLocaleString('en', { useGrouping: true }) || '?';
    numMediaFilesString = `${numImagesString} images`;
  } else if (viewModality == 'Video' || viewModality == 'VideoAudio') {
    const numVideosString = projectInfo.media_file_counts?.video?.toLocaleString('en', { useGrouping: true }) || '?';
    const numMinutesString: string = projectInfo.total_duration ? Math.round(projectInfo.total_duration / 60).toLocaleString('en-us') : '?';
    numMediaFilesString = `${numVideosString} videos (total ${numMinutesString} minutes)`;
  }
  let loadingMessage = <></>;
  if (isLoadingNewSearch) {
    loadingMessage = <p className="wise-loading-message">Searching on {numMediaFilesString} <LoadingOutlined /></p>;
  } else if (!isHomePage && !isLoadingNewSearch) {
    loadingMessage = <p className="wise-loading-message">Search completed in {searchLatency.toFixed(2)} seconds on {numMediaFilesString}</p>;
  } else if (isHomePage) {
    // Show a note about the 'featured images' for object search and face search
    let resultTypeName: 'image' | 'frame' | 'segment' | undefined;
    if (viewModality == 'Image') {
      resultTypeName = 'image';
    } else if (viewModality == 'Video') {
      if (viewMode == 'UnmergedSegments') {
        resultTypeName = 'frame';
      } else if (viewMode == 'Segments') {
        resultTypeName = 'segment';
      }
    }
    if (resultTypeName) {
      if (featureExtractorId.includes('owlv2')) {
        loadingMessage = <p className="wise-loading-message">Note: a random selection of objects in each {resultTypeName} is shown below</p>;
      } else if (featureExtractorId.includes('insightface')) {
        if (resultTypeName !== "segment") {
          loadingMessage = <p className="wise-loading-message">Note: a random selection of faces are shown below. Only one face is shown per {resultTypeName}; {resultTypeName}s containing multiple faces may be repeated.</p>;
        } else {
          loadingMessage = <p className="wise-loading-message">Note: a random selection of faces are shown below. Only one face is shown per {resultTypeName}.</p>
        }
      }
    }
  }

  const isLoadingFeaturedImages = (isHomePage && searchResultsHTML.length === 0);
  
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
    <Row justify="center">{loadingMessage}</Row>

    {
      (viewModality != 'Image') &&
      <Row justify="end">
        <Segmented
          options={[
            { label: 'Frames', value: 'UnmergedSegments', icon: <PictureOutlined /> },
            { label: 'Segments', value: 'Segments', icon: <AppstoreOutlined /> },
            { label: 'Videos', value: 'Videos', icon: <BarsOutlined /> },
          ]}
          value={viewMode} onChange={setViewMode}
        />
      </Row>
    }

    <section id="search-results">
      {(searchResultsHTML.length === 0) ? 
        <div className="wise-large-loading-screen"><LoadingOutlined /></div> : <></>
      }
      <div id="wise-image-grid" className="wise-image-grid">
        {searchResultsHTML}
      </div>
      {(searchResultsHTML.length === 0) ? <></> : pagination}
    </section>
    <ReportImageModal dataService={dataService} isHomePage={isHomePage}
                      selectedImageId={selectedImageId} setSelectedImageId={setSelectedImageId} />
    {
      imageDetails &&
      <ImageDetailsModal
        imageDetails={imageDetails}
        setImageDetails={setImageDetails}
        setSelectedImageId={setSelectedImageId}
        isHomePage={isHomePage}
        featureExtractorId={featureExtractorId}
        handleInternalSearchButtonClick={handleInternalSearchButtonClick}
      />
    }
  </>
};

export default SearchResults;
