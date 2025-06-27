import { useCallback, useRef, useState } from "react";
import { Button, Dropdown, Modal, Descriptions, List } from "antd";
import { MoreOutlined } from "@ant-design/icons";
import sanitizeHtml from "sanitize-html";
import '@vidstack/react/player/styles/default/theme.css';
import '@vidstack/react/player/styles/default/layouts/video.css';
import { MediaPlayer, MediaProvider, Track, type MediaPlayerInstance } from '@vidstack/react';
import { defaultLayoutIcons, DefaultVideoLayout } from '@vidstack/react/player/layouts/default';

import "./ImageDetailsModal.scss";
import { ASRSegment, ImageDetailsModalProps, ProcessedVideoSegment, ProcessedVectorInfo, isWithBBox } from "./types";
import StillImageView from "./StillImageView.tsx";
import VideoOccurrencesView from "./VideoOccurrencesView";
import { secondsToMinSecPadded } from "./utils.ts";

import config from '../config.ts';
import { BoundingBoxes } from "./BoundingBox.tsx";

interface ExternalMetadataProps {
  all_metadata: Record<string, string> | { asr_segments?: ASRSegment[] };
}
const ExternalMetadata = ({ all_metadata }: ExternalMetadataProps) => {
  const { asr_segments = undefined, ...rest } = all_metadata;
  if (Object.keys(rest).length === 0) {
    return <p>
      <b>Media Metadata</b>
      <br />
      <i>No metadata available</i>
    </p>
  }
  return (
    <>
      <p>
        <b>Media Metadata</b>
      </p>
      <Descriptions bordered column={1}>
        {Object.entries(rest).map(([key, value]) => (
          <Descriptions.Item key={key} label={key}>
            <span dangerouslySetInnerHTML={{ __html: sanitizeHtml(value || '') }} />
          </Descriptions.Item>
        ))}
      </Descriptions>
    </>
  );
}
const ImageDetailsModal = ({
  imageDetails,
  setImageDetails,
  setSelectedImageId,
  isHomePage,
  featureExtractorId,
  handleInternalSearchButtonClick,
}: ImageDetailsModalProps) => {
  const [isModalOpen, setIsModalOpen] = useState(true);
  console.log(imageDetails)
  const title = (
    <Button
      type="text"
      // href={imageDetails.mediaInfo.externalLink}
      target='_blank'
      size="large"
    >
      <b>{imageDetails.mediaInfo.title}</b>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        height="24"
        viewBox="0 0 24 24"
        width="24"
      >
        <path d="M0 0h24v24H0z" fill="none" />
        <path d="M19 19H5V5h7V3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z" />
      </svg>
    </Button>
  );
  const playerRef = useRef<MediaPlayerInstance>(null);

  let subtitles;
  // const width = imageDetails.info.width;
  // const height = imageDetails.info.height;
  if (imageDetails.mediaType === 'VIDEO') {
    const { asrSegments = [] } = imageDetails.mediaInfo;
    subtitles = {
      cues: asrSegments.map(ASRSegment => ({
        startTime: ASRSegment.start,
        endTime: ASRSegment.end,
        text: ASRSegment.text
      }))
    }
  }
  // Remove end time from timestamp (e.g. change "#t=16.0,20.0" to "#t=16.0") to prevent video from automatically pausing
  const videoSrc = imageDetails.link.replace(/(#t=[\d\.]+),[\d\.]+$/, '$1');

  const setStartTimestamp = () => {
    // This is needed because the video player doesn't automatically play the video from the start time in the URL (e.g. #t=16.0)
    if (imageDetails.mediaType == 'VIDEO' && !isHomePage && playerRef.current) playerRef.current.currentTime = imageDetails.ts;
  }

  const handleClickOccurrence = (videoSegment: ProcessedVideoSegment) => {
    const isVector = videoSegment.vector_id !== 'None';
    if (imageDetails.mediaType == 'VIDEO') {
      if (!isVector) {
        // Corresponds to a transcript result, must handle the same way as transcript item click
        if (playerRef.current) playerRef.current.currentTime = videoSegment.ts;
        setImageDetails(videoSegment);
      }
      else if (imageDetails.vector_id === videoSegment.vector_id) {
        // Handle click on the same vector
        if (playerRef.current) playerRef.current.currentTime = imageDetails.ts;
      } else {
        // Handle click on a different vector
        setImageDetails(videoSegment);
      }
    }
  }

  const doInternalSearchAndCloseDialog = useCallback((vector: ProcessedVectorInfo) => {
    handleInternalSearchButtonClick(vector);
    setIsModalOpen(false);
  }, [handleInternalSearchButtonClick]);

  const handleTranscriptItemClick = (item: ASRSegment) => {
    if (playerRef.current) playerRef.current.currentTime = item.start;
  }

  let image_viewer;
  if (imageDetails.mediaType == 'VIDEO') {
    let media_provider_track = <></>;
    if (!isHomePage) {
      /*
        TODO - chapter markers by default use thumbnails from
               storyboard - change this to use thumbnails from search
               results instead
      */
      media_provider_track = <>
        <Track
          content={{
            // @ts-ignore
            cues: [...imageDetails.mediaInfo.shots].sort((a, b) => a.ts - b.ts).map(shot => ({
              startTime: shot.ts + (shot.ts === 0 ? 0.1 : 0), /* if the first result is at 0 seconds,
                                                                                add 0.1s to the timestamp due to CSS rule
                                                                                requiring the matching chapter elements to be 'even' rather than odd */
              endTime: shot.te,
              text: 'Match found'
            })),
          }}
          kind="chapters"
          lang="en-US"
          default
        />
        <Track content={subtitles} label="English" kind="captions" lang="en-US" type="json" default />
      </>
    }
    image_viewer = (
      <MediaPlayer
        src={videoSrc}
        viewType="video"
        playsInline
        autoPlay
        ref={playerRef}
        onLoadedMetadata={setStartTimestamp}
        clipEndTime={imageDetails.mediaInfo.duration} // This is needed due to a bug with the chapter markers https://github.com/vidstack/player/issues/1022
      >
        <MediaProvider>
          {media_provider_track}
        </MediaProvider>
        <DefaultVideoLayout
          thumbnails={imageDetails.mediaInfo.timeline_hover_thumbnails}
          icons={defaultLayoutIcons}
          noScrubGesture={false}
          seekStep={5}
        />
      </MediaPlayer>
    );
  } else {
    image_viewer = <StillImageView
      imageDetails={imageDetails}
      isModalView={true}
      boundingBoxes={
        <BoundingBoxes
          imageDetails={imageDetails}
          isModalView={true}
          featureExtractorId={featureExtractorId}
          handleInternalSearchButtonClick={doInternalSearchAndCloseDialog}
        />
      }
    />;
  }

  return (
    <Modal
      title={title}
      open={isModalOpen}
      closable={true}
      maskClosable={true}
      destroyOnClose={true}
      footer={
        <>
            {config.ENABLE_REPORT_MEDIA && (
            <Dropdown
              menu={{
              items: [
                {
                label: "Report media", // TODO: send timestamp if a audio/video is being reported
                key: imageDetails.mediaInfo.filename || '',
                },
              ],
              onClick: ({ key }) => {
                setSelectedImageId(key);
              },
              }}
              placement="topLeft"
              trigger={["click"]}
              arrow
            >
              <Button
              shape="circle"
              icon={<MoreOutlined />}
              style={{ float: "left" }}
              />
            </Dropdown>
            )}
          <Button type="primary" onClick={() => setIsModalOpen(false)}>
            Close
          </Button>
        </>
      }
      zIndex={500} // The default zIndex is 1000. Setting this to 500 allows the ReportImageModal to be shown on top / in front of this modal, rather than behind
      onCancel={() => setIsModalOpen(false)}
      afterOpenChange={(is_open) => {
        if (!is_open) setImageDetails();
      }}
      width="90vw"
      className="wise-image-details-modal"
    >
      <div className="wise-image-wrapper">
        {image_viewer}
      </div>
      {
        (!isHomePage && imageDetails.mediaType == 'VIDEO') &&
        <VideoOccurrencesView
          featureExtractorId={featureExtractorId}
          shots={imageDetails.mediaInfo.shots}
          handleClickOccurrence={handleClickOccurrence}
          customHeaderSingular='search match in this video'
          customHeaderPlural='search matches in this video'
        />
      }
      <div>
        <p>
          <b>Filename</b>
          <br />
          <span>{imageDetails?.mediaInfo.filename}</span>
        </p>
      </div>
      {
        (imageDetails?.mediaType == 'VIDEO' && imageDetails.mediaInfo.asrSegments?.length) &&
        <div>
          <p>
            <b>Transcript</b>
          </p>
          <List
            size="small"
            bordered
            dataSource={imageDetails?.mediaType == 'VIDEO' && imageDetails.mediaInfo.asrSegments || []}
            renderItem={(item) => <List.Item onClick={() => handleTranscriptItemClick(item)}>
              <span className="wise-asr-timestamp">{secondsToMinSecPadded(item.start)}</span> <span dangerouslySetInnerHTML={{ __html: sanitizeHtml(item.text || '') }} />
            </List.Item>}
          />
        </div> || <></>
      }



      {imageDetails.mediaInfo.external_metadata && <ExternalMetadata all_metadata={imageDetails.mediaInfo.external_metadata} />}

    </Modal>
  );
};
export default ImageDetailsModal;
