import { useRef } from "react";
import { Button, Dropdown, Modal } from "antd";
import { MoreOutlined } from "@ant-design/icons";
// import sanitizeHtml from "sanitize-html";
import '@vidstack/react/player/styles/default/theme.css';
import '@vidstack/react/player/styles/default/layouts/video.css';
import { MediaPlayer, MediaProvider, Track, type MediaPlayerInstance } from '@vidstack/react';
import { defaultLayoutIcons, DefaultVideoLayout } from '@vidstack/react/player/layouts/default';

import "./ImageDetailsModal.scss";
import { ImageDetailsModalProps, ProcessedVideoSegment } from "./types";
import VideoOccurrencesView from "./VideoOccurrencesView";

const ImageDetailsModal = ({
  imageDetails,
  setImageDetails,
  setSelectedImageId,
}: ImageDetailsModalProps) => {
  let title;
  // let caption, author, copyright;

  console.log(imageDetails)
  
  const playerRef = useRef<MediaPlayerInstance>(null);
  
  if (imageDetails) {
    title = (
      <Button
        type="text"
        // href={imageDetails.videoInfo.externalLink}
        target='_blank'
        size="large"
      >
        <b>{imageDetails.videoInfo.title}</b>
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
    // ({ caption, author, copyright } = imageDetails.info);

    // const width = imageDetails.info.width;
    // const height = imageDetails.info.height;
  }
  // Remove end time from timestamp (e.g. change "#t=16.0,20.0" to "#t=16.0") to prevent video from automatically pausing
  const videoSrc = imageDetails?.link.replace(/(#t=[\d\.]+),[\d\.]+$/, '$1');
  
  const setStartTimestamp = () => {
    // This is needed because the video player doesn't automatically play the video from the start time in the URL (e.g. #t=16.0)
    if (imageDetails && playerRef.current) playerRef.current.currentTime = imageDetails?.ts;
  }

  const handleClickOccurrence = (videoSegment: ProcessedVideoSegment) => {
    if (imageDetails?.vector_id === videoSegment.vector_id) {
      if (imageDetails && playerRef.current) playerRef.current.currentTime = imageDetails?.ts;
    } else {
      setImageDetails(videoSegment);
    }
  }

  return (
    <Modal
      title={title}
      open={!!imageDetails}
      closable={true}
      maskClosable={true}
      destroyOnClose={true}
      footer={
        <>
          <Dropdown
            menu={{
              items: [
                {
                  label: "Report image",
                  key: imageDetails?.videoInfo.filename || '',
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
          <Button type="primary" onClick={() => setImageDetails()}>
            Close
          </Button>
        </>
      }
      zIndex={500} // The default zIndex is 1000. Setting this to 500 allows the ReportImageModal to be shown on top / in front of this modal, rather than behind
      onCancel={() => setImageDetails()}
      width="90vw"
      className="wise-image-details-modal"
    >
      <div className="wise-image-wrapper">
        {imageDetails && ['av', 'video'].includes(imageDetails.videoInfo.media_type) ? (
          <MediaPlayer
            src={videoSrc}
            viewType="video"
            playsInline
            autoPlay
            ref={playerRef}
            onLoadedMetadata={setStartTimestamp}
            clipEndTime={imageDetails.videoInfo.duration} // This is needed due to a bug with the chapter markers https://github.com/vidstack/player/issues/1022
          >
            {/* 
            TODO - chapter markers by default use thumbnails from storyboard - change this to use thumbnails from search results instead
            */}
            <MediaProvider>
              <Track content={{
                // @ts-ignore
                cues: [...imageDetails.videoInfo.shots].sort((a, b) => a.ts - b.ts).map(shot => ({
                    startTime: shot.ts + (shot.ts === 0 ? 0.1 : 0), /* if the first result is at 0 seconds,
                                                                add 0.1s to the timestamp due to CSS rule
                                                                requiring the matching chapter elements to be 'even' rather than odd */
                    endTime: shot.te,
                    text: 'Match found'
                  })),
              }} kind="chapters" lang="en-US" default />;

            </MediaProvider>
            <DefaultVideoLayout
              thumbnails={imageDetails.videoInfo.timeline_hover_thumbnails}
              icons={defaultLayoutIcons} noScrubGesture={false} seekStep={5} />
          </MediaPlayer>
        ) : (
          <img
            src={imageDetails?.link}
            // title={
            //   imageDetails?.videoInfo.filename +
            //   (imageDetails?.distance
            //     ? ` | Distance = ${imageDetails.distance.toFixed(2)}`
            //     : "")
            // }
          />
        )}
      </div>
      {
        imageDetails ?
        <VideoOccurrencesView videoInfo={imageDetails.videoInfo}
          handleClickOccurrence={handleClickOccurrence}
          customHeaderSingular='search match in this video'
          customHeaderPlural='search matches in this video'
        />
        : <></>
      }
      <p>
        <b>Filename</b>
        <br />
        <span>{imageDetails?.videoInfo.filename}</span>
      </p>

      {/* <div className="wise-image-details-metadata">
        <p>
          <b>Description</b>
          <br />
          <span dangerouslySetInnerHTML={{ __html: sanitizeHtml(caption) }} />
        </p>
        <p>
          <b>Author</b>
          <br />
          <span dangerouslySetInnerHTML={{ __html: sanitizeHtml(author) }} />
        </p>
        <p>
          <b>License</b>
          <br />
          <span dangerouslySetInnerHTML={{ __html: sanitizeHtml(copyright) }} />
        </p>
      </div> */}
    </Modal>
  );
};
export default ImageDetailsModal;
