import { useState } from "react";
import { Button, Collapse } from "antd";
import { CaretRightOutlined, LeftOutlined, RightOutlined, StarFilled } from "@ant-design/icons";

import './VideoOccurrencesView.scss';
import { ProcessedVideoSegment, VideoOccurrencesViewProps } from "./types";

const secondsToMinSecPadded = (time: number) => {
  const minutes = Math.floor(time / 60);
  const seconds = `${Math.floor(time % 60)}`.padStart(2, "0");
  return `${minutes}:${seconds}`;
};

const VideoOccurrencesView: React.FunctionComponent<VideoOccurrencesViewProps> = ({videoInfo, handleClickOccurrence, customHeaderSingular = 'occurrence', customHeaderPlural = 'occurrences'}) => {
  if (videoInfo.shots.length === 0) {
    return <></>;
  }

  const occurrences = [...videoInfo.shots].sort((a, b) => a.ts - b.ts); // Sort occurrences by timestamp
  const topMatch = occurrences.reduce((maxScoreOccurrence, currentOccurrence) => {
    return (maxScoreOccurrence.distance > currentOccurrence.distance) ? maxScoreOccurrence : currentOccurrence;
  });
  
  const occurrencesHTML = occurrences.map(searchResult => {
    return <div className="wise-occurrence" onClick={() => handleClickOccurrence(searchResult as ProcessedVideoSegment)} key={searchResult.vector_id}>
      <img
        src={searchResult.thumbnail}
        title={searchResult.distance ? `Distance = ${searchResult.distance.toFixed(2)}` : ''}
      />
      <span className="wise-occurrence-timestamp">{secondsToMinSecPadded(searchResult.ts)}</span>
      { (videoInfo.shots.length >= 2 && searchResult.vector_id === topMatch.vector_id) ? <span className="wise-top-match-label"><StarFilled /> Top match</span> : <></> }
    </div>
  });

  const [isOpen, setIsOpen] = useState(true);

  return (
    <div onClick={(e) => e.stopPropagation()}>
      <Collapse
        bordered={false}
        expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? -90 : 90} />}
        expandIconPosition="end"
        defaultActiveKey={0}
        items={[
          {
            key: '0',
            label: <>
              { isOpen ?
                <></> :
                <img src={topMatch.thumbnail} className="wise-occurrences-preview-thumbnail" />
              }
              <span>{videoInfo.shots.length} {videoInfo.shots.length === 1 ? customHeaderSingular : customHeaderPlural}</span>
            </>,
            children: (<>
              <div className="wise-occurrences-wrapper">
                {occurrencesHTML}
                <div className="wise-occurrences-timeline" style={{width: (videoInfo.shots.length - 1) * 181}} />
              </div>
              <Button className="wise-occurrences-left-arrow" shape="circle" icon={<LeftOutlined />} />
              <Button className="wise-occurrences-right-arrow" shape="circle" icon={<RightOutlined />} />
              {/* TODO only display the arrows when needed
                - using a window resize listener, check if the .wise-occurrences-wrapper is overflowing horizontally, and if so, display the right arrow
                - with an onScroll listener attached to the .wise-occurrences-wrapper, show/hide the left/right arrow depending on the user's scroll position
              */}
            </>),
          }
        ]}
        onChange={(e) => setIsOpen(e.length !== 0)}
      />
    </div>
  );
}

export default VideoOccurrencesView;