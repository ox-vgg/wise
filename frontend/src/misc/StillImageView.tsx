import React, { useEffect, useRef } from "react";

import "./StillImageView.scss";
import "./BoundingBox.scss";

import { ProcessedVectorInfo, isVideoSegment, isResult } from "./types";


export interface StillImageViewProps {
  imageDetails: ProcessedVectorInfo;
  isModalView: boolean;
  // If handleInternalSearchButtonClick is missing, the "Find Similar"
  // button is omitted.
  handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
  boundingBoxes?: React.ReactNode;
};


const StillImageView: React.FunctionComponent<StillImageViewProps> = ({
  imageDetails,
  isModalView,
  boundingBoxes,
}: StillImageViewProps) => {
  const imgref = useRef<HTMLImageElement | null>(null);
  const isVideoModal = isModalView && isVideoSegment(imageDetails);
  const img_src = isVideoModal ? imageDetails.thumbnail : isModalView ? imageDetails.link : imageDetails.thumbnail;

  // If there is no distance, we are showing for a vector that is not
  // a search result (e.g., landing page or thumbnail on WiseHeader).
  const distance_str = isResult(imageDetails)
                       ? `Similarity: ${imageDetails.distance.toFixed(2)}`
                       : "";
  // If there are bounding boxes, the "title" is for the image,
  // otherwise it is for the bounding box.
  const img_title = imageDetails.bbox ? "" : distance_str;

  const width = imageDetails.mediaInfo.width;
  const height = imageDetails.mediaInfo.height;

  /**
   * Try loading the high resolution image if the image is a thumbnail
   * and we are in modal view. On error, fall back to the original
   * image source.
   */
  useEffect(() => {
    const is_thumbnail = img_src.includes('thumbnail');
    if (!imgref.current || !is_thumbnail || !isModalView) {
      return;
    }
    const _url = new URL(imgref.current.src);
    const _params = _url.searchParams
    const high_res = _params.get('high_res')
    if (high_res === 'true') {
      return;
    }
    // try high res
    imgref.current.onerror = () => {
      if (!imgref.current) {
        return;
      }
      imgref.current.src = img_src;
    }
    _params.set('high_res', 'true');
    imgref.current.src = _url.toString()
  }, [imageDetails]);

  return (
    <div
      className="wise-still-image-wrapper"
      style={{
        aspectRatio: `${width} / ${height}`,
        zIndex: isVideoModal && 1 || undefined, // to make it appear above the video in modal
        pointerEvents: isVideoModal ? 'none' : 'auto',
      }}
    >
      <img ref={imgref} src={img_src} title={img_title} />
      {boundingBoxes && <div className="wise-bounding-boxes">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
          {boundingBoxes}
        </svg>
      </div>
      }
    </div>
  );
}

export default StillImageView;
