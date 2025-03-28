import { Popover } from "antd";

import "./StillImageView.scss";
import { StillImageViewProps } from "./types";

const StillImageView: React.FunctionComponent<StillImageViewProps> = ({
  imageDetails,
  isModalView
}: StillImageViewProps) => {

  const img_src = isModalView ? imageDetails.link : imageDetails.thumbnail;

  // If there are bounding boxes, the "title" (which shows the
  // distance to the search) is for the image, otherwise it is for the
  // bounding box.
  const distance_str = imageDetails.distance ?
        `Distance = ${imageDetails.distance.toFixed(2)}`
        : ""
  const img_title = imageDetails.bbox ? "" : distance_str;

  let bounding_boxes;
  if (imageDetails.bbox) {
    bounding_boxes = (
      <div className="wise-bounding-boxes">
        <Popover content={distance_str}>
          <div
            className="wise-bounding-box"
            style={{
              left: `${100*imageDetails.bbox.x}%`,
              top: `${100*imageDetails.bbox.y}%`,
              width: `${100*imageDetails.bbox.w}%`,
              height: `${100*imageDetails.bbox.h}%`,
            }}
          />
        </Popover>
      </div>
    );
  }

  const width = imageDetails.mediaInfo.width;
  const height = imageDetails.mediaInfo.height

  return (
    <div
      className="wise-still-image-wrapper"
      style={{aspectRatio: `${width} / ${height}`}}
    >
      <img src={img_src} title={img_title}/>
      { bounding_boxes }
    </div>
  );
}

export default StillImageView;
