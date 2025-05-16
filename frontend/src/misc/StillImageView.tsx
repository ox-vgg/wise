import { Button, Popover } from "antd";
import { SearchOutlined } from "@ant-design/icons";

import "./StillImageView.scss";
import { StillImageViewProps } from "./types";

const StillImageView: React.FunctionComponent<StillImageViewProps> = ({
  imageDetails,
  isModalView,
  handleInternalSearchButtonClick,
}: StillImageViewProps) => {

  const img_src = isModalView ? imageDetails.link : imageDetails.thumbnail;

  // If there are bounding boxes, the "title" (which shows the
  // distance to the search) is for the image, otherwise it is for the
  // bounding box.
  const distance_str = `Similarity: ${imageDetails.distance.toFixed(2)}`;
  const img_title = imageDetails.bbox ? "" : distance_str;

  let bounding_boxes;
  if (imageDetails.bbox) {
    bounding_boxes = (
      <div className="wise-bounding-boxes">
        <Popover
          overlayClassName="wise-bounding-box-tooltip"
          content={
            <>
              <span>{distance_str}</span>
              <div className="wise-bounding-box-tooltip-divider" />
              <Button
                type="link"
                size="small"
                icon={<SearchOutlined />}
                onClick={
                  (e) => {
                      e.stopPropagation();
                      handleInternalSearchButtonClick(imageDetails);
                  }
                }
              >
                Find Similar
              </Button>
            </>
          }
        >
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
