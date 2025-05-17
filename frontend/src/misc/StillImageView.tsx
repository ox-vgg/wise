import { Button, Popover } from "antd";
import { SearchOutlined } from "@ant-design/icons";

import { interleaveArrayWithElement } from "./utils.ts";

import "./StillImageView.scss";
import { ProcessedImageVector, ProcessedVectorInfo, StillImageViewProps } from "./types";


type ProcessedVectorInfoWithBBox = ProcessedVectorInfo & {
  bbox: NonNullable<ProcessedVectorInfo['bbox']>
};

const isWithBBox = (
  vector_info: ProcessedVectorInfo
): vector_info is ProcessedVectorInfoWithBBox => {
  return Boolean(vector_info.bbox);
}


const isResult = (
  vector: ProcessedVectorInfo
): vector is ProcessedImageVector => {
  return Boolean("distance" in vector);
}


const BoundingBox: React.FunctionComponent<{
  vector: ProcessedVectorInfoWithBBox;
  bbox_text: string;
  handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
}> = ({
  vector,
  bbox_text,
  handleInternalSearchButtonClick,
}) => {
  const innerBBox = <div
    className="wise-bounding-box"
    style={{
      left: `${100*vector.bbox.x}%`,
      top: `${100*vector.bbox.y}%`,
      width: `${100*vector.bbox.w}%`,
      height: `${100*vector.bbox.h}%`,
    }}
  />;

  const tooltip_divider = <div className="wise-bounding-box-tooltip-divider" />;
  const contents = [];
  if (bbox_text)
    contents.push(<span>{bbox_text}</span>);
  if (handleInternalSearchButtonClick) {
    contents.push(
      <Button
        type="link"
        size="small"
        icon={<SearchOutlined />}
        onClick={
          (e) => {
            e.stopPropagation();
            handleInternalSearchButtonClick(vector);
          }
        }
      >
        Find Similar
      </Button>
    );
  }

  if (contents.length === 0) {
    return innerBBox;
  } else {
    return (
      <Popover
        overlayClassName="wise-bounding-box-tooltip"
        content={interleaveArrayWithElement(contents, tooltip_divider)}
      >
        { innerBBox }
      </Popover>
    );
  }
};

const StillImageView: React.FunctionComponent<StillImageViewProps> = ({
  imageDetails,
  isModalView,
  handleInternalSearchButtonClick,
}: StillImageViewProps) => {

  const img_src = isModalView ? imageDetails.link : imageDetails.thumbnail;

  // If there is no distance, we are showing for a vector that is not
  // a search result (e.g., landing page or thumbnail on WiseHeader).
  const distance_str = isResult(imageDetails)
                       ? `Similarity: ${imageDetails.distance.toFixed(2)}`
                       : "";
  // If there are bounding boxes, the "title" is for the image,
  // otherwise it is for the bounding box.
  const img_title = imageDetails.bbox ? "" : distance_str;

  let bounding_box;
  if (isWithBBox(imageDetails))
    bounding_box = <BoundingBox
      vector={imageDetails}
      bbox_text={distance_str}
      handleInternalSearchButtonClick={handleInternalSearchButtonClick}
    />;

  const width = imageDetails.mediaInfo.width;
  const height = imageDetails.mediaInfo.height

  return (
    <div
      className="wise-still-image-wrapper"
      style={{aspectRatio: `${width} / ${height}`}}
    >
      <img src={img_src} title={img_title}/>
      <div className="wise-bounding-boxes">
        { bounding_box }
      </div>
    </div>
  );
}

export default StillImageView;
