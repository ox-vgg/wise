import { Button, Popover } from "antd";
import { SearchOutlined } from "@ant-design/icons";

import { interleaveArrayWithElement } from "./utils.ts";

import "./StillImageView.scss";
import { ProcessedImageVector, StillImageViewProps } from "./types";


type ProcessedImageVectorWithBBox = ProcessedImageVector & {
  bbox: NonNullable<ProcessedImageVector['bbox']>
};

const isWithBBox = (
  vector_info: ProcessedImageVector
): vector_info is ProcessedImageVectorWithBBox => {
  return Boolean(vector_info.bbox);
}


const BoundingBox: React.FunctionComponent<{
  vector: ProcessedImageVectorWithBBox;
  bbox_text: string;
  handleInternalSearchButtonClick?: (vector: ProcessedImageVector) => void;
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

  return (
    <Popover
      overlayClassName="wise-bounding-box-tooltip"
      content={interleaveArrayWithElement(contents, tooltip_divider)}
    >
      { innerBBox }
    </Popover>
  );
};

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
