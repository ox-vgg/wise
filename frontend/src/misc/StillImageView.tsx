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
  is_primary: boolean;
  handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
}> = ({
  vector,
  bbox_text,
  is_primary,
  handleInternalSearchButtonClick,
}) => {
  const innerBBox = <div
    className={`wise-bounding-box ${is_primary ? "wise-bounding-box-primary" : ""}`}
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

  // An image may have any number of bounding boxes.  If there is a
  // bbox in imageDetails then that bbox is the result of a search and
  // is the "primary" bbox.  The primary bbox is shown in the search
  // results page and is coloured in the modal dialog.  In addition to
  // imageDetails.bbox, there may be other bboxes associated to the
  // image.  Those are only shown in the modal dialog to not overcrowd
  // the search results page.  They are also only fetched when the
  // user selects the image for viewing on the modal dialog.
  const bounding_boxes = [];
  if (isWithBBox(imageDetails)) {
    bounding_boxes.push(
      <BoundingBox
        vector={imageDetails}
        bbox_text={distance_str}
        is_primary={true}
        handleInternalSearchButtonClick={handleInternalSearchButtonClick}
      />
    );
  }

  if (isModalView && imageDetails.related_vectors) {
    for (const vector of imageDetails.related_vectors) {
      // We don't need to skip the "primary bbox" because it is not
      // supposed to be on the related vectors.
      if (isWithBBox(vector)) {
        bounding_boxes.push(
          <BoundingBox
            vector={vector}
            bbox_text={""}
            is_primary={false}
            handleInternalSearchButtonClick={handleInternalSearchButtonClick}
          />
        );
      }
    }
  }


  const width = imageDetails.mediaInfo.width;
  const height = imageDetails.mediaInfo.height

  return (
    <div
      className="wise-still-image-wrapper"
      style={{aspectRatio: `${width} / ${height}`}}
    >
      <img src={img_src} title={img_title}/>
      <div className="wise-bounding-boxes">
        { bounding_boxes }
      </div>
    </div>
  );
}

export default StillImageView;
