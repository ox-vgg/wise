import { useMemo } from "react";
import { Button, Popover } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import {
  BBoxXYWH,
  ProcessedVectorInfo,
  ProcessedVectorInfoWithBBox,
  isResult,
  isVideoSegment,
  isWithBBox,
  isWithVectors,
} from "./types.ts";
import { interleaveArrayWithElement, clamp_bbox } from "./utils.ts";

// The following cannot be a React component and must be used as a function
// because of how popover works in antd
// Creates a single bounding box element
type BBoxType = "html" | "svg" | "svg_popover"
const BBox = (bbox: BBoxXYWH, box_type: BBoxType = "html", is_primary = false) => {
    const _clamped_bbox: BBoxXYWH = {
        x: 100 * clamp_bbox(bbox.x),
        y: 100 * clamp_bbox(bbox.y),
        w: 100 * clamp_bbox(bbox.w),
        h: 100 * clamp_bbox(bbox.h),
    }
    const box_class = `wise-bounding-box ${is_primary ? "wise-bounding-box-primary" : ""}`
    if (box_type == 'html') {
        return <div
            className={box_class}
            style={{
                left: `${_clamped_bbox.x}%`,
                top: `${_clamped_bbox.y}%`,
                width: `${_clamped_bbox.w}%`,
                height: `${_clamped_bbox.h}%`,
            }}
        />;
    }

    return <rect
        className={box_type === 'svg' ? box_class : undefined}
        x={`${_clamped_bbox.x}%`}
        y={`${_clamped_bbox.y}%`}
        width={`${_clamped_bbox.w}%`}
        height={`${_clamped_bbox.h}%`} />
}

// Wrapper component that puts a bounding box in a popover
interface BoundingBoxProps {
    vector: ProcessedVectorInfoWithBBox;
    bbox_text?: string;
    is_primary: boolean;
    bbox_type?: BBoxType;
    handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
}
const BoundingBox = ({
    vector,
    bbox_text = "",
    bbox_type = "html",
    is_primary,
    handleInternalSearchButtonClick,
}: BoundingBoxProps) => {
    const innerBBox = BBox(vector.bbox, bbox_type, is_primary);

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
                fresh={true}
                destroyTooltipOnHide={true}
            >
                {innerBBox}
            </Popover>
        );
    }
};

const useBBox = (imageDetails: ProcessedVectorInfo) => {
    const boundingBoxVectors = useMemo(() => {
        // Bounding box vectors filtered (if applicable) and sorted by size
        if (
            isWithVectors(imageDetails.mediaInfo) &&
            imageDetails.mediaInfo.vectors.every(v => isWithBBox(v))
        ) {
            let vectors = [...imageDetails.mediaInfo.vectors];
            if (isVideoSegment(imageDetails) && vectors.every(v => isVideoSegment(v))) {
                // Only show the bounding boxes from the same video frame
                vectors = vectors.filter(v => v.thumbnail_ts === imageDetails.thumbnail_ts);
            }
            return vectors.sort((vecA, vecB) => (
                vecB.bbox.w * vecB.bbox.h - vecA.bbox.w * vecA.bbox.h
            ));
        }
        return [];
    }, [imageDetails]);
    return boundingBoxVectors
}

export interface BoundingBoxesProps {
    imageDetails: ProcessedVectorInfo;
    isModalView: boolean;
    featureExtractorId: string;
    bboxType?: "svg" | "svg_popover";
    // If handleInternalSearchButtonClick is missing, the "Find Similar"
    // button is omitted.
    handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
}
export const BoundingBoxes = ({
    imageDetails,
    isModalView,
    featureExtractorId,
    bboxType = 'svg',
    handleInternalSearchButtonClick
}: BoundingBoxesProps) => {
    const distance_str = isResult(imageDetails)
        ? `Similarity: ${imageDetails.distance.toFixed(2)}`
        : "";
    const boundingBoxVectors = useBBox(imageDetails);

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
        if (featureExtractorId.includes("insightface")) {
            bounding_boxes.push(
                <BoundingBox
                    vector={imageDetails}
                    is_primary={true}
                    bbox_text={distance_str}
                    bbox_type={bboxType}
                    handleInternalSearchButtonClick={handleInternalSearchButtonClick}
                    key={`${imageDetails.media_id}-${imageDetails.vector_id}`}
                />
            );
        } else if (boundingBoxVectors.length > 0) {
            bounding_boxes.push(
                ...boundingBoxVectors.map(vec => (
                    <BoundingBox
                        vector={vec}
                        is_primary={true}
                        bbox_text={
                            isResult(vec)
                                ? `Similarity: ${vec.distance.toFixed(2)}`
                                : ""
                        }
                        bbox_type={bboxType}
                        handleInternalSearchButtonClick={handleInternalSearchButtonClick}
                        key={`${vec.media_id}-${vec.vector_id}`}
                    />
                ))
            );
        }
    }

    if (isModalView && imageDetails.related_vectors) {
        for (const vector of imageDetails.related_vectors) {
            // We don't need to skip the "primary bbox" because it is not
            // supposed to be on the related vectors.
            if (isWithBBox(vector)) {
                bounding_boxes.push(
                    <BoundingBox
                        vector={vector}
                        is_primary={false}
                        bbox_text={""}
                        bbox_type={bboxType}
                        handleInternalSearchButtonClick={handleInternalSearchButtonClick}
                        key={`${vector.media_id}-${vector.vector_id}`}
                    />
                );
            }
        }
    }
    return <>
        {bounding_boxes}
    </>

}
