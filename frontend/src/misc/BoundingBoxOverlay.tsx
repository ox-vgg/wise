// Copyright 2026 University of Oxford
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React from "react";
import { ProcessedVectorInfo,  } from "./types.ts";
import './BoundingBox.scss';
import { BoundingBoxes } from "./BoundingBox.tsx";
import StillImageView from "./StillImageView.tsx";



interface BoundingBoxOverlayProps {
    imageDetails: ProcessedVectorInfo
    displayOverlay?: boolean
    isModalView: boolean
    featureExtractorId: string
    showImage?: boolean
    // If handleInternalSearchButtonClick is missing, the "Find Similar"
    // button is omitted.
    handleInternalSearchButtonClick?: (vector: ProcessedVectorInfo) => void;
    children?: React.ReactNode
}
const BoundingBoxOverlay = ({ imageDetails, isModalView, displayOverlay = true, showImage = true, featureExtractorId, handleInternalSearchButtonClick, children }: BoundingBoxOverlayProps) => {
    const width = imageDetails.mediaInfo.width;
    const height = imageDetails.mediaInfo.height

    return <>
        {displayOverlay && showImage && <StillImageView imageDetails={imageDetails} isModalView={isModalView} />}
        {displayOverlay && <div className={"wise-bounding-boxes " + (isModalView && "wise-bounding-boxes-modal")} style={{ zIndex: !!children ? 1 : 'unset' }}>
            <svg viewBox={`0 0 ${100*width/height} 100`} preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
                <BoundingBoxes
                    featureExtractorId={featureExtractorId}
                    imageDetails={imageDetails}
                    isModalView={isModalView}
                    bboxType="svg"
                />
            </svg>
        </div>}
        {children}
        {displayOverlay && <div className={"wise-bounding-boxes " + (isModalView && "wise-bounding-boxes-modal")} style={{ zIndex: 99 }}>
            <svg viewBox={`0 0 ${100*width/height} 100`} preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
                < BoundingBoxes
                    featureExtractorId={featureExtractorId}
                    imageDetails={imageDetails}
                    isModalView={isModalView}
                    bboxType="svg_popover"
                    handleInternalSearchButtonClick={handleInternalSearchButtonClick}
                />
            </svg>
        </div>
        }
    </>
}
export default BoundingBoxOverlay
