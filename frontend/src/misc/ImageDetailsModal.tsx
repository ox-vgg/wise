import { useEffect } from "react";
import { Button, Dropdown, Modal } from "antd";
import { MoreOutlined } from "@ant-design/icons";
import sanitizeHtml from 'sanitize-html';
import { ImageDetailsModalProps, ProcessedSearchResult } from "./types";
import { useInternalSearchDataService } from "../DataService";
import './ImageDetailsModal.scss';

const ImageDetailsModal = ({imageDetails, setImageDetails, setSelectedImageId}: ImageDetailsModalProps) => {
  let title;
  let caption, author, copyright;

  const isImageDetails = imageDetails && Object.keys(imageDetails).length > 0;

  const internalSearchDataService = useInternalSearchDataService();
  let relatedImages: ProcessedSearchResult[] = [];

  if (isImageDetails) {
    title = <Button type="text" href="" onClick={(e) => {e.preventDefault()}} size="large">
      <b>{imageDetails.info.title}</b>
      <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24"><path d="M0 0h24v24H0z" fill="none"/><path d="M19 19H5V5h7V3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg>
    </Button>;
    ({ caption, author, copyright } = imageDetails.info);

    // const width = imageDetails.info.width;
    // const height = imageDetails.info.height;

    if (internalSearchDataService.internalImageId === imageDetails.info.id) {
      relatedImages = internalSearchDataService.searchResults;
    }
  }

  useEffect(() => {
    if (isImageDetails) {
      internalSearchDataService.performInternalSearch(imageDetails.info.id);
    }
  }, [imageDetails])

  return (
    <Modal title={title}
      open={isImageDetails} closable={true} maskClosable={true}
      footer={
        <>
          <Dropdown menu={{
            items: [{
              label: 'Report image',
              key: imageDetails.info?.id
            }],
            onClick: ({key}) => { setSelectedImageId(key) }
          }}
            placement="topLeft" trigger={['click']} arrow>
            <Button shape="circle" icon={<MoreOutlined />} style={{float: 'left'}} />
          </Dropdown>
          <Button type="primary" onClick={() => setImageDetails({})}>Close</Button>
        </>
      }
      onCancel={() => setImageDetails({})}
      width='90vw'
      zIndex={500} // The default zIndex is 1000. Setting this to 500 allows the ReportImageModal to be shown on top / in front of this modal, rather than behind
      className="wise-image-details-modal"
    >
      <div className="wise-image-details-modal-main-section">
        <a>
          <img src={imageDetails.link || undefined}
            key={imageDetails.link || undefined}
            title={imageDetails.info?.title + (imageDetails.distance ? ` | Distance = ${imageDetails.distance.toFixed(2)}` : '')}
          />
        </a>
        <div className="wise-image-details-metadata">
          <p>
            <b>Description</b><br />
            <span dangerouslySetInnerHTML={{__html: sanitizeHtml(caption || '')}} />
          </p>
          <p>
            <b>Author</b><br />
            <span dangerouslySetInnerHTML={{__html: sanitizeHtml(author || '')}} />
          </p>
          <p>
            <b>License</b><br />
            <span dangerouslySetInnerHTML={{__html: sanitizeHtml(copyright || '')}} />
          </p>
        </div>
      </div>
      <div style={{borderTop: '1px solid #e3e3e3', marginTop: 20, marginBottom: 20}} />
      <div className="wise-image-details-modal-related-images">
        {
          relatedImages.map(x => <img src={x.thumbnail} onClick={() => setImageDetails(x)} />)
        }
      </div>
    </Modal>
  )
}
export default ImageDetailsModal;
