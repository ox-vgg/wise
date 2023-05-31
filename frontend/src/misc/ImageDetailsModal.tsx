import { Button, Modal } from "antd";
import './ImageDetailsModal.scss';

/*
TODO
- figure out:
  - difference between DateTime vs DateTimeOriginal
  - LicenseShortName, License, UsageTerms, LicenseUrl
  - find out what is Restrictions field

- sanitize HTML before dangerously setting?
- add disclaimer
- include report image button
*/

const ImageDetailsModal = ({imageDetails, setImageDetails}: any) => {
  let img_link, img_link_tok, img_filename, /*img_filename_decoded,*/ title;
  let external_link;
  let caption, author, copyright;

  const isImageDetails = imageDetails && Object.keys(imageDetails).length > 0;

  if (isImageDetails) {
    console.log('ASDF', imageDetails)
    img_link = imageDetails.link;
    img_link_tok = img_link.split('/');
    img_filename = img_link_tok[img_link_tok.length - 2];
    // img_filename_decoded = decodeURIComponent(img_filename); // Decode filename to show special characters / utf-8 characters

    external_link = 'https://commons.wikimedia.org/wiki/File:' + img_filename;
    title = <Button type="text" href={external_link} target='_blank' size="large">
      <b>{imageDetails.info.title}</b>
      <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24"><path d="M0 0h24v24H0z" fill="none"/><path d="M19 19H5V5h7V3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg>
    </Button>;
    ({ caption, author, copyright } = imageDetails.info);

    // const width = imageDetails.info.width;
    // const height = imageDetails.info.height;
  }

  return (
    <Modal title={title}
      open={isImageDetails} closable={true} maskClosable={true}
      footer={<Button type="primary" onClick={() => setImageDetails({})}>Close</Button>}
      onCancel={() => setImageDetails({})}
      width='90vw'
      className="wise-image-details-model"
    >
      <a href={external_link} target='_blank'>
        <img src={imageDetails.link || undefined}
          title={imageDetails.info?.title + (imageDetails.distance ? ` | Distance = ${imageDetails.distance.toFixed(2)}` : '')}
        />
      </a>
      <div className="wise-image-details-metadata">
        <p>
          <b>Description</b><br />
          <span dangerouslySetInnerHTML={{__html: caption}} />
        </p>
        <p>
          <b>Author</b><br />
          <span dangerouslySetInnerHTML={{__html: author}} />
        </p>
        {/* <p>
          <b>Source</b><br />
          <span dangerouslySetInnerHTML={{__html: json_metadata['Credit']['value']}} />
        </p> */}
        <p>
          <b>License</b><br />
          <span dangerouslySetInnerHTML={{__html: copyright}} />
        </p>
      </div>
    </Modal>
  )
}
export default ImageDetailsModal;
