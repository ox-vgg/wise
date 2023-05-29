import { Button, Modal } from "antd";
import './ImageDetailsModal.scss';

// Placeholder data (TODO use actual metadata later)
const jsonMetadata = {
  "ns": 6,
  "title": "File:31063-Benque-argile.jpg",
  "pageid": 122713270,
  "imageinfo": [
      {
          "url": "https://upload.wikimedia.org/wikipedia/commons/8/8e/31063-Benque-argile.jpg",
          "extmetadata": {
              "Artist": {
                  "value": "<a href=\"//commons.wikimedia.org/wiki/User:Roland45\" title=\"User:Roland45\">Roland45</a>",
                  "source": "commons-desc-page"
              },
              "Credit": {
                  "value": "<span class=\"int-own-work\" lang=\"en\">Own work</span> avec les données : <br>* Découpage administratif communal et départemental : <a rel=\"nofollow\" class=\"external text\" href=\"https://geoservices.ign.fr/adminexpress#telechargement\">Admin Express</a> sur le site de l'IGN. <br>* Fond de plan (Infrastructures, eau, sols, relief) : <a rel=\"nofollow\" class=\"external text\" href=\"https://geoservices.ign.fr/planign\">Plan IGN</a> (<a rel=\"nofollow\" class=\"external text\" href=\"https://geoservices.ign.fr/presentation\">licence ouverte Etalab 2.0 depuis le 1er janvier 2021</a>)<br>* aléa argile : <a rel=\"nofollow\" class=\"external text\" href=\"https://www.georisques.gouv.fr/donnees/bases-de-donnees/retrait-gonflement-des-argiles\">périmètres de l'aléa</a> publié par l'« Observatoire des risques naturels »<br><br>Assemblé et enrichi dans <a href=\"https://fr.wikipedia.org/wiki/QGis\" class=\"extiw\" title=\"fr:QGis\">fr:QGis</a>.",
                  "source": "commons-desc-page"
              },
              "License": {
                  "value": "cc-by-sa-4.0",
                  "hidden": "",
                  "source": "commons-templates"
              },
              "DateTime": {
                  "value": "2022-09-06 07:38:53",
                  "hidden": "",
                  "source": "mediawiki-metadata"
              },
              "Categories": {
                  "value": "Maps of Benque|Self-published work|Uploaded with pattypan",
                  "hidden": "",
                  "source": "commons-categories"
              },
              "LicenseUrl": {
                  "value": "https://creativecommons.org/licenses/by-sa/4.0",
                  "source": "commons-desc-page"
              },
              "ObjectName": {
                  "value": "31063-Benque-argile",
                  "source": "mediawiki-metadata"
              },
              "UsageTerms": {
                  "value": "Creative Commons Attribution-Share Alike 4.0",
                  "source": "commons-desc-page"
              },
              "Assessments": {
                  "value": "",
                  "hidden": "",
                  "source": "commons-categories"
              },
              "Copyrighted": {
                  "value": "True",
                  "hidden": "",
                  "source": "commons-desc-page"
              },
              "Restrictions": {
                  "value": "",
                  "hidden": "",
                  "source": "commons-desc-page"
              },
              "DateTimeOriginal": {
                  "value": "2022-09-06",
                  "source": "commons-desc-page"
              },
              "ImageDescription": {
                  "value": "Carte de l'aléa retrait-gonflement des argiles sur la commune de <a href=\"https://fr.wikipedia.org/wiki/Benque\" class=\"extiw\" title=\"fr:Benque\">fr:Benque</a> (France).",
                  "source": "commons-desc-page"
              },
              "LicenseShortName": {
                  "value": "CC BY-SA 4.0",
                  "hidden": "",
                  "source": "commons-desc-page"
              },
              "AttributionRequired": {
                  "value": "true",
                  "hidden": "",
                  "source": "commons-desc-page"
              },
              "CommonsMetadataExtension": {
                  "value": 1.2,
                  "hidden": "",
                  "source": "extension"
              }
          },
          "descriptionurl": "https://commons.wikimedia.org/wiki/File:31063-Benque-argile.jpg",
          "descriptionshorturl": "https://commons.wikimedia.org/w/index.php?curid=122713270"
      }
  ],
  "imagerepository": "local"
};

/*
TODO
- figure out:
  - difference between DateTime vs DateTimeOriginal
  - LicenseShortName, License, UsageTerms, LicenseUrl
  - find out what is Restrictions field

- sanitize HTML before dangerously setting?
- TODO add button to open image link in new tab
- TODO add disclaimer
- include report image button
*/

const ImageDetailsModal = ({imageDetails, setImageDetails}: any) => {
  let img_link, img_link_tok, img_filename, img_filename_decoded, title;

  const isImageDetails = imageDetails && Object.keys(imageDetails).length > 0;

  if (isImageDetails) {
    console.log('ASDF', imageDetails)
    img_link = imageDetails.link;
    img_link_tok = img_link.split('/');
    img_filename = img_link_tok[img_link_tok.length - 2];
    img_filename_decoded = decodeURIComponent(img_filename); // Decode filename to show special characters / utf-8 characters

    // const width = imageDetails.info.width;
    // const height = imageDetails.info.height;
  
    title = img_filename_decoded.replaceAll('_', ' '); // Temporary hack for now. TODO display actual title from metadata
  }

  return (
    <Modal title={title}
      open={isImageDetails} closable={true} maskClosable={true}
      footer={<Button type="primary" onClick={() => setImageDetails({})}>Close</Button>}
      onCancel={() => setImageDetails({})}
      width='80vw'
      className="wise-image-details-model"
    >
      <a href={'https://commons.wikimedia.org/wiki/File:' + img_filename} target='_blank'>
        <img src={imageDetails.link || undefined}
          title={title + (imageDetails.distance ? ` | Distance = ${imageDetails.distance.toFixed(2)}` : '')}
        />
      </a>
      <div className="wise-image-details-metadata">
        <p>
          <b>Description</b><br />
          <span dangerouslySetInnerHTML={{__html: jsonMetadata['imageinfo'][0]['extmetadata']['ImageDescription']['value']}} />
        </p>
        <p>
          <b>Author</b><br />
          <span dangerouslySetInnerHTML={{__html: jsonMetadata['imageinfo'][0]['extmetadata']['Artist']['value']}} />
        </p>
        <p>
          <b>Source</b><br />
          <span dangerouslySetInnerHTML={{__html: jsonMetadata['imageinfo'][0]['extmetadata']['Credit']['value']}} />
        </p>
        <p>
          <b>License</b><br />
          <span dangerouslySetInnerHTML={{__html: jsonMetadata['imageinfo'][0]['extmetadata']['License']['value']}} />
        </p>
      </div>
    </Modal>
  )
}
export default ImageDetailsModal;
