// Config for production mode
const productionConfig = {
  MAX_SEARCH_RESULTS: 1000, // Maximum number of search rsults
  PAGE_SIZE: 500, // Number of images in each page when displaying search results
  NUM_PAGES_PER_REQUEST: 1, // Number of pages (each with having size PAGE_SIZE) to fetch in each API call
  FETCH_THUMBS: 1,
  FETCH_TIMEOUT: 60000, // Number of milliseconds to wait when fetching search results / featured images
  REPO_URL: "https://gitlab.com/vgg/wise/wise",
  WISE_OVERVIEW_CARD: {
    OVERVIEW: `
      <p>This audiovisual search engine is based on nearly 40 hours of videos released as part of the <a href="https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html">Let’s Make a Cinephile Search Tool!</a> challenge in 2025 and it enables:
      <ul>
        <li>Visual search : find scenes based on visual similarity.</li>
        <li>Face search : retrieve regions featuring specific human faces.</li>
        <li>Object search : retrieve regions featuring specific objects.</li>
        <li>Metadata search : find media based on textual metadata descriptions.</li>
        <li>Audio search : find scenes based on audio content.</li>
      </ul>
      </p>
    `,
    ABOUT: `
     <p>This audiovisual search engine is based on the WISE Search Engine (<a href="https://www.robots.ox.ac.uk/~vgg/software/wise/">WISE</a>) software developed and maintained by the Visual Geometry Group (<a href="https://www.robots.ox.ac.uk/~vgg/">VGG</a>) of Department of Engineering Science at University of Oxford.</p>
     <p>More details about WISE can be found in the <a href="https://www.robots.ox.ac.uk/~vgg/software/wise/">project page</a> and the <a href="https://gitlab.com/vgg/wise/wise/-/tree/wise2/">code repository</a>.</p>
    `,
    DISCLAIMER: `
      <p>The media showcased in this demo are for research purposes only. All rights to the original content are held by their respective copyright owners.</p>
    `
  },
  // Example queries shown in Overview card
  EXAMPLE_QUERIES: [],
  // Multimodal example queries shown in search dropdown
  MULTIMODAL_EXAMPLE_QUERIES: {
    'Video:open_clip': [
      {
        url: 'https://thor.robots.ox.ac.uk/wise/assets/cinephile/example-queries/Zwolle,_Museum_de_Fundatie.jpg'
      },
      {
        text: 'horse in snow'
      },
      {
        text: 'building with large columns'
      },
      {
        text: 'flag',
      },
    ],
    'Video:insightface': [
      {
        url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Albert_Plesman_%281953%29.jpg/960px-Albert_Plesman_%281953%29.jpg',
      },
      {
        url: 'https://upload.wikimedia.org/wikipedia/commons/8/8a/Mussolini_mezzobusto.jpg'
      },
      {
        url: 'https://thor.robots.ox.ac.uk/wise/assets/cinephile/example-queries/KoopmanBreeuwer1929-CROPPED.jpg'
      }
    ],
    'VideoAudio:clap': [
      {
        text: 'shouting'
      },
      {
        text: 'camera noise'
      },
      {
        text: 'horn'
      },
      {
        text: 'trumpet'
      }
      {
        url: 'https://thor.robots.ox.ac.uk/wise/assets/cinephile/example-queries/La_Donna_e_mobile.mp3'
      }
    ],
    'Video:owlv2': [
      {
        text: 'gun',
      },
      {
        text: 'suitcase',
      },
      {
        text: 'bicycle',
      },
      {
        text: 'gate',
      },
      {
        text: 'propeller',
      },
      {
        text: 'medal'
      },
      {
        text: 'flag',
      },
    ]
  },
  ENABLE_REPORT_MEDIA: true, // shows a "Report media" button in the image details page
  SHOT_SCALE_FILTER_LABEL: {
    0: "Extreme close-up",
    1: "Close-up",
    2: "Medium shot",
    3: "Full shot",
    4: "Long shot"
  },
  METADATA_TABLE_COLUMNS: [
    "provider", 
    "title", 
    "type", 
    "year", 
    "country", 
    "language", 
    "data_provider", 
    "dc_contributor", 
    "dc_description", 
    "edm_timespan_label", 
    "edm_preview", 
    "edm_place_latitude", 
    "edm_place_longitude", 
    "edm_place_label", "edm_place_alt_label", "edm_dataset_name", "edm_concept_label"
  ],
  METADATA_FILTER_PLACEHOLDER: "e.g. (a) soldaat OR soldat OR soldier (b) language:nl AND year:195*",
  METADATA_FILTER_HELP: "Use AND/OR to combine filters, Ctrl + Space key to show metadata columns.",
  PREFERRED_SEARCH_TARGETS_NAME: {
    "open_clip": "Visual",
    "insightface": "Faces",
    "metadata": "Metadata",
    "owlv2": "Objects",
    "clap": "Audio"
  }
};

// Config for development mode
const devConfig = {
  ...productionConfig,
  PAGE_SIZE: 500,
  NUM_PAGES_PER_REQUEST: 1,
  FETCH_THUMBS: 1,
};

window.wiseConfig = {
  productionConfig,
  devConfig
};