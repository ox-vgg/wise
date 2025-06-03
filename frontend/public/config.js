// Config for production mode
const productionConfig = {
  MAX_SEARCH_RESULTS: 1000, // Maximum number of search rsults
  PAGE_SIZE: 500, // Number of images in each page when displaying search results
  NUM_PAGES_PER_REQUEST: 1, // Number of pages (each with having size PAGE_SIZE) to fetch in each API call
  FETCH_THUMBS: 1,
  FETCH_TIMEOUT: 60000, // Number of milliseconds to wait when fetching search results / featured images
  REPO_URL: "https://gitlab.com/vgg/wise/wise",
  WISE_OVERVIEW_CARD: {
    ABOUT: `
     <p>
      WISE Search Engine (WISE) is an open source software that enables search of
      large collections of images, audio and video using natural language. The search is
      based solely on audiovisual content.
      </p>
      For example, the search query "hand holding a cup" can be used to
      immediately retrieve images or video clips showing relevant results,
      where the search results are obtained solely by analysis of the visual content.
      Similarly, sounds, faces and particular objects can also be searched for using the respective modes.
    </p>
     <p>
      WISE is developed and maintained by the Visual Geometry Group (<a href="https://www.robots.ox.ac.uk/~vgg/">VGG</a>)
      of Department of Engineering Science at University of Oxford. </p>
      <p>More details about WISE can be found in the <a href="https://gitlab.com/vgg/wise/wise/-/tree/wise2/">code repository</a> and from the <a href="https://www.robots.ox.ac.uk/~vgg/software/wise/">project page</a>.</p>
    `,
    DISCLAIMER: `
      <ul>
        <li>Feel free to write a custom disclaimer here</li>
        <li>Lorem ipsum</li>
      </ul>
    `
  },
  // Example queries shown in Overview card
  EXAMPLE_QUERIES: [
    // 'Cute puppy', 'Bees feeding on flower', 'Hot air balloon above a mountain',
    // 'Penguin with wings raised', 'Dolphin playing with ball', 'People taking pictures of mona lisa',
    // 'Painting of a naval battle', 'Panda chewing on bamboo', 'Plane refuelling another plane',
    // 'Mount Fuji during sunset',  'Car with a bicycle on top', 'Squirrel eating a nut',
    // 'People on a roller coaster', 'Running on a hill', 'A peculiar airplane', 'Busy street in Paris',
    // 'Singer next to a piano', 'Black and white photo of a steam train', 'First lady and her husband',
    // 'Cubist painting of a violin'
  ],
  // Multimodal example queries shown in search dropdown
  MULTIMODAL_EXAMPLE_QUERIES: [
    {
      url: 'https://images.unsplash.com/photo-1559562328-bc48b8b32e2b?fm=jpg&w=640&fit=crop&q=80',
      text: 'in snow'
    },
    {
      url: 'https://images.unsplash.com/photo-1588064011404-57a7bc7133f5?auto=format&fit=crop&w=640&q=80',
      text: 'at night'
    },
  ],
  ENABLE_REPORT_MEDIA: true, // shows a "Report media" button in the image details page
  SHOT_SCALE_FILTER_LABEL: {
    0: "Extreme close-up",
    1: "Close-up",
    2: "Medium shot",
    3: "Full shot",
    4: "Long shot"
  },
  METADATA_TABLE_COLUMNS: [],
  METADATA_FILTER_PLACEHOLDER: "",
  METADATA_FILTER_HELP: "Use AND/OR to combine filters, Ctrl + Space key to show metadata columns.",
  PREFERRED_SEARCH_TARGETS_NAME: {
    "open_clip": "Visual Search",
    "insightface": "Face Search",
    "metadata": "Metadata Search",
    "owlv2": "Object Search",
    "clap": "Audio Search"
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