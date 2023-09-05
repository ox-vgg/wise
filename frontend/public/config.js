// Config for production mode
const productionConfig = {
  // Note: API_BASE_URL must contain a trailing slash in order for the frontend code to work
  API_BASE_URL: "./", // "./" is relative to the <base href> defined in index.html
  MAX_SEARCH_RESULTS: 1000, // Maximum number of search rsults
  PAGE_SIZE: 50, // Number of images in each page when displaying search results
  NUM_PAGES_PER_REQUEST: 1, // Number of pages (each with having size PAGE_SIZE) to fetch in each API call
  FETCH_THUMBS: 1,
  FETCH_TIMEOUT: 60000, // Number of milliseconds to wait when fetching search results / featured images
  REPO_URL: "https://gitlab.com/vgg/wise/wise",
  WISE_OVERVIEW_CARD: {
    ABOUT: `
      <p>WISE Image Search Engine (WISE) is an open-source image search engine which leverages recent advances in machine learning and vision-language models that enable search based on image content using natural language. The expressive power of natural language allows the user to flexibly describe what they are looking for.</p>
      <!-- TODO include something like: WISE uses a language model to understand the meaning behind your query, allowing you to flexibly describe what you are looking for. Moreover, WISE uses a vision model to understand what's being depicted in an image (i.e. it searches by image content rather than metadata such as keywords, tags, or descriptions), so the images do not need to be manually tagged or labelled with text captions. -->

      <p><a href="https://gitlab.com/vgg/wise/wise#how-it-works" target='_blank'>How WISE works</a></p>
      <p><a href="https://gitlab.com/vgg/wise/wise" target='_blank'>Code repository</a></p>
      <p><a href="https://www.robots.ox.ac.uk/~vgg/software/wise/" target='_blank'>Project webpage</a></p>
      <p>WISE is developed at the Visual Geometry Group, University of Oxford.</p>
    `,
    DISCLAIMER: `
      <ul>
        <li>The images shown below are hosted on Wikimedia Commons and this website only provides search. The images belong to their respective authors and they are not the property of the University of Oxford.</li>
        <li>We currently do not use cookies on this website</li>
      </ul>
    `
  },
  EXAMPLE_QUERIES: [
    'Cute puppy', 'Bees feeding on flower', 'Hot air balloon above a mountain',
    'Penguin with wings raised', 'Dolphin playing with ball', 'People taking pictures of mona lisa',
    'Painting of a naval battle', 'Panda chewing on bamboo', 'Plane refuelling another plane',
    'Mount Fuji during sunset',  'Car with a bicycle on top', 'Squirrel eating a nut',
    'People on a roller coaster', 'Running on a hill', 'A peculiar airplane', 'Busy street in Paris',
    'Singer next to a piano', 'Black and white photo of a steam train', 'First lady and her husband',
    'Cubist painting of a violin'
  ]
};

// Config for development mode
const devConfig = {
  ...productionConfig,
  API_BASE_URL: "http://localhost:9670/wikimedia/", // Note: API_BASE_URL must contain a trailing slash in order for the frontend code to work
  PAGE_SIZE: 25,
  NUM_PAGES_PER_REQUEST: 2,
  FETCH_THUMBS: 1,
};

window.wiseConfig = {
  productionConfig,
  devConfig
};