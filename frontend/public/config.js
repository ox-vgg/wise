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
        WISE Search Engine (WISE) is an open-source multi-modal AI-powered image search engine.
        Recent advances in machine learning and vision-language models have enabled search based on image content using natural language.
        With the expressive power of natural language, users can flexibly describe what they are looking for.
        Furthermore, WISE uses a vision model to understand what's being depicted in an image – search results are retrieved based on image content rather than metadata such as keywords, tags, or descriptions, so the images do not need to be manually tagged or labelled with text captions.
      </p>
      <p><a href="https://gitlab.com/vgg/wise/wise#how-it-works" target='_blank'>How WISE works</a></p>
      <p><a href="https://gitlab.com/vgg/wise/wise" target='_blank'>Code repository</a></p>
      <p><a href="https://www.robots.ox.ac.uk/~vgg/software/wise/" target='_blank'>Project webpage</a></p>
      <p>WISE is developed at the Visual Geometry Group, University of Oxford.</p>
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