# Misc

Miscellaneous notes

## Featured images (how featured images are chosen)
The backend first loads a list of all the image IDs from the metadata SQLite table, and then it selects a random subset of 10,000 ids. 
When the `/featured` endpoint is called, a random subset of 1,000 ids are then selected from the aforementioned 10,000 subset, and these 1,000 ids along with some image metadata are then returned as the HTTP response. The endpoint supports pagination so that only 1 page (e.g. 50 images) of the featured images and metadata needs to be served to the frontend when the home page is opened, to ensure that it loads quickly. A random seed can be passed as a URL parameter to the `/featured` endpoint to change the random subset of 1,000 ids that are selected. Each time someone opens the WISE homepage, a different random seed is used, so that a different set of featured images is shown to the user, while the random seed is kept constant while users are paginating through the featured images to ensure consistency between pages.

## Multi-modal queries (images + text)
For multi-modal queries (images + text), the feature vectors of the individual images/text that make up the query are added together in a weighted sum, with 2x weight assigned to the text query/queries. The weighted sum is then normalised and used as the query vector for the nearest neighbour search.

## Negative examples
For negative examples, the feature vectors of the negative examples are multiplied by -0.2 and added together with the other query vector(s). The resulting vector is then normalised and used as the query vector for the nearest neighbour search.