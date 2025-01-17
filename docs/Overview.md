# Overview of WISE Image Search Engine (WISE) Code

 * data sources are created wise.py::parse_and_validate_input_datasets()
 * all the data structures exchanged between different modules (e.g. ImageInfo which is used to store individual search results) are defined src/data_models.py
   - Inheriting form BaseModel (of pydantic) allows us to define the data type and the pydantic performs data validations for us
 * src/ioutils.py contains code for reading input data sources, writing features h5 file, generating thumbnails, etc
 *src/inference.py contains everything related to the OpenCLIP model
   - the open_clip_torch library manages the download of models
   - extract_image_features() takes in a list of PIL images, or a batch of preprocessed images in the form of a tensor, and returns features/embeddings as a numpy array
   - extract_text_features() takes in a list of text queries and returns features/embeddings as a numpy array
 * src/projects.py : all actions related to a WISE project
   - create_wise_project_tree() creates the WISE folder structure
 * wise.py::_update()
   - An h5 file is produced for each input data source; each folder is a h5 file, each tar is a h5 file (e.g. folder of images or tar file)
   - If the image is malformed, it is written to a failedsamples webdataset containing all failed images
   - If an error occurred when extracting features or generating thumbnails, the h5 file is deleted and the user will need to re-run the init command again
