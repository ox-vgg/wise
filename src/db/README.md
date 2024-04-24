# Internal Metadata Database

The internal metadata database is an SQLite database storing information about the source collections (i.e. input folders or webdatasets), media files (e.g. images, videos, or audio files), vectors, and extra metadata. The database file is located in `{project_name}.db` within the project folder.

The diagram below shows the database schema. For more details, please refer to the [tables/\_\_init__.py](tables/__init__.py) file.

[![Database diagram](../../docs/assets/WISE%202%20internal%20metadata.svg)](https://dbdiagram.io/d/WISE-2-internal-metadata-65f3512eb1f3d4062cf6be68)