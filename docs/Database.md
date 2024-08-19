# Internal Metadata Database

The internal metadata database is an SQLite database storing information about the source collections (i.e. input folders or webdatasets), media files (e.g. images, videos, or audio files), vectors, and extra metadata. The database file is located in `metadata/internal.db` within the project folder.

WISE uses [SQLAlchemy](https://www.sqlalchemy.org/) (Core API) to interact with this database, and [Pydantic](https://docs.pydantic.dev/latest/) to parse / validate the data going into the DB.

The diagram below shows the database schema. The table structures are defined [here](reference/datatables.md) and the validation models are defined [here](reference/datamodels.md)


[![Database diagram](assets/WISE%202%20internal%20metadata.svg)](https://dbdiagram.io/d/WISE-2-internal-metadata-65f3512eb1f3d4062cf6be68)

In addition to the tables defined above, WISE stores the thumbnails as well in a separate database (`thumbs.db`) within the project folder. Thumbnails are explained in more detail [here](Thumbnails.md)

WISE uses the [repository pattern](https://www.cosmicpython.com/book/chapter_02_repository.html) to abstract the DB access. The CRUD methods are defined [here](reference/repository.md) and repository for each metadata table is instantiated in `src/wise/repository/__init__.py`
