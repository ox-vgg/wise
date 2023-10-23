# Migrating WISE Projects
A WISE project stores absolute path of many assets (e.g. image location) in a SQLite database. If the WISE project is moved to a different computer or if the location of assets change, we need to manually update (for now) these values. Here are a list of updates required to migrate a WISE project.

* Update `location` entry in SQLite database

Here is an example.

```
sqlite3 wise-store/project-name-here/
sqlite> select * from datasets;
1|/data/disk1/project-name-here/data|IMAGE_DIR
sqlite> UPDATE datasets \
  SET location="/data/new-disk/project-name-here/data" \
  WHERE id=1;
sqlite> (Note: press Ctrl +D to exit)
```

Note: This document is a work in progress and does not contain all the required information.
