from .base import SQLAlchemyRepository
from ..data_models import Project
import sqlalchemy as sa
from typing import Any, Optional


def get_version(project: Optional[Project] = None):
    # If project, get version and increment by 1
    # else set it to 1
    return project and project.version or 0


class WiseProjectsSQLAlchemyRepository(SQLAlchemyRepository[Project, Project, Project]):
    """
    - Ensure next version is larger than previous version
      when writing projects
    """

    def create(self, conn: sa.Connection, *, data: Project):

        if data.version is None:
            data.version = 0

        return super().create(conn, data=data)

    def update(self, conn: sa.Connection, id: Any, *, data: Project):
        # Get project
        project = self.get(conn, data.id)
        current_version = get_version(project)

        if data.version and (data.version <= current_version):
            raise ValueError("New version cannot be smaller than current version")

        return super().update(conn, id, data=data)

    pass
