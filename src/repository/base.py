from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Iterable, Any, Type, Optional

from pydantic import BaseModel
import sqlalchemy as sa


Entity = TypeVar("Entity", bound=BaseModel)
EntityCreate = TypeVar("EntityCreate", bound=BaseModel)
EntityUpdate = TypeVar("EntityUpdate", bound=BaseModel)


class EntityNotFoundException(Exception):
    pass


class Repository(ABC, Generic[Entity, EntityCreate, EntityUpdate]):
    @abstractmethod
    def get(self, id: Any) -> Optional[Entity]:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> Iterable[Entity]:
        raise NotImplementedError

    @abstractmethod
    def create(self, obj: EntityCreate):
        raise NotImplementedError

    @abstractmethod
    def update(self, id: Any, obj: EntityUpdate) -> Entity:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: Any) -> Entity:
        raise NotImplementedError

    @abstractmethod
    def delete_all(self):
        raise NotImplementedError


class SQLAlchemyRepository(Repository[Entity, EntityCreate, EntityUpdate]):
    def __init__(self, table: sa.Table, model: Type[Entity]):
        self._table = table
        self.model = model

    def get(self, conn: sa.Connection, id: Any) -> Optional[Entity]:
        result = conn.execute(sa.select(self._table).where(self._table.c.id == id))
        for row in result.mappings():
            return self.model.model_validate(row)
        return None

    def list(self, conn: sa.Connection):
        result = conn.execute(sa.select(self._table))
        for row in result.mappings():
            yield self.model.model_validate(row)

    def get_columns(self, conn: sa.Connection, column_names: Tuple[str]):
        result = conn.execute(sa.select(self._table.c[column_names]))
        yield from result.mappings()

    def create(self, conn: sa.Connection, *, data: EntityCreate):
        result = conn.execute(
            sa.insert(self._table).returning(self._table.c.id), [data.model_dump()]
        )
        obj = self.get(conn, next(result)[0])
        if not obj:
            raise RuntimeError()
        return obj

    # TODO: Need to be careful with update since we can also re-assign the id key
    # 1. Could remove the id key and check, but how do we find the name of the id column?
    # 2. Could let the database handle error with db specific
    def update(self, conn: sa.Connection, id: Any, *, data: EntityUpdate):
        current_entity = self.get(conn, id)
        if current_entity is None:
            raise EntityNotFoundException()

        update_data = data.model_dump(exclude_unset=True)
        updated_entity = current_entity.model_copy(update=update_data)

        conn.execute(
            sa.update(self._table)
            .where(self._table.c.id == id)
            .values(**updated_entity.model_dump())
        )

    def delete(self, conn: sa.Connection, id: Any):
        conn.execute(sa.delete(self._table).where(self._table.c.id == id))

    def delete_all(self, conn: sa.Connection):
        conn.execute(sa.delete(self._table))
