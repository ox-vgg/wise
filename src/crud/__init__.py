from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Iterable, Any, Type, Optional

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
            return self.model.from_orm(row)
        return None

    def list(self, conn: sa.Connection):
        result = conn.execute(sa.select(self._table))
        for row in result.mappings():
            yield self.model.from_orm(row)

    def create(self, conn: sa.Connection, *, data: EntityCreate):
        result = conn.execute(
            sa.insert(self._table).returning(self._table.c.id), [data.dict()]
        )
        obj = self.get(conn, next(result)[0])
        if not obj:
            raise RuntimeError()
        return obj

    def update(self, conn: sa.Connection, id: Any, *, data: EntityUpdate):
        current_data = self.get(conn, id)
        if current_data is None:
            raise EntityNotFoundException()

        updated_data = current_data.copy(update=data.dict(exclude_unset=True))

        conn.execute(
            sa.update(self._table)
            .where(self._table.c.id == id)
            .values(**updated_data.dict())
        )

    def delete(self, conn: sa.Connection, id: Any):
        conn.execute(sa.delete(self._table).where(self._table.c.id == id))

    def delete_all(self, conn: sa.Connection):
        result = conn.execute(sa.delete(self._table))
        print("count", result.rowcount)
