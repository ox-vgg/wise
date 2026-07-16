#!/usr/bin/env python3

## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import builtins
from typing import Any, Generic, Iterable, Optional, Type, TypeVar

import sqlalchemy as sa
from pydantic import BaseModel

Entity = TypeVar("Entity", bound=BaseModel)
EntityCreate = TypeVar("EntityCreate", bound=BaseModel)
EntityUpdate = TypeVar("EntityUpdate", bound=BaseModel)


class EntityNotFoundException(Exception):
    pass


## XXX: Type parameter lists are only supported in Python 3.12.  When
## we depend on python 3.12+, we can drop Generic.
class SQLAlchemyRepository(Generic[Entity, EntityCreate, EntityUpdate]):
    def __init__(self, table: sa.Table, model: Type[Entity]):
        self._table = table
        self.model = model

    def get(self, conn: sa.Connection, id: Any) -> Optional[Entity]:
        result = conn.execute(
            sa.select(self._table).where(self._table.c.id == id)
        )
        for row in result.mappings():
            return self.model.model_validate(row)
        return None

    # see https://docs.sqlalchemy.org/en/20/_modules/examples/performance/large_resultsets.html
    def list(
        self,
        conn: sa.Connection,
        batch_size: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ):
        _conn = conn
        if batch_size is not None:
            _conn = conn.execution_options(stream_results=True)

        result = _conn.execute(
            sa.select(self._table).limit(limit).offset(offset)
        )
        if batch_size is None:
            yield from map(self.model.model_validate, result.mappings())

        else:
            while True:
                chunk = result.mappings().fetchmany(batch_size)
                if not chunk:
                    break
                yield from map(self.model.model_validate, chunk)

    def get_row_by_column_match(
        self, conn: sa.Connection, column_name_to_match, column_value
    ):
        """
        Performs query equivalent to:
        ```
        SELECT * FROM table WHERE {col_name} = {col_value}
        ```
        """
        result = conn.execute(
            sa.select(self._table).where(
                self._table.c[column_name_to_match] == column_value
            )
        )
        for row in result.mappings():
            return self.model.model_validate(row)
        return None

    def list_by_column_match(
        self,
        conn,
        *,
        column_to_match: str,
        value_to_match: Any,
        select_columns: Optional[tuple[str]] = None,
        order_by_column: str,
        desc: bool = False,
        batch_size: int = 10000,
    ):
        """
        Performs a query equivalent to:
        ```
        SELECT {select_columns} FROM table WHERE {column_to_match} = {value_to_match}
        ORDER BY {order_by_column} {DESC if desc else ASC}
        ```

        If select_columns is None, all columns in the table are selected (`SELECT * ...`)
        """
        select_column_specs = (
            self._table
            if select_columns is None
            else self._table.c[select_columns]
        )
        result = conn.execution_options(stream_results=True).execute(
            sa.select(select_column_specs)
            .where(self._table.c[column_to_match] == value_to_match)
            .order_by(
                self._table.c[order_by_column].desc()
                if desc
                else self._table.c[order_by_column].asc()
            )
        )
        while True:
            chunk = result.mappings().fetchmany(batch_size)
            if not chunk:
                return

            for row in chunk:
                if select_columns:
                    yield row
                else:
                    yield self.model.model_validate(row)

    def get_columns(self, conn: sa.Connection, column_names: tuple[str]):
        result = conn.execute(sa.select(self._table.c[column_names]))
        yield from result.mappings()

    def get_count(self, conn: sa.Connection) -> int:
        return conn.execute(
            sa.select(sa.func.count(self._table.c.id))
        ).scalar()

    def create(self, conn: sa.Connection, *, data: EntityCreate):
        return self.create_many(conn, data=[data], return_objs=True)[0]

    def create_many(
        self,
        conn: sa.Connection,
        *,
        data: Iterable[EntityCreate],
        return_objs=False,
        return_ids=False,
        ## builtins.list on type annotations avoids clash with list method
    ) -> Optional[builtins.list[int] | builtins.list[Entity]]:
        """Create multiple entities on the repository.

        This method can optionally return the ids, or the whole "data
        model", of the entity created.  It is "cheaper" to return
        nothing.  The number of rows inserted will be checked even if
        nothing is returned.

        """
        if return_objs and return_ids:
            raise ValueError(
                "`return_objs` and `return_ids` options are mutually exclusive"
            )
        stmt = sa.insert(self._table)
        if return_ids:
            stmt = stmt.returning(self._table.id)
        elif return_objs:
            stmt = stmt.returning(self._table.columns)

        cur = conn.execute(stmt, [x.model_dump() for x in data])
        if return_ids:
            ids = [int(x) for x in cur]
            assert len(data) == len(ids)
            return ids
        elif return_objs:
            objs = [
                self.model.model_validate(x, from_attributes=True) for x in cur
            ]
            assert len(data) == len(objs)
            return objs
        else:
            assert len(data) == cur.rowcount

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
