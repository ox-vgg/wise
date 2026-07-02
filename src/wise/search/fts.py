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

import bisect
import functools
import itertools
import json
import logging
from collections import defaultdict
from typing import Annotated, Literal

import sqlalchemy as sa
from pydantic import Field, RootModel
from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement

from wise import db
from wise.data_models import (
    MediaMetadata,
    MediaType,
    ModalityType,
    VectorAndMediaMetadata,
)

logger = logging.getLogger(__name__)

COMMON_COLUMNS = ["media_id", "timestamp", "end_timestamp", "vector_id"]


def update_segment_text_with_highlights(
    highlighted_text: str, segments: list[dict]
):
    """
    Replace text in segments with the highlighted text, while taking care of the length mismatch
    due to the highlighting, and closing the dangling highlights
    """
    if len(segments) == 0:
        return segments

    if highlighted_text == "":
        # no highlights, return original segments
        return segments

    TOKENS = ("<b>", "</b>")
    TOKEN_LENGTH = 7  # sum(map(len, TOKENS))

    segment_lengths = list(map(lambda x: len(x["text"]), segments))
    cummulative_lengths = list(
        itertools.accumulate(segment_lengths, initial=0)
    )
    # account for 1 space between segments
    cummulative_lengths = list(map(sum, enumerate(cummulative_lengths)))

    offset = 0
    token_idx = 0
    current_token = TOKENS[token_idx]

    # Scan the highlighted text for tokens and update the segment lengths
    while (idx := highlighted_text.find(current_token, offset)) != -1:
        token_length = len(current_token)

        arr_idx = bisect.bisect(cummulative_lengths, idx)
        segment_lengths[arr_idx - 1] += token_length
        cummulative_lengths[arr_idx:] = map(
            lambda x: x + token_length, cummulative_lengths[arr_idx:]
        )
        offset = idx
        token_idx = 1 - token_idx
        current_token = TOKENS[token_idx]

    # get the updated segments from the highlighted text, account for space between segments
    updated_segment_texts = [
        highlighted_text[start : end - 1]
        for (start, end) in itertools.pairwise(cummulative_lengths)
    ]

    # replace the text in the segments with the highlighted one. Take care of dangling tags
    for idx, s in enumerate(segments):
        updated_text = updated_segment_texts[idx]
        original_length = len(s["text"])
        new_length = len(updated_text)

        if (new_length - original_length) % TOKEN_LENGTH != 0:
            # If the diff is not divisible by 7 (length of token pair), then there are dangling tags
            # Close tag in current segment and update next segment with an open tag, and update length
            updated_text += "</b>"
            updated_segment_texts[idx + 1] = (
                "<b>" + updated_segment_texts[idx + 1]
            )
        s["text"] = updated_text
    return segments


def merge_close_segments(segments: list[dict], threshold: float = 4):
    def merge(x, y):
        # first iteration
        if not x:
            x.append(y)
            return x

        # <b> tag not present in previous / current segment
        last = x[-1]
        if "<b>" not in last["text"] or "<b>" not in y["text"]:
            x.append(y)
            return x

        # segments are far apart
        if y["start"] - last["end"] > threshold:
            x.append(y)
            return x

        # merge
        last["end"] = y["end"]
        last["text"] += f" {y['text']}"
        return x

    merged_segments = list(functools.reduce(merge, segments, []))
    return merged_segments


def get_cte_from_ids(ids: list[tuple]):
    sub_queries = [
        sa.select(
            sa.literal(i).label("id"),
            sa.literal(m).label("media_id"),
            sa.literal(t).label("timestamp"),
            sa.literal(e).label("end_timestamp"),
            sa.literal(v).label("vector_id"),
        )
        for i, m, t, e, v in [
            (idx, *(vals + (None,) * (4 - len(vals))))
            for idx, vals in enumerate(ids)
        ]
    ]
    cte = sa.union_all(*sub_queries).cte("cte")
    return cte


# def get_cte_from_media_ids(media_ids: list[int]):
#     """
#     Create a CTE from a list of media_ids
#     """
#     sub_queries = [
#         sa.select(
#             sa.literal(i).label("rank"),
#             sa.literal(m).label("media_id"),
#         )
#         for i, m in enumerate(media_ids)
#     ]
#     cte = sa.union_all(*sub_queries).cte("cte")
#     return cte


def get_cte_from_media_ids(media_ids: list[int]):
    """
    Create a CTE from a list of media_ids using the values expression in SQLAlchemy
    """
    cte = (
        sa.values(
            sa.column("rank", sa.Integer),
            sa.column("media_id", sa.Integer),
        )
        .data([(i, m) for i, m in enumerate(media_ids)])
        .cte("cte")
    )

    return cte


def get_join_onclause(
    left: sa.FromClause,
    right: sa.FromClause,
    on_columns: list[str] = [],
    isouter: bool = False,
):
    """
    get the on_clause for the join operation, based on the 4 id columns
    usually used with the cte ids
    select * from cte join a on cte.media_id = a.media_id AND ...
    leftclause is assumed to have the standard names for the 4 columns
    """
    if not on_columns:
        on_columns = COMMON_COLUMNS
    return sa.and_(
        left.c[COMMON_COLUMNS[0]] == right.c[on_columns[0]],
        *[
            sa.or_(
                left.c[left_col] == right.c[col],
                (
                    sa.and_(left.c[left_col] is None, right.c[col] is None)
                    if isouter
                    else False
                ),
            )
            for left_col, col in zip(COMMON_COLUMNS[1:], on_columns[1:])
        ],
    )


def parse_fts_config(fts_config: dict):
    # generates a mapping from old name to new name and vice-versa
    column2fts = defaultdict(dict)
    fts2column = defaultdict(dict)
    for table, fts_cols in fts_config.items():
        for c in fts_cols:
            if ":" in c:
                old_name, new_name = c.split(":", 1)
            else:
                old_name, new_name = c, c

            column2fts[table][old_name] = new_name
            fts2column[table][new_name] = old_name

    return column2fts, fts2column


def get_metadata_selectable_from_fts5_config(
    extra_metadata_tables: dict[str, sa.Table], fts_config: dict
):
    """
    Provide an fts_config object with the metadata tables as keys and a list of columns as values
    """
    media_table = db.media_table
    if len(extra_metadata_tables) == 0:
        raise ValueError("No metadata tables found")
    # Based on the config, create a view and fts5 table pair
    from_clause = media_table
    columns = []
    cols2fts, _ = parse_fts_config(fts_config)
    for name, col2fts_map in cols2fts.items():
        m = extra_metadata_tables[name]
        from_clause = from_clause.join(
            m, m.c["media_id"] == media_table.c.id, isouter=True
        )
        mcols = [
            m.c[old_name].label(new_name)
            for old_name, new_name in col2fts_map.items()
        ]
        columns.extend(mcols)

    if len(columns) == 0:
        raise ValueError(
            "No valid columns found in the metadata tables for building fts5 index"
        )

    return sa.select(media_table.c.id, *columns).select_from(from_clause)


class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@compiler.compiles(CreateView)
def _create_view(element, compiler, **kw):
    return "CREATE VIEW %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@compiler.compiles(DropView)
def _drop_view(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


def view_exists(ddl, target, connection, **kw):
    return ddl.name in sa.inspect(connection).get_view_names()


def view_doesnt_exist(ddl, target, connection, **kw):
    return not view_exists(ddl, target, connection, **kw)


def create_fts_table_from_selectable(name, metadata, selectable):
    t = FTS5Table(
        name,
        metadata,
        *(
            sa.Column(c.name, c.type)
            for c in selectable.selected_columns
            if c.name != "id"
        ),
    )
    view_name = f"{name}_view"
    sa.event.listen(
        t,
        "before_create",
        DropView(view_name).execute_if(callable_=view_exists),
    )
    sa.event.listen(
        t,
        "before_create",
        CreateView(view_name, selectable),
    )
    sa.event.listen(
        t,
        "after_drop",
        DropView(view_name),
    )
    return t


class FTS5Table(sa.Table):
    pass


@compiler.compiles(sa.schema.CreateTable, "sqlite")
def _compile(element: sa.schema.CreateTable, compiler, **kw):
    if not isinstance(element.target, FTS5Table):
        return compiler.visit_create_table(element, **kw)
    name = compiler.preparer.format_table(element.target)
    cols = ", ".join(
        compiler.preparer.format_column(col) for col in element.target.columns
    )

    # content
    content = f"{name}_view"
    rowid = "id"
    # rowid

    return f"CREATE VIRTUAL TABLE {name} USING fts5({cols}, content='{content}', content_rowid='{rowid}')"


Operators = Literal["$match"]
OperatorQuery = Annotated[
    dict[Operators, str],
    Field(max_length=1, min_length=1),
]


class WISEFTSQuery(RootModel[OperatorQuery]):
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def convert_query_to_sql(self, fts_table: sa.TableClause):
        # { '$match': '...' }
        op, val = next(iter(self.root.items()))
        assert op == "$match"
        return fts_table.table_valued().match(val)


class FTSSearch:
    is_internal_search_supported = False
    table_name = db._WISE_FTS_TABLE

    def __init__(self, project: "WiseProject", metadata):
        self.project = project
        self.db_metadata = metadata

        self.tables = {t.name: t for t in project.external_metadata_tables()}
        self.fts_table = metadata.tables.get(self.table_name)

        try:
            with self.project.fts_config_file.open() as f:
                self.fts_config = json.load(f)
        except Exception:
            raise ValueError("fts_config could not be found!")

    def build_index(self, conn: sa.Connection):
        fts_selectable = get_metadata_selectable_from_fts5_config(
            self.tables, self.fts_config
        )

        self.fts_table = create_fts_table_from_selectable(
            self.table_name, self.db_metadata, fts_selectable
        )
        logger.info("Dropping existing fts5 table")
        self.fts_table.drop(conn, checkfirst=True)
        logger.info("Building fts5 index")
        self.fts_table.create(conn)
        stmt = sa.text(
            f"INSERT INTO [{self.fts_table.name}] ([{self.fts_table.name}]) VALUES ('rebuild')"
        )
        conn.execute(stmt)

    def search(
        self,
        conn: sa.Connection,
        q: WISEFTSQuery,
        start: int | None = None,
        end: int | None = None,
        ids_only: bool = False,
    ):
        """
        FTS search function

        TODO, make it fine-grained and return the segment to allow further filtering
        """
        if self.fts_table is None:
            raise ValueError(
                "Cannot use match operator without the fts table - build it with create_index.py and make sure the table is reflected from the db before calling this function"
            )

        col2fts, _ = parse_fts_config(self.fts_config)
        where_clause = q.convert_query_to_sql(self.fts_table)
        from_clause = db.media_table.join(
            self.fts_table,
            db.media_table.c.id
            == sa.literal_column(f"[{self.fts_table.name}].rowid"),
        )
        if end is None:
            limit = None
        elif start is None:
            limit = end
        else:
            limit = end - start

        # search on fts table and get all media_ids
        stmt = (
            sa.select(
                db.media_table.c.id,
            )
            .select_from(from_clause)
            .where(where_clause)
            .order_by(sa.text("rank"))
            .limit(limit)
            .offset(start)
        )

        ids = conn.execute(stmt).scalars().all()
        if ids_only or not ids:
            return ids

        cte = get_cte_from_media_ids(ids)
        from_clause = cte.join(
            db.media_table,
            cte.c.media_id == db.media_table.c.id,
        ).join(
            self.fts_table,
            db.media_table.c.id
            == sa.literal_column(f"[{self.fts_table.name}].rowid"),
        )

        # use the media_ids to select the matching rows and highlight the fts columns
        columns = []
        col_count = {}
        fts_cols = list(dict.fromkeys([x.name for x in self.fts_table.c]))
        for t, m in self.tables.items():
            from_clause = from_clause.join(
                m, m.c["media_id"] == db.media_table.c.id, isouter=True
            )

            _count = 0
            for x in (c for c in m.c if c.name not in COMMON_COLUMNS):
                fts_name = col2fts[m.name].get(x.name, x.name)

                column = (
                    sa.column(
                        f"highlight([{self.fts_table.name}], {fts_cols.index(fts_name)}, '<b>', '</b>')",
                        is_literal=True,
                    ).label(fts_name)
                    if fts_name in fts_cols
                    else x
                )
                _count += 1
                columns.append(column)
            col_count[t] = _count

        stmt = (
            sa.select(
                db.media_table,
                *columns,
            )
            .select_from(from_clause)
            .where(where_clause)
            .order_by(cte.c.rank)
            .limit((end - start))
            .offset(start)
        )

        res = conn.execute(stmt)

        responses = []

        # extra_metadata = [{}] * len(res.mappings())conn.execute(get_query(ids, []))
        def get_columns(cols, r):
            return r[: len(cols)], r[len(cols) :]
            # return {c.name: r[stmt.c.corresponding_column(c).key] for c in cols}

        for r in res.all():
            row = r[:]
            media_cols, row = get_columns(db.media_table.c, row)
            media_metadata = MediaMetadata.model_validate(
                dict(zip(db.media_table.c.keys(), media_cols))
            )
            extra_metadata = {}
            for m in self.tables.values():
                cols, row = get_columns(
                    [c for c in m.c.keys() if c not in COMMON_COLUMNS], row
                )
                extra_metadata[m.name] = dict(
                    zip(
                        [c for c in m.c.keys() if c not in COMMON_COLUMNS],
                        cols,
                    )
                )
            vals = extra_metadata.pop(db._WISE_ASR_TABLE, {})
            text, segments = (
                vals.pop("asr", ""),
                vals.pop("segments", []),
            )

            updated_segments = update_segment_text_with_highlights(
                text, segments
            )
            # Merge nearby segments
            # merged_segments = merge_close_segments(updated_segments)
            # Note: dont merge segments as they can become very long. Best to display them next to each other on the table
            merged_segments = updated_segments
            vector_media_metadata = VectorAndMediaMetadata.model_validate(
                media_metadata.model_dump()
                | {
                    "modality": ModalityType.TEXT,
                    "media_id": media_metadata.id,
                    "id": None,
                    "timestamp": None,
                    "end_timestamp": None,
                    "feature_extractor_id": "wise/metadata",
                }
                | {
                    "external_metadata": {
                        "asr_segments": merged_segments,
                        **{
                            k: v
                            for m in extra_metadata.values()
                            for k, v in m.items()
                            if v
                        },
                    }
                }
            )
            filtered_segments = list(
                filter(lambda x: "<b>" in x["text"], merged_segments)
            )
            if filtered_segments:
                for s in filtered_segments:
                    m = vector_media_metadata.model_copy(
                        update={
                            "modality": (
                                media_metadata.media_type
                                if media_metadata.media_type != MediaType.AV
                                else MediaType.VIDEO
                            ),
                            "timestamp": s["start"],
                            "end_timestamp": s["end"],
                        }
                    )
                    responses.append(m)
                    if media_metadata.media_type == MediaType.AV:
                        m_audio = m.model_copy(
                            update={"modality": ModalityType.AUDIO}
                        )
                        responses.append(m_audio)
            else:
                # if no segments, add the media metadata
                m = vector_media_metadata.model_copy(
                    update={
                        "modality": (
                            media_metadata.media_type
                            if media_metadata.media_type != MediaType.AV
                            else MediaType.VIDEO
                        ),
                        "timestamp": 0,
                        "end_timestamp": None,
                    }
                )
                responses.append(m)
                if media_metadata.media_type == MediaType.AV:
                    m_audio = m.model_copy(
                        update={"modality": ModalityType.AUDIO}
                    )
                    responses.append(m_audio)
        logger.info("Num responses: %d", len(responses))
        return responses


if __name__ == "__main__":
    pass
