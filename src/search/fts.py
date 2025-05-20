import bisect
import itertools
import functools
import logging
from typing import Literal, Annotated

from src import db
from src.data_models import VectorAndMediaMetadata, ModalityType

from pydantic import Field, RootModel
import sqlalchemy as sa

logger = logging.getLogger(__name__)

COMMON_COLUMNS = ["media_id", "timestamp", "end_timestamp", "vector_id"]


def update_segment_text_with_highlights(highlighted_text: str, segments: list[dict]):
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
    cummulative_lengths = list(itertools.accumulate(segment_lengths, initial=0))
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
            updated_segment_texts[idx + 1] = "<b>" + updated_segment_texts[idx + 1]
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
            (idx, *(vals + (None,) * (4 - len(vals)))) for idx, vals in enumerate(ids)
        ]
    ]
    cte = sa.union_all(*sub_queries).cte("cte")
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
                    sa.and_(left.c[left_col] == None, right.c[col] == None)
                    if isouter
                    else False
                ),
            )
            for left_col, col in zip(COMMON_COLUMNS[1:], on_columns[1:])
        ],
    )


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
    def __init__(self, metadata):
        self.tables = {
            t: metadata.tables[t]
            for t in metadata.tables
            if t.startswith("metadata-") and "fts" not in t
        }
        self.fts_table = metadata.tables["metadata_fts"]

    def search(
        self,
        conn: sa.Connection,
        q: WISEFTSQuery,
        start: int | None = None,
        end: int | None = None,
    ):
        """
        FTS search function

        TODO, make it fine-grained and return the segment to allow further filtering
        """
        if self.fts_table is None:
            raise ValueError("Cannot use match operator without the fts clause")

        where_clause = q.convert_query_to_sql(self.fts_table)
        from_clause = db.media_table.join(
            self.fts_table,
            db.media_table.c.id == sa.literal_column(f"[{self.fts_table.name}].rowid"),
        )
        columns = []
        fts_counter = 0
        col_count = {}
        fts_cols = list(dict.fromkeys([x.name for x in self.fts_table.c]))
        for t, m in self.tables.items():
            from_clause = from_clause.join(
                m, m.c["media_id"] == db.media_table.c.id, isouter=True
            )

            _count = 0
            for x in (c for c in m.c if c.name not in COMMON_COLUMNS):
                if x.name in fts_cols:
                    column = sa.column(
                        f"highlight([{self.fts_table.name}], {fts_counter}, '<b>', '</b>')",
                        is_literal=True,
                    ).label(x.name)
                    fts_counter += 1
                else:
                    column = x
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
            .order_by(sa.text("rank"))
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
            media_metadata = dict(zip(db.media_table.c.keys(), media_cols))
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
            vals = extra_metadata.pop("metadata-asr", {})
            text, segments = (
                vals.pop("asr", ""),
                vals.pop("segments", []),
            )

            updated_segments = update_segment_text_with_highlights(text, segments)
            # Merge nearby segments
            merged_segments = merge_close_segments(updated_segments)
            vector_media_metadata = VectorAndMediaMetadata.model_validate(
                media_metadata
                | {
                    "modality": ModalityType.TEXT,
                    "media_id": media_metadata["id"],
                    "id": None,
                    "timestamp": None,
                    "end_timestamp": None,
                    "feature_extractor_id": "metadata"
                }
                | {
                    "external_metadata": {
                        "asr_segments": merged_segments,
                        **{k: v for m in extra_metadata.values() for k, v in m.items()},
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
                            "timestamp": s["start"],
                            "end_timestamp": s["end"],
                        }
                    )
                    responses.append(m)
            else:
                # if no segments, add the media metadata
                responses.append(
                    vector_media_metadata.model_copy(
                        update={
                            "timestamp": 0,
                            "end_timestamp": None,
                        }
                    )
                )
        logger.info(f"Num responses: {len(responses)}")
        return responses


if __name__ == "__main__":
    pass
