from __future__ import annotations
from collections import defaultdict
import logging

from pathlib import Path
from typing import Annotated, Literal

from src.wise_project import WiseProject
from src import db
from src.utils import batched
from pydantic import BaseModel, Field, RootModel, field_serializer

import pandas as pd
import sqlalchemy as sa
from tqdm import tqdm
import typer

app = typer.Typer()
app_state = {"verbose": True}
logger = logging.getLogger()


class Narration(BaseModel):
    unique_narration_id: str
    participant_id: str
    video_id: str
    narration: str
    start_timestamp: float
    end_timestamp: float
    nouns: list[str]
    verbs: list[str]
    pairs: list[tuple[str, str]]
    main_actions: Annotated[list[tuple[str, str]], Field(min_length=1, max_length=1)]
    verb_classes: list[int]
    noun_classes: list[int]
    pair_classes: list[tuple[int, int]]
    main_action_classes: Annotated[
        list[tuple[int, int]], Field(min_length=1, max_length=1)
    ]
    hands: list[str]
    narration_timestamp: float


NA_ = Literal["N/A"]


class Ingredient(BaseModel):
    name: str
    amount: float | NA_
    amount_unit: str | NA_
    calories: float | NA_
    carbs: float | NA_
    fat: float | NA_
    protein: float | NA_
    weigh: list[VideoSegment]
    add: list[VideoSegment]


class Capture(BaseModel):
    videos: list[str]
    ingredients: dict[str, Ingredient]
    step_times: dict[str, list[VideoSegment]]
    prep_times: dict[str, list[VideoSegment]]


class Recipe(BaseModel):
    participant: str
    name: str
    type: str
    source: str
    steps: dict[str, str]
    captures: list[Capture]


class RecipeCollection(RootModel):
    root: dict[str, Recipe]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class VideoSegment(BaseModel):
    video: str
    start: float
    end: float


class VideoMetadata(BaseModel):
    video_id: str
    participant: str
    recipe: str
    recipe_id: str
    recipe_source: str
    steps: list[str] = []
    ingredients: list[str] = []

    @field_serializer("steps")
    def serialize_steps(self, steps: list[str], _info) -> str:
        return "\n".join([f"{idx}. {x}" for idx, x in enumerate(steps, start=1)])

    @field_serializer("ingredients")
    def serialize_ingredients(self, ingredients: list[str], _info) -> str:
        return ", ".join([x.capitalize() for x in ingredients])


metadata_table = sa.Table(
    "metadata-activities",
    db.project_metadata_obj,
    sa.Column(
        "media_id", sa.Integer, sa.ForeignKey("media.id"), index=True, nullable=False
    ),
    *[sa.Column(x, sa.String, default="") for x in VideoMetadata.model_fields.keys()],
)

narrations_table = sa.Table(
    "metadata-asr",
    db.project_metadata_obj,
    sa.Column(
        "media_id",
        sa.Integer,
        sa.ForeignKey("media.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    sa.Column("asr", sa.String),
    sa.Column("segments", sa.JSON),
)


class ASRSegment(BaseModel):
    start: float
    end: float
    text: str


class ASR(BaseModel):
    asr: str
    segments: list[ASRSegment]


def asr_from_narrations(df: pd.DataFrame):
    for video_id, group in df.sort_values(
        ["video_id", "start_timestamp"], ascending=True
    ).groupby("video_id"):
        segments = []
        for _, row in group.iterrows():
            segments.append(
                ASRSegment(
                    start=row["start_timestamp"],
                    end=row["end_timestamp"],
                    text=row["narration"].strip(),
                )
            )

        yield video_id, ASR(
            asr=" ".join(map(lambda x: x.text, segments)), segments=segments
        )


def video_metadata_from_recipes(recipes: RecipeCollection):
    def handle_recipe(recipe_id: str, recipe: Recipe):
        video_to_ingredient: dict[str, list] = defaultdict(list)
        video_to_step: dict[str, list] = defaultdict(list)
        all_videos = []
        for capture in recipe.captures:
            all_videos.extend(capture.videos)
            for ingredient in capture.ingredients.values():
                any(
                    map(
                        lambda s: video_to_ingredient[s.video].append(ingredient.name),
                        ingredient.weigh + ingredient.add,
                    )
                )

            for step_id, segments in capture.step_times.items():
                any(map(lambda s: video_to_step[s.video].append(step_id), segments))

        all_videos = sorted(list(dict.fromkeys(all_videos)))
        for v in all_videos:
            yield VideoMetadata(
                video_id=v,
                participant=recipe.participant,
                recipe=recipe.name,
                recipe_id=recipe_id,
                recipe_source=recipe.source,
                steps=[
                    recipe.steps[step_id]
                    for step_id in list(dict.fromkeys(video_to_step[v]))
                ],
                ingredients=list(dict.fromkeys(video_to_ingredient[v])),
            )

    for rid in recipes:
        yield from handle_recipe(rid, recipes[rid])


def get_video_filename_to_media_id_mapping(conn: sa.Connection):
    media_table = db.media_table
    source_collection_table = db.source_collections_table
    stmt = sa.select(
        media_table.c.id, media_table.c.path, source_collection_table.c.location
    ).select_from(media_table.join(source_collection_table))
    _mapping = {}
    for media_id, path, location in conn.execute(stmt).all():
        full_path = Path(location) / path
        _mapping[full_path.stem] = media_id
    return _mapping


@app.callback()
def base(verbose: bool = False):
    """
    WISE CLI
    Search through collections of images with Text / Image
    """
    app_state["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    global logger
    logger = logging.getLogger()


def import_narrations(
    hd_epic_narrations: Path,
):
    """
    Import HD-EPIC annotations into wise project as
    media metadata
    """
    assert hd_epic_narrations.is_file()

    logger.info(f"Reading from - {hd_epic_narrations}")
    df = pd.read_pickle(hd_epic_narrations)

    logger.info(df.head())
    logger.info(f"Columns: {df.columns}")

    return df


def import_recipes(hd_epic_complete_recipes: Path):
    assert hd_epic_complete_recipes.is_file()
    logger.info(f"Reading from - {hd_epic_complete_recipes}")
    return RecipeCollection.model_validate_json(hd_epic_complete_recipes.read_text())


@app.command(name="import")
def import_(
    hd_epic_annotations_dir: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            readable=True,
            exists=True,
            dir_okay=True,
            help="Path to hd-epic annotations directory",
        ),
    ],
    project_dir: Annotated[Path, typer.Option(..., help="Path to wise project")],
):
    HD_EPIC_NARRATIONS_PKL = "HD_EPIC_Narrations.pkl"
    HD_EPIC_NARRATIONS_DIR = "narrations-and-action-segments"
    df = import_narrations(
        hd_epic_annotations_dir / HD_EPIC_NARRATIONS_DIR / HD_EPIC_NARRATIONS_PKL
    )

    HD_EPIC_ACTIVITIES_DIR = "high-level"
    HD_EPIC_RECIPES_JSON = "complete_recipes.json"
    recipes = import_recipes(
        hd_epic_annotations_dir / HD_EPIC_ACTIVITIES_DIR / HD_EPIC_RECIPES_JSON
    )

    project = WiseProject(project_dir)
    db_engine = db.init_project(project.dburi)

    with db_engine.connect() as conn, tqdm() as pbar:
        narrations_table.drop(conn, checkfirst=True)
        narrations_table.create(conn)

        metadata_table.drop(conn, checkfirst=True)
        metadata_table.create(conn)

        video_filename_to_media_id = get_video_filename_to_media_id_mapping(conn)

    with db_engine.connect() as conn, tqdm(
        desc="Narrations", total=len(df.video_id.unique())
    ) as pbar:
        stmt = sa.insert(narrations_table)
        for batch in batched(asr_from_narrations(df), 16):
            conn.execute(
                stmt,
                [
                    {"media_id": video_filename_to_media_id[vid]} | asr.model_dump()
                    for vid, asr in batch
                ],
            )
            pbar.update(len(batch))
            conn.commit()

    with db_engine.connect() as conn, tqdm(desc="Metadata") as pbar:
        stmt = sa.insert(metadata_table)
        for video_metadata in batched(video_metadata_from_recipes(recipes), 1024):
            conn.execute(
                stmt,
                [
                    {"media_id": video_filename_to_media_id[x.video_id]}
                    | x.model_dump()
                    for x in video_metadata
                ],
            )
            pbar.update(len(video_metadata))
            conn.commit()

    return df, recipes


if __name__ == "__main__":
    app()
