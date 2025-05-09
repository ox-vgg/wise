import typer
from pathlib import Path
from typing import Optional, List, Dict
import logging
import json
from src.wise_project import WiseProject
from src.enums import IndexType, SearchTarget

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
)

def parse_search_targets(raw_search_targets: List[str]) -> Dict[SearchTarget, str]:
    targets = {}
    for item in raw_search_targets:
        if ":" not in item:
            raise typer.BadParameter(f"Invalid format for --search-target: '{item}' (expected key:value)")
        key, value = item.split(":", 1)
        try:
            target_enum = SearchTarget(key.lower())
        except ValueError:
            raise typer.BadParameter(f"Invalid search target type: '{key}'")
        targets[target_enum] = value
    return targets

app = typer.Typer()
@app.command(
    help="Serve the REST API and frontend UI for WISE.",
    epilog="For more details about WISE, visit https://www.robots.ox.ac.uk/~vgg/software/wise/",
    no_args_is_help=True,
)
def main(
    project_dir: Path = typer.Option(
        help="Project directory path"
    ),
    theme_asset_dir: Path = typer.Option(
        "frontend/dist",
        exists=True,
        dir_okay=True,
        file_okay=False,
        help=(
            "Static HTML assets related to the user interface are served "
            "from this folder."
        ),
    ),
    index_type: Optional[IndexType] = typer.Option(
        None,
        help="The faiss index to use for serving"
    ),
    search_targets: List[str] = typer.Option(
        [],
        "--search-target",
        help=(
            "One or more TARGET:FEATURE-ID pairs, where TARGET is one of {video, audio, face, image} "
            "and FEATURE-ID is the name (can be partial) of {audio,video,image}-feature-id. "
            "For example: --search-target video:open_clip --search-target audio:clap --search-target face:insightface"
        ),
    ),
    query_blocklist: Path = typer.Option(
        None,
        '--query-blocklist',
        '--query-blacklist',
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help=(
            "A text file containing a list of words/phrases (each separated by a line break) "
            "that users should be blocked from searching. When the user enters a query that matches "
            "one of the terms in the blocklist, an error message will be returned"
        ),
    )
):
    from api import serve

    project = WiseProject(
        project_dir,
        create_project=False
    )
    project_assets = project.discover_assets()

    search_targets = parse_search_targets(search_targets)
    if not search_targets:
        # initialise defaults for backward compatibility
        # TODO: remove this when the frontend starts using the new search target format
        if 'video' in project_assets:
            for feature_extractor_id in project_assets['video']:
                if 'insightface' in feature_extractor_id:
                    search_targets[SearchTarget.Video] = "insightface"
                    break
        if 'audio' in project_assets:
            search_targets[SearchTarget.Audio] = "clap"
        if SearchTarget.Video not in search_targets and 'video' in project_assets:
            search_targets[SearchTarget.Video] = "open_clip"
        print("No search targets specified. Using the following default values (for backward compatibility):")
        print(json.dumps(search_targets, indent=4))
        print("In future, default search targets will be taken from config.py")

    serve(
        project_dir,
        theme_asset_dir,
        index_type.value if index_type else None,
        search_targets,
        query_blocklist
    )


if __name__ == "__main__":
    app()
