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

import argparse
import logging
import sys

import wise._main.create_index
import wise._main.extract_features
import wise._main.media_metadata
import wise._main.serve


_logger = logging.getLogger(__name__)


def main(argv: list[str]):
    logging.basicConfig(
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="wise",
        description="A multimodal search engine for images, videos, and audio",
        epilog=(
            "For more details about WISE, visit"
            " https://www.robots.ox.ac.uk/~vgg/software/wise/"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        description=(
            "The following commands are available.  Use 'wise <command> --help'"
            " for more details about each command."
        ),
        metavar="COMMAND",
    )

    create_index_subparser = subparsers.add_parser(
        "create-index",
        help=(
            "Create a nearest neighbour search index for features extracted"
            " from images and videos."
        ),
        add_help=False,
    )

    extract_features_subparser = subparsers.add_parser(
        "extract-features",
        help=(
            "Initialise a WISE project by extracting features from images,"
            " audio, and videos."
        ),
        add_help=False,
    )

    metadata_subparser = subparsers.add_parser(
        "media-metadata",
        help=(
            "Manage metadata associated with media files contained in a"
            " WISE project."
        ),
        add_help=False,
    )

    serve_subparser = subparsers.add_parser(
        "serve",
        help="Serve the REST API and frontend UI for WISE.",
        add_help=False,
    )

    args, _ = parser.parse_known_args(argv[1:])
    if args.command == "create-index":
        return wise._main.create_index.main(argv[1:])
    elif args.command == "extract-features":
        return wise._main.extract_features.main(argv[1:])
    elif args.command == "media-metadata":
        return wise._main.media_metadata.main(argv[1:])
    elif args.command == "serve":
        return wise._main.serve.main(argv[1:])
    else:
        ## Panic!  argparse should never let us get here.
        _logger.critical("Unknown command '%s'", args.command)
        return 1


def script_entrypoint():
    ## Script entrypoints for Python distributions must be a function,
    ## we can't just pass this module.  Those functions are called
    ## without any argument --- it is up to them to get sys.argv.  In
    ## main() we expect argv as an argument so we can't use main() as
    ## the entrypoint.  We want to keep main() as a function that
    ## takes argv as argument so it can be called from Python code.
    return main(sys.argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
