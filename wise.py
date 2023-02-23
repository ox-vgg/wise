## Command line interface for WISE Image Search Engine (WISE)
##
##
## Author : Abhishek Dutta <adutta@robots.ox.ac.uk>, Prasanna Sridhar <prasanna@robots.ox.ac.uk>, Horace Lee <horacelee@robots.ox.ac.uk>
## Date   : 2023-Feb-08
##
## [ Example Usage ]
## wise search --for "dog" --in /data/images/
## wise search --query "picture of dog" --in /data/images/
## wise search --like /Pictures/mydog.jpg --in /data/images
## wise search --like /Pictures/dogs/german-shephard/ --in /data/images

import os
import argparse
from pathlib import Path


def search_using_text(args):
    if getattr(args, "query") is not None:
        query = getattr(args, "query")
    else:
        query = "picture of " + getattr(args, "for")
    print(
        'TODO: searching images in %s using text query "%s"'
        % (getattr(args, "in"), query)
    )
    return


def search_using_an_image(args):
    print(
        "TODO: searching for images similar to %s in the folder %s"
        % (getattr(args, "like"), getattr(args, "in"))
    )
    return


def search_using_images(args):
    print(
        "TODO: searching folder [%s] for class of objects represented in folder [%s]"
        % (getattr(args, "in"), getattr(args, "like"))
    )
    return


def init(args):
    # wise init [PROJECT_ID] --from IMAGE_DIR --from WEBDATASET_URL
    # example
    # wise init wikimedia --from /data/images/
    # wise init wikimedia --from /data/images/{00001..00010}.tar
    # wise init wikimedia --from /data/images/00001.tar, /data/images/00002.tar
    # wise init wikimedia --from https://...tar
    #
    # Notes
    # - [ ] Initialise the project tree
    # - [ ] Get the list of valid images / webdataset tars to process
    # - [ ] Create a db entry for the dataset source
    # - [ ] Extract and store features from the sources
    # - [ ] Store thumbnails
    # - [ ] Building index (optional, Brute force kNN for now)

    print("TODO: preparing folder [%s] for search" % (getattr(args, "in")))
    print(
        "This process involves extracting features, preparing approximate nearest neighbourhood search object, etc. and therefore may take long time to complete."
    )
    return


def test(args):
    print("TODO: running tests")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="wise",
        description="WISE Image Search Engine",
        epilog="See https://robots.ox.ac.uk/~vgg/wise/ for more details.",
    )

    parser.add_argument(
        "command",
        type=str,
        choices=["search", "test", "init"],
        help="the command to execute",
    )
    parser.add_argument(
        "--query", type=str, required=False, help="search query (e.g. a black swan)"
    )
    parser.add_argument(
        "--for",
        type=str,
        required=False,
        metavar="OBJECT-NAME",
        help='search for an object like dog, bus, etc. (same as --query "picture of OBJECT-NAME")',
    )

    parser.add_argument(
        "--like",
        type=str,
        required=False,
        metavar="IMAGE(S)",
        help="provide an image filename or a folder containing images",
    )
    parser.add_argument(
        "--in",
        type=str,
        required=True,
        metavar="FOLDER",
        help="the search operation is carried out in the images contained in this folder",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=os.path.join(Path.home(), ".wise"),
        metavar="FOLDER",
        required=False,
        help="by default, WISE uses $HOME/.wise/ folder for storing features, config, etc.",
    )

    args = parser.parse_args()
    if args.command == "search":
        if getattr(args, "query") or getattr(args, "for"):
            search_using_text(args)
        elif getattr(args, "like"):
            if os.path.isdir(getattr(args, "like")):
                search_using_images(args)
            elif os.path.exists(getattr(args, "like")) and os.path.isfile(
                getattr(args, "like")
            ):
                search_using_an_image(args)
            else:
                print("--like takes path of an image or a folder containing images")
        else:
            print(
                "search command requires one of the following: --query, --for, --like"
            )
    elif args.command == "test":
        test(args)
    elif args.command == "init":
        init(args)
    else:
        print("unknown command " + args.command)
