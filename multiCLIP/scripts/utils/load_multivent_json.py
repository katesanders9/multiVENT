"""Parse infact video json."""

import argparse
import csv
import inspect
import json
import logging
import os
import re
import sys
import time
import timeit
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOG = logging.getLogger(__name__)


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    """Create a command line arguments parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=inspect.cleandoc(description),
    )
    parser.add_argument("--input", type=str, help="Input directory", default="")
    parser.add_argument("--output", type=str, help="Output path", default="")
    return parser


def main(args: Optional[argparse.Namespace] = None) -> Optional[int]:
    """Execute the command with the given arguments."""

    if not args:
        parser = build_parser()
        args, _ = parser.parse_known_args()
    start_time = timeit.default_timer()

    vargs = vars(args)
    for varg in vargs.keys():
        LOG.info("arg %s: %s", varg, vargs[varg])

    #
    # json format
    #
    # key = video
    # sub_key = language, category, event, description
    #
    metadata = json.load(open(args.input, "r"))
    LOG.info("Total: %s", len(metadata.keys()))

    csv_file = open(os.path.join(args.output, "test.csv"), "w", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["video_path", "category", "language", "event", "description"])
    write_ctr = 0
    for video_key in metadata.keys():
        # for sub_key in metadata[key].keys():

        try:
            lang = metadata[video_key]["language"]
            category = metadata[video_key]["category"]
            event = metadata[video_key]["event"]
            desc = metadata[video_key]["description"]

            if lang and category and event and desc:
                # if desc:
                if (
                    len(lang.strip()) > 0
                    and len(category.strip()) > 0
                    and len(event.strip()) > 0
                    and len(desc.strip()) > 1
                ):
                    # if len(desc.strip()) > 1:
                    line = desc
                    line = line.replace("\n", " ").replace("\r", "")
                    line = re.sub(" +", " ", line)
                    line = line.strip()
                    csv_writer.writerow([video_key, category, lang, event, line])
                    write_ctr += 1
        except:
            LOG.info("...ERROR %s %s", video_key, metadata[video_key])

    csv_file.close()

    LOG.info("...write %s", write_ctr)
    total_time = timeit.default_timer() - start_time
    LOG.info("Ran the script in %.3f seconds", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
