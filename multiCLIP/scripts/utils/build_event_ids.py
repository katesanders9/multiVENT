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

import pandas as pd


def main(input_csv, out):

    test_csv = pd.read_csv(input_csv)

    event_list = test_csv.event.unique()
    with open(os.path.join(out, "event.csv"), "w", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["id", "event"])
        for idx, event in enumerate(event_list):
            csv_writer.writerow([idx, event])

    category_list = test_csv.category.unique()
    with open(os.path.join(out, "category.csv"), "w", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["id", "category"])
        for idx, category in enumerate(category_list):
            csv_writer.writerow([idx, category])

    language_list = test_csv.language.unique()
    with open(os.path.join(out, "language.csv"), "w", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["id", "language"])
        for idx, language in enumerate(language_list):
            csv_writer.writerow([idx, language])

    return 0
