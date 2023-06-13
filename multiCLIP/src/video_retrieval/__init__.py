"""Populates the top-level project namespace."""
import os
from pathlib import Path

PACKAGE_NAME = Path(__file__).resolve().parent.name
UTILS_DIR = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DIR = os.path.dirname(UTILS_DIR)
