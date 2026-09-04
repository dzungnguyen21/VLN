import argparse
import json
import math
import os
import re
import sqlite3
import struct
import sys
import traceback
from io import BytesIO
from pathlib import Path


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image



ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path=ENV_PATH):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()


def env_str(key, default=None):
    value = os.environ.get(key, "").strip()
    return value if value else default


def env_float(key, default=None):
    value = env_str(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{key} in .env must be a number, got {value!r}")


def env_int(key, default=None):
    value = env_str(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{key} in .env must be a whole number, got {value!r}")


def env_bool(key, default=False):
    value = env_str(key)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


def env_json(key, default=None):
    value = env_str(key)
    if value is None:
        return {} if default is None else default
    try:
        return json.loads(value)
    except ValueError as exception:
        raise ValueError(f"{key} in .env must be valid JSON on one line ({exception})")
