import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Same root .env as convert/ — DATASET_NAME, SCENE, RIG_HEIGHT_CM etc. must match whatever
# was used to build the dataset, since the <tag> on pose/goal columns is derived from the rig.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Lets finetuning/ modules import from sibling packages (e.g. system2_train.py importing
# Cosmos3Reasoner.DETECT_PROMPT_TEMPLATE from vln_subgoal_pipeline/models/cosmos3_reasoner.py,
# to train against the LITERAL deployed prompt rather than a copy that can drift). Same
# pattern vln_subgoal_pipeline/test/policy.py already uses for its own cross-directory import.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def env_list(key, default=()):
    value = env_str(key)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
