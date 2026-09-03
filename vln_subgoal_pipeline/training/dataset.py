import json
import random
import sys
from pathlib import Path

import config

sys.path.insert(0, str(config.PIPELINE_DIR))
from models.cosmos3_reasoner import Cosmos3Reasoner  # noqa: E402 — needs sys.path set first

from torch.utils.data import Dataset


def build_prompt(instruction):
    """Byte-for-byte the same content decompose() builds for a text-only call (image=None) —
    reusing Cosmos3Reasoner.SUBGOAL_SYSTEM_PROMPT as a class attribute needs no model load."""
    return (f"{Cosmos3Reasoner.SUBGOAL_SYSTEM_PROMPT}\n\n"
           f"Instruction: \"{instruction}\"\nDecompose into JSON subgoals:")


def build_target(subgoals):
    """Compact JSON array — matches the plain `response_text.strip()` fallback branch of
    Cosmos3Reasoner._parse_json_subgoals, so a correctly-formatted generation always parses."""
    return json.dumps(subgoals, ensure_ascii=False)


def load_examples(labels_path=config.LABELS_PATH):
    entries = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    return [
        {
            "run_id": entry["run_id"],
            "prompt": build_prompt(entry["instruction"]),
            "target": build_target(entry["subgoals"]),
        }
        for entry in entries
    ]


def split_examples(examples, val_fraction=config.VAL_FRACTION, seed=config.SEED):
    """Split by instruction (run_id) — each is independent here, unlike the frame-level
    navigation dataset where repeat takes of the same task had to stay together."""
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]  # train, val


class SubgoalDecompositionDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def build_train_val_datasets(labels_path=config.LABELS_PATH):
    examples = load_examples(labels_path)
    train_examples, val_examples = split_examples(examples)
    print(f"decompose() SFT examples: {len(train_examples)} train, {len(val_examples)} val "
         f"(of {len(examples)} total instructions)")
    return SubgoalDecompositionDataset(train_examples), SubgoalDecompositionDataset(val_examples)
