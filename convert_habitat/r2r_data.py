"""Loads a VLN-CE R2R split (*.json.gz) and computes this pipeline's episode/task numbering.

Deliberately does NOT touch habitat/habitat_sim — it only parses the raw split JSON (gzip'd),
so the numbering scheme is a pure function of that file and needs no persisted id-assignment
log. rollout.py is the only module that drives the actual simulator; it looks up the matching
real habitat Episode object (needed to step the sim) by episode_id against these job dicts.
"""
import gzip

from libs import *
from config import HABITAT_LIMIT_EPISODES, HABITAT_R2R_DATA_ROOT, HABITAT_REPEATS_PER_EPISODE, \
    HABITAT_SCENES


def split_path(split):
    return HABITAT_R2R_DATA_ROOT / split / f"{split}.json.gz"


def load_split_episodes(split):
    """Raw episode dicts straight from the split file — goals/instruction stay plain dicts
    here (unlike habitat-lab's own parsed Episode objects), since this module never asks
    habitat-lab to parse anything."""
    path = split_path(split)
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["episodes"]


def scan_id_of(episode):
    """'mp3d/<scan>/<scan>.glb' -> '<scan>'."""
    return Path(episode["scene_id"]).parent.name


def load_jobs(split):
    """One job dict per R2R episode, in this pipeline's deterministic numbering.

    task_index = dense rank after a stable sort on (scene_id, episode_id) over the WHOLE
    split file — episode_id is NOT globally contiguous in train.json.gz (verified: 10,819
    episodes, ids run 1..10,837 with gaps), so "task_index = episode_id - 1" would leave
    holes. One R2R episode = one instruction = one task_index (verified: val_seen has 259
    unique trajectory_id x 3 instructions =~ 778 episodes) — the correct analogue of
    convert/'s "one instruction, one run" unit is the episode, not the trajectory_id.
    """
    episodes = load_split_episodes(split)
    ordered = sorted(episodes, key=lambda ep: (ep["scene_id"], ep["episode_id"]))

    if HABITAT_SCENES:
        wanted_scans = set(HABITAT_SCENES)
        ordered = [ep for ep in ordered if scan_id_of(ep) in wanted_scans]

    jobs = []
    for task_index, episode in enumerate(ordered):
        repeat_index = 0
        jobs.append({
            "episode_id": str(episode["episode_id"]),
            "scan_id": scan_id_of(episode),
            "scene_id": episode["scene_id"],
            "trajectory_id": episode.get("trajectory_id"),
            "instruction": episode["instruction"]["instruction_text"].strip(),
            "task_index": task_index,
            "repeat_index": repeat_index,
            "episode_index": task_index * HABITAT_REPEATS_PER_EPISODE + repeat_index,
        })

    if HABITAT_LIMIT_EPISODES is not None:
        jobs = jobs[:HABITAT_LIMIT_EPISODES]

    return jobs


def jobs_by_scan(jobs):
    by_scan = {}
    for job in jobs:
        by_scan.setdefault(job["scan_id"], []).append(job)
    return by_scan


def episode_info_by_index(jobs):
    """episode_index -> {instruction, task_index, repeat_index, run_id} — same shape
    convert/episode.py's rebuild_meta expects. run_id is the R2R episode_id here (there's no
    ROS "run" concept), kept only for the meta/episodes.jsonl 'run_id' field's provenance."""
    return {
        job["episode_index"]: {
            "instruction": job["instruction"],
            "task_index": job["task_index"],
            "repeat_index": job["repeat_index"],
            "run_id": job["episode_id"],
        }
        for job in jobs
        if job["instruction"]
    }
