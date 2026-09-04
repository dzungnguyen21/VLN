# `convert_habitat/` — VLN-CE R2R + Habitat-Sim → training data

Builds two things from Habitat-Sim rollouts of VLN-CE R2R episodes on MP3D scans, at far
larger scale and with ground-truth (not pseudo-labeled) precision than the real-recording
`convert/` pipeline can offer:

- **Task D — pixel-goal**: the same LeRobot-v2.1-style dataset schema `convert/` produces
  (`action`, `pose.<tag>`, `goal.<tag>`, `relative_goal_frame_id.<tag>`), loadable by
  `finetuning/data.py`'s `PixelGoalTextDataset`/`ActionFrameDataset` **unmodified**.
- **Task B/C — object-pointing + scene understanding**: a new JSONL manifest matching
  `Cosmos3Reasoner.DETECT_PROMPT_TEMPLATE`'s exact response schema (`visible[]`,
  `guess_pixel`/`guess_label`/`guess_confidence`, `current_location`), so a checkpoint
  trained on it is a direct fit for the deployed `detect_landmarks()` — the actual capability
  this pipeline exists to improve.

Every MP3D scan under `HABITAT_SCENES_DIR` has a ground-truth `_semantic.ply`/`.house`, so
Task B/C's object/room labels come from Habitat's semantic sensor directly — no detector, no
pseudo-labeling error.

## Usage

```bash
cd convert_habitat
python main.py build                 # builds HABITAT_SPLIT (default: train)
python main.py build --force         # redo work already on disk

# One-off overrides (same convention as convert/ — real env vars win over .env):
HABITAT_SPLIT=val_seen HABITAT_SCENES=17DRP5sb8fy HABITAT_LIMIT_EPISODES=5 python main.py build
```

There's no separate `--split`/`--scenes`/`--limit` CLI flags: those are read once at
`config.py` import time from `.env`/env vars (`HABITAT_SPLIT`, `HABITAT_SCENES`,
`HABITAT_LIMIT_EPISODES`), matching `convert/`'s own established pattern
(`TARGET_FPS=3 python main.py build`) rather than introducing a second config surface.
`--force` is the one true "how to run this batch" flag, mirroring `convert/main.py`'s.

Without `--force`, an episode whose Task D parquet already exists is skipped entirely (Task D
+ Task B/C are always built together per episode, so one skip check covers both).

## Output layout

```
<OUT_DIR>/<HABITAT_DATASET_NAME>/<HABITAT_SPLIT>/
├── data/chunk-000/episode_000032.parquet          # Task D, convert/'s exact schema
├── videos/chunk-000/
│   ├── observation.images.rgb.125cm_0deg/episode_000032_0.jpg
│   └── observation.images.depth.125cm_0deg/episode_000032_0.png   # REAL depth, mm, uint16
├── detect/chunk-000/episode_000032.jsonl          # Task B/C, new schema (see below)
└── meta/
    ├── info.json / episodes.jsonl / tasks.jsonl / episodes_stats.jsonl   # convert/'s schema
```

One `SCENE_DIR` per R2R split, spanning every scan in it (not one per scan) — this is what
lets `finetuning/.env` point at Habitat data with just `DATASET_NAME=habitat_r2r`,
`SCENE=train`, `RIG_HEIGHT_CM=125`, `RIG_PITCH_DEG=0`, no code changes.

### Task B/C row schema

```json
{
  "episode_index": 12345, "frame_index": 7,
  "image_path": "videos/chunk-000/observation.images.rgb.125cm_0deg/episode_012345_7.jpg",
  "scan_id": "17DRP5sb8fy", "instruction": "...", "guided_direction": "None",
  "candidates": ["cushion", "sofa", "lamp", "plant", "picture", "chest_of_drawers"],
  "visible": [{"landmark": "sofa", "pixel_norm": [612, 430]}],
  "guess_pixel_norm": [340, 800], "guess_label": "toward the hallway",
  "guess_confidence": 0.85, "current_location": "living room"
}
```

`pixel_norm`/`guess_pixel_norm` are `[y, x]` on a 0–1000 grid (matches
`DETECT_PROMPT_TEMPLATE`'s convention), stored as raw ints — not the literal `"[y, x]"`
string the live model emits; that formatting is a one-line step at training-collation time.
`guided_direction` is always `"None"` for now (Task A, see below, isn't built yet).

## Numbering

`task_index` = dense rank after a stable sort on `(scene_id, episode_id)` over the whole
split file — `episode_id` is **not** globally contiguous in `train.json.gz` (verified: 10,819
episodes, ids run 1..10,837 with gaps), so `task_index = episode_id - 1` would leave holes.
One R2R episode = one instruction = one `task_index` (verified: `val_seen` has 259 unique
`trajectory_id` x 3 instructions ≈ 778 episodes). `episode_index = task_index *
HABITAT_REPEATS_PER_EPISODE + repeat_index` — multiplier defaults to 1 (no augmentation).

## Design notes worth knowing before touching this code

- **Goal projection uses the real 3D floor point, not `(x, y, 0)`.** `convert/episode.py`
  assumes a flat floor at world `z=0` (valid there — height is added only to the synthesized
  camera pose). Habitat is Y-up and every kept frame's `floor_position` (the AGENT's
  position, not the elevated RGB sensor's) is already the true floor height at that point —
  `episode.py`'s `build_labels` projects `frames[goal_frame]["floor_position"]` directly.
  Zeroing a coordinate the way `convert/` does would silently mis-project every goal on any
  scene where the floor isn't exactly at the world origin's height (virtually always).
- **One `habitat.Env` for the whole split**, not one per scan. habitat-lab already reloads
  the scene mesh transparently on `env.reset()` when the next episode's `scene_id` differs
  (verified directly) — restricting to one episode is just
  `env.episode_iterator = iter([episode])`. This avoided relying on the untested
  `habitat.dataset.content_scenes` Hydra override the original design sketch considered.
- **The rollout is a generator** (`rollout.walk_episode`), consumed lazily by
  `episode.select_keyframes` — holding a whole episode's raw per-step rgb/depth/semantic
  frames in memory before downsampling could reach ~500MB for a single long episode.
- **`ShortestPathFollower.get_next_action()` returns STOP both on success and, silently, on
  an internal `GreedyFollowerError`** (verified). `rollout.walk_episode` never trusts an early
  STOP without checking the actual distance to the target first.

## Failure modes

| Message | Cause | Scope |
|---|---|---|
| `waypoint N snaps X m off the navmesh` | R2R annotation point not navigable in this scan | Episode rejected |
| `leg N exceeded HABITAT_MAX_STEPS_PER_LEG` | Follower stuck/oscillating | Episode rejected |
| `episode exceeded HABITAT_MAX_EPISODE_STEPS` | Global step budget hit mid-leg | Episode rejected |
| `follower returned STOP early (dist > goal_radius)` | `GreedyFollowerError` swallowed by `stop_on_error=True` | Episode rejected |
| `follower found no path to the target` | `get_next_action` returned `None` | Episode rejected |
| `only N keyframe(s), need HABITAT_MIN_FRAMES` | Degenerate short path | Episode rejected |
| (no Task B/C rows written for an episode) | Scan's semantic vocabulary too small / nothing visible anywhere | Task D still built; not an error |

A single bad episode never aborts the batch — one `try/except ValueError` per job, same as
`convert/main.py`.

## Task A (subgoal decomposition) — not built here

FGR2R's `path_id` (`vln_subgoal_pipeline/FGR2R/FGR2R_<split>.json`) is the same id space as
VLN-CE's `trajectory_id`, which `r2r_data.py`'s job dicts already carry. A future phase would
join on that, use each sub-instruction's start/end `reference_path` indices to slice this
pipeline's already-rolled-out trajectory into per-subgoal chunks, and emit rows shaped like
`Cosmos3Reasoner.SUBGOAL_SYSTEM_PROMPT`'s JSON schema — reusing this pipeline's rollout/
keyframe machinery entirely, adding only the FGR2R join and a labeling policy for
`target_location`/`guided_direction` (FGR2R doesn't hand those out directly).

## Requires

The `habitat` conda env (`habitat-lab` + `habitat-sim`, already set up in this repo — see
repo-root `environment_habitat.yml`/`requirements_habitat.txt`), plus `numpy`, `pyarrow`,
`pillow`, `numpy-quaternion`.
