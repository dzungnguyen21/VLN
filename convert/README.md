# `convert/` — ZED rosbags → LeRobot v2.1 dataset

A self-contained, `.env`-driven pipeline that turns ROS 2 recordings (`.db3`) from a single
forward-facing ZED camera, plus `instruction_log.json`, into a **LeRobot v2.1** dataset for
language-conditioned navigation (System-2 / VLN) training.

```
instruction_log.json ─┐
                      ├─► stage 1: track ─► results/<bag>/traj_head_front.txt
record/<bag>/*.db3 ───┘     (cuVSLAM mono)          │  poses, UNKNOWN scale
                      ├─────────────────────────────►├─► stage 2: build
                                                        ─► traj_data/<name>/<scene>/
                                                           (parquet + jpg + png + meta/)
```

The two stages are split on purpose: stage 1 needs CUDA and is slow; stage 2 is pure CPU and
gets re-run often while tuning labels. Each writes its own artifacts, so either can be redone
alone.

## Naming contract

A bag folder must be named `zed_record_run_<run>_<take>`, and `<run>` must have a matching
`id: "run_<run>"` entry in `instruction_log.json`. One run = one instruction filmed
`TAKES_PER_INSTRUCTION` times.

```
task_index    = run  - 1
repeat_index  = take - 1
episode_index = task_index * TAKES_PER_INSTRUCTION + repeat_index
```

So `zed_record_run_07_03` is instruction 7, take 3, episode `6 * 5 + 2 = 32`. Bags that do not
match the pattern, have no instruction, or whose take is out of range are silently skipped.

## Usage

```bash
python main.py track            # stage 1 only: bags -> trajectories
python main.py build            # stage 2 only: trajectories -> episodes
python main.py all              # both

python main.py all --runs 1,2,7 # restrict to these runs
python main.py all --force      # redo work already on disk
```

Run from inside `convert/` (the modules import each other flat, e.g. `from config import *`).

Without `--force`, stage 1 skips a bag whose `traj_head_front.txt` exists and stage 2 skips an
episode whose parquet exists. Exit code is `1` if anything failed, `0` otherwise; a single bad
bag never aborts the batch.

## Configuration

Every knob lives in `../.env`, read once at import by `libs.load_env()`. Real environment
variables take precedence (`load_env` uses `setdefault`), so `TARGET_FPS=3 python main.py build`
works for a one-off.

**Must be set — no default:**

| Key | Meaning |
|---|---|
| `RIG_HEIGHT_CM` | Camera optical centre above the floor, in cm |
| `RIG_PITCH_DEG` | Camera downward tilt, in degrees |

These cannot be recovered from the bags (no `/tf`). Every goal label is a floor point projected
through them, so a wrong value builds cleanly, passes every check, and trains wrong. They are
also what turns auto-scale's unitless height into metres.

**Paths:** `DATA_ROOT` (and the derived `RECORD_DIR`, `RESULTS_DIR`, `OUT_DIR`),
`INSTRUCTION_LOG`.

**Dataset identity:** `DATASET_NAME`, `SCENE`, `TAKES_PER_INSTRUCTION`. Changing
`TAKES_PER_INSTRUCTION` after the first run renumbers every episode while the parquets already
written stay put — don't.

**Stage 1:** `TRACK_FPS` (default 15) — frames are downsampled to this rate before tracking;
downsampling is skipped if the bag is already slower.

**Stage 2:** `OUT_W`/`OUT_H` (640×480), `JPEG_QUALITY`, `TARGET_FPS` (2.0), `MIN_MOVE` (0.25 m),
`MIN_TURN_DEG` (10°), `TOL_MS` (60 ms pose↔image match window), `SUBGOAL_DIST` (2.5 m),
`MAX_GOAL_FRAMES` (30), `MIN_FRAMES` (4).

**Scale:** `AUTO_SCALE`, `MAX_SPREAD`, `SPEED_MIN`/`SPEED_MAX` sanity band, and
`SCALE_OVERRIDES` — per-bag pins as one-line JSON:

```
SCALE_OVERRIDES={"zed_record_run_07_02": {"path_length": 24.5}}
```

## Metric scale — the one thing that is not automatic

Mono VO recovers translation only up to an unknown multiplier, and that multiplier is
**independent per recording**, so no single global number works. `resolve_scale` tries, in order:

1. `SCALE_OVERRIDES[bag]["traj_scale"]` — direct multiplier.
2. `SCALE_OVERRIDES[bag]["path_length"]` — real metres walked, divided by raw path length.
3. Auto-scale (`scale.py`): matches a virtual floor grid between frame pairs across ~90
   log-spaced candidate heights, picks the height whose reprojection cost is minimal, and
   converts via `RIG_HEIGHT_CM`.
4. Otherwise **raise**. Defaulting to 1.0 would put every label off by a constant while every
   check still passed.

Auto-scale needs a textured, visible floor. It fails loudly (fewer than 5 strong frame pairs)
and warns when the median-absolute spread exceeds `MAX_SPREAD` — pin those bags by hand. A
second guard warns when the implied walking speed falls outside `SPEED_MIN`–`SPEED_MAX`.

## Labels

Keyframes are poses on the `TARGET_FPS` grid that have an image within `TOL_MS` **and** moved
`MIN_MOVE` metres or turned `MIN_TURN_DEG` degrees since the last kept frame.

- **`action`** — `0 STOP, 1 FORWARD, 2 LEFT, 3 RIGHT`, from the yaw delta. `action[i]` is what
  was done to *arrive at* frame `i`; frame 0 is `STOP` because the loader shifts this column
  left and discards it.
- **`goal.<tag>`** — pixel `(u, v)` of the next subgoal's floor point, projected through the
  synthesized camera pose using post-crop/resize intrinsics. `(-1, -1)` when there is no valid
  goal or it falls off-image.
- **`relative_goal_frame_id.<tag>`** — how many frames ahead that subgoal is. Subgoals are every
  FORWARD→turn transition, every `SUBGOAL_DIST` metres, and the last frame. Only goals at least
  `MIN_GOAL_K = 3` and at most `MAX_GOAL_FRAMES` frames ahead are kept.
- **`pose.<tag>`** — a 4×4 camera-to-world matrix, synthesized from `(x, y, yaw)` plus the rig
  height and pitch rather than taken raw from VO.

`<tag>` is `<H>cm_<P>deg`, e.g. `pose.120cm_15deg`, derived from the rig settings so datasets
built with different mounts never silently mix.

`SUBGOAL_DIST` is tied to `TARGET_FPS`: at 2 Hz and ~1.2 m/s one frame covers ~0.6 m, so a
subgoal distance under ~1.8 m puts every goal below the loader's `k >= 3` filter and the dataset
comes out empty.

## Output layout

```
traj_data/<DATASET_NAME>/<SCENE>/
├── data/chunk-000/episode_000032.parquet
├── videos/chunk-000/
│   ├── observation.images.rgb.<tag>/episode_000032_0.jpg
│   └── observation.images.depth.<tag>/episode_000032_0.png
└── meta/
    ├── info.json            # v2.1 header: counts, fps, feature schema
    ├── episodes.jsonl       # per episode: instruction, length, task/repeat index, run_id
    ├── tasks.jsonl          # task_index -> instruction
    └── episodes_stats.jsonl # min/max/mean/std for action, goal, relative_goal_frame_id
```

The depth PNGs are 16-bit **all-zero** images: this robot has one camera and no depth sensor,
but the loader indexes both streams and crashes on a missing path. Chunk size is 1000 episodes.

`meta/` is rebuilt from the parquets on disk after every run, never from in-memory state — so a
bag recorded weeks later can be appended and the metadata stays correct. `rebuild_meta` raises
if a parquet on disk has no instruction in the log, or if one `task_index` maps to two different
instructions.

Two dtypes in the parquet are load-bearing: `action` and `relative_goal_frame_id.<tag>` are
int64 **scalars**, not `(1,)` lists (as lists the loader raises `unhashable type: numpy.ndarray`),
and `pose.<tag>` stays a nested 4×4 because the loader does not reshape.

## Modules

| File | Responsibility |
|---|---|
| `main.py` | CLI, job discovery from folder names + instruction log, batch loops for both stages |
| `config.py` | Every tunable, read from `../.env` with defaults |
| `libs.py` | Shared imports, `.env` loader, typed `env_*` getters |
| `bag.py` | Read-only SQLite access to `.db3` + hand-written CDR decoder for `CompressedImage` and `CameraInfo` |
| `track.py` | Stage 1: cuVSLAM mono tracker → `traj_head_front.txt` |
| `scale.py` | Auto-scale: floor-grid photometric matching → camera height in trajectory units |
| `geometry.py` | Quaternions, pose synthesis, crop/resize intrinsics, time-nearest lookup, resampling, trajectory I/O |
| `episode.py` | Stage 2: keyframe selection, action/goal labels, image + parquet writing, `meta/` rebuild |

`bag.py` decodes CDR by hand, so the `rosbags` package is not needed. Only `track.py` and
`scale.py` import `cv2`, and only `track.py` imports `cuvslam` — but `main.py` imports `track`
unconditionally, so stage 2 currently still needs both installed.

Requires a `cuvslam`-capable environment plus `numpy`, `pyarrow`, `pillow`, `opencv-python`.
See the repo-root `setup_env.sh`.

## Failure modes worth recognising

| Message | Cause |
|---|---|
| `set RIG_HEIGHT_CM and RIG_PITCH_DEG` | The mount was never measured. Nothing can be produced. |
| `the trajectory never leaves the origin` | Mono VO never initialised — no motion, or no texture. |
| `<folder> is a split recording` | Multiple `.db3` in one bag folder; only single-file bags are read. |
| `images are not rectified (D != 0)` | Tracking is fed the rectified topic only. |
| `auto-scale failed: only N of M frame pairs` | Featureless floor — pin the bag in `SCALE_OVERRIDES`. |
| `only N keyframe(s), need MIN_FRAMES` | Too little motion, or `TOL_MS` too tight to pair poses with images. |
| `no metric scale for this bag` | `AUTO_SCALE=false` and no override for that bag. |
