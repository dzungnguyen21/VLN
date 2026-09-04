from libs import *

# --- Where the existing convert/ pipeline's real-recording dataset lives, shared -----------
# Reused verbatim: HABITAT_DATASET_NAME below is just a different sub-path under the same
# OUT_DIR root, so the two datasets never collide on disk.
OUT_DIR = Path(env_str("OUT_DIR", "/home/longht16/longht/data/office/recorded/traj_data"))

# --- VLN-CE R2R + MP3D scan locations -------------------------------------------------------
# Defaults resolved against the repo root (this pipeline is run from inside convert_habitat/,
# like convert/ is run from inside convert/, but data/ lives at the repo root) — same pattern
# convert/config.py uses for its own DEFAULT_INSTRUCTION_LOG default.
REPO_ROOT = Path(__file__).resolve().parent.parent
HABITAT_R2R_DATA_ROOT = Path(env_str("HABITAT_R2R_DATA_ROOT", str(REPO_ROOT / "data/vln_ce/raw_data/r2r")))
HABITAT_SCENES_DIR = Path(env_str("HABITAT_SCENES_DIR", str(REPO_ROOT / "data/scene_data/")))
HABITAT_SPLIT = env_str("HABITAT_SPLIT", "train")

# Optional restriction, for smoke tests: comma list of scan ids, and/or a cap on episode count.
HABITAT_SCENES = env_list("HABITAT_SCENES", ())
HABITAT_LIMIT_EPISODES = env_int("HABITAT_LIMIT_EPISODES")

# --- Dataset identity --------------------------------------------------------------------
HABITAT_DATASET_NAME = env_str("HABITAT_DATASET_NAME", "habitat_r2r")
# Seeds the candidate shuffle/negative sampling in landmarks.py, for reproducible builds.
HABITAT_SEED = env_int("HABITAT_SEED", 0)
# episode_index = task_index * this + repeat_index. 1 = one deterministic oracle rollout per
# R2R episode, no augmentation. See convert_habitat/README.md.
HABITAT_REPEATS_PER_EPISODE = env_int("HABITAT_REPEATS_PER_EPISODE", 1)

# --- Camera geometry — matches benchmark/nav/vln_r2r.yaml's rgb_sensor.position=[0,1.25,0], --
# orientation=[0,0,0] (measured from the habitat-lab config itself, not invented): the sensor
# sits 1.25m above the floor, perfectly level. Drives setting_tag(), same <H>cm_<P>deg
# convention convert/episode.py uses — deliberately different from the real-recording
# dataset's tag (whatever RIG_HEIGHT_CM/RIG_PITCH_DEG resolve to there) so the two datasets'
# columns can never collide even if ever placed under one scene_dir by mistake.
HABITAT_RIG_HEIGHT_CM = env_float("HABITAT_RIG_HEIGHT_CM", 125.0)
HABITAT_RIG_PITCH_DEG = env_float("HABITAT_RIG_PITCH_DEG", 0.0)

# --- Sensor render resolution — rendered natively at this size, no crop/resize step needed ---
# (unlike convert/, which only crops because a physical ZED camera has a fixed native res).
HABITAT_OUT_W = env_int("HABITAT_OUT_W", 640)
HABITAT_OUT_H = env_int("HABITAT_OUT_H", 480)
HABITAT_HFOV = env_float("HABITAT_HFOV", 90.0)
HABITAT_JPEG_QUALITY = env_int("HABITAT_JPEG_QUALITY", 90)
HABITAT_MAX_DEPTH_M = env_float("HABITAT_MAX_DEPTH_M", 10.0)

# --- Task D: keyframe / subgoal selection ---------------------------------------------------
# NOT the same values as convert/'s MIN_MOVE=0.25/MIN_TURN_DEG=10 on purpose: Habitat's own
# atomic forward step is already 0.25m, so those defaults would keep every single sim step.
HABITAT_MIN_MOVE = env_float("HABITAT_MIN_MOVE", 0.5)
HABITAT_MIN_TURN_DEG = env_float("HABITAT_MIN_TURN_DEG", 30.0)
HABITAT_SUBGOAL_DIST = env_float("HABITAT_SUBGOAL_DIST", 2.5)
HABITAT_MAX_GOAL_FRAMES = env_int("HABITAT_MAX_GOAL_FRAMES", 20)
HABITAT_MIN_FRAMES = env_int("HABITAT_MIN_FRAMES", 4)

# --- Rollout / ShortestPathFollower ----------------------------------------------------------
HABITAT_GOAL_RADIUS_LEG = env_float("HABITAT_GOAL_RADIUS_LEG", 0.25)
HABITAT_GOAL_RADIUS_FINAL = env_float("HABITAT_GOAL_RADIUS_FINAL", 1.0)
HABITAT_SNAP_TOLERANCE_M = env_float("HABITAT_SNAP_TOLERANCE_M", 1.0)
HABITAT_MAX_STEPS_PER_LEG = env_int("HABITAT_MAX_STEPS_PER_LEG", 200)
HABITAT_MAX_EPISODE_STEPS = env_int("HABITAT_MAX_EPISODE_STEPS", 500)

# --- Task B/C: object-pointing + scene understanding ------------------------------------------
HABITAT_MIN_INSTANCE_PIXELS = env_int("HABITAT_MIN_INSTANCE_PIXELS", 100)
# Guards against MP3D's well-documented mesh-reconstruction artifacts around mirrors/glass —
# verified directly: a scan's "sink" instance mesh bled into a bathroom mirror's reflection
# and covered ~11% of frame in a thin, implausible vertical strip (spanning 78% of frame
# HEIGHT but only 31% of width — the tall-thin-touching-an-edge shape typical of this class
# of artifact, not a real object's silhouette). Every legitimately-sized object actually
# observed in that same frame stayed under 5%, so 8% has real margin either side — but this
# is one data point, not a calibrated statistic; treat it as a coarse safety net and spot-
# check a wider sample (README's "Scale check" verification step) before large-scale
# training, not as a guarantee this class of artifact is fully screened out.
HABITAT_MAX_INSTANCE_FRACTION = env_float("HABITAT_MAX_INSTANCE_FRACTION", 0.08)
HABITAT_EXCLUDE_CATEGORIES = set(env_list(
    "HABITAT_EXCLUDE_CATEGORIES",
    ("wall", "ceiling", "floor", "misc", "void", "unlabeled", "objects",
     "railing", "column", "beam", "board_panel"),
))
HABITAT_N_POSITIVES = env_int("HABITAT_N_POSITIVES", 3)
HABITAT_N_NEGATIVES = env_int("HABITAT_N_NEGATIVES", 3)
HABITAT_N_CANDIDATES_MAX = env_int("HABITAT_N_CANDIDATES_MAX", 6)
HABITAT_GUESS_MAX_LOOKAHEAD = env_int("HABITAT_GUESS_MAX_LOOKAHEAD", 15)
