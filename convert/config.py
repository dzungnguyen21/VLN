from libs import *


DATA_ROOT = Path(env_str("DATA_ROOT", "/home/longht16/longht/data/office/recorded"))
RECORD_DIR = Path(env_str("RECORD_DIR", str(DATA_ROOT / "record")))
RESULTS_DIR = Path(env_str("RESULTS_DIR", str(DATA_ROOT / "results")))
OUT_DIR = Path(env_str("OUT_DIR", str(DATA_ROOT / "traj_data")))

DEFAULT_INSTRUCTION_LOG = Path(__file__).resolve().parent.parent / "instruction_log.json"
INSTRUCTION_LOG = Path(env_str("INSTRUCTION_LOG", str(DEFAULT_INSTRUCTION_LOG)))

DATASET_NAME = env_str("DATASET_NAME", "office_recording")
SCENE = env_str("SCENE", "scene_0001")
TAKES_PER_INSTRUCTION = env_int("TAKES_PER_INSTRUCTION", 5)
SCALE_OVERRIDES = env_json("SCALE_OVERRIDES", {})

TRACK_FPS = env_float("TRACK_FPS", 15.0)

RIG_HEIGHT_CM = env_float("RIG_HEIGHT_CM")
RIG_PITCH_DEG = env_float("RIG_PITCH_DEG")

OUT_WIDTH, OUT_HEIGHT = env_int("OUT_W", 640), env_int("OUT_H", 480)
JPEG_QUALITY = env_int("JPEG_QUALITY", 90)

TARGET_FPS = env_float("TARGET_FPS", 2.0)
MIN_MOVE = env_float("MIN_MOVE", 0.25)
MIN_TURN_DEG = env_float("MIN_TURN_DEG", 10.0)
TOL_MS = env_float("TOL_MS", 60.0)

SUBGOAL_DIST = env_float("SUBGOAL_DIST", 2.5)
MAX_GOAL_FRAMES = env_int("MAX_GOAL_FRAMES", 30)
MIN_FRAMES = env_int("MIN_FRAMES", 4)

AUTO_SCALE = env_bool("AUTO_SCALE", True)
MAX_SPREAD = env_float("MAX_SPREAD", 0.10)
SPEED_MIN = env_float("SPEED_MIN", 0.5)
SPEED_MAX = env_float("SPEED_MAX", 2.0)
