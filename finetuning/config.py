from libs import *

# --- Where the dataset built by convert/ lives -------------------------------------------
# Same identity keys as convert/config.py — point these at the same .env values used to
# build the dataset, or the <tag> on pose.<tag>/goal.<tag> below won't match what's on disk.
DATA_ROOT = Path(env_str("DATA_ROOT", "/home/longht16/longht/data/office/recorded"))
OUT_DIR = Path(env_str("OUT_DIR", str(DATA_ROOT / "traj_data")))
DATASET_NAME = env_str("DATASET_NAME", "office_recording")
SCENE = env_str("SCENE", "scene_0001")
SCENE_DIR = OUT_DIR / DATASET_NAME / SCENE

RIG_HEIGHT_CM = env_float("RIG_HEIGHT_CM")
RIG_PITCH_DEG = env_float("RIG_PITCH_DEG")

# --- Train/val split -----------------------------------------------------------------------
# Split by task_index, never by episode_index: a task's repeats (takes) must land on the same
# side of the split, or validation leaks near-duplicate trajectories of a training instruction.
VAL_FRACTION = env_float("VAL_FRACTION", 0.1)
SEED = env_int("SEED", 0)
NUM_WORKERS = env_int("NUM_WORKERS", 4)
DEVICE = env_str("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_ROOT = Path(env_str("CHECKPOINT_ROOT", str(Path(__file__).resolve().parent.parent
                                                       / "checkpoints")))

# =============================================================================================
# System 2 — Cosmos3-Edge Reasoner Tower, fine-tuned as a text-based pixel-goal VLM.
#
# This reproduces the I/O *contract* InternVLA-N1's own System 2 uses (a Qwen2.5-VL model
# prompted to emit the next waypoint's pixel coordinates as text, parsed back out with a
# regex) so a checkpoint trained here stays a drop-in replacement if the InternVLA-N1 pipeline
# is ever wired up around it later — see internnav/model/basemodel/internvla_n1/internvla_n1_policy.py
# upstream. We do NOT reproduce its `generate_latents` -> DiT-conditioning path: that consumes
# continuous trajectory data this dataset doesn't have (only discrete STOP/FORWARD/LEFT/RIGHT +
# pixel goal). System 1 here is a separate, lightweight controller trained directly on that.
# =============================================================================================
COSMOS3_EDGE_CHECKPOINT = env_str("COSMOS3_EDGE_CHECKPOINT", "nvidia/Cosmos3-Edge")
SYSTEM2_DTYPE = env_str("SYSTEM2_DTYPE", "bfloat16")  # float32 | float16 | bfloat16

SYSTEM2_PROMPT_TEMPLATE = env_str(
    "SYSTEM2_PROMPT_TEMPLATE",
    "You are an autonomous navigation assistant. Your task is to {instruction}. Where should "
    "you go next to stay on track? Please output the next waypoint's coordinates in the image."
)
# Our own convention: target text is the literal "u,v" pixel pair from the goal.<tag> column
# (the same (u, v) project_to_pixel/build_labels writes in convert/episode.py), parsed back
# with a plain int regex — see system2_infer.py. This is NOT necessarily the same digit order
# InternVLA-N1's own regex expects; verify before treating a checkpoint as literally drop-in.
SYSTEM2_TARGET_TEMPLATE = env_str("SYSTEM2_TARGET_TEMPLATE", "{u},{v}")

# --- Multi-task mix ------------------------------------------------------------------------
# One LoRA run supervises several text tasks over the SAME frames and the same task-level
# train/val split. Names must be keys of system2_train.TASK_SPECS. Dropping back to a single
# task is just SYSTEM2_TASKS=pixel_goal — the trainer takes no other shape.
SYSTEM2_TASKS = env_list("SYSTEM2_TASKS", ["pixel_goal", "action"])
# Sampling share per TASK, not per sample: weights are divided by each task's own length before
# sampling, so the mix stays at these proportions no matter how the frame counts differ. Empty
# (the default) means an equal share for every task in SYSTEM2_TASKS.
SYSTEM2_TASK_WEIGHTS = [float(weight) for weight in env_list("SYSTEM2_TASK_WEIGHTS", [])] \
    or [1.0] * len(SYSTEM2_TASKS)

# The action task: same frame, but the goal pixel is GIVEN in the prompt and the model names
# the discrete action instead. This is the same (frame, goal) -> action mapping System 1 learns
# from scratch — training it here as text too is what makes the mix multi-task rather than two
# unrelated heads, and gives a check on whether the Reasoner Tower can drive itself end to end.
SYSTEM2_ACTION_PROMPT_TEMPLATE = env_str(
    "SYSTEM2_ACTION_PROMPT_TEMPLATE",
    "You are an autonomous navigation assistant. Your task is to {instruction}. The next "
    "waypoint is at pixel {u},{v} in the image. Which single action should you take now? "
    "Answer with exactly one of: STOP, FORWARD, LEFT, RIGHT."
)
SYSTEM2_ACTION_TARGET_TEMPLATE = env_str("SYSTEM2_ACTION_TARGET_TEMPLATE", "{action}")
# Index order is load-bearing: it must match data.STOP/FORWARD/LEFT/RIGHT (0/1/2/3), which is
# the order convert/episode.py's discretize_actions writes into the `action` column.
SYSTEM2_ACTION_NAMES = env_list("SYSTEM2_ACTION_NAMES", ["STOP", "FORWARD", "LEFT", "RIGHT"])

SYSTEM2_LORA_R = env_int("SYSTEM2_LORA_R", 16)
SYSTEM2_LORA_ALPHA = env_int("SYSTEM2_LORA_ALPHA", 32)
SYSTEM2_LORA_DROPOUT = env_float("SYSTEM2_LORA_DROPOUT", 0.05)
# A regex over FULL module names (peft accepts target_modules as a single regex string, not
# just a list of substrings) — verified directly against the real checkpoint's
# named_modules(): text-side attention is model.language_model.layers.N.self_attn.{q,k,v,o}_proj,
# but the vision encoder ALSO has q_proj/k_proj/v_proj (just out_proj, not o_proj) — a plain
# substring list like the old ["q_proj","k_proj","v_proj","o_proj"] silently LoRA-wraps the
# vision encoder's Q/K/V too, since peft matches by name suffix across the WHOLE model. This
# regex is scoped to language_model only; confirmed via get_peft_model + named_modules() that
# it wraps exactly the 28 layers' 4 attention projections (112 modules) and nothing in .visual.
SYSTEM2_LORA_TARGET_MODULES = env_str(
    "SYSTEM2_LORA_TARGET_MODULES",
    r"^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
)

SYSTEM2_LEARNING_RATE = env_float("SYSTEM2_LEARNING_RATE", 1e-4)
SYSTEM2_WEIGHT_DECAY = env_float("SYSTEM2_WEIGHT_DECAY", 0.01)
SYSTEM2_NUM_EPOCHS = env_int("SYSTEM2_NUM_EPOCHS", 5)
SYSTEM2_BATCH_SIZE = env_int("SYSTEM2_BATCH_SIZE", 4)
SYSTEM2_GRAD_ACCUM_STEPS = env_int("SYSTEM2_GRAD_ACCUM_STEPS", 8)
SYSTEM2_WARMUP_RATIO = env_float("SYSTEM2_WARMUP_RATIO", 0.03)
SYSTEM2_MAX_GRAD_NORM = env_float("SYSTEM2_MAX_GRAD_NORM", 1.0)
SYSTEM2_MAX_NEW_TOKENS = env_int("SYSTEM2_MAX_NEW_TOKENS", 16)  # inference only — "123,45" is short

SYSTEM2_LOG_EVERY_N_STEPS = env_int("SYSTEM2_LOG_EVERY_N_STEPS", 20)
SYSTEM2_EVAL_EVERY_N_EPOCHS = env_int("SYSTEM2_EVAL_EVERY_N_EPOCHS", 1)
SYSTEM2_CHECKPOINT_DIR = CHECKPOINT_ROOT / env_str("SYSTEM2_RUN_NAME", "system2_cosmos_lora")

# =============================================================================================
# System 1 — small discrete-action controller: (rgb frame, predicted pixel goal) -> action.
# Deliberately not a VLM: trained from scratch on this dataset's own action labels, decoupled
# from System 2's weights so the two can be trained, versioned, and swapped independently.
# =============================================================================================
SYSTEM1_IMAGE_SIZE = env_int("SYSTEM1_IMAGE_SIZE", 224)
SYSTEM1_BACKBONE = env_str("SYSTEM1_BACKBONE", "resnet18")  # any torchvision backbone name
SYSTEM1_IMAGE_EMBED_DIM = env_int("SYSTEM1_IMAGE_EMBED_DIM", 512)
SYSTEM1_GOAL_EMBED_DIM = env_int("SYSTEM1_GOAL_EMBED_DIM", 64)
SYSTEM1_HIDDEN_DIM = env_int("SYSTEM1_HIDDEN_DIM", 256)
SYSTEM1_N_ACTIONS = env_int("SYSTEM1_N_ACTIONS", 4)  # STOP, FORWARD, LEFT, RIGHT

SYSTEM1_LEARNING_RATE = env_float("SYSTEM1_LEARNING_RATE", 3e-4)
SYSTEM1_WEIGHT_DECAY = env_float("SYSTEM1_WEIGHT_DECAY", 1e-4)
SYSTEM1_NUM_EPOCHS = env_int("SYSTEM1_NUM_EPOCHS", 30)
SYSTEM1_BATCH_SIZE = env_int("SYSTEM1_BATCH_SIZE", 64)

SYSTEM1_LOG_EVERY_N_STEPS = env_int("SYSTEM1_LOG_EVERY_N_STEPS", 50)
SYSTEM1_EVAL_EVERY_N_EPOCHS = env_int("SYSTEM1_EVAL_EVERY_N_EPOCHS", 1)
SYSTEM1_CHECKPOINT_DIR = CHECKPOINT_ROOT / env_str("SYSTEM1_RUN_NAME", "system1_controller")
