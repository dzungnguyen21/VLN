from pathlib import Path
import os

TRAINING_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TRAINING_DIR.parent
LABELS_PATH = Path(os.environ.get("SUBGOAL_LABELS_PATH", str(TRAINING_DIR / "subgoal_labels.json")))

# Matches configs/pipeline_config.yaml's models.cosmos3 section — keep in sync so the
# fine-tuned adapter is trained against the exact checkpoint the live pipeline loads.
MODEL_ID = os.environ.get("COSMOS3_MODEL_ID", "nvidia/Cosmos3-Edge")
DTYPE = os.environ.get("COSMOS3_DTYPE", "bfloat16")
DEVICE = os.environ.get("COSMOS3_DEVICE", "cuda")

# 2.4B bf16 weights (~4.9GB) alone leave too little headroom for activations on an 8GB GPU —
# confirmed by a real OOM during a smoke-test forward/backward pass at batch_size=2. 4-bit
# (QLoRA) loading cuts the resident weight footprint to ~1.2GB; disable only on a bigger GPU.
LOAD_IN_4BIT = os.environ.get("REASONER_LOAD_IN_4BIT", "true").lower() in ("1", "true", "yes")
GRADIENT_CHECKPOINTING = os.environ.get("REASONER_GRADIENT_CHECKPOINTING", "true").lower() in (
    "1", "true", "yes",
)

LORA_R = int(os.environ.get("REASONER_LORA_R", 16))
LORA_ALPHA = int(os.environ.get("REASONER_LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("REASONER_LORA_DROPOUT", 0.05))
# Placeholder — verify against model.named_modules() on the actual loaded Reasoner Tower
# before training; wrong names make peft silently attach zero adapters instead of raising.
LORA_TARGET_MODULES = os.environ.get(
    "REASONER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"
).split(",")

VAL_FRACTION = float(os.environ.get("REASONER_VAL_FRACTION", 0.1))
SEED = int(os.environ.get("REASONER_SEED", 0))

LEARNING_RATE = float(os.environ.get("REASONER_LEARNING_RATE", 1e-4))
WEIGHT_DECAY = float(os.environ.get("REASONER_WEIGHT_DECAY", 0.01))
NUM_EPOCHS = int(os.environ.get("REASONER_NUM_EPOCHS", 20))
# batch_size=1: even with 4-bit weights, an 8GB GPU has little headroom left for a 2.4B model's
# activations + logits over ~700-token sequences — see LOAD_IN_4BIT comment above.
BATCH_SIZE = int(os.environ.get("REASONER_BATCH_SIZE", 1))
GRAD_ACCUM_STEPS = int(os.environ.get("REASONER_GRAD_ACCUM_STEPS", 8))
WARMUP_RATIO = float(os.environ.get("REASONER_WARMUP_RATIO", 0.05))
MAX_GRAD_NORM = float(os.environ.get("REASONER_MAX_GRAD_NORM", 1.0))
MAX_SEQ_LEN = int(os.environ.get("REASONER_MAX_SEQ_LEN", 768))

LOG_EVERY_N_STEPS = int(os.environ.get("REASONER_LOG_EVERY_N_STEPS", 5))
EVAL_EVERY_N_EPOCHS = int(os.environ.get("REASONER_EVAL_EVERY_N_EPOCHS", 1))
CHECKPOINT_DIR = Path(os.environ.get(
    "REASONER_CHECKPOINT_DIR", str(PIPELINE_DIR / "checkpoints" / "cosmos3_reasoner_lora")
))
