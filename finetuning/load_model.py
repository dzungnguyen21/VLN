from libs import *
from config import (COSMOS3_EDGE_CHECKPOINT, SYSTEM2_DTYPE, SYSTEM2_LORA_ALPHA,
                    SYSTEM2_LORA_DROPOUT, SYSTEM2_LORA_R, SYSTEM2_LORA_TARGET_MODULES)

DTYPE_BY_NAME = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_reasoner(checkpoint=COSMOS3_EDGE_CHECKPOINT, dtype=SYSTEM2_DTYPE):
def load_reasoner(checkpoint=COSMOS3_EDGE_CHECKPOINT, dtype=SYSTEM2_DTYPE, device=None):
    """Cosmos3-Edge's Reasoner Tower as a standalone VLM.

    `AutoModelForImageTextToText` on this checkpoint loads ONLY the Reasoner Tower — the
    diffusion Generator/VAE/scheduler are separate Diffusers components this project never
    touches, since System 2's job here is text-based pixel-goal grounding, not action/video
    generation. See https://huggingface.co/docs/transformers/main/model_doc/cosmos3_edge.

    When `device` is a CUDA device, `device_map="auto"` is used so the transformers/accelerate
    backend streams weights directly onto the GPU instead of loading to CPU first and then doing
    a monolithic `.to(device)` that temporarily doubles the GPU memory requirement.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    use_device_map = device is not None and str(device).startswith("cuda")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, torch_dtype=DTYPE_BY_NAME[dtype], device_map=None
        checkpoint,
        dtype=DTYPE_BY_NAME[dtype],
        device_map="auto" if use_device_map else None,
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor


def apply_lora(model, r=SYSTEM2_LORA_R, alpha=SYSTEM2_LORA_ALPHA, dropout=SYSTEM2_LORA_DROPOUT,
               target_modules=SYSTEM2_LORA_TARGET_MODULES):
    """Freeze the backbone and wrap `target_modules` in LoRA adapters.

    `target_modules` is a regex over FULL module names (peft accepts either a list of name
    substrings or a single regex string) — verified directly against the real checkpoint's
    named_modules(): the vision encoder ALSO has q_proj/k_proj/v_proj (only its output
    projection differs, out_proj vs the text decoder's o_proj), so a plain substring list
    would silently also LoRA-wrap the vision encoder. The default in config.py is scoped to
    `model.language_model.layers.N.self_attn.*` only — confirmed via get_peft_model +
    named_modules() that it wraps exactly the 28 layers' 4 attention projections (112
    modules) and nothing under `.visual`.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_system2_model(checkpoint=COSMOS3_EDGE_CHECKPOINT, device=None):
    """load_reasoner + apply_lora, moved to `device`. The single entry point system2_train.py
    and system2_infer.py call, so the two-step load/wrap process lives in exactly one place."""
    model, processor = load_reasoner(checkpoint)
    and system2_infer.py call, so the two-step load/wrap process lives in exactly one place.

    Passes `device` to load_reasoner so weights are placed directly on the GPU via
    device_map="auto", avoiding the CPU→GPU copy that OOMs when the GPU is nearly full.
    The explicit .to(device) is kept as a fallback for CPU-only or non-CUDA targets.
    """
    model, processor = load_reasoner(checkpoint, device=device)
    model = apply_lora(model)
    if device is not None:
    # If device_map="auto" was used, the model is already on GPU; .to() is a no-op then.
    # For CPU or non-CUDA devices it still does the right thing.
    if device is not None and not str(device).startswith("cuda"):
        model = model.to(device)
    return model, processor


def save_adapter(model, output_dir):
    """LoRA adapters only — not the frozen Reasoner Tower weights."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)


def load_adapter(base_checkpoint, adapter_dir, device=None):
    """Reload a fine-tuned run: fresh Reasoner Tower + the saved LoRA adapter on top."""
    from peft import PeftModel

    model, processor = load_reasoner(base_checkpoint)
    model = PeftModel.from_pretrained(model, adapter_dir)
    if device is not None:
        model = model.to(device)
    return model, processor
