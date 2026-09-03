"""Fine-tunes Cosmos3Reasoner.decompose() — the "Fine-tune Cosmos 3 Reasoner" step named in
../pipeline.md — via LoRA SFT on hand-labeled subgoal decompositions (subgoal_labels.json).

Text-only by design: decompose() already supports image=None, subgoal decomposition is a
language task, and pairing (instruction, unrelated first-bag-frame) risks teaching the model a
wrong association — the first frame of a recording routinely shows nothing related to the
instruction's actual path (confirmed by inspection: run_02's fridge/pantry start frame has
nothing to do with "walk down the corridor... white door").
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import config
from dataset import build_train_val_datasets

from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def load_reasoner_for_training():
    """Same attn_implementation/trust_remote_code/dtype as Cosmos3Reasoner._load_model (the live
    pipeline's loader) so the fine-tuned adapter targets exactly what configs/pipeline_config.yaml
    deploys — but loaded directly here, not through Cosmos3Reasoner, because 4-bit quantization
    (config.LOAD_IN_4BIT) needs a quantization_config kwarg that class doesn't expose, and adding
    a training-only concern to the live pipeline's model wrapper isn't worth the coupling.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, config.DTYPE)
    load_kwargs = dict(attn_implementation="sdpa", trust_remote_code=True, dtype=dtype)

    if config.LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = config.DEVICE
    elif config.DEVICE == "cuda":
        load_kwargs["device_map"] = config.DEVICE

    model = AutoModelForImageTextToText.from_pretrained(config.MODEL_ID, **load_kwargs)
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    return model, processor


def apply_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.LOAD_IN_4BIT:
        # Casts norms to fp32, enables input-grad hooks for the frozen embedding layer, and
        # (with gradient checkpointing) makes k-bit LoRA training numerically stable — without
        # this, loss reliably NaNs a few steps in on quantized weights.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING
        )
    elif config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # incompatible with gradient checkpointing during training
    model.print_trainable_parameters()
    return model


def encode_example(processor, example, max_seq_len=config.MAX_SEQ_LEN):
    """(prompt, target) -> (input_ids, attention_mask, labels), with the prompt portion masked
    out of labels so loss only scores the assistant's JSON response — same masking pattern as
    any instruction-tuning SFT, applied text-only (no images, no vision tokens to worry about).
    """
    prompt_messages = [{"role": "user", "content": [{"type": "text", "text": example["prompt"]}]}]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": example["target"]}]}
    ]

    # Mirrors Cosmos3Reasoner.decompose()'s own text-only branch exactly: apply_chat_template
    # returns a formatted string (tokenize=False by default for this processor), then the
    # processor itself does the actual tokenization.
    prompt_text = processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False)

    prompt_ids = processor(text=prompt_text, return_tensors="pt").input_ids[0]
    full = processor(text=full_text, return_tensors="pt")
    input_ids = full.input_ids[0][:max_seq_len]
    attention_mask = full.attention_mask[0][:max_seq_len]

    labels = input_ids.clone()
    prompt_len = min(prompt_ids.shape[0], input_ids.shape[0])
    labels[:prompt_len] = -100
    return input_ids, attention_mask, labels


def pad_stack(sequences, pad_value, dtype):
    max_len = max(sequence.shape[0] for sequence in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=dtype)
    for row, sequence in enumerate(sequences):
        padded[row, :sequence.shape[0]] = sequence
    return padded


def build_collate_fn(processor):
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id  # common fallback for causal LMs

    def collate(batch_examples):
        encoded = [encode_example(processor, example) for example in batch_examples]
        input_ids = [item[0] for item in encoded]
        attention_masks = [item[1] for item in encoded]
        labels = [item[2] for item in encoded]

        return {
            "input_ids": pad_stack(input_ids, pad_token_id, input_ids[0].dtype),
            "attention_mask": pad_stack(attention_masks, 0, attention_masks[0].dtype),
            "labels": pad_stack(labels, -100, labels[0].dtype),
        }

    return collate


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    loss_sum, n_batches = 0.0, 0
    for batch in val_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        loss_sum += float(model(**batch).loss)
        n_batches += 1
    model.train()
    return loss_sum / max(n_batches, 1)


def main():
    torch.manual_seed(config.SEED)
    device = config.DEVICE

    model, processor = load_reasoner_for_training()
    model = apply_lora(model)
    collate_fn = build_collate_fn(processor)

    train_set, val_set = build_train_val_datasets()
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)

    steps_per_epoch = max(len(train_loader) // config.GRAD_ACCUM_STEPS, 1)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.WARMUP_RATIO), max(total_steps, 1)
    )

    best_eval_loss = float("inf")
    optimizer.zero_grad()
    for epoch in range(1, config.NUM_EPOCHS + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / config.GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += float(loss) * config.GRAD_ACCUM_STEPS

            if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % config.LOG_EVERY_N_STEPS == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                     f"loss {float(loss) * config.GRAD_ACCUM_STEPS:.4f}")

        print(f"epoch {epoch} done — mean train loss {running_loss / len(train_loader):.4f}")

        if epoch % config.EVAL_EVERY_N_EPOCHS == 0:
            eval_loss = evaluate(model, val_loader, device)
            print(f"epoch {epoch} eval loss {eval_loss:.4f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(config.CHECKPOINT_DIR / "best")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.CHECKPOINT_DIR / "final")
    print(f"adapters saved under {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
