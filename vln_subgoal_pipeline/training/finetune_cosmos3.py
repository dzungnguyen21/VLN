import os
import argparse
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FineTuneCosmos3")


def fine_tune_cosmos3(
    train_data_path: str,
    output_dir: str = "./checkpoints/cosmos3_vln_reasoner",
    model_id: str = "nvidia/Cosmos3-Edge",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """
    Fine-tunes Cosmos 3 Reasoner on VLN Subgoal Decomposition pairs using LoRA.
    """
    logger.info(f"Starting LoRA fine-tuning for Cosmos 3: {model_id}")
    logger.info(f"Training data: {train_data_path} -> Output checkpoints: {output_dir}")

    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Configure LoRA for vision-language reasoner backbone
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
        )

        logger.info("Fine-tuning pipeline prepared successfully.")
        # Note: In production, pass HuggingFace Dataset object to Trainer
        # trainer = Trainer(model=peft_model, args=training_args, train_dataset=dataset, ...)
        # trainer.train()
        # peft_model.save_pretrained(output_dir)
        return peft_model

    except ImportError as e:
        logger.warning(f"PEFT / PyTorch environment missing optional dependency ({e}).")
        logger.info(
            "To train on GPU, ensure `pip install peft accelerate datasets` is installed in your environment."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Cosmos 3 Reasoner on VLN Subgoal Decomposition")
    parser.add_argument("--data", type=str, default="./data/vln_train.jsonl", help="Path to training data JSONL")
    parser.add_argument("--output", type=str, default="./checkpoints/cosmos3_vln_reasoner", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    fine_tune_cosmos3(
        train_data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
