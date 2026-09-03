"""Attaches a LoRA adapter trained by train_reasoner.py onto a live Cosmos3Reasoner instance —
the drop-in path back into run_pipeline.py / benchmark_r2r.py / the closed-loop test harness,
none of which need to know the reasoner underneath was fine-tuned.

Usage:
    from training.load_finetuned import attach_adapter
    reasoner = Cosmos3Reasoner(model_id=config.MODEL_ID, device="cuda")
    attach_adapter(reasoner, "checkpoints/cosmos3_reasoner_lora/best")
    reasoner.decompose(instruction)  # now uses the fine-tuned weights
"""
from peft import PeftModel


def attach_adapter(reasoner, adapter_dir):
    if reasoner.model is None:
        raise RuntimeError("Cosmos3Reasoner model is not loaded (use_mock=True?) — "
                           "construct it with use_mock=False before attaching an adapter.")
    reasoner.model = PeftModel.from_pretrained(reasoner.model, adapter_dir)
    return reasoner
