"""Render metrics.jsonl (written by system2_train.MetricsLogger) as a loss-curve PNG.

    python plot_metrics.py                  # plots the current SYSTEM2 run
    python plot_metrics.py path/to/metrics.jsonl out.png
"""
from libs import *
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def main():
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else config.SYSTEM2_CHECKPOINT_DIR / "metrics.jsonl"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 \
        else metrics_path.with_name("loss_curve.png")

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train = [r for r in rows if r.get("event") == "train"]
    evals = [r for r in rows if r.get("event") == "eval"]
    if not train:
        raise SystemExit(f"no train rows in {metrics_path} yet")

    total_steps = train[-1]["total_steps"]
    x_train = [(r["epoch"] - 1) + r["step"] / total_steps for r in train]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_train, [r["loss"] for r in train], label="train loss (window mean)", lw=1,
            color="0.4")
    if evals:
        ax.plot([r["epoch"] for r in evals], [r["eval_loss"] for r in evals],
                "o-", label="eval loss", color="0.1")

    # Per-task curves for a multi-task run. The blended loss above can hide one task improving
    # while another degrades, which is the whole reason system2_train logs task_loss — so when
    # a row carries it, plot each task as well. A row may omit a task the sampler didn't draw
    # in that window, hence the per-task x lists rather than one shared axis.
    task_names = sorted({name for r in train for name in r.get("task_loss", {})})
    for name in task_names:
        points = [(x, r["task_loss"][name]) for x, r in zip(x_train, train)
                  if name in r.get("task_loss", {})]
        ax.plot([x for x, _ in points], [loss for _, loss in points],
                lw=1, alpha=0.8, label=f"train loss — {name}")
        eval_points = [(r["epoch"], r["task_eval_loss"][name]) for r in evals
                       if name in r.get("task_eval_loss", {})]
        if eval_points:
            ax.plot([e for e, _ in eval_points], [loss for _, loss in eval_points],
                    "o--", lw=1, alpha=0.8, label=f"eval loss — {name}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"System 2 Cosmos LoRA — {metrics_path.parent.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path} ({len(train)} train points, {len(evals)} evals)")


if __name__ == "__main__":
    main()
