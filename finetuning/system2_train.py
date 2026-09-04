from libs import *
import time
from collections import defaultdict

import config
from data import ActionFrameDataset, DetectLandmarksDataset, PixelGoalTextDataset, build_split_datasets
from load_model import build_system2_model, save_adapter
from system2_infer import build_action_prompt_messages, build_detect_prompt_messages, build_prompt_messages

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup


# =============================================================================================
# The task registry.
#
# Every task in the mix is the same shape — a dataset of frames, plus two pure functions
# turning one of its samples into the prompt the model sees and the text it should emit. Nothing
# below this block knows what a pixel goal or an action IS: encoding, prompt masking, sampling,
# per-task metrics and checkpointing are all driven off these specs, so adding a third task is
# adding one TaskSpec (and its prompt builder in system2_infer.py), not touching the trainer.
#
# All tasks share the SAME task-level train/val split — build_split_datasets is called per task
# with the same seed and val_fraction, so an instruction held out for one task is held out for
# every task, and no task can leak a val frame in through another's training set.
# =============================================================================================
class TaskSpec:
    def __init__(self, name, dataset_class, to_prompt, to_target, dataset_kwargs=None):
        self.name = name
        self.dataset_class = dataset_class
        self.to_prompt = to_prompt
        self.to_target = to_target
        self.dataset_kwargs = dataset_kwargs or {}

    def build_datasets(self):
        return build_split_datasets(
            self.dataset_class, scene_dir=config.SCENE_DIR,
            val_fraction=config.VAL_FRACTION, seed=config.SEED, **self.dataset_kwargs,
        )


# Plain module-level functions, not lambdas: with a "spawn" DataLoader start method the
# collate_fn that closes over these has to pickle, and a lambda doesn't.
def pixel_goal_prompt(sample):
    return build_prompt_messages(sample["image"], sample["instruction"])


def pixel_goal_target(sample):
    return config.SYSTEM2_TARGET_TEMPLATE.format(u=sample["goal_u"], v=sample["goal_v"])


def action_prompt(sample):
    return build_action_prompt_messages(sample["image"], sample["instruction"],
                                        int(sample["goal_uv"][0]), int(sample["goal_uv"][1]))


def action_target(sample):
    return config.SYSTEM2_ACTION_TARGET_TEMPLATE.format(
        action=config.SYSTEM2_ACTION_NAMES[int(sample["action"])])


def detect_prompt(sample):
    return build_detect_prompt_messages(sample["image"], sample["candidates"],
                                        sample["guided_direction"])


def detect_target(sample):
    """The exact JSON detect_landmarks() itself parses (raw [y, x] arrays, not the formatted
    strings that method converts them to only AFTER parsing — see
    convert_habitat/landmarks.py's module docstring for why the source data is stored this
    way). json.dumps rather than a template string since the shape (a variable-length
    `visible` list) doesn't fit one."""
    target = {
        "current_location": sample["current_location"],
        "visible": [{"landmark": item["landmark"], "pixel": item["pixel_norm"]}
                    for item in sample["visible"]],
        "guess_pixel": sample["guess_pixel_norm"],
        "guess_label": sample["guess_label"],
        "guess_confidence": sample["guess_confidence"],
    }
    return json.dumps(target, ensure_ascii=False)


TASK_SPECS = {
    # (image, instruction) -> "u,v": the original System 2 objective.
    "pixel_goal": TaskSpec("pixel_goal", PixelGoalTextDataset,
                           to_prompt=pixel_goal_prompt, to_target=pixel_goal_target),
    # (image, instruction, goal pixel) -> "FORWARD": the same mapping System 1 learns from
    # scratch, phrased as text. `require_valid_goal` because this task's prompt states the goal
    # pixel, so a frame that has none (relative_goal_frame_id < 0) has nothing to condition on.
    "action": TaskSpec("action", ActionFrameDataset,
                       to_prompt=action_prompt, to_target=action_target,
                       dataset_kwargs={"require_valid_goal": True}),
    # (image, candidate landmarks) -> visible[]/guess_pixel/guess_label/guess_confidence/
    # current_location: object-pointing + scene understanding, trained against the LITERAL
    # deployed Cosmos3Reasoner.detect_landmarks() prompt/response contract so a checkpoint
    # trained with this task in the mix is a direct fit for what's actually called at
    # inference time. Data comes from convert_habitat/ (Habitat-Sim rollouts of VLN-CE R2R
    # with ground-truth semantic labels) — a real-recording SCENE_DIR built by convert/ has
    # no detect/ directory, so DetectLandmarksDataset comes back empty there rather than
    # raising (see its docstring), and this task should simply be left out of SYSTEM2_TASKS
    # for those runs.
    "detect": TaskSpec("detect", DetectLandmarksDataset,
                       to_prompt=detect_prompt, to_target=detect_target),
}


def selected_tasks():
    """The configured mix as (specs, weight-by-name), validated up front — a typo in
    SYSTEM2_TASKS should fail before a model is loaded, not after."""
    unknown = [name for name in config.SYSTEM2_TASKS if name not in TASK_SPECS]
    if unknown:
        raise ValueError(f"unknown SYSTEM2_TASKS entries {unknown}; "
                         f"known tasks are {sorted(TASK_SPECS)}")
    if len(config.SYSTEM2_TASK_WEIGHTS) != len(config.SYSTEM2_TASKS):
        raise ValueError("SYSTEM2_TASK_WEIGHTS must have one weight per SYSTEM2_TASKS entry "
                         f"({len(config.SYSTEM2_TASKS)}), got {len(config.SYSTEM2_TASK_WEIGHTS)}")
    specs = [TASK_SPECS[name] for name in config.SYSTEM2_TASKS]
    weights = dict(zip(config.SYSTEM2_TASKS, config.SYSTEM2_TASK_WEIGHTS))
    return specs, weights


class MultiTaskDataset(Dataset):
    """Flattens the per-task datasets into one index of (task name, row in that task). The task
    name rides along with each sample so the collate_fn can pick the right prompt/target
    builders and the training loop can attribute the loss to the task that produced it."""

    def __init__(self, datasets_by_task):
        self.datasets_by_task = datasets_by_task
        self.index = [(name, row)
                      for name, dataset in datasets_by_task.items()
                      for row in range(len(dataset))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, row = self.index[index]
        return name, self.datasets_by_task[name][row]


def build_task_sampler(dataset, weights_by_task):
    """Sampling shares are per TASK, not per sample: each sample's weight is its task's weight
    divided by that task's length, so the expected mix over an epoch is exactly the configured
    proportions however lopsided the frame counts are. Without this, mixing a 20k-frame task
    with a 2k-frame one would silently train ~91% on the first.

    Sampling is with replacement (WeightedRandomSampler's only mode), so an "epoch" here is
    len(dataset) draws rather than a strict pass over every frame — over several epochs the
    coverage evens out, and the epoch boundary stays a useful place to evaluate and checkpoint.
    """
    per_sample = [weights_by_task[name] / max(len(dataset.datasets_by_task[name]), 1)
                  for name, _ in dataset.index]
    return WeightedRandomSampler(per_sample, num_samples=len(dataset), replacement=True)


def encode_example(processor, prompt_messages, target_text):
    """(prompt messages, target text) -> tokenized model inputs with the prompt portion masked
    out of `labels`, so the loss only scores the assistant's response — exactly the standard
    instruction-tuning masking pattern. Task-agnostic: every task in TASK_SPECS is trained
    through this one function, differing only in the two arguments.
    """
    full_messages = prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
    ]

    prompt_only = processor.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    full = processor.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )

    # Cosmos3-Edge's chat template is thinking-style: the generation prompt ends with
    # "<think>\n" while a completed assistant turn renders "<think>\n\n</think>{text}", and the
    # tokenizer merges the newlines at that boundary differently in the two encodings. So the
    # prompt-only LENGTH is not where the response starts — the last prompt token differs.
    # Mask by the longest common token prefix instead: everything the two encodings share is
    # prompt, everything after ("</think>{target}<|im_end|>") gets the loss.
    prompt_ids = prompt_only["input_ids"][0]
    full_ids = full["input_ids"][0]
    n_common = min(prompt_ids.shape[0], full_ids.shape[0])
    diverge = int(torch.argmax((prompt_ids[:n_common] != full_ids[:n_common]).int())) \
        if bool((prompt_ids[:n_common] != full_ids[:n_common]).any()) else n_common
    labels = full["input_ids"].clone()
    labels[:, :diverge] = -100
    full["labels"] = labels
    return full


def build_collate_fn(processor):
    """Real batching (batch_size > 1) here would need padding a batch of variable per-image
    vision-token counts, which is model/processor-specific and easy to get subtly wrong.
    Keeping the DataLoader at batch_size=1 and using SYSTEM2_GRAD_ACCUM_STEPS for the effective
    batch size sidesteps that entirely — revisit once Cosmos3EdgeProcessor's exact batched
    apply_chat_template behavior (padding side, vision key shapes) is confirmed.

    It also means every accumulation window is a random draw across tasks rather than a
    single-task batch, so each optimizer step already sees a mixed gradient.
    """

    def collate(samples):
        assert len(samples) == 1, "use DataLoader(batch_size=1) with this collate_fn"
        task_name, sample = samples[0]
        spec = TASK_SPECS[task_name]
        return task_name, encode_example(processor, spec.to_prompt(sample), spec.to_target(sample))

    return collate


class MetricsLogger:
    """Append-only JSONL next to the checkpoints — one line per logged train step / eval, so
    the run can be watched live (tail -f) and plotted later without parsing stdout."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log(self, **fields):
        fields["elapsed_s"] = round(time.time() - self.start_time, 1)
        with self.path.open("a") as f:
            f.write(json.dumps(fields) + "\n")


class TaskLossTracker:
    """Running mean loss overall and per task. Per-task numbers are the point of the whole
    exercise: one blended loss curve can't tell "both tasks improving" from "the cheap task
    improving while the other quietly degrades", which is the usual way a multi-task mix fails.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.total_loss, self.total_count = 0.0, 0

    def update(self, task_name, loss):
        self.loss_sums[task_name] += loss
        self.counts[task_name] += 1
        self.total_loss += loss
        self.total_count += 1

    def mean(self):
        return self.total_loss / max(self.total_count, 1)

    def per_task(self):
        return {name: self.loss_sums[name] / self.counts[name] for name in sorted(self.counts)}

    def weighted_mean(self, weights_by_task):
        """Model selection metric: per-task means recombined at the CONFIGURED task weights, not
        at whatever mix the sampler happened to draw. Keeps "best" from drifting as the sampled
        proportions wobble, and keeps a task that is merely more frequent from owning the score.
        """
        per_task = self.per_task()
        total_weight = sum(weights_by_task[name] for name in per_task)
        if not total_weight:
            return self.mean()
        return sum(weights_by_task[name] * loss for name, loss in per_task.items()) / total_weight


def build_optimizer_and_scheduler(model, n_optimizer_steps):
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.SYSTEM2_LEARNING_RATE,
                                  weight_decay=config.SYSTEM2_WEIGHT_DECAY)
    warmup_steps = int(n_optimizer_steps * config.SYSTEM2_WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max(n_optimizer_steps, 1))
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, val_loader, input_device):
    """Teacher-forced val loss, overall and per task — not generation accuracy. Checking whether
    the model actually EMITS a correct, parseable answer at inference is system2_infer's
    parse_pixel_goal / parse_action run over model.generate() outputs, which is slow enough to
    run separately, not every epoch.

    Note the two tasks' losses are not comparable in magnitude — "u,v" is several tokens of a
    fairly open range, an action name is one of four — so read each one's TREND, not the gap.
    """
    model.eval()
    tracker = TaskLossTracker()
    for task_name, batch in val_loader:
        batch = {key: value.to(input_device) for key, value in batch.items()}
        tracker.update(task_name, float(model(**batch).loss))
    model.train()
    return tracker


def main():
    set_seed(config.SEED)
    device = config.DEVICE

    specs, weights_by_task = selected_tasks()
    model, processor = build_system2_model(config.COSMOS3_EDGE_CHECKPOINT, device)

    # When device_map="auto" shards the model across multiple GPUs, the config DEVICE
    # ("cuda") may not be where the model's first layer actually lives. Always derive the
    # input device from the model itself so batches land on the right GPU.
    input_device = next(model.parameters()).device

    collate_fn = build_collate_fn(processor)

    train_sets, val_sets = {}, {}
    for spec in specs:
        train_sets[spec.name], val_sets[spec.name] = spec.build_datasets()
        print(f"task {spec.name}: {len(train_sets[spec.name])} train frames, "
              f"{len(val_sets[spec.name])} val frames (weight {weights_by_task[spec.name]})")

    train_set, val_set = MultiTaskDataset(train_sets), MultiTaskDataset(val_sets)
    print(f"system2 multi-task total: {len(train_set)} train, {len(val_set)} val")

    # Train draws at the configured task proportions; val is a plain deterministic pass over
    # every frame of every task, so the per-task eval losses are exact rather than sampled.
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=collate_fn,
                              sampler=build_task_sampler(train_set, weights_by_task),
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            num_workers=config.NUM_WORKERS)

    steps_per_epoch = max(len(train_loader) // config.SYSTEM2_GRAD_ACCUM_STEPS, 1)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, steps_per_epoch * config.SYSTEM2_NUM_EPOCHS
    )

    metrics = MetricsLogger(config.SYSTEM2_CHECKPOINT_DIR / "metrics.jsonl")
    metrics.log(event="start", tasks=list(config.SYSTEM2_TASKS), task_weights=weights_by_task,
                n_train_by_task={name: len(dataset) for name, dataset in train_sets.items()},
                n_val_by_task={name: len(dataset) for name, dataset in val_sets.items()},
                n_train=len(train_set), n_val=len(val_set),
                epochs=config.SYSTEM2_NUM_EPOCHS, grad_accum=config.SYSTEM2_GRAD_ACCUM_STEPS,
                lr=config.SYSTEM2_LEARNING_RATE)

    best_eval_loss = float("inf")
    optimizer.zero_grad()
    for epoch in range(1, config.SYSTEM2_NUM_EPOCHS + 1):
        epoch_tracker, window_tracker = TaskLossTracker(), TaskLossTracker()
        window_t0 = time.time()
        for step, (task_name, batch) in enumerate(train_loader):
            batch = {key: value.to(input_device) for key, value in batch.items()}
            loss = model(**batch).loss / config.SYSTEM2_GRAD_ACCUM_STEPS
            loss.backward()
            sample_loss = float(loss) * config.SYSTEM2_GRAD_ACCUM_STEPS
            epoch_tracker.update(task_name, sample_loss)
            window_tracker.update(task_name, sample_loss)

            if (step + 1) % config.SYSTEM2_GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.SYSTEM2_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % config.SYSTEM2_LOG_EVERY_N_STEPS == 0:
                per_task = window_tracker.per_task()
                samples_per_s = window_tracker.total_count / max(time.time() - window_t0, 1e-6)
                task_summary = " ".join(f"{name} {loss:.4f}" for name, loss in per_task.items())
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {window_tracker.mean():.4f} [{task_summary}] "
                      f"({samples_per_s:.2f} samples/s)", flush=True)
                metrics.log(event="train", epoch=epoch, step=step, total_steps=len(train_loader),
                            loss=round(window_tracker.mean(), 5),
                            task_loss={name: round(loss, 5) for name, loss in per_task.items()},
                            task_samples=dict(window_tracker.counts),
                            lr=scheduler.get_last_lr()[0],
                            samples_per_s=round(samples_per_s, 2))
                window_tracker.reset()
                window_t0 = time.time()

        print(f"epoch {epoch} done — mean train loss {epoch_tracker.mean():.4f} "
              f"[{epoch_tracker.per_task()}]", flush=True)
        metrics.log(event="epoch", epoch=epoch, train_loss=round(epoch_tracker.mean(), 5),
                    task_train_loss={name: round(loss, 5)
                                     for name, loss in epoch_tracker.per_task().items()},
                    task_samples=dict(epoch_tracker.counts))

        if epoch % config.SYSTEM2_EVAL_EVERY_N_EPOCHS == 0:
            eval_tracker = evaluate(model, val_loader, input_device)
            eval_loss = eval_tracker.weighted_mean(weights_by_task)
            per_task = eval_tracker.per_task()
            print(f"epoch {epoch} eval loss {eval_loss:.4f} [{per_task}]", flush=True)
            metrics.log(event="eval", epoch=epoch, eval_loss=round(eval_loss, 5),
                        task_eval_loss={name: round(loss, 5) for name, loss in per_task.items()},
                        best=eval_loss < best_eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_adapter(model, config.SYSTEM2_CHECKPOINT_DIR / "best")

    save_adapter(model, config.SYSTEM2_CHECKPOINT_DIR / "final")
    metrics.log(event="done", best_eval_loss=round(best_eval_loss, 5))


if __name__ == "__main__":
    main()
