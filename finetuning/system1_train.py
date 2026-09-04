from libs import *
import config
from data import ActionFrameDataset, build_split_datasets
from system1_model import PixelGoalController

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize, to_tensor

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)


def image_transform(image):
    tensor = to_tensor(resize(image, [config.SYSTEM1_IMAGE_SIZE, config.SYSTEM1_IMAGE_SIZE]))
    return normalize(tensor)


def collate(samples):
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "goal_uv": torch.stack([sample["goal_uv"] for sample in samples]),
        "goal_valid": torch.stack([sample["goal_valid"] for sample in samples]),
        "action": torch.stack([sample["action"] for sample in samples]),
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    n_correct, n_total, loss_sum, n_batches = 0, 0, 0.0, 0
    for batch in val_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["image"], batch["goal_uv"], batch["goal_valid"])
        loss_sum += float(F.cross_entropy(logits, batch["action"]))
        n_batches += 1
        n_correct += int((logits.argmax(-1) == batch["action"]).sum())
        n_total += batch["action"].numel()
    model.train()
    return {"eval_loss": loss_sum / max(n_batches, 1), "accuracy": n_correct / max(n_total, 1)}


def main():
    set_seed(config.SEED)
    device = config.DEVICE

    train_set, val_set = build_split_datasets(
        ActionFrameDataset, scene_dir=config.SCENE_DIR,
        val_fraction=config.VAL_FRACTION, seed=config.SEED, image_transform=image_transform,
    )
    print(f"system1 train frames: {len(train_set)}, val frames: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config.SYSTEM1_BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.SYSTEM1_BATCH_SIZE, shuffle=False,
                            collate_fn=collate, num_workers=config.NUM_WORKERS)

    model = PixelGoalController().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SYSTEM1_LEARNING_RATE,
                                  weight_decay=config.SYSTEM1_WEIGHT_DECAY)

    best_eval_loss = float("inf")
    for epoch in range(1, config.SYSTEM1_NUM_EPOCHS + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["image"], batch["goal_uv"], batch["goal_valid"])
            loss = F.cross_entropy(logits, batch["action"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss)

            if step % config.SYSTEM1_LOG_EVERY_N_STEPS == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {float(loss):.4f}")

        print(f"epoch {epoch} done — mean train loss {running_loss / len(train_loader):.4f}")

        if epoch % config.SYSTEM1_EVAL_EVERY_N_EPOCHS == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"epoch {epoch} eval: {metrics}")
            config.SYSTEM1_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            if metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = metrics["eval_loss"]
                torch.save(model.state_dict(), config.SYSTEM1_CHECKPOINT_DIR / "best.pt")

    config.SYSTEM1_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.SYSTEM1_CHECKPOINT_DIR / "final.pt")


if __name__ == "__main__":
    main()
