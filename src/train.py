import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, configure_logging, file_hash, seed_everything, seed_worker, write_json
from src.data import PREPROCESSING, InspectionDataset, load_manifest
from src.metrics import metrics, select_threshold
from src.model import build_model, load_checkpoint

LOG = logging.getLogger(__name__)


def loader(
    rows: list[dict], root: Path, batch_size: int, training: bool = False, workers: int = 0, seed: int = 42
) -> DataLoader:
    return DataLoader(
        InspectionDataset(rows, root, training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def predict(
    model: nn.Module, batches: DataLoader, device: str, criterion: nn.Module | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels, probabilities, total_loss, loss_denominator = [], [], 0.0, 0.0
    for images, target in batches:
        logits = model(images.to(device))
        if criterion is not None:
            denominator = (
                float(criterion.weight[target.to(device)].sum()) if criterion.weight is not None else len(target)
            )
            total_loss += float(criterion(logits, target.to(device))) * denominator
            loss_denominator += denominator
        labels.extend(target.tolist())
        probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
    return np.array(labels), np.array(probabilities), total_loss / max(loss_denominator, 1)


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        raise ValueError("Epochs, batch size, learning rate, patience must be positive")
    if not 0 <= args.freeze_epochs < args.epochs:
        raise ValueError("freeze-epochs must be nonnegative and less than epochs")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh run output directory to preserve model/evaluation provenance")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rows = load_manifest(args.manifest, args.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    train_batches = loader(train_rows, args.data, args.batch_size, True, args.workers, args.seed)
    val_batches = loader(val_rows, args.data, args.batch_size, workers=args.workers)
    counts = np.bincount([CLASS_NAMES.index(r["label"]) for r in train_rows], minlength=2)
    imbalance_ratio = float(counts.max() / counts.min())
    weights = len(train_rows) / (2 * counts) if imbalance_ratio >= 1.5 else np.ones(2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    LOG.info("device=%s train_counts=%s imbalance_ratio=%.3f loss_weights=%s", device, counts, imbalance_ratio, weights)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "device_used": device,
            "class_counts": counts.tolist(),
            "class_weights": weights.tolist(),
            "manifest_sha256": file_hash(args.manifest),
            "initialization": "ImageNet1K_V1",
        }
    )
    write_json(args.output / "config.json", config)
    model = build_model(pretrained=True).to(device)
    optimizer, scheduler = None, None
    best_f1, best_loss, stale = -1.0, float("inf"), 0
    checkpoint_path = args.output / "best.pt"
    history = []
    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        if epoch == 0 or epoch == args.freeze_epochs:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = not frozen or name.startswith("fc.")
            groups = [{"params": model.fc.parameters(), "lr": args.lr}]
            if not frozen:
                groups.append(
                    {
                        "params": [p for name, p in model.named_parameters() if not name.startswith("fc.")],
                        "lr": args.lr * 0.1,
                    }
                )
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            stale = 0
        model.train()
        # Keep pretrained BatchNorm statistics fixed; small inspection batches are noisy.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        total_loss, loss_denominator = 0.0, 0.0
        train_labels, train_probabilities = [], []
        for images, target in train_batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, target.to(device))
            loss.backward()
            optimizer.step()
            denominator = float(criterion.weight[target.to(device)].sum())
            total_loss += float(loss.detach()) * denominator
            loss_denominator += denominator
            train_labels.extend(target.tolist())
            train_probabilities.extend(logits.detach().softmax(dim=1)[:, 1].cpu().tolist())
        labels, probabilities, val_loss = predict(model, val_batches, device, criterion)
        score = metrics(labels, probabilities, 0.5)
        record = {
            "epoch": epoch + 1,
            "frozen_backbone": frozen,
            "train_loss": total_loss / loss_denominator,
            "train_metrics": metrics(np.array(train_labels), np.array(train_probabilities), 0.5),
            "val_loss": val_loss,
            **{k: score[k] for k in ("precision", "recall", "f1", "accuracy")},
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)
        LOG.info("epoch %s", json.dumps(record))
        write_json(args.output / "history.json", history)
        scheduler.step(score["f1"])
        improved = score["f1"] > best_f1 or (score["f1"] == best_f1 and val_loss < best_loss)
        if improved:
            best_f1, best_loss, stale = score["f1"], val_loss, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": "resnet18",
                    "class_names": list(CLASS_NAMES),
                    "preprocessing": PREPROCESSING,
                    "threshold": 0.5,
                    "model_version": args.output.name,
                    "epoch": epoch + 1,
                    "manifest_sha256": config["manifest_sha256"],
                    "validation_metrics": score,
                    "training_config": config,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        # Do not stop before at least one unfrozen epoch has run.
        if stale >= args.patience and epoch >= args.freeze_epochs:
            LOG.info("Early stopping after epoch %s", epoch + 1)
            break
    best_model, checkpoint = load_checkpoint(checkpoint_path, device)
    labels, probabilities, _ = predict(best_model, val_batches, device)
    threshold, tradeoff = select_threshold(labels, probabilities, args.target_recall)
    checkpoint["threshold"] = threshold
    checkpoint["validation_metrics"] = metrics(labels, probabilities, threshold)
    torch.save(checkpoint, checkpoint_path)
    write_json(
        args.output / "threshold_selection.json",
        {
            "policy": "maximize precision subject to validation recall target; ties use F1 then highest threshold",
            "target_recall": args.target_recall,
            "selected_threshold": threshold,
            "metrics": checkpoint["validation_metrics"],
            "tradeoff": tradeoff,
            "validation_predictions": [
                {"path": row["path"], "label": int(label), "defect_probability": float(prob)}
                for row, label, prob in zip(val_rows, labels, probabilities, strict=True)
            ],
        },
    )
    LOG.info("Best model saved: %s, validation threshold %.6f", checkpoint_path, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet ResNet18 two-stage fine-tuning")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run-001"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    configure_logging()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
