from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import load_config
from .losses import classification_loss
from .train_cloud import assert_cloud_training_authorized, load_audit


@dataclass(frozen=True)
class PairExample:
    d1: str
    d2: str
    labels: tuple[int, ...]
    record_ids: tuple[int, ...]


def load_pca50(path: Path) -> tuple[dict[str, np.ndarray], int]:
    profiles: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        feature_count = len(header) - 1
        if feature_count <= 0:
            raise ValueError(f"No PCA features found in {path}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != feature_count + 1:
                raise ValueError(
                    f"{path}:{line_number} has {len(row)} columns; "
                    f"expected {feature_count + 1}"
                )
            drug_id = row[0].strip()
            if not drug_id:
                raise ValueError(f"Blank drug ID at {path}:{line_number}")
            if drug_id in profiles:
                raise ValueError(f"Duplicate PCA profile for {drug_id}")
            profiles[drug_id] = np.asarray(row[1:], dtype=np.float32)
    if not profiles:
        raise ValueError(f"No drug profiles found in {path}")
    return profiles, feature_count


def load_pair_examples(
    records_path: Path,
    split_path: Path,
    split: str,
    num_classes: int,
) -> list[PairExample]:
    record_splits: dict[int, str] = {}
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = int(row["record_id"])
            if record_id in record_splits:
                raise ValueError(f"Duplicate split row for record {record_id}")
            record_splits[record_id] = row["split"]

    grouped: OrderedDict[
        tuple[str, str], dict[str, set[int] | list[int]]
    ] = OrderedDict()
    pair_splits: dict[tuple[str, str], str] = {}
    seen_record_ids: set[int] = set()
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = int(row["record_id"])
            if record_id in seen_record_ids:
                raise ValueError(f"Duplicate record ID {record_id}")
            seen_record_ids.add(record_id)
            if record_id not in record_splits:
                raise ValueError(f"Record {record_id} has no split assignment")
            row_split = record_splits[record_id]
            pair = (row["d1"], row["d2"])
            previous_split = pair_splits.setdefault(pair, row_split)
            if previous_split != row_split:
                raise ValueError(
                    f"Directed pair {pair} crosses {previous_split}/{row_split}"
                )
            if row_split != split:
                continue
            label = int(row["type_internal"])
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} is outside [0, {num_classes - 1}]"
                )
            bucket = grouped.setdefault(
                pair, {"labels": set(), "record_ids": []}
            )
            labels = bucket["labels"]
            ids = bucket["record_ids"]
            assert isinstance(labels, set)
            assert isinstance(ids, list)
            labels.add(label)
            ids.append(record_id)

    unknown_split_records = set(record_splits) - seen_record_ids
    if unknown_split_records:
        first = min(unknown_split_records)
        raise ValueError(f"Split contains unknown record ID {first}")
    if not grouped:
        raise ValueError(f"No examples found for split={split!r}")
    return [
        PairExample(
            d1=d1,
            d2=d2,
            labels=tuple(sorted(bucket["labels"])),
            record_ids=tuple(bucket["record_ids"]),
        )
        for (d1, d2), bucket in grouped.items()
    ]


class DeepDDIPairDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        profiles: dict[str, np.ndarray],
        feature_count: int,
        num_classes: int,
    ) -> None:
        self.examples = examples
        self.num_classes = num_classes
        missing = sorted(
            {
                drug
                for example in examples
                for drug in (example.d1, example.d2)
                if drug not in profiles
            }
        )
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{len(missing)} drugs lack PCA50 profiles: {preview}"
            )

        features = np.empty(
            (len(examples), feature_count * 2), dtype=np.float32
        )
        targets = np.zeros(
            (len(examples), num_classes), dtype=np.float32
        )
        for index, example in enumerate(examples):
            features[index, :feature_count] = profiles[example.d1]
            features[index, feature_count:] = profiles[example.d2]
            targets[index, list(example.labels)] = 1.0
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index], index


class DeepDDI(nn.Module):
    """PyTorch reproduction of the architecture stored by DeepDDI's authors."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 2048,
        hidden_layers: int = 9,
        batch_norm_epsilon: float = 1e-3,
        batch_norm_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")
        self.linears = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        current_dim = input_dim
        for _ in range(hidden_layers):
            linear = nn.Linear(current_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)
            self.normalizations.append(
                nn.BatchNorm1d(
                    hidden_dim,
                    eps=batch_norm_epsilon,
                    momentum=batch_norm_momentum,
                )
            )
            current_dim = hidden_dim
        self.output = nn.Linear(current_dim, num_classes)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        # The released Keras model applies Dense(relu) before BatchNormalization.
        for linear, normalization in zip(
            self.linears, self.normalizations
        ):
            hidden = normalization(torch.relu(linear(hidden)))
        return self.output(hidden)


def positive_class_weights(
    dataset: DeepDDIPairDataset,
    max_weight: float | None,
) -> torch.Tensor:
    positives = dataset.targets.sum(dim=0, dtype=torch.float64)
    negatives = len(dataset) - positives
    weights = (negatives / positives.clamp_min(1)).float()
    return weights.clamp_max(max_weight) if max_weight is not None else weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    dataset: DeepDDIPairDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": generator,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
    return DataLoader(**options)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    indices = []
    with torch.inference_mode():
        for features, target, index in tqdm(
            loader, desc=description, leave=False, dynamic_ncols=True
        ):
            features = features.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(features)
            probabilities.append(torch.sigmoid(logits.float()).cpu())
            targets.append(target)
            indices.append(index)
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(indices).numpy(),
    )


def macro_f1_at_half(
    target: np.ndarray, probability: np.ndarray
) -> float:
    return float(
        f1_score(
            target,
            probability >= 0.5,
            average="macro",
            zero_division=0,
        )
    )


def build_datasets(
    config: dict, requested_splits: tuple[str, ...]
) -> tuple[dict[str, DeepDDIPairDataset], dict, int]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    if audit["task_mode"] != "multilabel":
        raise ValueError("DeepDDI baseline requires task_mode=multilabel")
    num_classes = int(audit["type_count"])
    profiles, feature_count = load_pca50(paths["pca50"])
    datasets = {}
    for split in requested_splits:
        examples = load_pair_examples(
            paths["records"],
            paths["split"],
            split,
            num_classes,
        )
        datasets[split] = DeepDDIPairDataset(
            examples,
            profiles,
            feature_count,
            num_classes,
        )
    return datasets, audit, feature_count * 2


def make_model(
    config: dict, input_dim: int, num_classes: int
) -> DeepDDI:
    model_config = dict(config.get("model", {}))
    return DeepDDI(
        input_dim=input_dim,
        num_classes=num_classes,
        **model_config,
    )


def train(config: dict) -> None:
    assert_cloud_training_authorized()
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit, input_dim = build_datasets(
        config, ("train", "val")
    )
    train_set = datasets["train"]
    val_set = datasets["val"]
    training = config["training"]
    batch_size = int(training.get("batch_size", 512))
    num_workers = int(training.get("num_workers", 0))
    train_loader = make_loader(
        train_set, batch_size, True, num_workers, seed
    )
    val_loader = make_loader(
        val_set, batch_size, False, num_workers, seed + 1
    )

    device = torch.device("cuda")
    precision = str(training.get("mixed_precision", "none")).lower()
    if precision not in {"none", "bf16"}:
        raise ValueError("training.mixed_precision must be 'none' or 'bf16'")
    use_bf16 = precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but is unsupported on this GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_classes = int(audit["type_count"])
    model = make_model(config, input_dim, num_classes).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 2)),
        min_lr=float(training.get("min_learning_rate", 1e-5)),
    )
    loss_config = config.get("loss", {})
    max_weight_value = loss_config.get("max_class_weight")
    max_weight = (
        float(max_weight_value) if max_weight_value is not None else None
    )
    weights = positive_class_weights(train_set, max_weight).to(device)
    print(
        f"Model: DeepDDI, parameters={parameter_count:,}, "
        f"hidden_layers={len(model.linears)}, input_dim={input_dim}"
    )
    print(
        f"Examples: train={len(train_set)}, val={len(val_set)}; "
        f"batch_size={batch_size}, mixed_precision={precision}"
    )
    print(
        "Class weights: "
        f"min={weights.min().item():.4f}, max={weights.max().item():.4f}, "
        f"cap={max_weight}"
    )

    paths = {key: Path(value) for key, value in config["paths"].items()}
    checkpoint_dir = paths["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.csv"
    history: list[dict] = []
    best_f1 = -1.0
    stale = 0
    max_epochs = int(training.get("epochs", 40))
    patience = int(training.get("patience", 8))
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        running_loss = torch.zeros((), device=device)
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1:03d}/{max_epochs:03d}",
            dynamic_ncols=True,
        )
        for step, (features, target, _) in enumerate(progress, start=1):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(features)
                loss = classification_loss(
                    logits,
                    target,
                    "multilabel",
                    weights,
                    loss_name=str(
                        loss_config.get("classification", "weighted")
                    ),
                    focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, step={step}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training.get("gradient_clip", 5.0)),
            )
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(
                    f"Non-finite gradient at epoch={epoch + 1}, step={step}"
                )
            optimizer.step()
            running_loss += loss.detach()
            if step == 1 or step % 20 == 0:
                progress.set_postfix(
                    loss=f"{(running_loss / step).item():.6f}",
                    refresh=False,
                )
        progress.close()
        epoch_loss = float(
            (running_loss / max(len(train_loader), 1)).item()
        )
        probability, target, _ = collect_predictions(
            model, val_loader, device, use_bf16, "validation"
        )
        val_f1 = macro_f1_at_half(target, probability)
        scheduler.step(val_f1)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed = time.perf_counter() - started
        print(
            f"epoch={epoch + 1:03d} loss={epoch_loss:.6f} "
            f"val_macro_f1={val_f1:.6f} lr={learning_rate:.2e}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_macro_f1": val_f1,
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
            }
        )
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            stale = 0
        else:
            stale += 1
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1,
            "stale": stale,
            "elapsed_seconds": elapsed,
            "parameter_count": parameter_count,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "task_mode": "multilabel",
            "config": config,
            "validation": {
                "macro_f1": val_f1,
                "examples": len(val_set),
                "threshold": 0.5,
            },
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if improved:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        if stale >= patience:
            print("Early stopping.")
            break


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    config: dict,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires a CUDA GPU.")
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit, input_dim = build_datasets(config, (split,))
    dataset = datasets[split]
    training = config["training"]
    loader = make_loader(
        dataset,
        int(training.get("batch_size", 512)),
        False,
        int(training.get("num_workers", 0)),
        seed + 2,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    num_classes = int(audit["type_count"])
    if int(checkpoint["input_dim"]) != input_dim:
        raise ValueError("Checkpoint input dimension does not match PCA data")
    if int(checkpoint["num_classes"]) != num_classes:
        raise ValueError("Checkpoint class count does not match audit")
    model = make_model(config, input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    precision = str(training.get("mixed_precision", "none")).lower()
    use_bf16 = precision == "bf16"
    probability, target, indices = collect_predictions(
        model, loader, device, use_bf16, f"evaluate {split}"
    )
    prediction = (probability >= 0.5).astype(np.int8)
    macro_f1 = macro_f1_at_half(target, probability)
    metrics = {
        "macro_f1": macro_f1,
        "examples": int(len(dataset)),
        "classes": num_classes,
        "threshold": 0.5,
        "dataset": audit.get("dataset"),
        "split": split,
        "task_mode": "multilabel",
        "baseline": "DeepDDI",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation": checkpoint.get("validation"),
        "parameter_count": checkpoint.get("parameter_count"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    sample_rows = []
    for sample_index, dataset_index in enumerate(indices.tolist()):
        example = dataset.examples[dataset_index]
        sample_rows.append(
            {
                "sample_index": sample_index,
                "d1": example.d1,
                "d2": example.d2,
                "record_ids": ";".join(
                    str(value) for value in example.record_ids
                ),
            }
        )
    with (output_dir / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=target,
        probability=probability,
        prediction=prediction,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the external DeepDDI baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument(
        "--split", choices=("val", "test"), default="test"
    )
    evaluate_parser.add_argument(
        "--output-dir", required=True, type=Path
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "train":
        train(config)
    else:
        evaluate(
            config,
            args.checkpoint.resolve(),
            args.split,
            args.output_dir.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
