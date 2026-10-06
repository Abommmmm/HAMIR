from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import (
    DDIPairDataset,
    make_collate_fn,
    move_batch_to_device,
)
from .deepddi_baseline import file_sha256, macro_f1_at_half
from .losses import classification_loss
from .train_cloud import (
    assert_cloud_training_authorized,
    class_weights,
    load_audit,
    set_seed,
)


class ShiftedSoftplus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = math.log(2.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value) - self.shift


class GaussianDistance(nn.Module):
    def __init__(self, cutoff: float, num_gaussians: int) -> None:
        super().__init__()
        if num_gaussians < 2:
            raise ValueError("num_gaussians must be at least 2")
        centers = torch.linspace(0.0, cutoff, num_gaussians)
        spacing = float(centers[1] - centers[0])
        self.register_buffer("centers", centers)
        self.coefficient = -0.5 / spacing**2

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        offset = distance.unsqueeze(-1) - self.centers
        return torch.exp(self.coefficient * offset.square())


def segment_sum(
    value: torch.Tensor, index: torch.Tensor, size: int
) -> torch.Tensor:
    result = value.new_zeros((size, *value.shape[1:]))
    result.index_add_(0, index, value.to(result.dtype))
    return result


class SchNetInteraction(nn.Module):
    """Continuous-filter interaction used by the official 3DGT-DDI code."""

    def __init__(
        self,
        hidden_channels: int,
        num_filters: int,
        num_gaussians: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.atom_projection = nn.Linear(
            hidden_channels, num_filters, bias=False
        )
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.update = nn.Sequential(
            nn.Linear(num_filters, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self,
        node: torch.Tensor,
        edge_index: torch.Tensor,
        distance: torch.Tensor,
        radial: torch.Tensor,
    ) -> torch.Tensor:
        receiver, sender = edge_index
        cutoff_weight = 0.5 * (
            torch.cos(math.pi * distance / self.cutoff) + 1.0
        )
        filters = self.filter_network(radial) * cutoff_weight.unsqueeze(-1)
        message = self.atom_projection(node[sender]) * filters
        aggregate = segment_sum(message, receiver, len(node))
        return node + self.update(aggregate)


class SchNetMolecularEncoder(nn.Module):
    """SchNet branch matching the graph-only DrugBank branch of 3DGT-DDI."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_filters: int = 128,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        output_channels: int = 32,
        max_atomic_number: int = 118,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            max_atomic_number + 1, hidden_channels
        )
        self.distance_expansion = GaussianDistance(cutoff, num_gaussians)
        self.blocks = nn.ModuleList(
            SchNetInteraction(
                hidden_channels,
                num_filters,
                num_gaussians,
                cutoff,
            )
            for _ in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels // 2, output_channels),
        )

    def forward(self, molecule: dict[str, torch.Tensor]) -> torch.Tensor:
        atomic_number = molecule["atomic_numbers"].clamp(
            0, self.embedding.num_embeddings - 1
        )
        node = self.embedding(atomic_number)
        receiver, sender = molecule["edge_index"]
        displacement = (
            molecule["pos"][receiver] - molecule["pos"][sender]
        )
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        radial = self.distance_expansion(distance)
        for block in self.blocks:
            node = block(node, molecule["edge_index"], distance, radial)
        atom_output = self.output(node)
        graph_count = int(molecule["ptr"].numel() - 1)
        return segment_sum(atom_output, molecule["batch"], graph_count)


class GraphPairCNN(nn.Module):
    """Pair CNN used by 3DGT-DDI after its two SchNet branches."""

    def __init__(
        self, graph_channels: int, num_classes: int, hidden_channels: int = 64
    ) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * graph_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.stack([first, second], dim=1)
        features = self.convolutions(pair).flatten(start_dim=1)
        return self.classifier(features)


class ThreeDGTDDI(nn.Module):
    """
    Graph-only 3DGT-DDI adapted from binary DrugBank prediction to the
    project's 86-label event task.

    The official architecture uses independent SchNet branches for the two
    directed drug positions, followed by a one-dimensional CNN pair decoder.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_filters: int = 128,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        graph_channels: int = 32,
        pair_hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        encoder_options = {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "num_filters": num_filters,
            "num_gaussians": num_gaussians,
            "cutoff": cutoff,
            "output_channels": graph_channels,
        }
        self.first_encoder = SchNetMolecularEncoder(**encoder_options)
        self.second_encoder = SchNetMolecularEncoder(**encoder_options)
        self.geometric_block_count = num_layers * 2
        self.decoder = GraphPairCNN(
            graph_channels, num_classes, pair_hidden_channels
        )

    def forward(
        self,
        first: dict[str, torch.Tensor],
        second: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.decoder(
            self.first_encoder(first), self.second_encoder(second)
        )


class ContinuousFilterInteraction(nn.Module):
    """Distance-conditioned CFIM used in the Meta3D-DDI reproduction."""

    def __init__(
        self, hidden_channels: int, num_gaussians: int, dropout: float
    ) -> None:
        super().__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.source_projection = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.normalization = nn.LayerNorm(hidden_channels)

    def forward(
        self,
        node: torch.Tensor,
        edge_index: torch.Tensor,
        radial: torch.Tensor,
        cutoff_weight: torch.Tensor,
    ) -> torch.Tensor:
        receiver, sender = edge_index
        filters = (
            self.filter_network(radial) * cutoff_weight.unsqueeze(-1)
        )
        message = self.source_projection(node[sender]) * filters
        aggregate = segment_sum(message, receiver, len(node))
        return self.normalization(node + self.update(aggregate))


class Meta3DMolecularEncoder(nn.Module):
    """
    Rotation/translation-invariant 3DGNN with continuous-filter interactions.

    Only invariant atom-pair distances enter CFIM. The atom-gated sum pooling
    yields one representation per independently oriented drug conformer.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        atom_feature_dim: int = 7,
        dropout: float = 0.1,
        max_atomic_number: int = 118,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.embedding = nn.Embedding(
            max_atomic_number + 1, hidden_channels
        )
        self.atom_projection = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.distance_expansion = GaussianDistance(cutoff, num_gaussians)
        self.blocks = nn.ModuleList(
            ContinuousFilterInteraction(
                hidden_channels, num_gaussians, dropout
            )
            for _ in range(num_layers)
        )
        self.pool_gate = nn.Linear(hidden_channels, 1)
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
        )

    def forward(self, molecule: dict[str, torch.Tensor]) -> torch.Tensor:
        atomic_number = molecule["atomic_numbers"].clamp(
            0, self.embedding.num_embeddings - 1
        )
        node = self.embedding(atomic_number) + self.atom_projection(
            molecule["atom_features"]
        )
        receiver, sender = molecule["edge_index"]
        distance = torch.linalg.vector_norm(
            molecule["pos"][receiver] - molecule["pos"][sender], dim=-1
        )
        radial = self.distance_expansion(distance)
        cutoff_weight = 0.5 * (
            torch.cos(math.pi * distance / self.cutoff) + 1.0
        )
        for block in self.blocks:
            node = block(
                node,
                molecule["edge_index"],
                radial,
                cutoff_weight,
            )
        gate = torch.sigmoid(self.pool_gate(node))
        graph_count = int(molecule["ptr"].numel() - 1)
        numerator = segment_sum(
            gate * self.pool_projection(node),
            molecule["batch"],
            graph_count,
        )
        denominator = segment_sum(
            gate, molecule["batch"], graph_count
        ).clamp_min(1e-6)
        return numerator / denominator


class Meta3DDI(nn.Module):
    """
    Meta3D-DDI 3DGNN/CFIM adapted to the existing warm pair split.

    The original bilevel few-shot stage is intentionally omitted because it
    requires scaffold-disjoint support/query episodes. This checkpoint is
    therefore recorded as an adapted warm-split reproduction.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        dropout: float = 0.1,
        decoder_hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = Meta3DMolecularEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            dropout=dropout,
        )
        self.geometric_block_count = num_layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 4, decoder_hidden_channels),
            nn.LayerNorm(decoder_hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_channels, num_classes),
        )

    def forward(
        self,
        first: dict[str, torch.Tensor],
        second: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        first_graph = self.encoder(first)
        second_graph = self.encoder(second)
        pair = torch.cat(
            [
                first_graph,
                second_graph,
                torch.abs(first_graph - second_graph),
                first_graph * second_graph,
            ],
            dim=-1,
        )
        return self.decoder(pair)


def make_model(config: dict, num_classes: int) -> nn.Module:
    baseline = str(config["baseline"]).lower()
    options = dict(config.get("model", {}))
    if baseline == "3dgt_ddi":
        return ThreeDGTDDI(num_classes=num_classes, **options)
    if baseline == "meta3d_ddi":
        return Meta3DDI(num_classes=num_classes, **options)
    raise ValueError(f"Unsupported 3D baseline: {baseline}")


def build_datasets(
    config: dict, splits: tuple[str, ...]
) -> tuple[dict[str, DDIPairDataset], dict]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    if audit["task_mode"] != "multilabel":
        raise ValueError("3D DDI baselines require multilabel data")
    num_classes = int(audit["type_count"])
    seed = int(config.get("seed", 17))
    datasets = {
        split: DDIPairDataset(
            paths["records"],
            paths["split"],
            paths["conformers"],
            split,
            num_classes,
            "multilabel",
            seed + offset,
        )
        for offset, split in enumerate(splits)
    }
    return datasets, audit


def make_loader(
    dataset: DDIPairDataset,
    config: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    training = config["training"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(training.get("num_workers", 0))
    options = {
        "dataset": dataset,
        "batch_size": int(training.get("batch_size", 128)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "generator": generator,
        "collate_fn": make_collate_fn(
            "multilabel",
            cutoff=float(config.get("data", {}).get("cutoff", 5.0)),
        ),
    }
    if workers > 0:
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
    cursor = 0
    with torch.inference_mode():
        for batch in tqdm(
            loader, desc=description, leave=False, dynamic_ncols=True
        ):
            batch = move_batch_to_device(
                batch, device, include_alt=False
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(batch["d1"], batch["d2"])
            probabilities.append(torch.sigmoid(logits.float()).cpu())
            targets.append(batch["target"].cpu())
            batch_size = len(batch["target"])
            indices.append(
                torch.arange(cursor, cursor + batch_size, dtype=torch.long)
            )
            cursor += batch_size
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(indices).numpy(),
    )


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "learning_rate": float(row["learning_rate"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
            for row in csv.DictReader(handle)
        ]


def implementation_name(baseline: str) -> str:
    if baseline == "3dgt_ddi":
        return "graph_only_multilabel_adaptation"
    return "cfim_warm_split_adapted_reproduction"


def train(
    config: dict, resume_checkpoint: Path | None = None
) -> None:
    assert_cloud_training_authorized()
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit = build_datasets(config, ("train", "val"))
    train_set = datasets["train"]
    val_set = datasets["val"]
    train_loader = make_loader(train_set, config, True, seed)
    val_loader = make_loader(val_set, config, False, seed + 1)
    model = make_model(config, int(audit["type_count"]))
    training = config["training"]
    precision = str(training.get("mixed_precision", "none")).lower()
    if precision not in {"none", "bf16"}:
        raise ValueError("mixed_precision must be none or bf16")
    use_bf16 = precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training.get("lr_gamma", 0.98))
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    loss_config = config.get("loss", {})
    maximum_value = loss_config.get("max_class_weight")
    maximum = (
        None if maximum_value is None else float(maximum_value)
    )
    weights = class_weights(
        train_set,
        int(audit["type_count"]),
        "multilabel",
        maximum,
    ).to(device)
    baseline = str(config["baseline"]).lower()
    print(
        f"Model: {baseline.upper()}, parameters={parameter_count:,}, "
        f"geometric_blocks={model.geometric_block_count}, "
        f"implementation={implementation_name(baseline)}"
    )
    print(
        f"Examples: train={len(train_set)}, val={len(val_set)}; "
        f"batch_size={training.get('batch_size')}, "
        f"mixed_precision={precision}, "
        f"cutoff={config.get('data', {}).get('cutoff', 5.0)}"
    )
    print(
        f"Class weights: min={weights.min().item():.4f}, "
        f"max={weights.max().item():.4f}, cap={maximum}"
    )

    paths = {key: Path(value) for key, value in config["paths"].items()}
    checkpoint_dir = paths["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.csv"
    history: list[dict] = []
    best_f1 = -1.0
    stale = 0
    first_epoch = 0
    previous_elapsed = 0.0
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        if str(checkpoint.get("baseline", "")).lower() != baseline:
            raise ValueError("Resume checkpoint baseline mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        first_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint.get("best_f1", -1.0))
        stale = int(checkpoint.get("stale", 0))
        previous_elapsed = float(
            checkpoint.get("elapsed_seconds", 0.0)
        )
        history = load_history(history_path)
        print(
            f"Resumed from {resume_checkpoint}: epoch={first_epoch}, "
            f"best_val_macro_f1={best_f1:.6f}, stale={stale}"
        )

    max_epochs = int(training.get("epochs", 80))
    patience = int(training.get("patience", 12))
    if first_epoch >= max_epochs:
        raise ValueError("Resume epoch is not below training.epochs")
    started = time.perf_counter()
    for epoch in range(first_epoch, max_epochs):
        model.train()
        running_loss = torch.zeros((), device=device)
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1:03d}/{max_epochs:03d}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(
                batch, device, include_alt=False
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(batch["d1"], batch["d2"])
                loss = classification_loss(
                    logits,
                    batch["target"],
                    "multilabel",
                    weights,
                    loss_name=str(
                        loss_config.get("classification", "weighted")
                    ),
                    focal_gamma=float(
                        loss_config.get("focal_gamma", 2.0)
                    ),
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, step={step}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training.get("gradient_clip", 5.0)),
            )
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            running_loss += loss.detach()
            if step == 1 or step % 20 == 0:
                progress.set_postfix(
                    loss=f"{(running_loss / step).item():.6f}",
                    refresh=False,
                )

        mean_loss = running_loss / max(len(train_loader), 1)
        probability, target, _ = collect_predictions(
            model, val_loader, device, use_bf16, "validation"
        )
        val_f1 = macro_f1_at_half(target, probability)
        scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        elapsed = previous_elapsed + time.perf_counter() - started
        print(
            f"epoch={epoch + 1:03d} loss={mean_loss.item():.6f} "
            f"val_macro_f1={val_f1:.6f} lr={learning_rate:.2e}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(mean_loss.item()),
                "val_macro_f1": val_f1,
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
            }
        )
        with history_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(history[0])
            )
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
            "num_classes": int(audit["type_count"]),
            "task_mode": "multilabel",
            "baseline": baseline,
            "implementation": implementation_name(baseline),
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


def evaluate(
    config: dict,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires a CUDA GPU")
    seed = int(config.get("seed", 17))
    set_seed(seed)
    datasets, audit = build_datasets(config, (split,))
    dataset = datasets[split]
    loader = make_loader(dataset, config, False, seed + 2)
    model = make_model(config, int(audit["type_count"]))
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    baseline = str(config["baseline"]).lower()
    if str(checkpoint["baseline"]).lower() != baseline:
        raise ValueError("Checkpoint baseline mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    precision = str(
        config["training"].get("mixed_precision", "none")
    ).lower()
    probability, target, indices = collect_predictions(
        model,
        loader,
        device,
        precision == "bf16",
        f"evaluate {split}",
    )
    prediction = (probability >= 0.5).astype(np.int8)
    metrics = {
        "macro_f1": macro_f1_at_half(target, probability),
        "examples": int(len(dataset)),
        "classes": int(audit["type_count"]),
        "threshold": 0.5,
        "dataset": audit.get("dataset"),
        "split": split,
        "task_mode": "multilabel",
        "baseline": baseline,
        "implementation": checkpoint.get(
            "implementation", implementation_name(baseline)
        ),
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
    rows = []
    for sample_index, dataset_index in enumerate(indices.tolist()):
        example = dataset.examples[dataset_index]
        rows.append(
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=target,
        probability=probability,
        prediction=prediction,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results written to {output_dir}")


def verify_data(config: dict) -> None:
    datasets, audit = build_datasets(config, ("train", "val", "test"))
    missing = []
    conformer_dir = Path(config["paths"]["conformers"])
    drug_ids = {
        drug_id
        for dataset in datasets.values()
        for example in dataset.examples
        for drug_id in (example.d1, example.d2)
    }
    for drug_id in sorted(drug_ids):
        if not (conformer_dir / f"{drug_id}.npz").is_file():
            missing.append(drug_id)
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} drugs lack 3D conformers: {preview}"
        )
    print(
        f"3D data verified: drugs={len(drug_ids)}, "
        f"classes={audit['type_count']}, "
        + ", ".join(
            f"{split}={len(dataset)}"
            for split, dataset in datasets.items()
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train adapted 3DGT-DDI or Meta3D-DDI on DCIA splits."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify-data")
    verify_parser.add_argument("--config", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume-checkpoint", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument(
        "--checkpoint", required=True, type=Path
    )
    evaluate_parser.add_argument(
        "--split", choices=("val", "test"), default="test"
    )
    evaluate_parser.add_argument(
        "--output-dir", required=True, type=Path
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "verify-data":
        verify_data(config)
    elif args.command == "train":
        train(
            config,
            resume_checkpoint=(
                None
                if args.resume_checkpoint is None
                else args.resume_checkpoint.resolve()
            ),
        )
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
