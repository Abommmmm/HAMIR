from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import (
    DDIPairDataset,
    make_collate_fn,
    move_batch_to_device,
)
from .losses import (
    build_top_response_keep_masks,
    classification_loss,
    faithfulness_margin,
    js_divergence_from_logits,
    response_sparsity,
)
from .models import DCIR


CLOUD_MARKERS = (
    "COLAB_GPU",
    "KAGGLE_URL_BASE",
    "SLURM_JOB_ID",
    "AWS_EXECUTION_ENV",
    "AZUREML_RUN_ID",
    "VERTEX_PRODUCT",
    "DCIR_CLOUD_RUNTIME",
)


def assert_cloud_training_authorized() -> None:
    if os.environ.get("CLOUD_TRAINING") != "1":
        raise SystemExit(
            "Training refused: set CLOUD_TRAINING=1 only inside the authorized "
            "cloud runtime."
        )
    markers = [name for name in CLOUD_MARKERS if os.environ.get(name)]
    if not markers:
        raise SystemExit(
            "Training refused: no cloud runtime marker was detected. On a cloud "
            "GPU VM set DCIR_CLOUD_RUNTIME=1 explicitly."
        )
    if not torch.cuda.is_available():
        raise SystemExit("Training refused: CUDA GPU is not available.")
    print(f"Cloud authorization: {', '.join(markers)}")
    print(f"Host: {socket.gethostname()}, GPU: {torch.cuda.get_device_name(0)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audit(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_weights(
    dataset: DDIPairDataset,
    num_classes: int,
    task_mode: str,
    max_weight: float | None = None,
):
    if task_mode == "multiclass":
        counts = Counter(example.labels[0] for example in dataset.examples)
        total = sum(counts.values())
        weights = [
            total / max(num_classes * counts.get(index, 1), 1)
            for index in range(num_classes)
        ]
        result = torch.tensor(weights, dtype=torch.float32)
        return result.clamp_max(max_weight) if max_weight is not None else result
    positives = np.zeros(num_classes, dtype=np.float64)
    for example in dataset.examples:
        positives[example.labels] += 1
    negatives = len(dataset) - positives
    result = torch.tensor(
        negatives / np.maximum(positives, 1), dtype=torch.float32
    )
    return result.clamp_max(max_weight) if max_weight is not None else result


def load_initial_checkpoint(model, path: Path) -> float:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validation = checkpoint.get("validation") or {}
    baseline_f1 = float(validation.get("macro_f1", -1.0))
    print(
        f"Initialized model from {path} "
        f"(baseline val_macro_f1={baseline_f1:.6f}); optimizer is fresh."
    )
    return baseline_f1


def evaluate(
    model,
    loader,
    device,
    task_mode: str,
    use_bf16: bool = False,
) -> tuple[float, dict]:
    model.eval()
    logits_all = []
    target_all = []
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="validation",
            leave=False,
            dynamic_ncols=True,
        ):
            batch = move_batch_to_device(batch, device, include_alt=False)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(batch["d1"], batch["d2"])
            logits_all.append(output["interaction_logits"])
            target_all.append(batch["target"])
    logits = torch.cat(logits_all).cpu()
    target = torch.cat(target_all).cpu()
    if task_mode == "multiclass":
        prediction = logits.argmax(dim=-1).numpy()
        truth = target.numpy()
    else:
        prediction = (torch.sigmoid(logits) >= 0.5).numpy()
        truth = target.numpy()
    macro_f1 = float(f1_score(truth, prediction, average="macro", zero_division=0))
    return macro_f1, {
        "macro_f1": macro_f1,
        "examples": int(len(target)),
    }


def train(
    config: dict,
    init_checkpoint: Path | None = None,
    start_epoch: int = 0,
) -> None:
    assert_cloud_training_authorized()
    seed = int(config.get("seed", 17))
    set_seed(seed)
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    num_classes = int(audit["type_count"])
    task_mode = audit["task_mode"]
    data_config = config.get("data", {})
    cutoff = float(data_config.get("cutoff", 5.0))

    train_set = DDIPairDataset(
        paths["records"],
        paths["split"],
        paths["conformers"],
        "train",
        num_classes,
        task_mode,
        seed,
    )
    val_set = DDIPairDataset(
        paths["records"],
        paths["split"],
        paths["conformers"],
        "val",
        num_classes,
        task_mode,
        seed + 1,
    )
    collate = make_collate_fn(task_mode, cutoff=cutoff)
    batch_size = int(config["training"].get("batch_size", 8))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=collate,
        pin_memory=True,
    )

    model_config = dict(config["model"])
    model = DCIR(num_classes=num_classes, **model_config)
    device = torch.device("cuda")
    precision = str(config["training"].get("mixed_precision", "none")).lower()
    if precision not in {"none", "bf16"}:
        raise ValueError("training.mixed_precision must be 'none' or 'bf16'")
    use_bf16 = precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but is not supported by this GPU.")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        f"Numerics: mixed_precision={precision}, "
        "float32_matmul_precision=high, fused_adamw=true"
    )
    model.to(device)
    best_f1 = (
        load_initial_checkpoint(model, init_checkpoint)
        if init_checkpoint is not None
        else -1.0
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
        fused=True,
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    print(
        f"Model: ablation={model.ablation}, "
        f"trainable_parameters={trainable_parameter_count:,}, "
        f"stored_parameters={total_parameters:,}"
    )
    loss_config = config.get("loss", {})
    max_class_weight = loss_config.get("max_class_weight")
    max_class_weight = (
        float(max_class_weight) if max_class_weight is not None else None
    )
    weights = class_weights(
        train_set,
        num_classes,
        task_mode,
        max_weight=max_class_weight,
    ).to(device)
    print(
        "Class weights: "
        f"min={weights.min().item():.4f}, max={weights.max().item():.4f}, "
        f"cap={max_class_weight}"
    )
    lambda_faith = float(loss_config.get("lambda_faithfulness", 0.1))
    lambda_consistency = float(loss_config.get("lambda_consistency", 0.05))
    lambda_sparsity = float(loss_config.get("lambda_sparsity", 1e-4))
    warmup = int(loss_config.get("explanation_warmup", 10))
    ramp_epochs = max(
        1, int(loss_config.get("explanation_ramp_epochs", 1))
    )
    explanation_objective_enabled = model.use_crdm and any(
        value != 0.0
        for value in (
            lambda_faith,
            lambda_consistency,
            lambda_sparsity,
        )
    )
    patience = int(config["training"].get("patience", 20))
    max_epochs = int(config["training"].get("epochs", 150))
    checkpoint_dir = paths["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.csv"
    history: list[dict] = []
    training_started = time.perf_counter()
    phase2_epoch = config["training"].get("phase2_epoch")
    phase2_epoch = int(phase2_epoch) if phase2_epoch is not None else None
    phase2_learning_rate = config["training"].get("phase2_learning_rate")
    phase2_learning_rate = (
        float(phase2_learning_rate)
        if phase2_learning_rate is not None
        else None
    )

    if start_epoch < 0 or start_epoch >= max_epochs:
        raise ValueError(
            f"start_epoch must be in [0, {max_epochs - 1}], got {start_epoch}"
        )
    stale = 0
    for epoch in range(start_epoch, max_epochs):
        if (
            phase2_epoch is not None
            and phase2_learning_rate is not None
            and epoch >= phase2_epoch
        ):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = phase2_learning_rate
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        use_explanation_losses = (
            epoch >= warmup and explanation_objective_enabled
        )
        needs_alternate = (
            use_explanation_losses and lambda_consistency != 0.0
        )
        explanation_scale = (
            min(1.0, (epoch - warmup + 1) / ramp_epochs)
            if use_explanation_losses
            else 0.0
        )
        train_set.include_alternates = needs_alternate
        model.train()
        running = torch.zeros((), device=device)
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1:03d}/{max_epochs:03d}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(
                batch,
                device,
                include_alt=needs_alternate,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(batch["d1"], batch["d2"])
                loss = classification_loss(
                    output["interaction_logits"],
                    batch["target"],
                    task_mode,
                    weights,
                    loss_name=loss_config.get("classification", "weighted"),
                    focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
                )
                if use_explanation_losses:
                    if lambda_faith != 0.0:
                        keep1, keep2 = build_top_response_keep_masks(
                            output,
                            batch["target"],
                            batch["d1"]["ptr"],
                            batch["d2"]["ptr"],
                            task_mode,
                            fraction=float(
                                loss_config.get("mask_fraction", 0.1)
                            ),
                        )
                        masked = model(
                            batch["d1"],
                            batch["d2"],
                            keep1=keep1,
                            keep2=keep2,
                        )
                        faith = faithfulness_margin(
                            output["interaction_logits"],
                            masked["interaction_logits"],
                            batch["target"],
                            task_mode,
                            margin=float(
                                loss_config.get(
                                    "faithfulness_margin", 0.1
                                )
                            ),
                        )
                        loss = (
                            loss
                            + explanation_scale * lambda_faith * faith
                        )
                    if lambda_consistency != 0.0:
                        alternate = model(
                            batch["d1_alt"], batch["d2_alt"]
                        )
                        consistency = js_divergence_from_logits(
                            output["interaction_logits"],
                            alternate["interaction_logits"],
                            task_mode,
                        )
                        loss = (
                            loss
                            + explanation_scale
                            * lambda_consistency
                            * consistency
                        )
                    if lambda_sparsity != 0.0:
                        sparse = response_sparsity(
                            output["d1_atom_response"],
                            output["d2_atom_response"],
                        )
                        loss = (
                            loss
                            + explanation_scale * lambda_sparsity * sparse
                        )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, step={step}. "
                    "The optimizer step was skipped; the last checkpoint is safe."
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"].get("gradient_clip", 5.0)),
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm).item():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient at epoch={epoch + 1}, step={step}. "
                    "The optimizer step was skipped; the last checkpoint is safe."
                )
            optimizer.step()
            running += loss.detach()
            if step == 1 or step % 20 == 0:
                progress.set_postfix(
                    loss=f"{(running / step).item():.6f}",
                    refresh=False,
                )

        epoch_loss = (running / max(len(train_loader), 1)).item()
        progress.close()
        del progress
        macro_f1, metrics = evaluate(
            model,
            val_loader,
            device,
            task_mode,
            use_bf16=use_bf16,
        )
        dgbf_scale = (
            float(model.dgbf.residual_scale.detach().cpu())
            if model.use_dgbf
            else 0.0
        )
        msan_scale = (
            float(model.msan_residual_scale.detach().cpu())
            if model.use_msan_residual
            else 0.0
        )
        print(
            f"epoch={epoch + 1:03d} loss={epoch_loss:.6f} "
            f"val_macro_f1={macro_f1:.6f} "
            f"explanation_scale={explanation_scale:.2f} "
            f"dgbf_scale={dgbf_scale:+.4f} "
            f"msan_scale={msan_scale:+.4f} "
            f"lr={current_learning_rate:.2e}"
        )
        elapsed_seconds = time.perf_counter() - training_started
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_macro_f1": macro_f1,
                "explanation_scale": explanation_scale,
                "dgbf_residual_scale": dgbf_scale,
                "msan_residual_scale": msan_scale,
                "learning_rate": current_learning_rate,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_f1": max(best_f1, macro_f1),
            "stale": stale,
            "elapsed_seconds": elapsed_seconds,
            "stored_parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameter_count,
            "num_classes": num_classes,
            "task_mode": task_mode,
            "config": config,
            "validation": metrics,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            stale = 0
            checkpoint["best_f1"] = best_f1
            checkpoint["stale"] = stale
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping.")
                break


def main() -> int:
    parser = argparse.ArgumentParser(description="Cloud-only DCIR trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load model weights from a checkpoint and use a fresh optimizer.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Zero-based first epoch; use 10 to begin at displayed epoch 11.",
    )
    args = parser.parse_args()
    train(
        load_config(args.config),
        init_checkpoint=args.init_checkpoint,
        start_epoch=args.start_epoch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
