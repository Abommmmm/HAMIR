from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import DDIPairDataset, make_collate_fn, move_batch_to_device
from .models import DCIR
from .train_cloud import load_audit


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_average_precision(target: np.ndarray, score: np.ndarray):
    return (
        float(average_precision_score(target, score))
        if int(target.sum()) > 0
        else None
    )


def _safe_roc_auc(target: np.ndarray, score: np.ndarray):
    return (
        float(roc_auc_score(target, score))
        if len(np.unique(target)) == 2
        else None
    )


def compute_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    task_mode: str,
    threshold: float | np.ndarray,
    label_values: list[str],
) -> tuple[dict, list[dict], np.ndarray]:
    num_classes = probability.shape[1]
    if task_mode == "multiclass":
        prediction = probability.argmax(axis=1)
        target_matrix = np.eye(num_classes, dtype=np.int8)[target]
        prediction_matrix = np.eye(num_classes, dtype=np.int8)[prediction]
    else:
        threshold_array = np.asarray(threshold, dtype=np.float64)
        if threshold_array.ndim > 1:
            raise ValueError("threshold must be a scalar or one-dimensional")
        if threshold_array.ndim == 1 and len(threshold_array) != num_classes:
            raise ValueError(
                "per-class threshold count must match probability columns"
            )
        if (
            not np.isfinite(threshold_array).all()
            or (threshold_array <= 0.0).any()
            or (threshold_array >= 1.0).any()
        ):
            raise ValueError("all thresholds must be finite and between 0 and 1")
        target_matrix = target.astype(np.int8)
        prediction_matrix = (
            probability >= threshold_array
        ).astype(np.int8)
        prediction = prediction_matrix

    per_class = []
    average_precisions = []
    roc_aucs = []
    for index in range(num_classes):
        truth = target_matrix[:, index]
        predicted = prediction_matrix[:, index]
        ap = _safe_average_precision(truth, probability[:, index])
        auc = _safe_roc_auc(truth, probability[:, index])
        if ap is not None:
            average_precisions.append(ap)
        if auc is not None:
            roc_aucs.append(auc)
        per_class.append(
            {
                "class_internal": index,
                "class_raw": (
                    label_values[index]
                    if index < len(label_values)
                    else str(index)
                ),
                "support": int(truth.sum()),
                "precision": float(
                    precision_score(truth, predicted, zero_division=0)
                ),
                "recall": float(
                    recall_score(truth, predicted, zero_division=0)
                ),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "auprc": ap,
                "auroc": auc,
            }
        )

    scalar_threshold = np.asarray(threshold).ndim == 0
    common = {
        "examples": int(len(target)),
        "classes": int(num_classes),
        "threshold": (
            float(threshold)
            if task_mode != "multiclass" and scalar_threshold
            else None
        ),
        "threshold_mode": (
            "multiclass"
            if task_mode == "multiclass"
            else ("global" if scalar_threshold else "per_class")
        ),
        "macro_f1": float(
            f1_score(
                target_matrix,
                prediction_matrix,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                target_matrix,
                prediction_matrix,
                average="micro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                target_matrix,
                prediction_matrix,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                target_matrix,
                prediction_matrix,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_precision": float(
            precision_score(
                target_matrix,
                prediction_matrix,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                target_matrix,
                prediction_matrix,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                target_matrix,
                prediction_matrix,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_auprc_supported": (
            float(np.mean(average_precisions)) if average_precisions else None
        ),
        "micro_auprc": float(
            average_precision_score(
                target_matrix.ravel(), probability.ravel()
            )
        ),
        "macro_auroc_supported": (
            float(np.mean(roc_aucs)) if roc_aucs else None
        ),
        "micro_auroc": float(
            roc_auc_score(target_matrix.ravel(), probability.ravel())
        ),
        "auprc_supported_classes": int(len(average_precisions)),
        "auroc_supported_classes": int(len(roc_aucs)),
    }
    if task_mode == "multiclass":
        common["accuracy"] = float(accuracy_score(target, prediction))
    else:
        common["subset_accuracy"] = float(
            accuracy_score(target_matrix, prediction_matrix)
        )
        common["hamming_loss"] = float(
            hamming_loss(target_matrix, prediction_matrix)
        )
    return common, per_class, prediction


def evaluate(
    config: dict,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
    threshold: float,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires a CUDA GPU.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")

    seed = int(config.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    num_classes = int(audit["type_count"])
    task_mode = str(audit["task_mode"])
    dataset = DDIPairDataset(
        paths["records"],
        paths["split"],
        paths["conformers"],
        split,
        num_classes,
        task_mode,
        seed + 2,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 8)),
        collate_fn=make_collate_fn(
            task_mode,
            cutoff=float(config.get("data", {}).get("cutoff", 5.0)),
        ),
        pin_memory=True,
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = DCIR(num_classes=num_classes, **dict(config["model"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()

    precision = str(
        config["training"].get("mixed_precision", "none")
    ).lower()
    use_bf16 = precision == "bf16"
    probabilities = []
    targets = []
    sample_rows = []
    sample_index = 0
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"evaluate {split}",
            dynamic_ncols=True,
        ):
            drug_ids = batch["drug_ids"]
            record_ids = batch["record_ids"]
            batch = move_batch_to_device(batch, device, include_alt=False)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(
                    batch["d1"], batch["d2"]
                )["interaction_logits"]
            probability = (
                torch.softmax(logits.float(), dim=-1)
                if task_mode == "multiclass"
                else torch.sigmoid(logits.float())
            )
            probabilities.append(probability.cpu())
            targets.append(batch["target"].cpu())
            for drugs, ids in zip(drug_ids, record_ids):
                sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "d1": drugs[0],
                        "d2": drugs[1],
                        "record_ids": ";".join(str(value) for value in ids),
                    }
                )
                sample_index += 1

    probability_array = torch.cat(probabilities).numpy()
    target_array = torch.cat(targets).numpy()
    metrics, per_class, prediction = compute_metrics(
        target_array,
        probability_array,
        task_mode,
        threshold,
        [str(value) for value in audit.get("type_values", [])],
    )
    metrics.update(
        {
            "dataset": audit.get("dataset"),
            "split": split,
            "task_mode": task_mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_validation": checkpoint.get("validation"),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (output_dir / "per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    with (output_dir / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=target_array,
        probability=probability_array,
        prediction=prediction,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results written to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen DCIR checkpoint once on a fixed split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("val", "test"), default="test"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(
        load_config(args.config),
        args.checkpoint.resolve(),
        args.split,
        args.output_dir.resolve(),
        args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
