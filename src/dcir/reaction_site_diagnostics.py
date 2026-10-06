from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import Subset

from .config import load_config
from .reaction_sites import (
    ReactionSiteDataset,
    _amp_dtype,
    _build_model,
    _loader,
    _move_batch,
)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU is available")
    return device


def _load_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    config = load_config(config_path)
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = _build_model(config)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    return config, model, state


def _test_loader(
    config: dict[str, Any],
    *,
    max_examples: int | None = None,
    seed: int = 17,
):
    paths = {key: Path(value) for key, value in config["paths"].items()}
    dataset = ReactionSiteDataset(
        paths["test_index"], paths["molecule_cache"]
    )
    if max_examples is not None and max_examples < len(dataset):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), max_examples))
        dataset = Subset(dataset, indices)
    training = config["training"]
    return _loader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        workers=(0 if os.name == "nt" else int(training.get("num_workers", 4))),
        cutoff=float(config["model"].get("cutoff", 5.0)),
        shuffle=False,
    )


def _binary_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    truth = targets.astype(bool)
    prediction = probabilities >= threshold
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auprc": float(average_precision_score(targets, probabilities)),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _pool_atoms(
    probabilities: torch.Tensor,
    motif_index: torch.Tensor,
    motif_count: int,
    mode: str,
) -> torch.Tensor:
    if mode == "max":
        output = probabilities.new_zeros(motif_count)
        output.scatter_reduce_(
            0,
            motif_index,
            probabilities,
            reduce="amax",
            include_self=True,
        )
        return output
    if mode == "mean":
        output = probabilities.new_zeros(motif_count)
        counts = probabilities.new_zeros(motif_count)
        output.index_add_(0, motif_index, probabilities)
        counts.index_add_(0, motif_index, torch.ones_like(probabilities))
        return output / counts.clamp_min(1)
    raise ValueError(mode)


def _top1_group_hits(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    motif_batch: torch.Tensor,
) -> tuple[int, int]:
    hits = total = 0
    batch_size = int(motif_batch.max()) + 1 if len(motif_batch) else 0
    for index in range(batch_size):
        local = torch.nonzero(motif_batch == index).flatten()
        if not len(local) or not bool((targets[local] > 0.5).any()):
            continue
        selected = local[torch.argmax(probabilities[local])]
        hits += int(targets[selected] > 0.5)
        total += 1
    return hits, total


@torch.no_grad()
def group_analysis(
    models: list[tuple[str, Path, Path]],
    *,
    device_name: str,
    threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    device = _device(device_name)
    rows: list[dict[str, Any]] = []
    for model_name, config_path, checkpoint_path in models:
        config, model, state = _load_model(
            config_path, checkpoint_path, device
        )
        loader = _test_loader(config)
        amp_dtype = _amp_dtype(
            str(config["training"].get("mixed_precision", "fp16")),
            device,
        )
        accumulators: dict[str, dict[str, list[np.ndarray] | int]] = {}
        methods = ["atom_max_pool", "atom_mean_pool"]
        if getattr(model, "architecture", "") == "hierarchical_motif":
            methods.append("explicit_group_head")
        for method in methods:
            accumulators[method] = {
                "probabilities": [],
                "targets": [],
                "top1_hits": 0,
                "top1_total": 0,
            }
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                output = model(batch["d1"], batch["d2"])
            for role in ("1", "2"):
                atom_probability = torch.sigmoid(output[f"logits{role}"])
                molecule = batch[f"d{role}"]
                group_target = batch[f"motif_target{role}"]
                for mode, method in (
                    ("max", "atom_max_pool"),
                    ("mean", "atom_mean_pool"),
                ):
                    probability = _pool_atoms(
                        atom_probability,
                        molecule["motif_index"],
                        len(group_target),
                        mode,
                    )
                    hits, total = _top1_group_hits(
                        probability, group_target, molecule["motif_batch"]
                    )
                    item = accumulators[method]
                    item["probabilities"].append(
                        probability.float().cpu().numpy()
                    )
                    item["targets"].append(
                        group_target.float().cpu().numpy()
                    )
                    item["top1_hits"] += hits
                    item["top1_total"] += total
                if "group_logits1" in output:
                    probability = torch.sigmoid(
                        output[f"group_logits{role}"]
                    )
                    hits, total = _top1_group_hits(
                        probability, group_target, molecule["motif_batch"]
                    )
                    item = accumulators["explicit_group_head"]
                    item["probabilities"].append(
                        probability.float().cpu().numpy()
                    )
                    item["targets"].append(
                        group_target.float().cpu().numpy()
                    )
                    item["top1_hits"] += hits
                    item["top1_total"] += total
        for method, item in accumulators.items():
            probabilities = np.concatenate(item["probabilities"])
            targets = np.concatenate(item["targets"])
            metrics = _binary_metrics(targets, probabilities, threshold)
            metrics.update(
                {
                    "model": model_name,
                    "method": method,
                    "top1_group_hit_rate": (
                        item["top1_hits"] / max(item["top1_total"], 1)
                    ),
                    "groups": int(len(targets)),
                    "checkpoint_epoch": int(state["epoch"]),
                }
            )
            rows.append(metrics)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "threshold": threshold,
        "task": "BRICS group prediction",
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "group_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / "group_analysis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return result


def _clone_molecule(molecule: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in molecule.items()
    }


def _occlude(
    molecule: dict[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    result = _clone_molecule(molecule)
    result["occlusion_mask"] = mask
    return result


def _top_atom_mask(
    probabilities: torch.Tensor,
    ptr: torch.Tensor,
    *,
    random_selection: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        top = int(torch.argmax(probabilities[start:end]))
        if random_selection:
            count = end - start
            draw = int(
                torch.randint(max(count - 1, 1), (1,), generator=generator)
            )
            local = draw + int(count > 1 and draw >= top)
        else:
            local = top
        mask[start + local] = True
    return mask


def _top_motif_mask(
    group_probabilities: torch.Tensor,
    molecule: dict[str, torch.Tensor],
    *,
    random_selection: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    atom_mask = torch.zeros(
        len(molecule["atomic_numbers"]),
        dtype=torch.bool,
        device=group_probabilities.device,
    )
    batch_size = len(molecule["ptr"]) - 1
    for index in range(batch_size):
        local = torch.nonzero(molecule["motif_batch"] == index).flatten()
        top_offset = int(torch.argmax(group_probabilities[local]))
        if random_selection:
            if len(local) == 1:
                selected = local[top_offset]
            else:
                sizes = torch.stack(
                    [
                        (molecule["motif_index"] == motif).sum()
                        for motif in local
                    ]
                )
                candidates = torch.arange(len(local), device=local.device)
                candidates = candidates[candidates != top_offset]
                distance = torch.abs(
                    sizes[candidates] - sizes[top_offset]
                )
                closest = candidates[distance == distance.min()]
                draw = int(
                    torch.randint(
                        len(closest), (1,), generator=generator
                    )
                )
                selected = local[int(closest[draw])]
        else:
            selected = local[top_offset]
        atom_mask |= molecule["motif_index"] == selected
    return atom_mask


def _fixed_top_atom_indices(
    output: dict[str, torch.Tensor], batch: dict, k: int = 3
) -> dict[str, list[torch.Tensor]]:
    selected: dict[str, list[torch.Tensor]] = {"1": [], "2": []}
    for role in ("1", "2"):
        probability = torch.sigmoid(output[f"logits{role}"])
        ptr = batch[f"d{role}"]["ptr"]
        for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
            local = torch.topk(
                probability[start:end], min(k, end - start)
            ).indices
            selected[role].append(local + start)
    return selected


def _fixed_pair_confidence(
    output: dict[str, torch.Tensor],
    selected: dict[str, list[torch.Tensor]],
) -> np.ndarray:
    values = []
    batch_size = len(selected["1"])
    for index in range(batch_size):
        molecule_values = []
        for role in ("1", "2"):
            probability = torch.sigmoid(output[f"logits{role}"])
            molecule_values.append(
                probability[selected[role][index]].mean()
            )
        values.append(torch.stack(molecule_values).mean())
    return torch.stack(values).float().cpu().numpy()


def _pair_bce(
    output: dict[str, torch.Tensor], batch: dict
) -> np.ndarray:
    values = []
    batch_size = len(batch["d1"]["ptr"]) - 1
    for index in range(batch_size):
        molecule_values = []
        for role in ("1", "2"):
            ptr = batch[f"d{role}"]["ptr"]
            start, end = ptr[index].item(), ptr[index + 1].item()
            molecule_values.append(
                F.binary_cross_entropy_with_logits(
                    output[f"logits{role}"][start:end].float(),
                    batch[f"target{role}"][start:end].float(),
                )
            )
        values.append(torch.stack(molecule_values).mean())
    return torch.stack(values).cpu().numpy()


def _collect_atom_predictions(
    output: dict[str, torch.Tensor], batch: dict
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = torch.cat(
        [
            torch.sigmoid(output["logits1"]),
            torch.sigmoid(output["logits2"]),
        ]
    ).float().cpu().numpy()
    targets = torch.cat(
        [batch["target1"], batch["target2"]]
    ).float().cpu().numpy()
    return targets, probabilities


def _paired_mean_test(
    candidate: np.ndarray,
    random_baseline: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | bool]:
    difference = candidate - random_baseline
    rng = np.random.default_rng(seed)
    sample = rng.integers(
        0, len(difference), size=(iterations, len(difference))
    )
    means = difference[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(iterations, len(difference)),
    )
    null = (difference * signs).mean(axis=1)
    observed = float(difference.mean())
    p_value = float(
        (np.count_nonzero(np.abs(null) >= abs(observed)) + 1)
        / (iterations + 1)
    )
    return {
        "mean_candidate_effect": float(candidate.mean()),
        "mean_random_effect": float(random_baseline.mean()),
        "paired_difference": observed,
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "paired_sign_flip_p": p_value,
        "significant": bool(low > 0 and p_value < 0.05),
    }


@torch.no_grad()
def faithfulness_analysis(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device_name: str,
    max_examples: int,
    iterations: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    device = _device(device_name)
    config, model, state = _load_model(
        config_path, checkpoint_path, device
    )
    if getattr(model, "architecture", "") != "hierarchical_motif":
        raise ValueError("Faithfulness analysis requires A2 group outputs")
    loader = _test_loader(
        config, max_examples=max_examples, seed=seed
    )
    precision = str(config["training"].get("mixed_precision", "fp16"))
    if device.type == "cpu" and precision.lower() in {"fp16", "float16"}:
        # CPU autocast is used here only for reproducible diagnostics when a
        # CUDA checkpoint's mixed-precision effects must be reconstructed.
        amp_dtype = torch.float16
    else:
        amp_dtype = _amp_dtype(precision if device.type == "cuda" else "none", device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    effects: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"probability_drop": [], "bce_increase": []}
        for name in (
            "top_atom",
            "random_atom",
            "top_group",
            "random_group",
        )
    }
    prediction_targets: list[np.ndarray] = []
    prediction_probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name in ("original", *effects)
    }
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            original = model(batch["d1"], batch["d2"])
        fixed_targets = _fixed_top_atom_indices(original, batch)
        base_confidence = _fixed_pair_confidence(
            original, fixed_targets
        )
        base_bce = _pair_bce(original, batch)
        targets, probabilities = _collect_atom_predictions(
            original, batch
        )
        prediction_targets.append(targets)
        prediction_probabilities["original"].append(probabilities)
        masks: dict[str, dict[str, torch.Tensor]] = {
            name: {} for name in effects
        }
        for role in ("1", "2"):
            probability = torch.sigmoid(original[f"logits{role}"])
            cpu_generator = generator
            masks["top_atom"][role] = _top_atom_mask(
                probability,
                batch[f"d{role}"]["ptr"],
                random_selection=False,
                generator=cpu_generator,
            )
            masks["random_atom"][role] = _top_atom_mask(
                probability,
                batch[f"d{role}"]["ptr"],
                random_selection=True,
                generator=cpu_generator,
            )
            group_probability = torch.sigmoid(
                original[f"group_logits{role}"]
            )
            masks["top_group"][role] = _top_motif_mask(
                group_probability,
                batch[f"d{role}"],
                random_selection=False,
                generator=cpu_generator,
            )
            masks["random_group"][role] = _top_motif_mask(
                group_probability,
                batch[f"d{role}"],
                random_selection=True,
                generator=cpu_generator,
            )
        for condition in effects:
            first = _occlude(batch["d1"], masks[condition]["1"])
            second = _occlude(batch["d2"], masks[condition]["2"])
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                perturbed = model(first, second)
            effects[condition]["probability_drop"].append(
                base_confidence
                - _fixed_pair_confidence(perturbed, fixed_targets)
            )
            effects[condition]["bce_increase"].append(
                _pair_bce(perturbed, batch) - base_bce
            )
            _, probabilities = _collect_atom_predictions(
                perturbed, batch
            )
            prediction_probabilities[condition].append(probabilities)
    arrays = {
        condition: {
            metric: np.concatenate(values)
            for metric, values in metrics.items()
        }
        for condition, metrics in effects.items()
    }
    all_targets = np.concatenate(prediction_targets)
    atom_metrics = {
        condition: _binary_metrics(
            all_targets,
            np.concatenate(probabilities),
            threshold=0.5,
        )
        for condition, probabilities in prediction_probabilities.items()
    }
    result = {
        "examples": int(len(arrays["top_atom"]["probability_drop"])),
        "fixed_prediction_target": (
            "the original top-3 atom indices in each reactant"
        ),
        "occlusion": (
            "replace selected encoded atom states with the mean state of "
            "unmasked atoms from the same molecule; the chemical graph "
            "and 3D geometry remain unchanged"
        ),
        "random_group_control": (
            "a non-top BRICS group with the closest atom count"
        ),
        "checkpoint_epoch": int(state["epoch"]),
        "atom_occlusion": {
            "fixed_target_probability_drop": _paired_mean_test(
                arrays["top_atom"]["probability_drop"],
                arrays["random_atom"]["probability_drop"],
                iterations=iterations,
                seed=seed,
            ),
            "ground_truth_bce_increase": _paired_mean_test(
                arrays["top_atom"]["bce_increase"],
                arrays["random_atom"]["bce_increase"],
                iterations=iterations,
                seed=seed + 1,
            ),
        },
        "group_occlusion": {
            "fixed_target_probability_drop": _paired_mean_test(
                arrays["top_group"]["probability_drop"],
                arrays["random_group"]["probability_drop"],
                iterations=iterations,
                seed=seed + 2,
            ),
            "ground_truth_bce_increase": _paired_mean_test(
                arrays["top_group"]["bce_increase"],
                arrays["random_group"]["bce_increase"],
                iterations=iterations,
                seed=seed + 3,
            ),
        },
        "atom_metrics_at_0.5": atom_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "faithfulness_effects.npz",
        top_atom_probability_drop=arrays["top_atom"]["probability_drop"],
        random_atom_probability_drop=arrays["random_atom"]["probability_drop"],
        top_group_probability_drop=arrays["top_group"]["probability_drop"],
        random_group_probability_drop=arrays["random_group"]["probability_drop"],
        top_atom_bce_increase=arrays["top_atom"]["bce_increase"],
        random_atom_bce_increase=arrays["random_atom"]["bce_increase"],
        top_group_bce_increase=arrays["top_group"]["bce_increase"],
        random_group_bce_increase=arrays["random_group"]["bce_increase"],
    )
    (output_dir / "faithfulness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def _random_rotations(
    count: int, device: torch.device
) -> torch.Tensor:
    matrix = torch.randn(count, 3, 3, device=device)
    q, _ = torch.linalg.qr(matrix)
    determinant = torch.linalg.det(q)
    q[:, :, -1] *= determinant.sign().unsqueeze(-1)
    return q


def _transform_positions(
    molecule: dict[str, torch.Tensor],
    *,
    rotate: bool,
    translate: bool,
    noise_std: float,
) -> dict[str, torch.Tensor]:
    result = _clone_molecule(molecule)
    batch_size = len(molecule["ptr"]) - 1
    position = molecule["pos"]
    if rotate:
        rotation = _random_rotations(batch_size, position.device)
        position = torch.einsum(
            "ni,nij->nj", position, rotation[molecule["batch"]]
        )
    if translate:
        shift = torch.randn(batch_size, 3, device=position.device) * 5.0
        position = position + shift[molecule["batch"]]
    if noise_std:
        position = position + torch.randn_like(position) * noise_std
    result["pos"] = position
    return result


def _probability_differences(
    original: dict[str, torch.Tensor],
    changed: dict[str, torch.Tensor],
    mapping: tuple[str, str] = ("1", "2"),
) -> np.ndarray:
    values = []
    for original_role, changed_role in zip(("1", "2"), mapping):
        first = torch.sigmoid(original[f"logits{original_role}"])
        second = torch.sigmoid(changed[f"logits{changed_role}"])
        values.append((first - second).abs().float().cpu().numpy())
    return np.concatenate(values)


def _topk_consistency(
    original: dict[str, torch.Tensor],
    changed: dict[str, torch.Tensor],
    batch: dict,
    mapping: tuple[str, str] = ("1", "2"),
    k: int = 3,
) -> tuple[int, int]:
    equal = total = 0
    for original_role, changed_role in zip(("1", "2"), mapping):
        ptr = batch[f"d{original_role}"]["ptr"]
        first = original[f"logits{original_role}"]
        second = changed[f"logits{changed_role}"]
        for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
            count = min(k, end - start)
            a = set(torch.topk(first[start:end], count).indices.tolist())
            b = set(torch.topk(second[start:end], count).indices.tolist())
            equal += int(a == b)
            total += 1
    return equal, total


def _mapped_atom_predictions(
    output: dict[str, torch.Tensor],
    batch: dict,
    mapping: tuple[str, str] = ("1", "2"),
) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    probabilities = []
    for original_role, changed_role in zip(("1", "2"), mapping):
        targets.append(batch[f"target{original_role}"])
        probabilities.append(
            torch.sigmoid(output[f"logits{changed_role}"])
        )
    return (
        torch.cat(targets).float().cpu().numpy(),
        torch.cat(probabilities).float().cpu().numpy(),
    )


def _topk_target_hits(
    output: dict[str, torch.Tensor],
    batch: dict,
    mapping: tuple[str, str] = ("1", "2"),
    k: int = 3,
) -> tuple[int, int]:
    hits = total = 0
    for original_role, changed_role in zip(("1", "2"), mapping):
        ptr = batch[f"d{original_role}"]["ptr"]
        target = batch[f"target{original_role}"]
        logits = output[f"logits{changed_role}"]
        for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
            local_target = target[start:end]
            if not bool((local_target > 0.5).any()):
                continue
            count = min(k, end - start)
            selected = torch.topk(logits[start:end], count).indices
            hits += int(bool((local_target[selected] > 0.5).any()))
            total += 1
    return hits, total


@torch.no_grad()
def robustness_analysis(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device_name: str,
    max_examples: int,
    noise_std: float,
    seed: int,
    precision: str,
    output_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = _device(device_name)
    config, model, state = _load_model(
        config_path, checkpoint_path, device
    )
    loader = _test_loader(
        config, max_examples=max_examples, seed=seed
    )
    configured_precision = str(
        config["training"].get("mixed_precision", "fp16")
    )
    amp_dtype = (
        None
        if precision == "fp32"
        else _amp_dtype(configured_precision, device)
    )
    differences = {
        "rotation": [],
        "translation": [],
        "rotation_translation": [],
        "coordinate_noise": [],
        "reactant_swap": [],
    }
    topk = {key: [0, 0] for key in differences}
    target_blocks: list[np.ndarray] = []
    probability_blocks: dict[str, list[np.ndarray]] = {
        name: [] for name in ("original", *differences)
    }
    target_hits = {
        name: [0, 0] for name in ("original", *differences)
    }
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            original = model(batch["d1"], batch["d2"])
            variants = {
                "rotation": model(
                    _transform_positions(
                        batch["d1"],
                        rotate=True,
                        translate=False,
                        noise_std=0.0,
                    ),
                    _transform_positions(
                        batch["d2"],
                        rotate=True,
                        translate=False,
                        noise_std=0.0,
                    ),
                ),
                "translation": model(
                    _transform_positions(
                        batch["d1"],
                        rotate=False,
                        translate=True,
                        noise_std=0.0,
                    ),
                    _transform_positions(
                        batch["d2"],
                        rotate=False,
                        translate=True,
                        noise_std=0.0,
                    ),
                ),
                "rotation_translation": model(
                    _transform_positions(
                        batch["d1"],
                        rotate=True,
                        translate=True,
                        noise_std=0.0,
                    ),
                    _transform_positions(
                        batch["d2"],
                        rotate=True,
                        translate=True,
                        noise_std=0.0,
                    ),
                ),
                "coordinate_noise": model(
                    _transform_positions(
                        batch["d1"],
                        rotate=False,
                        translate=False,
                        noise_std=noise_std,
                    ),
                    _transform_positions(
                        batch["d2"],
                        rotate=False,
                        translate=False,
                        noise_std=noise_std,
                    ),
                ),
                "reactant_swap": model(batch["d2"], batch["d1"]),
            }
        targets, probabilities = _mapped_atom_predictions(
            original, batch
        )
        target_blocks.append(targets)
        probability_blocks["original"].append(probabilities)
        hits, total = _topk_target_hits(original, batch)
        target_hits["original"][0] += hits
        target_hits["original"][1] += total
        for name, changed in variants.items():
            mapping = ("2", "1") if name == "reactant_swap" else ("1", "2")
            differences[name].append(
                _probability_differences(original, changed, mapping)
            )
            hits, total = _topk_consistency(
                original, changed, batch, mapping
            )
            topk[name][0] += hits
            topk[name][1] += total
            _, probabilities = _mapped_atom_predictions(
                changed, batch, mapping
            )
            probability_blocks[name].append(probabilities)
            hits, total = _topk_target_hits(
                changed, batch, mapping
            )
            target_hits[name][0] += hits
            target_hits[name][1] += total
    all_targets = np.concatenate(target_blocks)
    task_metrics = {
        name: {
            **_binary_metrics(
                all_targets,
                np.concatenate(blocks),
                threshold=0.5,
            ),
            "top3_atom_hit_rate": (
                target_hits[name][0] / max(target_hits[name][1], 1)
            ),
        }
        for name, blocks in probability_blocks.items()
    }
    result: dict[str, Any] = {
        "examples": min(max_examples, len(loader.dataset)),
        "noise_std_angstrom": noise_std,
        "precision": (
            "fp32" if amp_dtype is None else configured_precision
        ),
        "checkpoint_epoch": int(state["epoch"]),
        "original_task_metrics": task_metrics["original"],
        "conditions": {},
    }
    for name, blocks in differences.items():
        values = np.concatenate(blocks)
        result["conditions"][name] = {
            "probability_mae": float(values.mean()),
            "probability_max_abs": float(values.max()),
            "top3_set_exact_rate": topk[name][0] / max(topk[name][1], 1),
            "task_metrics_at_0.5": task_metrics[name],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def _history_seconds(checkpoint_path: Path) -> dict[str, float | int] | None:
    path = checkpoint_path.parent / "history.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seconds = [float(row["seconds"]) for row in rows if row.get("seconds")]
    if not seconds:
        return None
    return {
        "epochs_recorded": len(seconds),
        "mean_epoch_seconds": float(np.mean(seconds)),
        "median_epoch_seconds": float(np.median(seconds)),
    }


@torch.no_grad()
def efficiency_analysis(
    models: list[tuple[str, Path, Path]],
    *,
    device_name: str,
    warmup: int,
    steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    device = _device(device_name)
    rows = []
    for name, config_path, checkpoint_path in models:
        config, model, state = _load_model(
            config_path, checkpoint_path, device
        )
        loader = _test_loader(config)
        batch = _move_batch(next(iter(loader)), device)
        batch_size = len(batch["d1"]["ptr"]) - 1
        amp_dtype = _amp_dtype(
            str(config["training"].get("mixed_precision", "fp16")),
            device,
        )

        def forward():
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                return model(batch["d1"], batch["d2"])

        for _ in range(warmup):
            forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(steps):
            forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        row: dict[str, Any] = {
            "model": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_megabytes": checkpoint_path.stat().st_size / 2**20,
            "batch_size": batch_size,
            "latency_ms_per_batch": seconds / steps * 1000,
            "latency_ms_per_reaction": seconds / steps / batch_size * 1000,
            "throughput_reactions_per_second": steps * batch_size / seconds,
            "checkpoint_epoch": int(state["epoch"]),
        }
        if device.type == "cuda":
            row["peak_gpu_megabytes"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
            )
        history = _history_seconds(checkpoint_path)
        if history:
            row.update(history)
        rows.append(row)
        del model, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {"warmup_steps": warmup, "timed_steps": steps, "rows": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "efficiency.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / "efficiency.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return result


def _model_argument(value: list[str]) -> tuple[str, Path, Path]:
    if len(value) != 3:
        raise ValueError("--model requires NAME CONFIG CHECKPOINT")
    return value[0], Path(value[1]), Path(value[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reaction-site group, faithfulness, robustness diagnostics."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    group = sub.add_parser("group")
    group.add_argument(
        "--model", nargs=3, action="append", required=True,
        metavar=("NAME", "CONFIG", "CHECKPOINT")
    )
    group.add_argument("--device", default="cuda")
    group.add_argument("--threshold", type=float, default=0.5)
    group.add_argument("--output-dir", type=Path, required=True)

    faith = sub.add_parser("faithfulness")
    faith.add_argument("--config", type=Path, required=True)
    faith.add_argument("--checkpoint", type=Path, required=True)
    faith.add_argument("--device", default="cuda")
    faith.add_argument("--max-examples", type=int, default=1000)
    faith.add_argument("--iterations", type=int, default=10000)
    faith.add_argument("--seed", type=int, default=17)
    faith.add_argument("--output-dir", type=Path, required=True)

    robust = sub.add_parser("robustness")
    robust.add_argument("--config", type=Path, required=True)
    robust.add_argument("--checkpoint", type=Path, required=True)
    robust.add_argument("--device", default="cuda")
    robust.add_argument("--max-examples", type=int, default=1000)
    robust.add_argument("--noise-std", type=float, default=0.05)
    robust.add_argument("--seed", type=int, default=17)
    robust.add_argument(
        "--precision",
        choices=("config", "fp32"),
        default="config",
        help="Use configured mixed precision or force FP32.",
    )
    robust.add_argument("--output-dir", type=Path, required=True)

    efficiency = sub.add_parser("efficiency")
    efficiency.add_argument(
        "--model", nargs=3, action="append", required=True,
        metavar=("NAME", "CONFIG", "CHECKPOINT")
    )
    efficiency.add_argument("--device", default="cuda")
    efficiency.add_argument("--warmup", type=int, default=10)
    efficiency.add_argument("--steps", type=int, default=50)
    efficiency.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "group":
        group_analysis(
            [_model_argument(value) for value in args.model],
            device_name=args.device,
            threshold=args.threshold,
            output_dir=args.output_dir,
        )
    elif args.command == "faithfulness":
        faithfulness_analysis(
            args.config,
            args.checkpoint,
            device_name=args.device,
            max_examples=args.max_examples,
            iterations=args.iterations,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.command == "robustness":
        robustness_analysis(
            args.config,
            args.checkpoint,
            device_name=args.device,
            max_examples=args.max_examples,
            noise_std=args.noise_std,
            seed=args.seed,
            precision=args.precision,
            output_dir=args.output_dir,
        )
    elif args.command == "efficiency":
        efficiency_analysis(
            [_model_argument(value) for value in args.model],
            device_name=args.device,
            warmup=args.warmup,
            steps=args.steps,
            output_dir=args.output_dir,
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
