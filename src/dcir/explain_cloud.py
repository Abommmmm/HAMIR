from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import (
    DDIPairDataset,
    collate_molecules,
    make_collate_fn,
    move_batch_to_device,
    move_molecule_to_device,
)
from .losses import build_top_response_keep_masks
from .models import DCIR
from .train_cloud import load_audit


def _distribution_metrics(values: np.ndarray, fraction: float) -> tuple[float, float]:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if len(values) <= 1 or float(values.sum()) <= 0:
        return 0.0, 0.0
    probability = values / values.sum()
    entropy = -float(
        np.sum(probability * np.log(np.maximum(probability, 1e-12)))
    ) / math.log(len(values))
    count = max(1, int(math.ceil(len(values) * fraction)))
    top_mass = float(np.sort(probability)[-count:].sum())
    return 1.0 - entropy, top_mass


def _bernoulli_js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0 - eps)
    midpoint = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def _mean(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _median(values) -> float | None:
    values = list(values)
    return float(np.median(values)) if values else None


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_model(config: dict, checkpoint_path: Path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    num_classes = int(checkpoint["num_classes"])
    model = DCIR(num_classes=num_classes, **dict(config["model"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def quantify_explanations(
    config: dict,
    checkpoint_path: Path,
    result_dir: Path,
    output_dir: Path,
    mask_fraction: float,
    device: torch.device,
    num_workers: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    num_classes = int(audit["type_count"])
    task_mode = str(audit["task_mode"])
    if task_mode != "multilabel":
        raise ValueError("DeepDDI explanation analysis expects multilabel mode.")
    seed = int(config.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, checkpoint = _load_model(config, checkpoint_path, device)

    dataset = DDIPairDataset(
        paths["records"],
        paths["split"],
        paths["conformers"],
        "test",
        num_classes,
        task_mode,
        seed + 2,
        include_alternates=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=(
            int(config["training"].get("num_workers", 8))
            if num_workers is None
            else num_workers
        ),
        collate_fn=make_collate_fn(
            task_mode,
            cutoff=float(config.get("data", {}).get("cutoff", 5.0)),
        ),
        pin_memory=True,
    )
    precision = str(
        config["training"].get("mixed_precision", "none")
    ).lower()
    use_bf16 = precision == "bf16"
    sample_rows: list[dict] = []
    label_rows: list[dict] = []
    sample_index = 0

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="quantify explanations",
            dynamic_ncols=True,
        ):
            batch = move_batch_to_device(batch, device, include_alt=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                base = model(batch["d1"], batch["d2"])
                keep1, keep2 = build_top_response_keep_masks(
                    base,
                    batch["target"],
                    batch["d1"]["ptr"],
                    batch["d2"]["ptr"],
                    task_mode,
                    fraction=mask_fraction,
                )
                masked = model(
                    batch["d1"],
                    batch["d2"],
                    keep1=keep1,
                    keep2=keep2,
                )
                alternate = model(batch["d1_alt"], batch["d2_alt"])

            probability = torch.sigmoid(
                base["interaction_logits"].float()
            ).cpu().numpy()
            masked_probability = torch.sigmoid(
                masked["interaction_logits"].float()
            ).cpu().numpy()
            alternate_probability = torch.sigmoid(
                alternate["interaction_logits"].float()
            ).cpu().numpy()
            target = batch["target"].cpu().numpy().astype(bool)
            response1 = base["d1_atom_response"].float().cpu().numpy()
            response2 = base["d2_atom_response"].float().cpu().numpy()
            ptr1 = batch["d1"]["ptr"].numpy()
            ptr2 = batch["d2"]["ptr"].numpy()
            keep1_np = keep1.cpu().numpy()
            keep2_np = keep2.cpu().numpy()

            for local_index in range(len(target)):
                labels = np.where(target[local_index])[0]
                start1, end1 = ptr1[local_index : local_index + 2]
                start2, end2 = ptr2[local_index : local_index + 2]
                faith = (
                    probability[local_index, labels]
                    - masked_probability[local_index, labels]
                )
                consistency_abs = np.abs(
                    probability[local_index] - alternate_probability[local_index]
                )
                consistency_js = _bernoulli_js(
                    probability[local_index], alternate_probability[local_index]
                )
                concentrations = []
                top_masses = []
                for label in labels:
                    label_concentrations = []
                    label_top_masses = []
                    for response in (
                        response1[start1:end1, label],
                        response2[start2:end2, label],
                    ):
                        concentration, top_mass = _distribution_metrics(
                            response, mask_fraction
                        )
                        concentrations.append(concentration)
                        top_masses.append(top_mass)
                        label_concentrations.append(concentration)
                        label_top_masses.append(top_mass)
                    label_rows.append(
                        {
                            "sample_index": sample_index,
                            "class_internal": int(label),
                            "class_raw": str(
                                audit["type_values"][int(label)]
                            ),
                            "original_probability": float(
                                probability[local_index, label]
                            ),
                            "masked_probability": float(
                                masked_probability[local_index, label]
                            ),
                            "faithfulness_drop": float(
                                probability[local_index, label]
                                - masked_probability[local_index, label]
                            ),
                            "conformer_probability_abs_diff": float(
                                consistency_abs[label]
                            ),
                            "conformer_js": float(consistency_js[label]),
                            "response_concentration": float(
                                np.mean(label_concentrations)
                            ),
                            "top_response_mass": float(
                                np.mean(label_top_masses)
                            ),
                        }
                    )

                removed = int((~keep1_np[start1:end1]).sum()) + int(
                    (~keep2_np[start2:end2]).sum()
                )
                atom_count = int(end1 - start1 + end2 - start2)
                sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "positive_label_count": int(len(labels)),
                        "original_true_probability": float(
                            probability[local_index, labels].mean()
                        ),
                        "masked_true_probability": float(
                            masked_probability[local_index, labels].mean()
                        ),
                        "faithfulness_drop": float(faith.mean()),
                        "faithfulness_positive_fraction": float(
                            (faith > 0).mean()
                        ),
                        "conformer_positive_abs_diff": float(
                            consistency_abs[labels].mean()
                        ),
                        "conformer_all_abs_diff": float(
                            consistency_abs.mean()
                        ),
                        "conformer_positive_js": float(
                            consistency_js[labels].mean()
                        ),
                        "response_concentration": float(
                            np.mean(concentrations)
                        ),
                        "top_response_mass": float(np.mean(top_masses)),
                        "removed_atom_fraction": float(
                            removed / max(atom_count, 1)
                        ),
                    }
                )
                sample_index += 1

    sample_frame = pd.DataFrame(sample_rows)
    label_frame = pd.DataFrame(label_rows)
    per_class = (
        label_frame.groupby(["class_internal", "class_raw"], as_index=False)
        .agg(
            support=("sample_index", "count"),
            mean_original_probability=("original_probability", "mean"),
            mean_faithfulness_drop=("faithfulness_drop", "mean"),
            median_faithfulness_drop=("faithfulness_drop", "median"),
            positive_faithfulness_rate=(
                "faithfulness_drop",
                lambda values: float((values > 0).mean()),
            ),
            mean_conformer_abs_diff=(
                "conformer_probability_abs_diff",
                "mean",
            ),
            mean_conformer_js=("conformer_js", "mean"),
            mean_response_concentration=(
                "response_concentration",
                "mean",
            ),
            mean_top_response_mass=("top_response_mass", "mean"),
        )
        .sort_values("class_internal")
    )
    metrics = {
        "dataset": audit.get("dataset"),
        "split": "test",
        "examples": int(len(sample_frame)),
        "positive_label_instances": int(len(label_frame)),
        "classes": num_classes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation": checkpoint.get("validation"),
        "mask_fraction": mask_fraction,
        "mean_faithfulness_drop": float(
            sample_frame["faithfulness_drop"].mean()
        ),
        "median_faithfulness_drop": float(
            sample_frame["faithfulness_drop"].median()
        ),
        "positive_faithfulness_sample_rate": float(
            (sample_frame["faithfulness_drop"] > 0).mean()
        ),
        "mean_conformer_positive_abs_diff": float(
            sample_frame["conformer_positive_abs_diff"].mean()
        ),
        "mean_conformer_all_abs_diff": float(
            sample_frame["conformer_all_abs_diff"].mean()
        ),
        "mean_conformer_positive_js": float(
            sample_frame["conformer_positive_js"].mean()
        ),
        "mean_response_concentration": float(
            sample_frame["response_concentration"].mean()
        ),
        "mean_top_response_mass": float(
            sample_frame["top_response_mass"].mean()
        ),
        "mean_removed_atom_fraction": float(
            sample_frame["removed_atom_fraction"].mean()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(
        output_dir / "per_sample_explanation_metrics.csv", index=False
    )
    label_frame.to_csv(
        output_dir / "per_label_explanation_metrics.csv", index=False
    )
    per_class.to_csv(
        output_dir / "per_class_explanation_metrics.csv", index=False
    )
    _atomic_write_json(output_dir / "explanation_metrics.json", metrics)
    return sample_frame, per_class, metrics


def select_cases(
    result_dir: Path,
    output_dir: Path,
    count_correct: int,
    count_error: int,
    count_rare: int,
) -> pd.DataFrame:
    with np.load(
        result_dir / "predictions.npz", allow_pickle=False
    ) as archive:
        target = np.asarray(archive["target"]).astype(bool)
        probability = np.asarray(archive["probability"])
        prediction = np.asarray(archive["prediction"]).astype(bool)
    class_metrics = pd.read_csv(
        result_dir / "per_class_metrics.csv",
        dtype={"class_raw": str},
    )
    rows = []
    selected: set[int] = set()

    exact = np.all(target == prediction, axis=1)
    positive_confidence = np.where(target, probability, np.nan)
    correct_score = np.nanmean(positive_confidence, axis=1)
    correct_candidates = np.where(exact)[0]
    for index in correct_candidates[
        np.argsort(correct_score[correct_candidates])[::-1]
    ][:count_correct]:
        labels = np.where(target[index])[0]
        focus = int(labels[np.argmax(probability[index, labels])])
        rows.append(
            {
                "case_type": "high_confidence_correct",
                "sample_index": int(index),
                "focus_class_internal": focus,
                "focus_class_raw": str(focus + 1),
            }
        )
        selected.add(int(index))

    error_count = np.not_equal(target, prediction).sum(axis=1)
    for index in np.argsort(error_count)[::-1]:
        if len([row for row in rows if row["case_type"] == "hard_error"]) >= count_error:
            break
        if error_count[index] == 0 or int(index) in selected:
            continue
        false_negative = np.where(target[index] & ~prediction[index])[0]
        false_positive = np.where(~target[index] & prediction[index])[0]
        if len(false_negative):
            focus = int(
                false_negative[np.argmin(probability[index, false_negative])]
            )
        else:
            focus = int(
                false_positive[np.argmax(probability[index, false_positive])]
            )
        rows.append(
            {
                "case_type": "hard_error",
                "sample_index": int(index),
                "focus_class_internal": focus,
                "focus_class_raw": str(focus + 1),
            }
        )
        selected.add(int(index))

    rare_classes = class_metrics.sort_values(["support", "f1"])
    rare_added = 0
    for class_row in rare_classes.itertuples(index=False):
        label = int(class_row.class_internal)
        candidates = np.where(target[:, label] & prediction[:, label])[0]
        candidates = candidates[np.argsort(probability[candidates, label])[::-1]]
        for index in candidates:
            if int(index) in selected:
                continue
            rows.append(
                {
                    "case_type": "rare_class_correct",
                    "sample_index": int(index),
                    "focus_class_internal": label,
                    "focus_class_raw": str(class_row.class_raw),
                }
            )
            selected.add(int(index))
            rare_added += 1
            break
        if rare_added >= count_rare:
            break

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "case_selection.csv", index=False)
    return frame


def _molecule_from_archive(path: Path):
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    with np.load(path, allow_pickle=False) as archive:
        numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        bond_index = np.asarray(archive["bond_index"], dtype=np.int64)
        bond_features = np.asarray(
            archive["bond_features"], dtype=np.float32
        )
    editable = Chem.RWMol()
    for number in numbers:
        editable.AddAtom(Chem.Atom(int(number)))
    seen = set()
    types = (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    )
    for edge_index in range(bond_index.shape[1]):
        source = int(bond_index[0, edge_index])
        target = int(bond_index[1, edge_index])
        pair = tuple(sorted((source, target)))
        if source == target or pair in seen:
            continue
        seen.add(pair)
        kind = int(np.argmax(bond_features[edge_index, :4]))
        editable.AddBond(pair[0], pair[1], types[kind])
        if kind == 3:
            editable.GetAtomWithIdx(pair[0]).SetIsAromatic(True)
            editable.GetAtomWithIdx(pair[1]).SetIsAromatic(True)
    molecule = editable.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        molecule.UpdatePropertyCache(strict=False)
    rdDepictor.Compute2DCoords(molecule)
    return molecule


def draw_atom_response(path: Path, response: np.ndarray, output_path: Path) -> None:
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = _molecule_from_archive(path)
    response = np.maximum(np.asarray(response, dtype=np.float64), 0.0)
    scale = float(np.quantile(response, 0.95)) if len(response) else 0.0
    normalized = np.clip(response / max(scale, 1e-12), 0.0, 1.0)
    top = set(np.argsort(response)[-min(10, len(response)) :].tolist())
    for atom in molecule.GetAtoms():
        if atom.GetIdx() in top:
            atom.SetProp("atomNote", str(atom.GetIdx()))
    highlights = list(range(len(response)))
    colors = {
        index: (
            1.0,
            float(1.0 - 0.82 * normalized[index]),
            float(1.0 - 0.82 * normalized[index]),
        )
        for index in highlights
    }
    radii = {
        index: float(0.18 + 0.32 * normalized[index])
        for index in highlights
    }
    drawer = rdMolDraw2D.MolDraw2DSVG(900, 600)
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=highlights,
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")


def export_cases(
    config: dict,
    checkpoint_path: Path,
    result_dir: Path,
    output_dir: Path,
    selection: pd.DataFrame,
    mask_fraction: float,
    device: torch.device,
) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    audit = load_audit(paths["audit_report"])
    num_classes = int(audit["type_count"])
    task_mode = str(audit["task_mode"])
    seed = int(config.get("seed", 17))
    model, _ = _load_model(config, checkpoint_path, device)
    dataset = DDIPairDataset(
        paths["records"],
        paths["split"],
        paths["conformers"],
        "test",
        num_classes,
        task_mode,
        seed + 2,
        include_alternates=False,
    )
    with np.load(
        result_dir / "predictions.npz", allow_pickle=False
    ) as archive:
        frozen_target = np.asarray(archive["target"])
        frozen_probability = np.asarray(archive["probability"])
        frozen_prediction = np.asarray(archive["prediction"])
    cutoff = float(config.get("data", {}).get("cutoff", 5.0))
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(
        list(selection.itertuples(index=False)),
        desc="export explanation cases",
        dynamic_ncols=True,
    ):
        index = int(row.sample_index)
        label = int(row.focus_class_internal)
        item = dataset[index]
        mol1 = move_molecule_to_device(
            collate_molecules([item["d1"]], cutoff), device
        )
        mol2 = move_molecule_to_device(
            collate_molecules([item["d2"]], cutoff), device
        )
        target = torch.as_tensor(
            frozen_target[index : index + 1],
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            output = model.predict(
                mol1,
                mol2,
                task_mode,
                return_explanations=True,
            )
            keep1, keep2 = build_top_response_keep_masks(
                output,
                target,
                mol1["ptr"],
                mol2["ptr"],
                task_mode,
                fraction=mask_fraction,
            )
            masked = model(mol1, mol2, keep1=keep1, keep2=keep2)
        current_probability = torch.sigmoid(
            output["interaction_logits"].float()
        )[0].cpu().numpy()
        masked_probability = torch.sigmoid(
            masked["interaction_logits"].float()
        )[0].cpu().numpy()
        response1 = (
            output["d1_atom_response"][:, label].float().cpu().numpy()
        )
        response2 = (
            output["d2_atom_response"][:, label].float().cpu().numpy()
        )
        cross12 = (
            output["cross_atom_contributions_12"][0][label]
            .float()
            .cpu()
            .numpy()
        )
        flat_top = np.argsort(np.abs(cross12).ravel())[
            -min(20, cross12.size) :
        ][::-1]
        cross_pairs = [
            {
                "d1_atom": int(np.unravel_index(flat, cross12.shape)[0]),
                "d2_atom": int(np.unravel_index(flat, cross12.shape)[1]),
                "contribution": float(
                    cross12[np.unravel_index(flat, cross12.shape)]
                ),
            }
            for flat in flat_top
        ]
        case_name = (
            f"{row.case_type}_{index:05d}_class_{str(row.focus_class_raw)}"
        )
        case_dir = cases_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        d1, d2 = item["drug_ids"]
        payload = {
            "case_type": row.case_type,
            "sample_index": index,
            "record_ids": item["record_ids"],
            "d1": d1,
            "d2": d2,
            "focus_class_internal": label,
            "focus_class_raw": str(row.focus_class_raw),
            "frozen_test_probability": float(
                frozen_probability[index, label]
            ),
            "frozen_test_prediction": bool(
                frozen_prediction[index, label]
            ),
            "target": bool(frozen_target[index, label]),
            "explanation_pass_probability": float(
                current_probability[label]
            ),
            "masked_probability": float(masked_probability[label]),
            "faithfulness_drop": float(
                current_probability[label] - masked_probability[label]
            ),
            "d1_atom_response": response1.tolist(),
            "d2_atom_response": response2.tolist(),
            "top_d1_atoms": [
                int(value)
                for value in np.argsort(response1)[
                    -min(10, len(response1)) :
                ][::-1]
            ],
            "top_d2_atoms": [
                int(value)
                for value in np.argsort(response2)[
                    -min(10, len(response2)) :
                ][::-1]
            ],
            "top_cross_atom_pairs": cross_pairs,
            "interpretation_notice": (
                "Atom responses and cross-atom contributions are model-derived "
                "attributions, not experimentally verified mechanisms."
            ),
        }
        _atomic_write_json(case_dir / "explanation.json", payload)
        draw_atom_response(
            paths["conformers"] / f"{d1}.npz",
            response1,
            case_dir / "d1_atom_response.svg",
        )
        draw_atom_response(
            paths["conformers"] / f"{d2}.npz",
            response2,
            case_dir / "d2_atom_response.svg",
        )


def write_report(
    metrics: dict,
    per_class: pd.DataFrame,
    selection: pd.DataFrame,
    output_dir: Path,
) -> None:
    def write_class_chart(
        metric: str,
        filename: str,
        title: str,
        *,
        allow_negative: bool,
    ) -> None:
        ranked = per_class.sort_values(metric)
        row_height = 25
        width = 940
        height = 70 + row_height * len(ranked)
        values = ranked[metric].to_numpy(dtype=float)
        if allow_negative:
            scale = max(float(np.max(np.abs(values))), 1e-12)
            origin = 470
            span = 380
        else:
            scale = max(float(np.max(values)), 1e-12)
            origin = 90
            span = 760
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2}" y="28" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20">{title}</text>',
        ]
        if allow_negative:
            elements.append(
                f'<line x1="{origin}" y1="42" x2="{origin}" '
                f'y2="{height - 10}" stroke="#555"/>'
            )
        for row_index, row in enumerate(ranked.itertuples(index=False)):
            value = float(getattr(row, metric))
            y = 48 + row_index * row_height
            bar = abs(value) / scale * span
            x = origin - bar if allow_negative and value < 0 else origin
            color = "#b85c4b" if value < 0 else "#3973ac"
            elements.extend(
                [
                    f'<text x="75" y="{y + 13}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="11">'
                    f'{row.class_raw}</text>',
                    f'<rect x="{x:.1f}" y="{y}" width="{bar:.1f}" '
                    f'height="16" fill="{color}"/>',
                    f'<text x="{x + bar + 6:.1f}" y="{y + 13}" '
                    f'font-family="sans-serif" font-size="11">'
                    f'{value:.5f} (n={int(row.support)})</text>',
                ]
            )
        elements.append("</svg>")
        (output_dir / filename).write_text(
            "\n".join(elements), encoding="utf-8"
        )

    write_class_chart(
        "mean_faithfulness_drop",
        "class_faithfulness.svg",
        "Per-class faithfulness probability drop",
        allow_negative=True,
    )
    write_class_chart(
        "mean_conformer_abs_diff",
        "class_conformer_consistency.svg",
        "Per-class conformer probability difference",
        allow_negative=False,
    )
    weakest_faith = per_class.sort_values("mean_faithfulness_drop").iloc[0]
    strongest_faith = per_class.sort_values(
        "mean_faithfulness_drop", ascending=False
    ).iloc[0]
    report = f"""# DeepDDI 冻结模型解释性分析

## 设置

- 测试药物对：{metrics['examples']:,}
- 正标签实例：{metrics['positive_label_instances']:,}
- 类别数：{metrics['classes']}
- Checkpoint epoch：{metrics['checkpoint_epoch']}
- 原子掩蔽比例：{metrics['mask_fraction']:.2f}
- 案例数量：{len(selection)}

## 定量解释结果

- 平均 Faithfulness probability drop：{metrics['mean_faithfulness_drop']:.6f}
- Faithfulness drop 中位数：{metrics['median_faithfulness_drop']:.6f}
- 掩蔽后概率下降的样本比例：{metrics['positive_faithfulness_sample_rate']:.6f}
- 构象间正标签概率平均绝对差：{metrics['mean_conformer_positive_abs_diff']:.6f}
- 构象间全类别概率平均绝对差：{metrics['mean_conformer_all_abs_diff']:.6f}
- 构象间正标签 JS divergence：{metrics['mean_conformer_positive_js']:.6f}
- 原子响应平均集中度：{metrics['mean_response_concentration']:.6f}
- Top {metrics['mask_fraction']:.0%} 原子平均响应质量：{metrics['mean_top_response_mass']:.6f}
- 实际掩蔽原子比例：{metrics['mean_removed_atom_fraction']:.6f}

## 类别差异

- Faithfulness 最强类别：{strongest_faith['class_raw']}
  （mean drop={strongest_faith['mean_faithfulness_drop']:.6f}）
- Faithfulness 最弱类别：{weakest_faith['class_raw']}
  （mean drop={weakest_faith['mean_faithfulness_drop']:.6f}）

## 产物

- `explanation_metrics.json`
- `per_sample_explanation_metrics.csv`
- `per_label_explanation_metrics.csv`
- `per_class_explanation_metrics.csv`
- `case_selection.csv`
- `cases/*/explanation.json`
- `cases/*/d1_atom_response.svg`
- `cases/*/d2_atom_response.svg`
- `class_faithfulness.svg`
- `class_conformer_consistency.svg`

解释结果表示模型归因，不等同于真实反应中心、物理键变化或实验机制。
"""
    (output_dir / "explanation_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantify and export explanations from a frozen DCIR model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask-fraction", type=float, default=0.10)
    parser.add_argument("--correct-cases", type=int, default=5)
    parser.add_argument("--error-cases", type=int, default=5)
    parser.add_argument("--rare-cases", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override DataLoader workers; use 0 for local Windows runs.",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Explanation analysis requires a CUDA GPU.")
    if not 0.0 < args.mask_fraction < 1.0:
        raise ValueError("--mask-fraction must be strictly between 0 and 1.")
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    sample_frame, per_class, metrics = quantify_explanations(
        config,
        args.checkpoint.resolve(),
        args.result_dir.resolve(),
        output_dir,
        args.mask_fraction,
        device,
        args.num_workers,
    )
    selection = select_cases(
        args.result_dir.resolve(),
        output_dir,
        args.correct_cases,
        args.error_cases,
        args.rare_cases,
    )
    export_cases(
        config,
        args.checkpoint.resolve(),
        args.result_dir.resolve(),
        output_dir,
        selection,
        args.mask_fraction,
        device,
    )
    write_report(metrics, per_class, selection, output_dir)
    print(f"All explanation outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
