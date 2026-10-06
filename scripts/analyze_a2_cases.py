#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
source_path = str(SOURCE)
if source_path in sys.path:
    sys.path.remove(source_path)
sys.path.insert(0, source_path)

from dcir.config import load_config
from dcir.reaction_sites import (
    ReactionSiteDataset,
    _build_model,
    _loader,
    _move_batch,
)


CONFIG = ROOT / "configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml"
CHECKPOINT = ROOT / "outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt"
PREDICTIONS = ROOT / "outputs/results/reaction_sites/uspto50k/painn_a2_hierarchical_motif/test_predictions.npz"
INDEX = ROOT / "outputs/reaction_sites/uspto50k/index/test.jsonl"
MOLECULES = ROOT / "outputs/reaction_sites/uspto50k/molecules"
OUTPUT = ROOT / "outputs/analysis/reaction_sites/uspto50k/a2_case_analysis"


CASE_SPECS = [
    (134, "high_confidence_exact", "High-confidence exact localization"),
    (803, "complex_exact", "Exact localization in a large molecular pair"),
    (50, "multicenter_exact", "Exact multi-center localization"),
    (520, "threshold_edge_exact", "Correct but threshold-sensitive localization"),
    (173, "near_miss_false_positive", "Near miss with one extra atom"),
    (2434, "failure_multicenter", "Failure on a dense multi-center reaction"),
]


@dataclass
class SamplePrediction:
    atom_probabilities: dict[str, np.ndarray]
    group_probabilities: dict[str, np.ndarray]


def load_records() -> list[dict[str, Any]]:
    with INDEX.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def recover_saved_atom_probabilities(
    records: list[dict[str, Any]],
    selected: set[int],
    batch_size: int = 64,
) -> dict[int, dict[str, np.ndarray]]:
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        logits = np.asarray(archive["logits"], dtype=np.float64)
    result: dict[int, dict[str, np.ndarray]] = {}
    flat_offset = 0
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        first_size = sum(int(row["d1"]["n_atoms"]) for row in batch)
        second_size = sum(int(row["d2"]["n_atoms"]) for row in batch)
        first_logits = logits[flat_offset : flat_offset + first_size]
        second_logits = logits[
            flat_offset + first_size : flat_offset + first_size + second_size
        ]
        first_offset = second_offset = 0
        for local_index, row in enumerate(batch):
            global_index = batch_start + local_index
            n1 = int(row["d1"]["n_atoms"])
            n2 = int(row["d2"]["n_atoms"])
            if global_index in selected:
                result[global_index] = {
                    "d1": sigmoid(first_logits[first_offset : first_offset + n1]),
                    "d2": sigmoid(
                        second_logits[second_offset : second_offset + n2]
                    ),
                }
            first_offset += n1
            second_offset += n2
        flat_offset += first_size + second_size
    if flat_offset != len(logits):
        raise ValueError("Prediction archive was not consumed exactly")
    return result


@torch.no_grad()
def infer_explicit_group_probabilities(
    selected_indices: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    config = load_config(CONFIG)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = _build_model(config)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    paths = {key: Path(value) for key, value in config["paths"].items()}
    dataset = ReactionSiteDataset(paths["test_index"], paths["molecule_cache"])
    subset = Subset(dataset, selected_indices)
    loader = _loader(
        subset,
        batch_size=len(selected_indices),
        workers=0,
        cutoff=float(config["model"].get("cutoff", 5.0)),
        shuffle=False,
    )
    batch = _move_batch(next(iter(loader)), torch.device("cpu"))
    output = model(batch["d1"], batch["d2"])
    result: dict[int, dict[str, np.ndarray]] = {
        index: {} for index in selected_indices
    }
    for role in ("1", "2"):
        probabilities = torch.sigmoid(output[f"group_logits{role}"]).cpu().numpy()
        motif_batch = batch[f"d{role}"]["motif_batch"].cpu().numpy()
        for local_index, global_index in enumerate(selected_indices):
            result[global_index][f"d{role}"] = probabilities[
                motif_batch == local_index
            ].astype(np.float64)
    return result


def binary_counts(target: np.ndarray, probability: np.ndarray) -> tuple[int, int, int]:
    truth = target.astype(bool)
    prediction = probability >= 0.5
    return (
        int(np.count_nonzero(prediction & truth)),
        int(np.count_nonzero(prediction & ~truth)),
        int(np.count_nonzero(~prediction & truth)),
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def blend(stops: list[tuple[float, tuple[float, float, float]]], value: float):
    value = float(np.clip(value, 0.0, 1.0))
    for (left_x, left), (right_x, right) in zip(stops[:-1], stops[1:]):
        if value <= right_x:
            ratio = (value - left_x) / max(right_x - left_x, 1e-12)
            return tuple(
                float(left[i] + ratio * (right[i] - left[i])) for i in range(3)
            )
    return stops[-1][1]


ATOM_STOPS = [
    (0.0, (0.92, 0.95, 0.99)),
    (0.5, (0.99, 0.84, 0.45)),
    (1.0, (0.83, 0.12, 0.12)),
]
GROUP_STOPS = [
    (0.0, (0.95, 0.94, 0.98)),
    (0.5, (0.72, 0.59, 0.85)),
    (1.0, (0.35, 0.08, 0.55)),
]


def molecule_panel(
    smiles: str,
    *,
    mode: str,
    truth_indices: list[int],
    atom_probabilities: np.ndarray,
    motif_ids: list[int],
    group_probabilities: np.ndarray,
    width: int = 650,
    height: int = 390,
) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    AllChem.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.atomHighlightsAreCircles = True
    options.highlightBondWidthMultiplier = 10
    options.legendFontSize = 20

    if mode == "truth":
        atoms = list(truth_indices)
        colors = {index: (0.88, 0.12, 0.12) for index in atoms}
        radii = {index: 0.48 for index in atoms}
        legend = "Ground truth (red)"
    elif mode == "atom":
        atoms = list(range(molecule.GetNumAtoms()))
        colors = {
            index: blend(ATOM_STOPS, float(atom_probabilities[index]))
            for index in atoms
        }
        radii = {
            index: 0.52 if atom_probabilities[index] >= 0.5 else 0.30
            for index in atoms
        }
        legend = "A2 atom probability (large circle: p >= 0.5)"
    elif mode == "group":
        atoms = list(range(molecule.GetNumAtoms()))
        colors = {
            index: blend(
                GROUP_STOPS,
                float(group_probabilities[int(motif_ids[index])]),
            )
            for index in atoms
        }
        radii = {
            index: (
                0.50
                if group_probabilities[int(motif_ids[index])] >= 0.5
                else 0.30
            )
            for index in atoms
        }
        legend = "A2 explicit BRICS-group probability"
    else:
        raise ValueError(mode)

    drawer.DrawMolecule(
        molecule,
        legend=legend,
        highlightAtoms=atoms,
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    return Image.open(
        __import__("io").BytesIO(drawer.GetDrawingText())
    ).convert("RGB")


def top_indices(probabilities: np.ndarray, count: int = 3) -> list[int]:
    return np.argsort(-probabilities)[: min(count, len(probabilities))].astype(int).tolist()


def build_case_summary(
    index: int,
    slug: str,
    label: str,
    record: dict[str, Any],
    prediction: SamplePrediction,
) -> dict[str, Any]:
    tp = fp = fn = 0
    roles: dict[str, Any] = {}
    exact = True
    top3_pair = True
    true_probabilities: list[float] = []
    false_probabilities: list[float] = []
    for role in ("d1", "d2"):
        data = record[role]
        probability = prediction.atom_probabilities[role]
        target = np.zeros(len(probability), dtype=np.uint8)
        target[np.asarray(data["positive_indices"], dtype=np.int64)] = 1
        counts = binary_counts(target, probability)
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
        predicted = np.flatnonzero(probability >= 0.5).astype(int).tolist()
        exact &= predicted == sorted(data["positive_indices"])
        top3 = top_indices(probability)
        top3_pair &= bool(set(top3) & set(data["positive_indices"]))
        true_probabilities.extend(probability[target.astype(bool)].tolist())
        false_probabilities.extend(probability[~target.astype(bool)].tolist())
        group_probability = prediction.group_probabilities[role]
        true_motifs = sorted(
            {int(data["motif_ids"][i]) for i in data["positive_indices"]}
        )
        roles[role] = {
            "smiles": data["smiles"],
            "n_atoms": int(data["n_atoms"]),
            "true_atoms": list(map(int, data["positive_indices"])),
            "predicted_atoms_p_ge_0_5": predicted,
            "top3_atoms": top3,
            "top3_probabilities": [
                float(probability[item]) for item in top3
            ],
            "true_atom_probabilities": [
                float(probability[item]) for item in data["positive_indices"]
            ],
            "true_motifs": true_motifs,
            "top_group": int(np.argmax(group_probability)),
            "top_group_probability": float(np.max(group_probability)),
            "true_group_probabilities": {
                str(motif): float(group_probability[motif])
                for motif in true_motifs
            },
        }
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return {
        "case_index": index,
        "case_slug": slug,
        "case_label": label,
        "sample_id": record["sample_id"],
        "reaction_class": int(record["reaction_class"]),
        "total_atoms": int(record["d1"]["n_atoms"] + record["d2"]["n_atoms"]),
        "positive_atoms": int(
            len(record["d1"]["positive_indices"])
            + len(record["d2"]["positive_indices"])
        ),
        "atom_f1": float(f1),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "exact_pair": bool(exact),
        "top3_hit_both_reactants": bool(top3_pair),
        "minimum_true_atom_probability": float(min(true_probabilities)),
        "maximum_false_atom_probability": float(max(false_probabilities)),
        "roles": roles,
    }


def draw_case(summary: dict[str, Any], record: dict[str, Any], prediction: SamplePrediction):
    panel_width, panel_height = 650, 390
    margin = 34
    header_height = 155
    footer_height = 130
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + panel_width * 3,
            header_height + panel_height * 2 + footer_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 92), fill=(26, 54, 93))
    draw.text(
        (margin, 18),
        f"{summary['case_label']} — {summary['sample_id']}",
        fill="white",
        font=font(34, bold=True),
    )
    status = (
        f"Class {summary['reaction_class']}  |  Atom-F1 {summary['atom_f1']:.3f}"
        f"  |  Exact-Pair {'yes' if summary['exact_pair'] else 'no'}"
        f"  |  TP/FP/FN {summary['true_positives']}/{summary['false_positives']}/{summary['false_negatives']}"
    )
    draw.text((margin, 103), status, fill=(33, 48, 70), font=font(24))

    for row_index, role in enumerate(("d1", "d2")):
        data = record[role]
        panels = [
            molecule_panel(
                data["smiles"],
                mode="truth",
                truth_indices=data["positive_indices"],
                atom_probabilities=prediction.atom_probabilities[role],
                motif_ids=data["motif_ids"],
                group_probabilities=prediction.group_probabilities[role],
                width=panel_width,
                height=panel_height,
            ),
            molecule_panel(
                data["smiles"],
                mode="atom",
                truth_indices=data["positive_indices"],
                atom_probabilities=prediction.atom_probabilities[role],
                motif_ids=data["motif_ids"],
                group_probabilities=prediction.group_probabilities[role],
                width=panel_width,
                height=panel_height,
            ),
            molecule_panel(
                data["smiles"],
                mode="group",
                truth_indices=data["positive_indices"],
                atom_probabilities=prediction.atom_probabilities[role],
                motif_ids=data["motif_ids"],
                group_probabilities=prediction.group_probabilities[role],
                width=panel_width,
                height=panel_height,
            ),
        ]
        y = header_height + row_index * panel_height
        for column, panel in enumerate(panels):
            canvas.paste(panel, (margin + column * panel_width, y))
        draw.rounded_rectangle(
            (4, y + 10, margin - 6, y + 75),
            radius=8,
            fill=(47, 117, 181),
        )
        draw.text((9, y + 26), role.upper(), fill="white", font=font(18, bold=True))

    y = header_height + panel_height * 2 + 12
    footer_lines = []
    for role in ("d1", "d2"):
        role_data = summary["roles"][role]
        footer_lines.append(
            f"{role.upper()}: true atoms {role_data['true_atoms']} | "
            f"predicted p≥0.5 {role_data['predicted_atoms_p_ge_0_5']} | "
            f"top-3 {role_data['top3_atoms']} | "
            f"true motifs {role_data['true_motifs']} | "
            f"top group G{role_data['top_group']}={role_data['top_group_probability']:.3f}"
        )
    footer_lines.append(
        "Color scales: atom probability pale blue→amber→red; "
        "explicit group probability pale lavender→purple."
    )
    for offset, text in enumerate(footer_lines):
        draw.text((margin, y + offset * 34), text, fill=(38, 50, 69), font=font(20))
    return canvas


def narrative(summary: dict[str, Any]) -> str:
    slug = summary["case_slug"]
    if slug == "high_confidence_exact":
        return (
            "A2 isolates the two labeled atoms without introducing any false "
            "positive. Both true-site probabilities are close to one, while "
            "all non-center probabilities remain far below the decision boundary."
        )
    if slug == "complex_exact":
        return (
            "The pair contains 81 atoms, yet A2 retains exact localization. "
            "This case shows that high atom count alone does not force diffuse "
            "attention or spurious positive sites."
        )
    if slug == "multicenter_exact":
        return (
            "Six participating atoms are recovered exactly. Atom and explicit "
            "BRICS-group heatmaps concentrate on the same reactive neighborhoods, "
            "supporting hierarchical consistency in a multi-center reaction."
        )
    if slug == "threshold_edge_exact":
        return (
            "The discrete prediction is correct, but the weakest true-site "
            "probability is only slightly above 0.5. The chemical ranking is "
            "useful, while the exact classification is sensitive to calibration."
        )
    if slug == "near_miss_false_positive":
        return (
            "All five true atoms are recovered, but one additional atom crosses "
            "the threshold. This is a localized overprediction rather than a "
            "failure to identify the reactive region."
        )
    return (
        "The atom head places none of the six labeled atoms above 0.5, although "
        "Top-3 still intersects the true region in both reactants. In contrast, "
        "the explicit group head assigns near-unit probability to the correct "
        "BRICS groups. This exposes a cross-level inconsistency: motif recognition "
        "succeeds, but the signal is not converted into calibrated atom scores."
    )


def build_3d_case(
    summary: dict[str, Any],
    record: dict[str, Any],
    prediction: SamplePrediction,
) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("d1", "d2"):
        data = record[role]
        with np.load(MOLECULES / f"{data['key']}.npz") as molecule:
            positions = np.asarray(molecule["pos"], dtype=np.float64)
            atomic_numbers = np.asarray(
                molecule["atomic_numbers"], dtype=np.int64
            )
            bond_index = np.asarray(molecule["bond_index"], dtype=np.int64)

        positions -= positions.mean(axis=0, keepdims=True)
        unique_bonds = sorted(
            {
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in bond_index.T
                if int(left) != int(right)
            }
        )
        truth = set(map(int, data["positive_indices"]))
        atom_probability = prediction.atom_probabilities[role]
        group_probability = prediction.group_probabilities[role]
        motif_ids = list(map(int, data["motif_ids"]))
        roles[role] = {
            "smiles": data["smiles"],
            "atoms": [
                {
                    "index": index,
                    "z": int(atomic_numbers[index]),
                    "position": [
                        round(float(value), 4) for value in positions[index]
                    ],
                    "atom_probability": round(
                        float(atom_probability[index]), 6
                    ),
                    "group_probability": round(
                        float(group_probability[motif_ids[index]]), 6
                    ),
                    "motif": motif_ids[index],
                    "truth": index in truth,
                    "predicted": bool(atom_probability[index] >= 0.5),
                }
                for index in range(len(positions))
            ],
            "bonds": [list(pair) for pair in unique_bonds],
        }
    return {
        "slug": summary["case_slug"],
        "label": summary["case_label"],
        "sample_id": summary["sample_id"],
        "reaction_class": summary["reaction_class"],
        "atom_f1": round(float(summary["atom_f1"]), 4),
        "counts": {
            "tp": summary["true_positives"],
            "fp": summary["false_positives"],
            "fn": summary["false_negatives"],
        },
        "interpretation": summary["interpretation"],
        "roles": roles,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    selected_indices = [item[0] for item in CASE_SPECS]
    atom_probabilities = recover_saved_atom_probabilities(
        records, set(selected_indices)
    )
    group_probabilities = infer_explicit_group_probabilities(selected_indices)

    summaries = []
    cases_3d = []
    for index, slug, label in CASE_SPECS:
        prediction = SamplePrediction(
            atom_probabilities=atom_probabilities[index],
            group_probabilities=group_probabilities[index],
        )
        summary = build_case_summary(
            index, slug, label, records[index], prediction
        )
        summary["interpretation"] = narrative(summary)
        cases_3d.append(build_3d_case(summary, records[index], prediction))
        figure = draw_case(summary, records[index], prediction)
        figure_path = OUTPUT / f"{slug}.png"
        figure.save(figure_path, quality=95)
        summary["figure"] = figure_path.name
        summaries.append(summary)

    with (OUTPUT / "case_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "A2 hierarchical motif",
                "threshold": 0.5,
                "cases": cases_3d,
            },
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    with (OUTPUT / "case_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "A2 hierarchical motif",
                "threshold": 0.5,
                "selection_policy": (
                    "Prespecified representative categories: high-confidence "
                    "exact, large-pair exact, multi-center exact, threshold-edge "
                    "exact, near miss, and clear failure."
                ),
                "cases": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    fields = [
        "case_slug",
        "case_label",
        "sample_id",
        "reaction_class",
        "total_atoms",
        "positive_atoms",
        "atom_f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "exact_pair",
        "top3_hit_both_reactants",
        "minimum_true_atom_probability",
        "maximum_false_atom_probability",
        "figure",
        "interpretation",
    ]
    with (OUTPUT / "case_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item[key] for key in fields})

    lines = [
        "# A2 case analysis",
        "",
        "All cases use the fixed test threshold of 0.5. Atom probabilities "
        "come from the saved A2 test predictions; explicit BRICS-group "
        "probabilities are recovered from the A2 checkpoint.",
        "",
    ]
    for number, item in enumerate(summaries, 1):
        lines.extend(
            [
                f"## Case {number}: {item['case_label']}",
                "",
                f"- Sample: `{item['sample_id']}`; reaction class: "
                f"{item['reaction_class']}; total atoms: {item['total_atoms']}.",
                f"- Atom-F1: {item['atom_f1']:.3f}; TP/FP/FN: "
                f"{item['true_positives']}/{item['false_positives']}/"
                f"{item['false_negatives']}; Exact-Pair: "
                f"{'yes' if item['exact_pair'] else 'no'}.",
                f"- Minimum true-site probability: "
                f"{item['minimum_true_atom_probability']:.4f}; maximum "
                f"non-site probability: {item['maximum_false_atom_probability']:.4f}.",
                f"- Interpretation: {item['interpretation']}",
                "",
                f"![{item['case_label']}]({item['figure']})",
                "",
            ]
        )
    (OUTPUT / "CASE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    # Compact 2×3 overview for manuscript planning.
    thumbnails = []
    for item in summaries:
        image = Image.open(OUTPUT / item["figure"]).convert("RGB")
        image.thumbnail((950, 480))
        card = Image.new("RGB", (970, 560), "white")
        card.paste(image, ((970 - image.width) // 2, 55))
        draw = ImageDraw.Draw(card)
        draw.text(
            (18, 12),
            item["case_label"],
            fill=(26, 54, 93),
            font=font(24, bold=True),
        )
        thumbnails.append(card)
    overview = Image.new("RGB", (1940, 1680), (245, 247, 250))
    for index, image in enumerate(thumbnails):
        overview.paste(image, ((index % 2) * 970, (index // 2) * 560))
    overview.save(OUTPUT / "a2_case_overview.png", quality=95)

    print(json.dumps({"output": str(OUTPUT), "cases": summaries}, indent=2))


if __name__ == "__main__":
    main()
