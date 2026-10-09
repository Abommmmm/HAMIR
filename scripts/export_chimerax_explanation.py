from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


BOND_TYPES = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


def atom_name(symbol: str, index: int) -> str:
    return f"{symbol.upper()}{index + 1}"[:4]


def orient_for_display(positions: np.ndarray) -> np.ndarray:
    """Rigidly orient a conformer so its two widest axes face the camera."""
    centered = np.asarray(positions, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
    oriented = centered @ principal_axes.T
    for axis in range(3):
        extreme = int(np.argmax(np.abs(oriented[:, axis])))
        if oriented[extreme, axis] < 0:
            oriented[:, axis] *= -1
    # Preserve handedness: this remains a proper rigid rotation, not a mirror.
    transform = np.linalg.lstsq(centered, oriented, rcond=None)[0]
    if np.linalg.det(transform) < 0:
        oriented[:, 2] *= -1
    return oriented.astype(np.float32)


def molecule_from_archive(
    archive_path: Path,
    responses: list[float],
    *,
    chain_id: str,
    residue_name: str,
) -> tuple[Chem.Mol, list[str], np.ndarray]:
    with np.load(archive_path, allow_pickle=False) as archive:
        atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        atom_features = np.asarray(archive["atom_features"], dtype=np.float32)
        bond_index = np.asarray(archive["bond_index"], dtype=np.int64)
        bond_features = np.asarray(archive["bond_features"], dtype=np.float32)
        positions = orient_for_display(
            np.asarray(archive["positions"], dtype=np.float32)[0]
        )

    response = np.asarray(responses, dtype=np.float64)
    if len(response) != len(atomic_numbers):
        raise ValueError(
            f"{archive_path.stem}: {len(response)} responses for "
            f"{len(atomic_numbers)} atoms"
        )

    maximum = float(response.max(initial=0.0))
    relative_importance = (
        np.zeros_like(response) if maximum <= 0 else 100.0 * response / maximum
    )

    editable = Chem.RWMol()
    names: list[str] = []
    for index, atomic_number in enumerate(atomic_numbers):
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(round(float(atom_features[index, 0]))))
        atom.SetIsAromatic(bool(round(float(atom_features[index, 1]))))
        editable.AddAtom(atom)
        names.append(atom_name(atom.GetSymbol(), index))

    for edge, features in zip(bond_index.T, bond_features):
        begin, end = map(int, edge)
        if begin >= end:
            continue
        kind = int(np.argmax(features[:4]))
        editable.AddBond(begin, end, BOND_TYPES.get(kind, Chem.BondType.SINGLE))
        bond = editable.GetBondBetweenAtoms(begin, end)
        if kind == 3:
            bond.SetIsAromatic(True)
        bond.SetIsConjugated(bool(round(float(features[4]))))

    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(atomic_numbers))
    for index, xyz in enumerate(positions):
        conformer.SetAtomPosition(index, tuple(map(float, xyz)))
    molecule.AddConformer(conformer, assignId=True)

    for index, atom in enumerate(molecule.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{names[index]:>4}")
        info.SetSerialNumber(index + 1)
        info.SetResidueName(residue_name[:3].upper())
        info.SetResidueNumber(1)
        info.SetChainId(chain_id)
        info.SetOccupancy(1.0)
        info.SetTempFactor(float(relative_importance[index]))
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)

    return molecule, names, relative_importance


def write_pdb(
    molecule: Chem.Mol,
    names: list[str],
    relative_importance: np.ndarray,
    path: Path,
) -> None:
    conformer = molecule.GetConformer()
    lines: list[str] = []
    for index, atom in enumerate(molecule.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        position = conformer.GetAtomPosition(index)
        name = names[index]
        element = atom.GetSymbol().upper()
        lines.append(
            f"HETATM{index + 1:5d} {name:>4} {info.GetResidueName():>3} "
            f"{info.GetChainId():1}{info.GetResidueNumber():4d}    "
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"{1.00:6.2f}{relative_importance[index]:6.2f}"
            f"          {element:>2}"
        )
    neighbours: dict[int, list[int]] = {i: [] for i in range(molecule.GetNumAtoms())}
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        neighbours[begin].append(end)
        neighbours[end].append(begin)
    for begin, ends in neighbours.items():
        if ends:
            lines.append(
                f"CONECT{begin + 1:5d}"
                + "".join(f"{end + 1:5d}" for end in sorted(ends))
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_case(case_dir: Path, conformer_dir: Path) -> Path:
    explanation_path = case_dir / "explanation.json"
    explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
    output_dir = case_dir / "chimerax"
    output_dir.mkdir(parents=True, exist_ok=True)

    drugs = [
        (explanation["d1"], explanation["d1_atom_response"], "A", "D1"),
        (explanation["d2"], explanation["d2_atom_response"], "B", "D2"),
    ]
    atom_names: list[list[str]] = []
    summaries: list[str] = []
    pdb_paths: list[Path] = []
    x_widths: list[float] = []

    for drug_id, responses, chain_id, residue_name in drugs:
        molecule, names, importance = molecule_from_archive(
            conformer_dir / f"{drug_id}.npz",
            responses,
            chain_id=chain_id,
            residue_name=residue_name,
        )
        pdb_path = output_dir / f"{drug_id}_atom_importance.pdb"
        write_pdb(molecule, names, importance, pdb_path)
        pdb_paths.append(pdb_path)
        atom_names.append(names)
        coordinates = np.asarray(molecule.GetConformer().GetPositions())
        x_widths.append(float(np.ptp(coordinates[:, 0])))
        ranked = np.argsort(-importance)[:10]
        summaries.append(
            f"{drug_id}\n"
            + "\n".join(
                f"  model_atom={int(i):>2}  pdb_atom={names[int(i)]:<4}  "
                f"relative_importance={importance[int(i)]:8.3f}"
                for i in ranked
            )
        )

    cross_path = output_dir / "top_cross_atom_pairs.pb"
    pb_lines = [
        "; Model-attribution edges, NOT physical contacts or chemical bonds.",
        "; halfbond = false",
        "; color = magenta",
        "; radius = 0.12",
        "; dashes = 6",
    ]
    for pair in explanation["top_cross_atom_pairs"][:10]:
        d1_index = int(pair["d1_atom"])
        d2_index = int(pair["d2_atom"])
        pb_lines.append(
            f"#1/A:1@{atom_names[0][d1_index]} " f"#2/B:1@{atom_names[1][d2_index]}"
        )
    cross_path.write_text("\n".join(pb_lines) + "\n", encoding="utf-8")

    top5_path = output_dir / "top5_cross_atom_pairs.pb"
    top5_lines = pb_lines[:5]
    for pair in explanation["top_cross_atom_pairs"][:5]:
        d1_index = int(pair["d1_atom"])
        d2_index = int(pair["d2_atom"])
        top5_lines.append(
            f"#1/A:1@{atom_names[0][d1_index]} " f"#2/B:1@{atom_names[1][d2_index]}"
        )
    top5_path.write_text("\n".join(top5_lines) + "\n", encoding="utf-8")

    aligned_path = output_dir / "top5_response_aligned_cross_atom_pairs.pb"
    top_d1 = set(map(int, explanation["top_d1_atoms"][:10]))
    top_d2 = set(map(int, explanation["top_d2_atoms"][:10]))
    aligned_pairs = [
        pair
        for pair in explanation["top_cross_atom_pairs"]
        if int(pair["d1_atom"]) in top_d1 and int(pair["d2_atom"]) in top_d2
    ][:5]
    aligned_lines = [
        "; Top cross contributions constrained to atom-response Top-10 endpoints.",
        "; These are model attributions, NOT physical contacts or chemical bonds.",
        "; halfbond = false",
        "; color = dodgerblue",
        "; radius = 0.12",
        "; dashes = 6",
    ]
    for pair in aligned_pairs:
        d1_index = int(pair["d1_atom"])
        d2_index = int(pair["d2_atom"])
        aligned_lines.append(
            f"#1/A:1@{atom_names[0][d1_index]} " f"#2/B:1@{atom_names[1][d2_index]}"
        )
    aligned_path.write_text("\n".join(aligned_lines) + "\n", encoding="utf-8")

    cxc_path = output_dir / "view_case.cxc"
    pair_name = f"{explanation['d1']}_{explanation['d2']}"
    image_path = output_dir / f"{pair_name}_explanation.png"
    session_path = output_dir / f"{pair_name}_explanation.cxs"
    to_cx = lambda path: path.resolve().as_posix()
    display_gap = 4.0
    d1_shift = -(x_widths[1] + display_gap) / 2.0
    d2_shift = (x_widths[0] + display_gap) / 2.0
    commands = [
        "close",
        f'open "{to_cx(pdb_paths[0])}"',
        f'open "{to_cx(pdb_paths[1])}"',
        "style #1,2 stick",
        "color byattribute bfactor #1,2 target a palette white:gold:red range 0,100 key true",
        "style #1,2@@bfactor>=10 ball",
        f"move x {d1_shift:.3f} models #1",
        f"move x {d2_shift:.3f} models #2",
        f'open "{to_cx(top5_path)}"',
        "set bgColor white",
        "lighting soft",
        "graphics silhouettes true",
        "view #1,2 pad 0.2",
        f'save "{to_cx(image_path)}" width 2400 height 1600 supersample 3',
        f'save "{to_cx(session_path)}"',
    ]
    cxc_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    summary_path = output_dir / "atom_importance_summary.txt"
    summary_path.write_text(
        (
            f"Case: {case_dir.name}\n"
            f"Drug pair: {explanation['d1']} + {explanation['d2']}\n"
            f"Class: {explanation['focus_class_raw']}\n"
            f"Predicted probability: {explanation['frozen_test_probability']:.8f}\n\n"
            "B-factor contains within-molecule relative atom importance:\n"
            "100 * atom_response / max(atom_response).\n"
            "Cross-molecule pseudobonds are model attributions, not physical contacts.\n\n"
            + "\n\n".join(summaries)
            + "\n"
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--conformer-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = write_case(args.case_dir.resolve(), args.conformer_dir.resolve())
    print(output_dir)


if __name__ == "__main__":
    main()
