# HAMIR: Hierarchical Atom-Motif Interaction for Reaction-Site Prediction

This repository contains the core implementation of HAMIR (named A2 in the
original implementation) for two-reactant atom participation prediction. HAMIR combines a shared 3D PaiNN
encoder, sparse bidirectional atom attention, hierarchical BRICS motif
interaction, and an explicit motif-level supervision head.

The source-only release intentionally excludes datasets, molecule caches,
checkpoints, saved predictions, logs, experimental results, A0/A1 ablations,
and external baseline implementations.

The internal HAMIR-2D control is included in `src/dcir/hamir_2d.py`. It replaces
the molecular encoder with a covalent-bond message-passing encoder while keeping
the hierarchical interaction modules. Its configuration is
`configs/reaction_sites/uspto50k_hamir_2d_seed17.yaml`; use the same training and
evaluation commands below with this configuration. Original filenames are kept
for compatibility with archived configurations and checkpoint parameter names.

## Included functionality

- USPTO-50K atom-mapped reaction preparation;
- deterministic ETKDGv3 conformer-cache construction;
- A2 training, validation, checkpointing, and test evaluation;
- atom- and motif-level faithfulness analysis;
- 3D rotation/translation/noise and reactant-swap robustness analysis;
- A2 case visualization and ChimeraX export;
- EnzymeMap A2 zero-shot preparation and pair-disjoint fine-tuning splits.

## Environment

Python 3.11 or 3.12 is recommended. Install the CUDA-enabled PyTorch wheel
appropriate for the host first, then install the remaining dependencies:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data placement

Data are not distributed in this repository. After obtaining USPTO-50K from
an authorized source, place the atom-mapped CSV files at:

```text
data/raw/reactions/uspto_50k/atom_mapped_train.csv
data/raw/reactions/uspto_50k/atom_mapped_valid.csv
data/raw/reactions/uspto_50k/atom_mapped_test.csv
```

## USPTO-50K workflow

```bash
export PYTHONPATH="$PWD/src"

python scripts/reaction_sites.py prepare   --config configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml

python scripts/reaction_sites.py build-3d   --config configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml   --workers 8

python scripts/reaction_sites.py train   --config configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml

python scripts/reaction_sites.py evaluate   --config configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml   --checkpoint outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt   --split test
```

## A2 diagnostics

```bash
python scripts/reaction_site_diagnostics.py faithfulness   --config configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml   --checkpoint outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt   --device cuda --max-examples 3557 --iterations 10000 --seed 17   --output-dir outputs/analysis/reaction_sites/uspto50k/faithfulness

python scripts/reaction_site_diagnostics.py robustness   --config configs/reaction_sites/uspto50k_painn_a2_hierarchical_motif_3080ti.yaml   --checkpoint outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt   --device cuda --max-examples 3557 --noise-std 0.05 --precision fp32   --seed 17 --output-dir outputs/analysis/reaction_sites/uspto50k/robustness
```

## Tests

```bash
python -m pytest -q
```

## Reproducibility and licensing

The fixed threshold is 0.5 and must not be tuned on the test set. Dataset and
checkpoint licenses are not implied by this source release. Add an explicit
software license before publishing the repository publicly.
