# A2 on USPTO-MIT and USPTO-FULL

This workflow evaluates the existing USPTO-50K A2 checkpoint and then
fine-tunes the same architecture on two larger USPTO collections. All reported
test metrics use threshold 0.5.

## Dataset protocol

- USPTO-MIT uses the atom-mapped `train.txt`, `valid.txt`, and `test.txt`
  released with `wengong-jin/nips17-rexgen`. The official fixed split is kept.
- USPTO-FULL uses DeepChem's public Lowe-USPTO mirror. This CSV is not atom
  mapped, so it is mapped with RXNMapper first. After mapping, unique canonical
  reaction identities are assigned to a deterministic 80/10/10 hash split.
- Exact canonical reaction identities found in any USPTO-50K split are removed
  from both large-dataset experiments before splitting/evaluation.
- Duplicate reaction identities are removed.
- The existing reaction-site filters are then applied: exactly two
  product-participating reactants, at least one changed atom in both molecules,
  and at most 128 atoms per molecule.
- The reaction-centre labels are derived only from atom-mapped bond/atom changes.
  BRICS components provide motif labels, and ETKDGv3 with MMFF94s/UFF fallbacks
  provides the input conformers.

The generated `preparation_report.json` is part of the experiment record and
must be retained with each reported result.

## 1. Download

```bash
cd ~/autodl-tmp/DCIA
python scripts/download_uspto_large.py all
```

Expected downloads are about 50 MB for the compressed mapped USPTO-MIT archive
and 265 MB for the raw USPTO-FULL CSV. The script records URLs and SHA256
digests. Raw data should not be committed to GitHub.

## 2. Atom-map USPTO-FULL

USPTO-MIT is already mapped. USPTO-FULL requires this one-time step. A separate
environment is recommended so RXNMapper dependencies cannot alter the A2
training environment.

```bash
conda create -n rxnmap python=3.10 -y
conda run -n rxnmap pip install "rxnmapper[rdkit]==0.4.3"

mkdir -p outputs/logs/reaction_sites/uspto_full
nohup conda run -n rxnmap python scripts/map_uspto_full.py \
  --input data/raw/reactions/uspto_full/USPTO_FULL.csv \
  --output data/raw/reactions/uspto_full/atom_mapped_all.csv \
  --batch-size 32 \
  > outputs/logs/reaction_sites/uspto_full/map.log 2>&1 &

tail -f outputs/logs/reaction_sites/uspto_full/map.log
```

The mapper output is append-only and row-aligned; rerunning the same command
resumes after the last written row.

## 3. Prepare indexes and audit overlap

Run in the normal A2 environment:

```bash
export PYTHONPATH="$PWD/src"

python scripts/prepare_uspto_large_reaction_sites.py mit
python scripts/prepare_uspto_large_reaction_sites.py full

cat outputs/reaction_sites/uspto_mit/preparation_report.json
cat outputs/reaction_sites/uspto_full/preparation_report.json
```

Do not report results until both reports show non-empty train, validation, and
test indexes. Record the filtered sample counts because they are not equal to
the raw dataset sizes.

## 4. Zero-shot evaluation

Only test-set conformers are required initially:

```bash
bash scripts/run_uspto_large_a2.sh mit build3d-test
bash scripts/run_uspto_large_a2.sh mit zero-shot

bash scripts/run_uspto_large_a2.sh full build3d-test
bash scripts/run_uspto_large_a2.sh full zero-shot
```

These commands load the frozen USPTO-50K checkpoint:

```text
outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt
```

## 5. Fine-tuning

The remaining train/validation geometries reuse any test cache already built.
Batch size remains 64 on the RTX 3080 Ti.

```bash
mkdir -p outputs/logs/reaction_sites/uspto_mit \
         outputs/logs/reaction_sites/uspto_full

bash scripts/run_uspto_large_a2.sh mit build3d-all
nohup bash scripts/run_uspto_large_a2.sh mit finetune \
  > outputs/logs/reaction_sites/uspto_mit/a2_finetune.log 2>&1 &
tail -f outputs/logs/reaction_sites/uspto_mit/a2_finetune.log

# Start FULL after MIT has released the GPU.
bash scripts/run_uspto_large_a2.sh full build3d-all
nohup bash scripts/run_uspto_large_a2.sh full finetune \
  > outputs/logs/reaction_sites/uspto_full/a2_finetune.log 2>&1 &
tail -f outputs/logs/reaction_sites/uspto_full/a2_finetune.log
```

After each run finishes:

```bash
bash scripts/run_uspto_large_a2.sh mit evaluate-finetuned
bash scripts/run_uspto_large_a2.sh full evaluate-finetuned
python scripts/summarize_uspto_large_results.py
```

## Reporting

Report four rows: MIT zero-shot, MIT fine-tuned, FULL zero-shot, and FULL
fine-tuned. Use Atom-F1 as the primary metric and include precision, recall,
Top3-Hit, Exact-Pair and Group-F1. Zero-shot measures cross-dataset domain
shift; fine-tuning measures transfer adaptability. Do not compare these
reaction-site scores directly with product-prediction or retrosynthesis top-k
accuracy reported in the original USPTO benchmark papers.
