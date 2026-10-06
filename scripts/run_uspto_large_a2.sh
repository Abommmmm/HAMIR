#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-}"
stage="${2:-}"
if [[ "$dataset" != "mit" && "$dataset" != "full" ]]; then
  echo "Usage: bash scripts/run_uspto_large_a2.sh {mit|full} {download|map|prepare|build3d-test|build3d-all|zero-shot|finetune|evaluate-finetuned}"
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"
zero_config="configs/reaction_sites/uspto_${dataset}_painn_a2_zero_shot_3080ti.yaml"
fine_config="configs/reaction_sites/uspto_${dataset}_painn_a2_finetune_3080ti.yaml"
source_checkpoint="outputs/checkpoints/reaction_sites/uspto50k/painn_a2_hierarchical_motif/best.pt"
fine_checkpoint="outputs/checkpoints/reaction_sites/uspto_${dataset}/painn_a2_finetune/best.pt"

case "$stage" in
  download)
    python scripts/download_uspto_large.py "$dataset"
    ;;
  map)
    if [[ "$dataset" != "full" ]]; then
      echo "USPTO-MIT is already atom mapped; no mapping stage is needed."
      exit 0
    fi
    python scripts/map_uspto_full.py --batch-size 32
    ;;
  prepare)
    python scripts/prepare_uspto_large_reaction_sites.py "$dataset"
    ;;
  build3d-test)
    python -m dcir.reaction_sites build-3d \
      --config "$zero_config" --workers 12 --splits test
    ;;
  build3d-all)
    python -m dcir.reaction_sites build-3d \
      --config "$fine_config" --workers 12
    ;;
  zero-shot)
    python -m dcir.reaction_sites evaluate \
      --config "$zero_config" --checkpoint "$source_checkpoint" --split test
    ;;
  finetune)
    python -m dcir.reaction_sites train \
      --config "$fine_config" --init-checkpoint "$source_checkpoint"
    ;;
  evaluate-finetuned)
    python -m dcir.reaction_sites evaluate \
      --config "$fine_config" --checkpoint "$fine_checkpoint" --split test
    ;;
  *)
    echo "Unknown stage: $stage"
    exit 2
    ;;
esac
