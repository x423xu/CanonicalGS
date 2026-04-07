#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depthsplat

python -m canonicalgs.main \
  mode=inspect_dataset \
  output_dir=outputs/canonicalgs-bootstrap
