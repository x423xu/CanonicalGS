#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
source /data0/xxy/miniconda3/etc/profile.d/conda.sh
conda activate depthsplat

cd /data0/xxy/code/CanonicalGS
export PYTHONPATH=src

python -m canonicalgs.main \
  mode=smoke_test_100scenes \
  wandb.mode=offline \
  runtime.device=cuda:0 \
  output_dir=outputs/smoke_test_100scenes \
  dataset.fixed_scene_count=100 \
  dataset.fixed_scene_seed=111123 \
  train.smoke_steps=500 \
  train.log_every=50
