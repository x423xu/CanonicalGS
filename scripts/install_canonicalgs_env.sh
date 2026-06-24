#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/data0/xxy/miniconda3/bin/conda}"
SOURCE_ENV="${SOURCE_ENV:-/data0/xxy/conda_envs/depthsplat}"
TARGET_ENV="${TARGET_ENV:-/data0/xxy/conda_envs/canonicalgs}"
CUDA_HOME="${CUDA_HOME:-$TARGET_ENV}"
MAX_JOBS="${MAX_JOBS:-8}"
FORCE=0

usage() {
  cat <<USAGE
Usage: $0 [--force]

Creates the CanonicalGS conda environment by cloning the existing DepthSplat
environment, initializes the Swin3D submodule, installs CanonicalGS editable,
and builds/installs Swin3D CUDA extensions.

Environment variables:
  CONDA_BIN   Path to conda executable. Default: /data0/xxy/miniconda3/bin/conda
  SOURCE_ENV  Existing environment to clone. Default: /data0/xxy/conda_envs/depthsplat
  TARGET_ENV  CanonicalGS environment path. Default: /data0/xxy/conda_envs/canonicalgs
  CUDA_HOME   CUDA/toolkit prefix. Default: TARGET_ENV
  MAX_JOBS    Parallel compile jobs. Default: 8

Options:
  --force     Remove TARGET_ENV before recreating it.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda executable not found: $CONDA_BIN" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_ENV" ]]; then
  echo "Source environment not found: $SOURCE_ENV" >&2
  exit 1
fi

if [[ -d "$TARGET_ENV" && "$FORCE" -eq 1 ]]; then
  "$CONDA_BIN" env remove -y -p "$TARGET_ENV"
fi

if [[ ! -d "$TARGET_ENV" ]]; then
  "$CONDA_BIN" create -y -p "$TARGET_ENV" --clone "$SOURCE_ENV"
else
  echo "Using existing environment: $TARGET_ENV"
fi

cd "$REPO_ROOT"
git submodule update --init --recursive third_party/Swin3D

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$TARGET_ENV/bin:$PATH"
export MAX_JOBS

"$TARGET_ENV/bin/python" -m pip install -e "$REPO_ROOT"
"$TARGET_ENV/bin/python" -m pip install -r "$REPO_ROOT/third_party/Swin3D/requirements.txt"
(
  cd "$REPO_ROOT/third_party/Swin3D"
  "$TARGET_ENV/bin/python" setup.py install
)

cd "$REPO_ROOT"
"$TARGET_ENV/bin/python" - <<'PY'
import canonicalgs
import torch
import Swin3D
from Swin3D.models import Swin3DUNet
from Swin3D.sparse_dl import attn_cuda, knn_cuda
print("canonicalgs:", canonicalgs.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("Swin3D:", Swin3D.__file__)
print("Swin3DUNet:", Swin3DUNet)
print("attn_cuda:", attn_cuda.__file__)
print("knn_cuda:", knn_cuda.__file__)
PY
