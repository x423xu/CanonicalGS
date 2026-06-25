#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CANONICALGS_ENV_PREFIX="${CANONICALGS_ENV_PREFIX:-$REPO_ROOT/.conda/canonicalgs}"
CANONICALGS_CUDA_HOME="${CANONICALGS_CUDA_HOME:-$CANONICALGS_ENV_PREFIX}"
CANONICALGS_TORCH_INDEX_URL="${CANONICALGS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CANONICALGS_PYG_WHEEL_URL="${CANONICALGS_PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.4.0+cu124.html}"
CANONICALGS_CUDA_ARCH_LIST="${CANONICALGS_CUDA_ARCH_LIST:-8.6}"
CANONICALGS_MINKOWSKI_REPO="${CANONICALGS_MINKOWSKI_REPO:-https://github.com/NVIDIA/MinkowskiEngine.git}"
CANONICALGS_MINKOWSKI_REF="${CANONICALGS_MINKOWSKI_REF:-v0.5.4}"
MAX_JOBS="${MAX_JOBS:-8}"
FORCE=0

usage() {
  cat <<USAGE
Usage: $0 [--force]

Creates a fresh CanonicalGS conda environment by installing packages from
conda and pip. This script does not clone or copy another conda environment.

Environment variables:
  CONDA_BIN                    Path to conda executable. Default: first conda on PATH.
  CANONICALGS_ENV_PREFIX       Target environment path. Default: <repo>/.conda/canonicalgs
  CANONICALGS_CUDA_HOME        CUDA/toolkit prefix. Default: CANONICALGS_ENV_PREFIX
  CANONICALGS_TORCH_INDEX_URL  PyTorch wheel index. Default: https://download.pytorch.org/whl/cu124
  CANONICALGS_PYG_WHEEL_URL    PyG wheel page for torch-scatter. Default: torch 2.4.0 + cu124
  CANONICALGS_CUDA_ARCH_LIST   CUDA architectures for extension builds. Default: 8.6
  CANONICALGS_MINKOWSKI_REPO   MinkowskiEngine git repo. Default: NVIDIA/MinkowskiEngine
  CANONICALGS_MINKOWSKI_REF    MinkowskiEngine git ref. Default: v0.5.4
  CANONICALGS_CC               C compiler. Default: conda env compiler
  CANONICALGS_CXX              C++ compiler. Default: conda env compiler
  MAX_JOBS                     Parallel compile jobs. Default: 8

Options:
  --force                      Remove CANONICALGS_ENV_PREFIX before recreating it.
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

if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Conda executable not found. Set CONDA_BIN or put conda on PATH." >&2
  exit 1
fi

if [[ "$FORCE" -eq 1 && -d "$CANONICALGS_ENV_PREFIX" ]]; then
  "$CONDA_BIN" env remove -y -p "$CANONICALGS_ENV_PREFIX"
fi

if [[ ! -d "$CANONICALGS_ENV_PREFIX" ]]; then
  "$CONDA_BIN" env create -y -p "$CANONICALGS_ENV_PREFIX" -f "$REPO_ROOT/environment.yml"
else
  echo "Using existing environment: $CANONICALGS_ENV_PREFIX"
fi

cd "$REPO_ROOT"
export CUDA_HOME="$CANONICALGS_CUDA_HOME"
export PATH="$CANONICALGS_ENV_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$CANONICALGS_CUDA_ARCH_LIST"
export MAX_JOBS

git submodule update --init --recursive third_party/Swin3D

PYTHON="$CANONICALGS_ENV_PREFIX/bin/python"
PIP=("$PYTHON" -m pip)
CANONICALGS_CC="${CANONICALGS_CC:-$CANONICALGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
CANONICALGS_CXX="${CANONICALGS_CXX:-$CANONICALGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++}"

if [[ ! -x "$CANONICALGS_CC" || ! -x "$CANONICALGS_CXX" ]]; then
  echo "Conda compilers not found. Check CANONICALGS_CC/CANONICALGS_CXX or environment.yml." >&2
  exit 1
fi

export CC="$CANONICALGS_CC"
export CXX="$CANONICALGS_CXX"
export LIBRARY_PATH="$CANONICALGS_ENV_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CANONICALGS_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

validate_swin3d_submodule() {
  local swin3d_root="$REPO_ROOT/third_party/Swin3D/Swin3D"
  local missing=0

  grep -q "other_down_stride=2" "$swin3d_root/models/Swin3D.py" || missing=1
  grep -q "stem_norm='bn'" "$swin3d_root/models/Swin3D.py" || missing=1
  grep -q "norm=stem_norm" "$swin3d_root/models/Swin3D.py" || missing=1
  grep -q "norm='bn'" "$swin3d_root/modules/mink_layers.py" || missing=1
  grep -q "MinkowskiInstanceNorm" "$swin3d_root/modules/mink_layers.py" || missing=1
  grep -q "torch.clamp(sum_coffs, min=1e-6)" "$swin3d_root/sparse_dl/attn/attn_coff.py" || missing=1

  if [[ "$missing" -ne 0 ]]; then
    echo "Swin3D submodule is missing CanonicalGS fixes. Run: git submodule update --init --recursive third_party/Swin3D" >&2
    exit 1
  fi
}


install_minkowski_engine() {
  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/canonicalgs-minkowski.XXXXXX")"
  trap 'rm -rf "$build_dir"' RETURN

  git clone --depth 1 --branch "$CANONICALGS_MINKOWSKI_REF" "$CANONICALGS_MINKOWSKI_REPO" "$build_dir"
  "$PYTHON" - "$build_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

def ensure_after_any(rel, anchors, additions):
    path = root / rel
    text = path.read_text()
    additions = ''.join(line + "\n" for line in additions if line not in text)
    if not additions:
        return
    for anchor in anchors:
        if anchor in text:
            path.write_text(text.replace(anchor, anchor + additions, 1))
            return
    lines = text.splitlines(keepends=True)
    last_include = max((i for i, line in enumerate(lines) if line.startswith("#include ")), default=-1)
    if last_include < 0:
        raise RuntimeError(f"no include anchor found in {rel}")
    lines.insert(last_include + 1, additions)
    path.write_text(''.join(lines))

ensure_after_any("src/3rdparty/concurrent_unordered_map.cuh", ["#include <thrust/count.h>\n", "#include <thrust/pair.h>\n"], ["#include <thrust/execution_policy.h>"])
ensure_after_any("src/convolution_kernel.cuh", ["#include <thrust/device_vector.h>\n", "#include <thrust/iterator/zip_iterator.h>\n"], ["#include <thrust/execution_policy.h>"])
ensure_after_any("src/coordinate_map_gpu.cu", ["#include <thrust/sort.h>\n"], ["#include <thrust/remove.h>", "#include <thrust/unique.h>"])
ensure_after_any("src/spmm.cu", ["#include <thrust/device_vector.h>\n", "#include <thrust/iterator/zip_iterator.h>\n"], ["#include <thrust/execution_policy.h>", "#include <thrust/reduce.h>", "#include <thrust/sort.h>"])

path = root / "src/3rdparty/concurrent_unordered_map.cuh"
text = path.read_text()
if "#define CUB_DISABLE_NVTX" not in text:
    text = text.replace(
        "#ifndef CONCURRENT_UNORDERED_MAP_CUH\n#define CONCURRENT_UNORDERED_MAP_CUH\n",
        "#ifndef CONCURRENT_UNORDERED_MAP_CUH\n#define CONCURRENT_UNORDERED_MAP_CUH\n\n#ifndef CUB_DISABLE_NVTX\n#define CUB_DISABLE_NVTX\n#endif\n",
        1,
    )
path.write_text(text)

path = root / "src/coordinate_map_gpu.cuh"
text = path.read_text()
old = '''      m_map = map_type::create(\n          compute_hash_table_size(size, m_hashtable_occupancy),\n          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);\n'''
new = '''      auto map_unique = map_type::create(\n          compute_hash_table_size(size, m_hashtable_occupancy),\n          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);\n      auto map_deleter = map_unique.get_deleter();\n      m_map = std::shared_ptr<map_type>(map_unique.release(), map_deleter);\n'''
if old not in text:
    raise RuntimeError("coordinate_map_gpu.cuh map creation block not found")
path.write_text(text.replace(old, new, 1))

path = root / "src/spmm.cu"
text = path.read_text()
if "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>" not in text:
    text = text.replace("#include <ATen/cuda/CUDAContext.h>\n", "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n", 1)
    text = text.replace("#include <torch/extension.h>\n#include <torch/script.h>\n", "#include <torch/script.h>\n", 1)
path.write_text(text)

for path in root.glob("MinkowskiEngine/**/*.py"):
    text = path.read_text()
    text = text.replace("from collections import Sequence, namedtuple", "from collections import namedtuple\nfrom collections.abc import Sequence")
    text = text.replace("from collections import Sequence", "from collections.abc import Sequence")
    path.write_text(text)
PY

  (
    cd "$build_dir"
    "$PYTHON" setup.py install --blas=openblas --force_cuda
  )
}

"${PIP[@]}" install --upgrade pip setuptools==69.5.1 wheel
"${PIP[@]}" install --index-url "$CANONICALGS_TORCH_INDEX_URL" torch==2.4.0 torchvision==0.19.0
"${PIP[@]}" install torch-scatter -f "$CANONICALGS_PYG_WHEEL_URL"
REQ_WITHOUT_LOCAL_BUILD="$(mktemp)"
grep -v 'diff-gaussian-rasterization-modified' "$REPO_ROOT/requirements.txt" > "$REQ_WITHOUT_LOCAL_BUILD"
"${PIP[@]}" install -r "$REQ_WITHOUT_LOCAL_BUILD"
rm -f "$REQ_WITHOUT_LOCAL_BUILD"
"${PIP[@]}" install --no-build-isolation git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
"${PIP[@]}" install timm==0.4.9

# MinkowskiEngine is required by Swin3D and CanonicalGS. Build it from source
# inside this conda environment; no existing environment or system toolkit is copied.
install_minkowski_engine
validate_swin3d_submodule

"${PIP[@]}" install -e "$REPO_ROOT"
(
  cd "$REPO_ROOT/third_party/Swin3D"
  "$PYTHON" setup.py install
)

cd "$REPO_ROOT"
"$PYTHON" - <<'PY'
import torch
from canonicalgs.model.encoder import ENCODERS
import Swin3D
from Swin3D.models import Swin3DUNet
from Swin3D.sparse_dl import attn_cuda, knn_cuda
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("encoders:", sorted(ENCODERS.keys()))
print("Swin3D:", Swin3D.__file__)
print("Swin3DUNet:", Swin3DUNet)
print("attn_cuda:", attn_cuda.__file__)
print("knn_cuda:", knn_cuda.__file__)
PY
