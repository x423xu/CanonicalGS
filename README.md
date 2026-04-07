# CanonicalGS

CanonicalGS is a feedforward multi-view reconstruction system where the canonical
scene representation lives in an accumulated 3D evidence state and Gaussian
splats are used only as local rendering primitives.

The implementation contract for this repository is defined by:

- `scientist/2026-03-17_canonicalgs_scientist_brief.md`
- `canonical_ffgs_pipeline_summary.txt`
- `order_invariant_depth_estimation_summary.txt`
- `eng-kitty/Running-statement.md`

## Current Status

This branch contains a clean implementation slice centered on
`mono_voxel_lite`, plus the supporting dataset, rendering, and evaluation code
needed to load a trained checkpoint and test it.

- `src/canonicalgs/model/mono_voxel_lite.py`: standalone `mono_voxel_lite`
  implementation
- `scripts/compare_mono_voxel_lite_reference.py`: evaluation against the
  Active-FFGS reference pipeline
- `scripts/evaluate_mono_voxel_lite_canonical_contract.py`: canonical-contract
  evaluation on sampled RE10K episodes
- `tests/test_mono_voxel_lite_equivalence.py`: parity tests for modules and
  checkpoint loading

The current verified checkpoint reproduces the reference output exactly in the
comparison script, with eval PSNR around `25`.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
python -m canonicalgs.main mode=inspect_dataset
```

## Test `mono_voxel_lite`

### 1. Required external assets

This repo expects the following external paths to exist when running the test
scripts with their default arguments:

- RE10K data root: `/data0/xxy/data/re10k`
- Reference repo: `/data0/xxy/code/Active-FFGS-streaming`
- Checkpoint:
  `/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints/epoch_9-step_300000.ckpt`
- W&B config:
  `/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/wandb/run-20260309_102424-awdebv94/files/config.yaml`
- Evaluation index:
  `/data0/xxy/code/Active-FFGS-streaming/assets/evaluation_index_re10k.json`

If your paths differ, pass them explicitly with `--root`, `--reference-repo`,
`--checkpoint`, `--wandb-config`, or `--evaluation-index`.

### 2. Recommended runtime

On the `malab` machine, the scripts were verified with:

```bash
/data0/xxy/conda_envs/depthsplat/bin/python
```

Run from the repository root with `PYTHONPATH=src` so the local package is
importable without installation.

### 3. Main test command

This is the main command to test the trained `mono_voxel_lite` checkpoint
against the reference implementation:

```bash
cd /data0/xxy/code/CanonicalGS
PYTHONPATH=src /data0/xxy/conda_envs/depthsplat/bin/python \
  scripts/compare_mono_voxel_lite_reference.py \
  --samples 100 \
  --report outputs/mono_voxel_lite_compare_100.json
```

What it does:

- loads the trained `mono_voxel_lite` checkpoint
- loads the matching reference encoder/decoder configuration
- runs evaluation samples from RE10K
- compares rendered outputs and Gaussian tensors
- writes a JSON report to `outputs/mono_voxel_lite_compare_100.json`

Expected result:

- mean PSNR is about `25`
- the canonical implementation should match the reference exactly for the tested
  outputs, with zero deltas in the report

Example verified result:

```json
{
  "samples_evaluated": 10,
  "reference_mean_psnr": 25.095772361755373,
  "canonical_mean_psnr": 25.095772361755373,
  "max_psnr_delta": 0.0
}
```

An existing 100-sample report in this repo also records:

```text
reference_mean_psnr = 25.43104263305664
canonical_mean_psnr = 25.43104263305664
```

### 4. Canonical-contract evaluation

To evaluate the checkpoint with the episode-builder based canonical contract:

```bash
cd /data0/xxy/code/CanonicalGS
PYTHONPATH=src /data0/xxy/conda_envs/depthsplat/bin/python \
  scripts/evaluate_mono_voxel_lite_canonical_contract.py \
  --samples 100 \
  --report outputs/mono_voxel_lite_canonical_contract_eval_100.json
```

This produces a second JSON report with per-sample summaries on the `eval`
split.

### 5. Unit tests

To run the parity tests for the implementation and checkpoint loader:

```bash
cd /data0/xxy/code/CanonicalGS
PYTHONPATH=src /data0/xxy/conda_envs/depthsplat/bin/python -m pytest \
  tests/test_mono_voxel_lite_equivalence.py -q
```

Verified result on `malab`:

```text
2 passed
```

### 6. Useful output files

- `outputs/mono_voxel_lite_compare_100.json`: reference-vs-canonical comparison
- `outputs/mono_voxel_lite_canonical_contract_eval_100.json`: canonical-contract
  eval report
- `outputs/mono_voxel_lite_compare_100.log`: stdout/stderr capture from a full
  comparison run

## Planned Delivery Order

1. dataset episode builder
2. per-view encoder and depth/confidence head
3. sparse evidence writer
4. commutative accumulator
5. canonical posterior readout
6. local Gaussian readout
7. renderer
8. loss stack and evaluation suite
