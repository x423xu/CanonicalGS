# Eng-Kitty Work Log

This folder stores task logs and implementation notes for work completed in this workspace.

## 2026-03-17

- Established `eng-kitty/` as the local execution-log folder.
- Rebuilt `Running-statement.md` into a formal CanonicalGS execution contract tied to the scientist brief, reference repos, remote runtime policy, logging requirements, training losses, evaluation benchmarks, and completion gates.
- Bootstrapped the previously empty upstream repo into the local workspace and attached it to `origin`.
- Added the first implementation slice: `pyproject.toml`, Hydra config skeleton, package scaffold, wandb bootstrap utilities, manifest loading, canonical episode dataclasses, deterministic nested-context episode building, and initial tests.
- Added live RE10K remote-dataset inspection against `/data0/xxy/data/re10k`, including chunk/index parsing and sample episode construction on `malab`.
- Implemented a minimal end-to-end CanonicalGS forward path: independent view encoder, voxel evidence writer, commutative evidence accumulator, canonical posterior readout, and one-Gaussian-per-cell local Gaussian readout.
- Implemented canonical loss scaffolding for convergence, monotone uncertainty, and low-evidence suppression, plus a bootstrap training loop with optional wandb logging.
- Verified remote execution in `depthsplat` with `CUDA_VISIBLE_DEVICES=7` for `inspect_forward`, `inspect_losses`, and a two-step `bootstrap_train` smoke run.
