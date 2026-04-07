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

The upstream repository was empty. This workspace now contains the first
implementation slice:

- typed project configuration
- a Hydra entrypoint
- wandb bootstrap utilities
- dataset manifest loading
- canonical episode data structures
- a deterministic nested-context episode builder for `C2` to `C6` with fixed `Q`

This is the required first module in the scientist brief and is the base for the
remaining pipeline modules.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
python -m canonicalgs.main mode=inspect_dataset
```

## Planned Delivery Order

1. dataset episode builder
2. per-view encoder and depth/confidence head
3. sparse evidence writer
4. commutative accumulator
5. canonical posterior readout
6. local Gaussian readout
7. renderer
8. loss stack and evaluation suite
