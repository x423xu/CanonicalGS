# CanonicalGS Running Statement

## Project Identity

- Project name: `CanonicalGS`
- Working mode: local-first development, remote GPU execution
- Engineering owner: `Eng-Kitty`
- Primary objective: implement the CanonicalGS feedforward multi-view reconstruction system defined in `scientist/2026-03-17_canonicalgs_scientist_brief.md`

## Source of Truth

This project must follow the scientist brief as the primary implementation contract.

Mandatory references:

1. `scientist/2026-03-17_canonicalgs_scientist_brief.md`
2. `canonical_ffgs_pipeline_summary.txt`
3. `order_invariant_depth_estimation_summary.txt`
4. `https://github.com/x423xu/CanonicalGS.git`
5. `https://github.com/x423xu/Active-FFGS-streaming.git` on branch `multiview`
6. `https://github.com/x423xu/depthsplat.git`

Priority rule:

- If implementation convenience conflicts with the scientist brief, the scientist brief wins.
- If reference repos conflict with the scientist brief, use the reference repos only as engineering templates, not as scientific authority.

## Execution Environment

### Local workspace

- Local project root: `E:\code\CanonicalGS`
- Local repo role: source code, configs, scripts, docs, lightweight assets only
- Local machine should not become a storage sink for large training artifacts or copied dataset chunks

### Remote execution

- Remote host alias: `malab`
- Remote project path: `/data0/xxy/code/CanonicalGS`
- GPU policy: always run on `cuda:7`
- Conda environment: `depthsplat`

### Development rule

- All code edits happen locally first.
- All training, profiling, and GPU validation happen remotely on `malab` with `cuda:7`.
- The local repo is the authoritative codebase.
- The remote repo is the authoritative execution site for data-heavy runs.

## Repository and Sync Policy

### Local repo bootstrap

The project should be pulled locally from:

- `https://github.com/x423xu/CanonicalGS.git`

### Remote sync contract

The remote path `/data0/xxy/code/CanonicalGS` must stay synchronized with the local source tree, with one exception:

- large datasets
- checkpoints
- cached features
- rendered outputs
- temporary artifacts

must remain excluded from code synchronization unless explicitly required.

### Git and ignore policy

The repository must enforce a strict separation between code and heavy artifacts.

Required categories to ignore:

- dataset roots
- downloaded scene caches
- training logs that are not required for source control
- checkpoints
- tensorboard or wandb offline blobs
- rendered media outputs
- profiler dumps
- scratch directories

Operational rule:

- never sync large data by accident
- never commit large data by accident
- never modify ignore rules in a way that makes the remote storage layout unsafe

## Scientific Contract

CanonicalGS is not allowed to degenerate into direct Gaussian prediction.

The implementation must preserve these non-negotiable properties:

1. canonicality
2. order invariance
3. monotone improvement with more evidence
4. separation between canonical state and renderer

### Required system graph

The implementation target is:

`unordered context views -> per-view evidence -> commutative sparse 3D evidence state -> canonical posterior readout -> local Gaussian readout -> differentiable rendering`

### Forbidden shortcuts

The following are explicitly disallowed:

- recurrent fusion across views before the canonical state
- order-sensitive attention before evidence accumulation
- learned global rewriting of the accumulated canonical state
- subset losses that force raw evidence equality across different context sizes
- target-view leakage into context sets
- evaluation with different target sets across context sizes within the same episode
- memory-saving tricks that materially damage model quality just to fit VRAM

## Implementation Scope

### Required first version

The first deliverable must be the minimal correct prototype, not an over-engineered version.

Implement:

1. dataset episode builder with nested contexts `C2` to `C6` and fixed target set `Q`
2. per-view image encoder
3. order-invariant depth / confidence estimation path
4. sparse per-view evidence writer
5. commutative sparse evidence accumulator
6. canonical posterior readout
7. one-Gaussian-per-active-cell local Gaussian readout
8. differentiable renderer
9. rendering loss
10. canonical convergence loss
11. monotone uncertainty loss
12. low-evidence suppression loss
13. evaluation and benchmark suite
14. wandb logging aligned with `depthsplat`

### Explicitly deferred for v1

Do not start with:

- adaptive subdivision
- multi-Gaussian-per-cell decoding
- heavy refinement blocks
- any module that rewrites canonical evidence rather than reading from it

## Architecture Statement

### 1. Episode construction

Each episode must come from a single clip and provide:

- a frame pool of at least 12 valid frames
- nested context sets `C2 subset C3 subset C4 subset C5 subset C6`
- a held-out target set `Q` with `|Q| = 6`

The target set must stay fixed while sweeping context sizes from 2 to 6.

### 2. Per-view independent processing

Each context view must be processed independently before fusion.

Per-view outputs must include:

- image features
- depth estimate or depth distribution
- uncertainty / confidence
- ray-aligned evidence writes into 3D cells

The depth path should follow the order-invariant design summary:

- independent image encoding
- independent pairwise construction where relevant
- symmetric source-view aggregation
- refinement only after symmetric aggregation

### 3. Evidence writing

The per-view writer must satisfy:

- cells before likely depth receive free-space evidence
- cells near likely depth receive surface evidence
- geometry moments are deposited near surface support
- appearance statistics are deposited near surface support

### 4. Canonical accumulation

The canonical state must store additive sufficient statistics only.

Required fields per active cell:

- `surface_evidence`
- `free_evidence`
- `total_confidence`
- `geo_moment_1`
- `geo_moment_2`
- `app_moment_1`
- `app_weight`

The merge operator must be additive and commutative for every stored field.

### 5. Canonical posterior readout

Readout must derive posterior quantities from the accumulated state, including:

- support
- unknownness
- confidence
- local surface mean
- local surface covariance or spread
- uncertainty

These posterior readouts are the correct objects for convergence supervision.

### 6. Gaussian readout

Gaussians are downstream rendering primitives only.

For each sufficiently supported active cell:

- center comes from local surface mean
- covariance comes from local geometry statistics with numerical clipping
- opacity is a monotone function of support and confidence
- appearance comes from local appearance readout

Gaussian generation must stay local to each cell.

### 7. Rendering

The renderer consumes:

- local Gaussian set
- target cameras

and produces:

- rendered RGB
- optional depth / opacity diagnostics

## Training Protocol

### Batch schedule

For each episode in a training batch:

1. sample one scene clip
2. build nested context sets `C2` to `C6`
3. build one fixed target set `Q`
4. run the same model five times using context sizes 2, 3, 4, 5, and 6

### Per-context forward

For each `k in {2, 3, 4, 5, 6}`:

1. independently process context views in `Ck`
2. write sparse evidence
3. accumulate canonical state `S_k`
4. compute posterior readout `R_k`
5. read out Gaussian set `G_k`
6. render every target in `Q`
7. cache render outputs and posterior state summaries for loss computation

### Loss stack

The full v1 objective must include:

1. `L_rend`
2. `L_conv`
3. `L_mono`
4. `L_null`

With the scientist brief semantics:

- `L_rend`: average L1 + SSIM + LPIPS across all context sizes and all target views
- `L_conv`: smaller-context posterior readouts converge toward stop-gradient `S_6` posterior readout
- `L_mono`: uncertainty should not increase from `S_k` to `S_{k+1}` in teacher-supported cells
- `L_null`: unsupported low-confidence cells should not produce occupancy fog or Gaussian floaters

Recommended starting loss weights:

- `lambda_rend = 1.0`
- `lambda_conv = 0.1 to 0.25`
- `lambda_mono = 0.05 to 0.1`
- `lambda_null = 1e-3 to 1e-2`

## Logging and Experiment Tracking

### Logger policy

Use `wandb` as the experiment logger.

Logging behavior must be aligned with `depthsplat` conventions as closely as practical, including:

- train loss breakdown
- validation loss breakdown
- PSNR
- SSIM
- LPIPS
- context-size-conditioned metrics
- learning rate
- timing / throughput where available
- qualitative renders where useful

### Required metric grouping

Track metrics separately for context sizes `2, 3, 4, 5, 6` whenever it matters.

Minimum required logged groups:

- rendering metrics by context size
- convergence metrics by context size
- monotonicity diagnostics by context transition
- low-evidence suppression diagnostics
- permutation stability diagnostics

## Performance and Resource Constraints

- Available GPU memory on `cuda:7`: `48 GB`
- Training must fit this budget without degrading the intended scientific behavior

This means:

- optimize implementation efficiency
- use sparse representations where intended
- keep the initial decoder simple
- avoid wasteful tensor retention

This does not mean:

- reduce quality-critical resolution blindly
- remove required losses to save memory
- collapse the canonical state into a cheaper but scientifically invalid shortcut

## Evaluation Contract

The benchmark suite must include the following.

### 1. Novel view synthesis

Report by context size `2` to `6`:

- PSNR
- SSIM
- LPIPS

### 2. Canonical convergence

Measure distance from `readout(S_k)` to `readout(S_6)` in reliable teacher-supported cells.

### 3. Permutation sensitivity

For a fixed context set, verify:

- accumulated raw evidence is invariant up to numerical tolerance
- rendered outputs remain stable across permutations

### 4. Incremental update stability

Measure update magnitude across:

- `S_2 -> S_3`
- `S_3 -> S_4`
- `S_4 -> S_5`
- `S_5 -> S_6`

Expected behavior:

- update magnitudes should shrink in sufficiently observed regions as evidence saturates

### 5. Robustness

Evaluate under:

- random context dropout
- uneven coverage
- varying baseline
- incremental arrival of new views

## Delivery Order

Implementation must proceed in this order unless a hard dependency forces a small local reorder:

1. repo bootstrap and ignore policy validation
2. dataset episode builder
3. view encoder and depth / confidence head
4. sparse evidence writer
5. commutative accumulator
6. canonical readout
7. local Gaussian readout
8. renderer integration
9. rendering-only smoke training
10. convergence loss integration
11. monotone uncertainty loss integration
12. low-evidence suppression integration
13. wandb instrumentation
14. benchmark suite
15. remote execution validation on `cuda:7`

## Required Sanity Checks

Before long runs, verify all of the following:

1. the same context set under different orderings yields identical accumulated raw evidence up to numerical tolerance
2. richer context sets do not reduce raw confidence in touched cells
3. target views are fully disjoint from context views
4. the system can overfit a tiny controlled subset
5. rendering improves as context grows from 2 to 6 on the tiny subset
6. convergence loss decreases as smaller-context readouts approach `S_6`
7. wandb logs render correctly and contain the required metric groups

## Completion Gate

Implementation is not considered finished until the codebase satisfies the scientist brief and this running statement.

Final sign-off requires:

1. the required module chain exists
2. the forbidden shortcuts are not present
3. all four required losses are implemented
4. wandb logging works
5. remote execution works on `malab` with `depthsplat` and `cuda:7`
6. evaluation covers rendering, convergence, permutation stability, and incremental stability
7. any conflict with the scientist brief is explicitly reported rather than silently worked around

## Current Operating Rule

Until the repository structure is populated and implementation starts, this document is the project execution contract for all future CanonicalGS work in this workspace.
