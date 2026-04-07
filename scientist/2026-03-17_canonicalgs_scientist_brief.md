# CanonicalGS Implementation Orchestration

## Scope

This document is the implementation handoff for CanonicalGS.

It is written to prevent engineering ambiguity. The goal is not only to describe the idea, but to define:

- what the system is
- what each module is allowed to do
- what data enters and leaves each stage
- what losses are optimized
- what training episodes look like
- how evaluation is run
- what must not be implemented because it violates the scientific design

The project target is a feedforward multi-view reconstruction and rendering system with:

- a canonical 3D scene state built from accumulated evidence
- Gaussian splats used only as local rendering primitives

## Non-Negotiable Scientific Principles

### Principle 1: Canonicality

The latent 3D scene state must be a function of the evidence set, not of the order in which views are processed.

Engineering consequence:

- views must be processed independently before fusion
- fusion must use commutative accumulation
- no recurrent fusion and no order-sensitive attention before the canonical state

### Principle 2: Monotone Improvement

As more valid views are added, the scene state should become more certain and more accurate in expectation.

Engineering consequence:

- raw evidence and confidence should accumulate, not be overwritten
- training should encourage convergence of smaller-context posterior readouts toward richer-context readouts
- uncertainty should decrease as evidence grows in sufficiently observed regions

### Principle 3: Separation of State and Renderer

The canonical scene state is the scientific object. Gaussians are only a rendering readout.

Engineering consequence:

- do not let the renderer become a second hidden scene representation
- do not let a learned refiner freely rewrite the canonical state

## System Overview

The system is:

`unordered context views -> per-view evidence -> commutative 3D evidence state -> canonical posterior readout -> local Gaussian generation -> target view rendering`

The canonical state must live in the 3D evidence layer, not in the Gaussian layer.

## Notation

This section defines all recurring variables used in this document.

### Scene and views

- `I_i`: RGB image of context frame `i`
- `K_i`: camera intrinsics of context frame `i`
- `T_i`: camera pose of context frame `i`
- `q`: index of a held-out target view
- `I_q`: RGB image of target view `q`
- `K_q`: camera intrinsics of target view `q`
- `T_q`: camera pose of target view `q`

### Context and target sets

- `Ck`: context set with `k` views
- `C2 subset C3 subset C4 subset C5 subset C6`: nested context sets; each larger set contains all views from the smaller set plus additional views
- `Q`: target set of held-out query views

### Spatial variables

- `v`: index of an active 3D cell or voxel
- `S_k`: canonical accumulated 3D evidence state built from context set `Ck`
- `R_k`: posterior readout derived from `S_k`
- `G_k`: Gaussian set read out from `S_k`

### Canonical state variables per cell

- `surface_evidence(v)`: accumulated support that cell `v` contains surface
- `free_evidence(v)`: accumulated support that cell `v` lies in free space
- `confidence(v)`: total accumulated evidence mass for cell `v`
- `geo_moment_1(v)`: first accumulated geometry moment for cell `v`
- `geo_moment_2(v)`: second accumulated geometry moment for cell `v`
- `app_moment_1(v)`: accumulated appearance statistic for cell `v`
- `app_weight(v)`: total appearance weight for cell `v`

### Posterior readout variables per cell

- `support_k(v)`: posterior support or occupancy score derived from `S_k`
- `unknown_k(v)`: unknownness score derived from `S_k`
- `mu_k(v)`: local surface mean for cell `v` derived from `S_k`
- `cov_k(v)`: local geometric covariance or spread for cell `v` derived from `S_k`
- `uncert_k(v)`: uncertainty score for cell `v` derived from `S_k`
- `conf_k(v)`: confidence score for cell `v` derived from `S_k`
- `alpha_k(v)`: Gaussian opacity read out from `S_k` for cell `v`

### Rendered outputs

- `G_k`: local Gaussian set generated from `S_k`
- `Ihat_q^(k)`: rendered target image for target view `q` using context set `Ck`

### Loss symbols

- `L_rend`: rendering loss
- `L_conv`: canonical convergence loss
- `L_mono`: monotone uncertainty loss
- `L_null`: low-evidence suppression loss
- `L_total`: total training objective

### Scalar coefficients and masks

- `lambda_*`: loss weights
- `m(v)`: teacher mask indicating that cell `v` is reliable in the richest-context state
- `tau_*`: threshold values used for support or confidence masking

## Module Graph

The intended module graph is:

1. episode sampler
2. per-view encoder
3. per-view geometric evidence predictor
4. per-view evidence writer
5. commutative sparse evidence accumulator
6. canonical posterior readout
7. optional adaptive subdivision proposal
8. local Gaussian readout
9. differentiable renderer
10. loss computation
11. evaluation and benchmark metrics

Module 7 is optional. Modules 1 to 6 and 8 to 10 are required.

## Data Organization

### Dataset

Primary dataset: RealEstate10K.

Each usable frame must provide:

- RGB image
- camera intrinsics
- camera pose
- frame index or timestamp

Optional metadata:

- clip id
- scene id
- quality flags

### Episode Definition

An episode is the atomic training and evaluation unit.

Each episode comes from one video clip and contains:

- a frame pool with at least 12 valid frames
- nested context sets
- one held-out target set

Required structure:

- `C2 subset C3 subset C4 subset C5 subset C6`
- `Q` is a disjoint target set
- `|Q| = 6`

Minimum usable frame pool for the standard benchmark:

- 6 context-capable frames
- 6 target frames
- total >= 12

If the frame pool is larger than 12, subsample consistently.

### Why This Episode Structure Matters

This design allows a single scene episode to answer all key questions:

- does rendering improve from 2 to 6 context views
- does the canonical state converge toward the richer-context state
- is the system robust to input order

### Episode Fields

Each episode object should expose:

```text
episode.scene_id
episode.clip_id
episode.frame_pool
episode.context_sets["2".."6"]
episode.target_set
```

Each frame record should expose:

```text
frame.image
frame.intrinsics
frame.pose
frame.frame_id
frame.timestamp
```

### Input Contract to the Model

For one forward pass at context size `k`, model input is:

```text
{
  "context_views": [(I_i, K_i, T_i) for i in Ck],
  "target_views":  [(I_q, K_q, T_q) for q in Q],
  "scene_bounds": optional,
  "voxel_spec": grid definition
}
```

Important:

- `context_views` are an unordered set for the model
- `target_views` are used only for rendering supervision and evaluation
- `k` is the number of context views in the current forward pass

## Canonical State Specification

The canonical state is a sparse set of active 3D cells.

Each active cell `v` should store additive sufficient statistics, not arbitrary hidden features.

Recommended fields:

```text
cell.surface_evidence
cell.free_evidence
cell.total_confidence
cell.geo_moment_1
cell.geo_moment_2
cell.app_moment_1
cell.app_weight
```

Optional fields:

```text
cell.normal_moment_1
cell.normal_moment_2
cell.conflict_score
cell.last_update_step
```

### Constraints on Canonical State

- evidence contributions must be non-negative
- accumulation must be commutative
- accumulation must be associative in implementation as far as numerically practical
- the state should never be globally rewritten by a learned module

### Derived Posterior Readouts

From the accumulated cell statistics, compute:

- support probability
- free-space probability or free-space score
- unknownness
- local surface mean
- local geometric covariance or spread
- uncertainty score

These derived readouts are the correct objects for convergence losses.

Raw counts are not the correct objects for subset equality losses.

## Per-View Evidence Writer

For each context view, the per-view network must output:

- image features
- depth distribution or depth band
- confidence / uncertainty

Then the evidence writer maps the view into 3D cell updates.

Required behavior along each ray:

- cells before the likely surface receive free-space evidence
- cells near the likely depth receive surface evidence
- geometry statistics are deposited near the surface
- appearance statistics are deposited near the surface

Forbidden behavior:

- writing global scene updates based on other views
- using view order inside the writer

## Accumulator

The accumulator merges per-view sparse cell updates into one scene state.

The merge operator must be additive field-wise for all evidence statistics.

Examples:

- surface evidence sums
- free-space evidence sums
- geometry moments sum
- appearance moments sum

The accumulator is the place where order invariance is made exact.

## Canonical Readout

This module converts additive evidence statistics into stable posterior quantities.

Recommended outputs:

```text
state_readout.support
state_readout.unknown
state_readout.local_surface_mean
state_readout.local_surface_cov
state_readout.uncertainty
state_readout.confidence
```

If a cell shows:

- high support
- high uncertainty
- high contradiction

then the correct next action is adaptive subdivision or local ambiguity handling, not arbitrary feature rewriting.

## Gaussian Readout

Gaussians are emitted locally from the canonical readout.

For each sufficiently supported cell:

- center comes from local surface mean
- covariance comes from local surface covariance, clipped for numerical stability
- opacity comes from a monotone function of support and confidence
- appearance comes from local appearance readout

Important restrictions:

- Gaussian generation must be local
- Gaussian generation must not use global recurrent fusion
- Gaussian generation must not mutate canonical cell evidence

Optional engineering choice:

- one Gaussian per active cell for the first version

Possible later extensions:

- adaptive number of Gaussians per cell
- subdivision for ambiguous cells

## Renderer

The renderer receives:

- Gaussian parameters
- target cameras

and returns:

- rendered RGB target images
- optional depth or opacity maps

The renderer must stay downstream of the canonical state.

## Training Orchestration

### Batch Construction

For each batch item:

1. sample one episode from one clip
2. construct nested context sets `C2, C3, C4, C5, C6`
3. construct one fixed target set `Q` of size 6
4. run separate forwards for context sizes 2 to 6

The target set must remain fixed across context sizes within the same episode.

### Forward Schedule

For each context size `k in {2,3,4,5,6}`:

1. run per-view evidence extraction independently on views in `Ck`
2. write sparse 3D evidence
3. accumulate to canonical state `S_k`
4. compute canonical posterior readout `R_k`
5. read out Gaussian set `G_k`
6. render all target views in `Q`
7. store rendered outputs and state readouts

### Important Training Clarification

The forward passes for different context sizes are not different models.

They are repeated evaluations of the same model under different amounts of evidence from the same scene episode.

## Objective Function

The objective should supervise:

- target-view rendering quality
- convergence of smaller-context posterior readouts toward richer-context posterior readouts
- monotone reduction of uncertainty as evidence accumulates
- suppression of unsupported floaters

### Loss 1: Rendering Loss

For each context size `k`, render all 6 target views.

Use:

- L1
- SSIM
- LPIPS

Definition:

```text
L_rend = average over k in {2..6} and q in Q of
         [lambda_l1 * L1 + lambda_ssim * SSIM_loss + lambda_lpips * LPIPS]
```

Variable explanation:

- `k`: context size, one of 2, 3, 4, 5, 6
- `q`: target-view index in the held-out target set `Q`
- `Ihat_q^(k)`: prediction for target view `q` using context set `Ck`
- `I_q`: ground-truth RGB image for target view `q`
- `lambda_l1`, `lambda_ssim`, `lambda_lpips`: internal weights for the three rendering terms

This is the primary task loss.

### Loss 2: Canonical Convergence Loss

Use the richest-context state `S_6` as the within-episode teacher.

Important:

- use stop-gradient on the teacher side
- compare posterior readouts, not raw additive evidence counts

Teacher mask:

- only evaluate where `S_6` has sufficient confidence and support

Recommended compared quantities:

- support probability
- local surface mean
- optional local covariance if stable

Definition:

```text
L_conv = average over k in {2,3,4,5} of
         masked distance between readout(S_k) and stopgrad(readout(S_6))
```

Variable explanation:

- `S_k`: canonical evidence state built from context set `Ck`
- `S_6`: richest-context canonical evidence state built from 6 context views
- `readout(S_k)`: posterior quantities computed from state `S_k`, such as support probability and local surface mean
- `stopgrad(...)`: stop-gradient operator; the richer-context teacher is treated as a target and is not updated through this loss
- `m(v)`: reliability mask selecting cells that have enough support and confidence in `S_6`

Recommended concrete compared variables:

- `support_k(v)` against `support_6(v)`
- `mu_k(v)` against `mu_6(v)`
- optionally `cov_k(v)` against `cov_6(v)` if covariance is numerically stable

This loss encodes convergence toward the fuller-evidence state.

### Loss 3: Monotone Uncertainty Loss

For nested context sizes, penalize increases in uncertainty in teacher-supported regions.

Definition:

```text
L_mono = sum over k in {2,3,4,5} of
         hinge( uncertainty(S_{k+1}) - uncertainty(S_k) )
```

Variable explanation:

- `uncertainty(S_k)`: uncertainty readout from the canonical state at context size `k`
- `S_{k+1}`: the next richer-context state in the nested chain
- `hinge(x)`: penalty applied only when `x` is positive, meaning uncertainty increased instead of decreased

This loss is applied only in teacher-supported regions selected by `m(v)`.

Interpretation:

- more evidence should not increase uncertainty in already supported regions, except for small tolerance

### Loss 4: Low-Evidence Suppression

In cells with very low confidence, penalize occupancy or rendered opacity.

Definition:

```text
L_null = penalty on occupancy or opacity where confidence is very low
```

Variable explanation:

- low-confidence cells are identified using a threshold on `conf_6(v)` or the corresponding low-evidence score from `S_6`
- the penalized quantity is either posterior support or Gaussian opacity `alpha_6(v)`

This reduces floaters and fog-like artifacts.

### Final Objective

```text
L_total = lambda_rend * L_rend
        + lambda_conv * L_conv
        + lambda_mono * L_mono
        + lambda_null * L_null
```

Variable explanation:

- `lambda_rend`: weight for rendering quality
- `lambda_conv`: weight for convergence toward the richer-context teacher state
- `lambda_mono`: weight for uncertainty monotonicity
- `lambda_null`: weight for low-evidence suppression

Recommended starting weights:

- `lambda_rend = 1.0`
- `lambda_conv = 0.1 to 0.25`
- `lambda_mono = 0.05 to 0.1`
- `lambda_null = 1e-3 to 1e-2`

### Why Earlier Subset Loss Was Rejected

Do not directly penalize equality of:

- raw surface evidence
- raw free-space evidence
- raw accumulated counts

across different subset sizes.

Reason:

- raw evidence should increase as more views are added
- forcing raw-count equality conflicts with monotone accumulation

The right convergence target is the posterior readout of the richer-context state.

## What Engineers Must Not Implement

- no recurrent view-order-sensitive fusion before the canonical state
- no transformer or attention block that mixes views before evidence writing unless it is provably permutation invariant and still preserves independent evidence semantics
- no learned module that freely overwrites the canonical state after accumulation
- no subset loss that forces raw evidence equality across context sizes
- no target leakage, meaning target views must never be inserted into the context set
- no evaluation where targets change when context count changes within the same episode

## Evaluation and Benchmark

The benchmark must measure both rendering and structural properties.

### Evaluation Split

Use held-out RE10K validation/test episodes with the same episode structure as training:

- one frame pool
- fixed target set of 6
- nested contexts from 2 to 6

### Benchmark 1: Novel View Synthesis

For each context size `2,3,4,5,6`, report:

- PSNR
- SSIM
- LPIPS

The main figure should be a curve of quality versus number of context views.

### Benchmark 2: Canonical Convergence

Measure:

- distance from `readout(S_k)` to `readout(S_6)`
- optionally per-cell error only in well-supported teacher regions

This is the key benchmark for the canonical-state claim.

### Benchmark 3: Permutation Sensitivity

For a fixed context set, randomly permute the context order and verify:

- canonical raw accumulation is invariant up to numerical precision
- rendered outputs are stable

### Benchmark 4: Incremental Update Stability

Measure change size from:

- `S_2 -> S_3`
- `S_3 -> S_4`
- `S_4 -> S_5`
- `S_5 -> S_6`

Expected pattern:

- updates should become smaller as the state saturates in observed regions

### Benchmark 5: Robustness

Evaluate under:

- random context dropout
- varying baseline
- uneven coverage
- incremental arrival of new views

The expectation is that CanonicalGS is more stable than direct Gaussian prediction under these perturbations.

## Recommended Engineering Delivery Order

Engineers should implement in this order:

1. dataset episode builder for nested contexts and fixed targets
2. view encoder and depth / confidence head
3. sparse per-view evidence writer
4. sparse commutative accumulator
5. canonical posterior readout
6. local Gaussian readout
7. renderer
8. rendering loss only
9. convergence loss
10. monotone uncertainty loss
11. low-evidence suppression
12. benchmark suite

Do not start with optional refinement modules.

## Sanity Checks Before Full Training

Before long training runs, verify:

1. the same context set in different orders gives identical accumulated raw evidence
2. larger context sets produce larger or equal raw confidence in touched cells
3. target views are fully disjoint from context views
4. the model can overfit a tiny subset of scenes
5. rendering improves from 2 to 6 context views on the tiny subset
6. convergence loss decreases when moving from `S_2` toward `S_6`

## Minimal Version Versus Full Version

### Minimal Version

Implement:

- one Gaussian per active cell
- no adaptive subdivision
- support, confidence, local geometry moments
- rendering loss plus convergence loss plus uncertainty loss

This is the correct first prototype.

### Full Version

Later add:

- adaptive subdivision
- better local geometry statistics
- richer appearance readout
- ambiguity handling for multi-surface cells

These are extensions, not the starting point.

## Final Implementation Summary

CanonicalGS is not a direct Gaussian predictor.

It is a canonical evidence-state system with the following contract:

- each context view is converted independently into uncertainty-aware 3D evidence
- evidence is accumulated commutatively into a sparse canonical 3D state
- the canonical state stores support, free-space, confidence, and local geometry statistics
- richer context sets should produce posterior states closer to the richer-evidence reference and lower uncertainty
- Gaussian splats are emitted locally only for rendering
- training is driven by rendering, canonical convergence, and monotone uncertainty reduction
- evaluation must show both novel-view quality and state convergence from 2 to 6 context views
