# Prospective cross-architecture JMLR protocol

## Registration boundary

This protocol was fixed before any checkpoint with the seeds below was trained
and before any corresponding strict-separation result was evaluated.  It was
designed after the earlier seed-20260714/class-12 headline result and the
retrospective five-checkpoint suite were known.  It is therefore prospective
for the new checkpoints and outcomes below, but it is not evidence that the
research programme as a whole was preregistered.

The local SHA-256 digest binds the protocol to the resulting checkpoints and
artifacts.  A local digest is not a trusted public timestamp.  The release must
preserve this file's history or archive it before the prospective result is
used as externally time-stamped evidence.

## Frozen design

- Dataset: GTSRB archive recorded as Zenodo record 13741936, archive MD5
  `7faafd0d78d831c7eb4d555641a83a62`.
- New training seeds: 20260801, 20260802, 20260803, 20260804, and 20260805.
- Architectures:
  - `cnn48`: Conv3x3(3,16)-ReLU-AvgPool-Conv3x3(16,32)-ReLU-
    AvgPool-FC128-ReLU-FC43.  The six mechanism groups partition the 32
    channels after the second convolution.
  - `mlp192`: Flatten-FC192-ReLU-FC128-ReLU-FC43.  The six mechanism groups
    partition the first 192 hidden units.
- Training: 15 epochs, batch size 256, AdamW with learning rate 0.002 and
  weight decay 0.0001, cosine annealing over 15 epochs, deterministic seeded
  gain augmentation, a seed-specific 4,000-row validation split, and the best
  validation checkpoint.  Weights are frozen to denominator 4096 before
  evaluation.  The empirical training device may be MPS or CPU and is recorded.
- Targets: GTSRB classes 1, 2, 12, 13, and 14.
- Contrast and anchors: for each checkpoint and target, choose the modal
  runner-up contrast among correctly classified test images, then the four
  highest-margin images with that contrast.  Lexical path order breaks ties.
  The first two are construction anchors and the next two are held out.
- Group discovery: rank the intervention-layer units by mean absolute change
  in the true-class logit under one-unit ablation on the first 256 validation
  rows.  Partition the fixed ranking round-robin into six groups.  Discovery
  sees no target group, anchor, mechanism triangle, candidate interface,
  stable error, or verifier outcome.

## Frozen mechanism triangles

The three triangles below were selected structurally, without access to a new
checkpoint.  Vertex coordinates are gates for the six discovered groups.

- `triangle_a`: `(0,0,0,1,1,1)`, `(1,0,0,1,0,1)`,
  `(0,0,1,1,0,1)`.
- `triangle_b`: `(0,1,1,0,1,0)`, `(1,1,0,0,1,0)`,
  `(0,1,0,1,1,0)`.
- `triangle_c`: `(0,1,1,0,0,1)`, `(1,0,1,0,0,1)`,
  `(0,0,1,1,0,1)`.

Every triangle uses the denominator-4 barycentric grid plus the denominator-32
grid on all three boundary edges, giving 99 distinct points per anchor.  These
are exact rational finite nets, not continuum certificates.

## Frozen intervention and selection rule

- Claim: target-versus-contrast logit effect of removing mechanism group 0.
- Hard interface: all five unit-allocation interventions on groups 1 through 5.
- Fractional library: all ten unordered hard-query pairs at mixture weights
  1/4, 1/2, and 3/4.  The lower-index endpoint is replaced, giving 30
  five-call hybrid interfaces.
- Cost: hard and hybrid interfaces match five calls and total unit
  attenuation five.  Group support, touched units, and any other physical cost
  are reported separately and are not called matched.
- Primary decoder sensitivity: `L=14`.
- Secondary sensitivity values: `L` in 4, 8, 14, and 20.
- Selection uses only the two construction anchors: minimum exact construction
  error, then minimum exact Lipschitz constant required for zero construction
  error, then lexical candidate identifier.
- Confirmation opens the two held-out anchors and includes all construction,
  held-out, and cross-anchor world pairs.

## Frozen outcomes and estimands

Each architecture-seed-class-triangle unit receives exactly one outcome:

1. `CONFIRMED_STRICT_WIN_ON_DECLARED_NET`: the hard all-anchor error is
   positive, the selected hybrid has zero construction error, and its
   all-anchor error is zero.
2. `SELECTED_INTERFACE_DID_NOT_CONFIRM`: a zero-error construction interface
   was selected but its all-anchor error is positive.
3. `NO_ZERO_ERROR_FRACTIONAL_CANDIDATE`: the selected construction error is
   positive while the hard all-anchor error is positive.
4. `NO_HARD_OBSTRUCTION_ON_DECLARED_NET`: the hard all-anchor error is zero.
5. `TRAINING_OR_ANCHOR_FAILURE`: the quantized checkpoint has test accuracy
   below 0.60 or a target has fewer than four eligible anchors.  Such failures
   remain in the denominator and are never silently replaced.

The primary descriptive estimand is the confirmed count and rate among all 150
frozen units (2 architectures x 5 seeds x 5 classes x 3 triangles).  Secondary
descriptions are per-architecture rates, the rate among units with both a hard
obstruction and a construction-valid selected candidate, and the number of
checkpoints, classes, and triangles with at least one confirmation.  Wilson
intervals are reported as descriptive binomial intervals only; repeated units
within a checkpoint are dependent, so no population-frequency or independent-
trial interpretation is permitted.

Every negative or inapplicable outcome, all 30 construction errors, selected
interface, exact stable errors, Jensen control, affine interpolation control,
cost record, model/input hashes, and exact rational world table must be retained.

## Decision rule and claim boundary

The study has no acceptance-contingent stopping rule and no result-dependent
change to the manuscript's theorem.  Any outcome is reported.  A positive
result broadens prospective finite-net evidence; a null result limits prevalence
claims.  Neither result upgrades a finite net to a continuum proof, supplies
human-readable channel semantics, validates external-only access, or proves an
acceptance probability at any venue.
