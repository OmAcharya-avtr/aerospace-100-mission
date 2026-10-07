# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-07

First release. Validation level 3 (research grade). Status: TESTING.
Contains a learned component, benchmarked against an exact computation and
reported as losing on that computation's own criterion.

### Added

- `simplexguard.polytope` — convex polytopes in halfspace form and the support
  function algebra the switching condition rests on: `Polytope` with an
  LP support function, `Box` with the closed form `c.m + abs(c).r`, the
  Pontryagin difference by the exact halfspace identity, preimage under a
  linear map, LP-based redundancy removal, a signed Chebyshev radius that is
  negative exactly when the set is empty, and 2-D vertex and area helpers for
  plotting only.
- `simplexguard.plant` — the declared model: `A`, `B`, the declared disturbance
  set `W`, the declared constraint sets `X` and `U`, and the four assumptions
  that bound what any result means, written down in the module docstring.
  `Plant.step` deliberately does **not** check the disturbance against `W`, so
  that the bound-violation experiment can break the assumption on purpose.
  `reference_plant` is an illustrative single-axis attitude loop, labelled as
  illustrative wherever it appears.
- `simplexguard.controllers` — discrete LQR gains, a conservative baseline and
  an aggressive performance controller, both saturated into the declared input
  box, with a `saturates_at` flag because the multi-step predictor's accuracy
  depends on how often the saturation binds.
- `simplexguard.invariant` — the maximal robust invariant set by the one-step-set
  recursion, and `verify_robust_invariance`, which checks the three certificate
  properties **exactly** by support function rather than by sampling, from
  scratch, so the set is produced by one path and confirmed by another. An empty
  set and a non-converged recursion each raise their own exception rather than
  returning something that is not a certificate.
- `simplexguard.guard` — the exact switching condition
  `c_j^T (A x + B u) + h_W(c_j) <= d_j`, with the eroded set precomputed once, a
  signed margin in state units, explicit flags for an inadmissible input and for
  a state outside the certificate, a brute-force vertex-enumeration path for
  cross-checking, and an optional minimum **baseline** dwell. There is no
  minimum performance dwell, by design: extending baseline authority is always
  safe and extending performance authority is not.
- `simplexguard.simulate` — guarded, unguarded and baseline-only episodes on a
  shared disturbance sequence, so every comparison is paired; three disturbance
  samplers (zero, uniform, worst-case vertex) and a scaled sampler for the
  bound-violation experiment.
- `simplexguard.accounting` — the six assurance quantities: switch rate,
  dwell-time distribution per authority, authority share, paired conservatism
  cost against both extremes, constraint-violation count and invariant-set exit
  count, with the switching-margin distribution. The report refuses unpaired
  episodes rather than computing a meaningless difference.
- `simplexguard.boundviolation` — the deliberate bound-violation sweep: the
  guard unchanged, the realised disturbance inflated, and the smallest scale at
  which each kind of failure first appears.
- `simplexguard.reachability` — the exact `L`-step worst-case and nominal
  predictors, by unrolling the linear performance loop and maximising each
  disturbance term by support function. Horizon 0 reproduces the one-step
  condition exactly.
- `simplexguard.predictor` — the learned switch predictor: episode-split
  datasets, an isotonic-calibrated random forest and logistic regression,
  confusion-matrix metrics, Brier score, expected calibration error, lead-time
  statistics and single-row latency measurement, all benchmarked against the
  exact predictors on the same held-out episodes.
- CLI `python -m simplexguard` with `plant`, `invariant`, `guard`, `run`,
  `accounting`, `bound-sweep` and `predict`, and distinct exit statuses for bad
  input (2) and an empty or non-converged invariant set (3).
- 218 tests including Hypothesis property tests for the support-function and
  set-algebra identities, hand-calculated known-answer tests with the arithmetic
  shown in the comments, input-validation and edge-case tests, an integration
  test and a regression suite that pins the reported numbers.
- Eight validation scripts with their committed raw output, five examples each
  producing a PNG in `screenshots/`, and `docs/REQUIREMENTS.md` with 77 numbered
  requirements, each mapped to the test or validation script that exercises it.

### Measured negative results in this release

Published because they are the most informative part of the package, not
despite being negative.

- **The learned switch predictor loses to the exact condition on the exact
  condition's own criterion**, on accuracy (F1 0.980685 against 1.000000) and on
  latency (a factor of 846). No forest size closes the latency gap; a 20-tree
  forest is still 198 times the cost. No retune was attempted.
- **The guarantee has essentially no margin.** A 10 % underestimate of the
  declared disturbance bound already produces constraint violations under a
  worst-case sampler. The reason is structural: the recursion computes the
  *maximal* robust invariant set, whose invariance holds with equality and whose
  facet-wise slack inside the constraint set is exactly zero.
- **The guard chatters**: 64.8 % of performance intervals and 31.4 % of baseline
  intervals last a single step on the reference scenario.
- **A minimum baseline dwell of 2 reduces the switch rate by nothing** while
  raising the authority fraction from 0.084 to 0.096 and the cost ratio from
  1.198341 to 1.201536. The validation check that asserted a strictly decreasing
  switch rate was weakened to non-increasing when this was measured, and the
  plateau is reported rather than removed.
- **The exact worst-case multi-step predictor is not sound on realised
  episodes** (recall 0.862841), because it ignores the saturation of the
  performance input (measured at 9.3729 % of steps) and the guard's own
  intervention inside the horizon.
- **Below a reference amplitude of 0.12 rad the guard never fires** and is pure
  overhead on this scenario.
- **At twelve times the declared bound the guard eliminates only 19.79 %** of
  the violations it eliminates completely at the declared bound.
- The exact switching condition and brute-force vertex enumeration **disagree on
  4 of 10000 states bisected onto the switching boundary**, at an absolute
  margin of 5.551e-17. Floating point, not algebra; reported rather than
  tolerance-tuned.

### Known limitations in this release

- Linear discrete-time plants only; no nonlinearity, no parametric uncertainty,
  no hybrid dynamics.
- Full state feedback with no observer, no measurement noise and no computation
  delay. One step of delay changes the switching condition and the one-step-ahead
  version is not implemented.
- The robust invariant set covers 85.5 % of the declared constraint set, so
  14.5 % of states that satisfy every declared constraint are outside the
  certificate and the guard will not operate there.
- The invariant-set recursion costs one linear programme per facet per
  iteration. It is 3.2 s on the shipped 2-state plant and grows sharply as the
  declared disturbance approaches the scale at which no certificate exists.
- Every reported accounting number comes from a 2-state plant. The set algebra
  is checked in dimensions 1 to 5, but nothing here establishes how the
  recursion behaves at high dimension.
- The shipped plant, controllers and tracking-cost weights are illustrative.

### Tool defects found and worked around

- `sklearn.calibration.CalibratedClassifierCV(base, cv="prefit")` raises
  `InvalidParameterError` on scikit-learn 1.9.1: the `"prefit"` value was
  removed in favour of wrapping the fitted estimator in
  `sklearn.frozen.FrozenEstimator`. `fit_switch_predictor` uses the
  `FrozenEstimator` form.
- `RandomForestClassifier` with `n_jobs > 1` is **slower** at single-row
  inference, not faster: 33.1 ms against 5.8 ms per `predict_proba` for a
  150-tree forest, a factor of 5.73, because joblib's per-call dispatch
  dominates. `fit_switch_predictor` sets `n_jobs = 1` after fitting so the
  learned model is benchmarked at its best latency.
