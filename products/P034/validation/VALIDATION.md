# faultinject — validation evidence (Level 2, research grade)

**Product:** P034 FaultInject · **Version:** 0.1.0 · **Date of run:** 2026-10-05

**Environment.** Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
pytest 9.1.1, Hypothesis 6.168.3, Ruff. Build container: **1 CPU core**, with
**five build agents sharing it** during this run. PyTorch is not available in
this container, which is why the learned model is a scikit-learn forest.

Every number in this file was produced by running the scripts in this directory
in the session that wrote it. Each script's raw stdout is committed beside it.
No number here was typed by hand, and no tolerance was changed after seeing a
result.

| # | Check | Script | Raw output | Result |
|---|---|---|---|---|
| 1 | Seeded replay is bit-identical over all 248 coverage cells | `validate_replay.py` | `replay_output.txt` | PASS |
| 2 | Coverage accounting vs a hand enumeration | `validate_coverage.py` | `coverage_output.txt` | PASS |
| 3 | Constant sensor bias vs the analytic innovation shift | `validate_analytic_bias.py` | `analytic_bias_output.txt` | PASS |
| 4 | NaN injection detected, including the absorbed case | `validate_nan_detection.py` | `nan_detection_output.txt` | PASS (one blind spot verified and documented) |
| 5 | Campaign search: two baselines vs the learned prioritiser | `validate_benchmark.py` | `benchmark_output.txt` | PASS structurally; **learned shows no measurable advantage over the non-learned kind-mean ablation** |
| 6 | Wrapper transparency and taxonomy distinctness | `validate_transparency.py` | `transparency_output.txt` | PASS (one non-monotonicity found and documented) |

Reproduce, from `products/P034/`:

```bash
python validation/validate_replay.py
python validation/validate_coverage.py
python validation/validate_analytic_bias.py
python validation/validate_nan_detection.py
python validation/validate_transparency.py
python validation/validate_benchmark.py
```

Wall clock on the shared 1-core container, including interpreter start: 17.1 s,
8.8 s, 5.9 s, 3.4 s, 3.4 s and 33.5 s respectively — **72.1 s in total**. The
test suite (`python -m pytest tests/ -q`) is **202 tests**, all passing, in
about 20–30 s on the same machine. These figures are from a contended machine
and will vary.

---

## 1. Seeded replay is bit-identical

**What "bit-identical" means here.** The comparison is over the IEEE 754
binary64 byte image of the whole trace: true position and velocity, both
estimator states, the applied command, the reference and every logged
innovation. A comparison to a tolerance would hide exactly the non-determinism
this check exists to find.

**Scope.** A pool of 248 cases, one in every coverage cell, so all sixteen fault
kinds, all four channels, every parameter bin and both the deterministic and the
stochastic handlers are exercised.

| Check | Result |
|---|---|
| 1a. Same case run twice, byte images compared | 248 cases, **2 622 456 bytes**, **0 mismatches** |
| 1b. Case → JSON → case → run, compared with the original run | 248 round trips, **0 mismatches** |
| 1c. Tampered case file rejected | `ValueError: case_id mismatch: file says bd32605725b734a5, content hashes to b7e9137ae2f99b98` |
| 1d. Negative control: `seed + 1` changes the byte image | **248 of 248** cases changed |
| 1e. Content-hash regression for a fixed case | `bd32605725b734a5`, unchanged |

Check 1d matters: without it, 1a would pass trivially if the seed had no effect.

## 2. Coverage accounting vs a hand enumeration

**Subset chosen by hand:** `{sensor_bias, timing_late_sample}` — two fault
classes, two channel sets, one log-scaled continuous parameter with three bins
and one integer parameter with two bins, in 32 cells.

| Check | Hand value | Computed | Result |
|---|---|---|---|
| 2a. Cell labels of the subset, in order | 32 typed-out labels | 32 | identical, cell for cell |
| 2b. Per-kind cell counts, all 16 kinds | products of `channels × 2 × 2 × param bins` | same | 16 of 16 match |
| 2b. Total cells | **248** | **248** | match |
| 2c. Coverage after a six-injection campaign (one repeated) | **5 of 32 = 0.156250000000** | 5 of 32 = 0.156250000000 | exact |
| 2c. Per-kind breakdown | `sensor_bias 3 of 24`, `timing_late_sample 2 of 8` | same | match |
| 2d. Every cell reachable by a constructed case | 248 | 248, **0 unreachable** | PASS |
| 2e. Out-of-subset injection rejected | — | `ValueError` raised | PASS |

The binning arithmetic for each of the six injections is printed step by step in
`coverage_output.txt`, so the hand computation can be checked line by line.

**Defect found and fixed by 2d.** The first run of this check reported 12
unreachable cells: the geometric midpoint of the top `numerical_overflow`
magnitude bin was computed as `sqrt(lo * hi)`, and `lo * hi` overflows to
infinity when `hi = 1e300`. The midpoint is now taken in log space.
`tests/test_taxonomy.py::test_representative_of_huge_log_bin_is_finite` guards it.

## 3. Constant sensor bias vs the analytic innovation shift

**Derivation** (full statement with assumptions in the script docstring). With
`M = A(I − KH)` and `Δ` denoting biased minus unbiased at the same seed,

```
Δε_{k+1} = M Δε_k − A K b,   Δε_0 = 0
Δe_k     = H Δε_k + b
         = b − H (I − M)^{-1} (I − M^k) A K b                 (1)
Δe_∞     = G b,   G = I − H (I − M)^{-1} A K                  (2)
mean_k Δe_k = G b + (1/N) H (I−M)^{-2} (I − M^N) A K b        (3)
```

The control input cancels in the prediction-error recursion, so the result holds
for any control law, including the saturating one this target uses. The noise
cancels because both runs use the same seed.

**Error dynamics.** Eigenvalues of `M`: **0.992030856181997** and
**0.609652027224596**, both inside the unit circle.

| Check | Measured | Analytic | Tolerance | Result |
|---|---|---|---|---|
| 3a. `Δe_0` for a 1.0 m position bias | error **0.000e+00** | `b` exactly | 1e-12 | PASS |
| 3b. Every step vs Eq. (1), position bias | max error **2.054e-15** | — | 1e-12 | PASS |
| 3b. Every step vs Eq. (1), velocity bias | max error **3.109e-15** | — | 1e-12 | PASS |
| 3c. Mean shift over 150 steps, position bias | **0.584598217** m | **0.584598217** m | 1e-12 (err 5.551e-16) | PASS |
| 3c. Mean shift over 150 steps, velocity bias | **−1.047688073** m | **−1.047688073** m | 1e-12 (err 2.220e-15) | PASS |
| 3d. Steady-state shift, position bias, Eq. (2) | — | **−3.553e-15** (analytically zero) | 1e-13 | PASS |
| 3d. Steady-state shift, velocity bias, Eq. (2) | — | **−2.509528032** m per m/s of bias | must be non-zero | PASS |
| 3e. Linearity of `Δe / b` over b ∈ {0.1, 0.25, 1, 4, 10} | worst deviation **7.772e-15** | constant | 1e-12 | PASS |
| 3f. Seed independence over 5 noise seeds | worst deviation **2.887e-15** | identical | 1e-12 | PASS |

**The result worth reading twice.** For a constant *position* bias the
steady-state innovation shift is exactly zero: the filter absorbs the bias into
its state estimate and an innovation-mean monitor eventually sees nothing. A
naive expectation that "a bias of `b` shifts the innovation mean by `b`" is
right only for the first step. For a *velocity* bias the steady-state position
innovation shift is **−2.509528032 m per m/s** and does not decay.

3f is checked to a tolerance, not bit-identically, and the script says why: the
two runs being differenced round differently, so bit identity across *seeds* is
not expected. Bit identity for the *same* case is check 1a.

## 4. NaN injection is detected rather than silently absorbed

| Check | Result |
|---|---|
| 4a. `classify_value` vs a 15-row hand table of IEEE 754 values | all rows match |
| 4b. NaN on `pos`, `vel`, `u`, plain target | detected on all three, **detection latency 0 steps**, severity exactly **1.000000**, trace non-finite |
| 4c. NaN on `pos` with a target that sanitises it | trace stays finite, severity **0.002414** (negligible), monitor still reports it, verdict **`absorbed`** |
| 4d. Negative control: fault-free run | nothing reported |
| 4e. All 12 `numerical_nan` coverage cells | 12 of 12 detected |
| 4f. Denormal on `pos`, `vel`, `u` | `subnormal` detected on all three |

**4c is the point of the check.** A target that replaces a NaN with zero
produces a severity of 0.002414 — the scoring function alone would call it
negligible. The boundary monitor is what makes it visible, and it labels the
case `absorbed` rather than `absent`.

**Documented blind spot, verified in 4f.** The monitor flags `nan`, `inf`,
`subnormal` and magnitudes above `sqrt(DBL_MAX) = 1.340781e+154` (the smallest
magnitude certain to overflow on one multiplication). An injected
`numerical_overflow` magnitude *below* that threshold is a finite normal number
and is **not** flagged:

| magnitude | monitor classes | severity |
|---|---|---|
| 1.0e+08 | none | 0.902667 |
| 1.0e+50 | none | 0.902667 |
| 1.0e+150 | none | 0.902667 |
| 1.0e+250 | `large` (verdict `absorbed`, the command saturation swallows it) | 0.902667 |

Those cases are caught by the severity score's divergence test instead, and the
script fails if a sub-threshold magnitude is neither flagged nor severe.

## 5. Campaign search: two baselines vs the learned prioritiser

**Protocol.** 3 case pools, each covering all 248 cells twice (496 cases); the
severity of every case in every pool computed exhaustively; 8 strategy seeds per
pool, so **24 runs per strategy**; budget **60 executions, 12.1 % of the pool**;
learned warm-up 32 random executions, **counted against the budget**; severe
means severity ≥ 0.6; 5000-resample percentile bootstrap.

**Ground truth.** 28.49 % of pool cases are severe, so a budget of 60 spent
uniformly at random has expectation **17.10** severe cases.

| Strategy | Mean severe found | 95 % interval | Mean severity of executed cases | Final coverage |
|---|---|---|---|---|
| uniform_random (baseline 1) | 16.583 | [15.208, 17.958] | 0.3258 | 0.2266 |
| coverage_greedy (baseline 2) | 16.708 | [15.458, 18.000] | 0.3299 | 0.2419 |
| kind_mean (non-learned ablation) | **34.292** | [32.792, 35.750] | 0.5985 | 0.1899 |
| learned (random forest) | 33.875 | [32.083, 35.583] | 0.5764 | 0.1892 |

**Verdicts under the specification's overlap rule** (the baseline's interval
contains the other strategy's mean ⇒ `no measurable advantage`):

| Comparison | Verdict |
|---|---|
| learned vs uniform_random | **measurable advantage** (33.875 is above 17.958) |
| learned vs coverage_greedy | measurable advantage |
| **learned vs kind_mean** | **no measurable advantage** |
| **coverage_greedy vs uniform_random** | **no measurable advantage** |
| kind_mean vs uniform_random | kind_mean ahead |

Paired bootstrap on the per-run difference (same pool, same seed), which is
tighter because it removes between-pool variance:

| Difference | Mean | 95 % interval | |
|---|---|---|---|
| coverage_greedy − uniform_random | 0.125 | [−0.292, 0.543] | contains zero |
| kind_mean − uniform_random | 17.708 | [16.583, 18.875] | excludes zero |
| learned − uniform_random | 17.292 | [15.917, 18.625] | excludes zero |

**What this means, stated plainly.** The learned prioritiser finds about twice
as many severe cases as uniform random on the same budget, and that gap is well
outside the random baseline's interval. But a four-line non-learned heuristic —
rank untried cases by the running mean severity of their fault *kind* — scores
34.292 against the model's 33.875, and the two intervals overlap heavily. On
this benchmark **the learned model shows no measurable advantage over the
non-learned ablation**: essentially all of the gain over random comes from
learning which fault kinds are dangerous, which is something a campaign engineer
also knows after twenty executions.

**Coverage-greedy is not a severity-finding strategy.** It reaches more cells
(0.2419 against 0.2266) but finds no more severe cases than random; the paired
interval contains zero. That is the expected behaviour and it is reported rather
than buried: coverage and severity are different objectives.

**Uncertainty output.** Empirical coverage of the nominal 95 % interval from the
forest's ensemble spread: **0.947581**, mean interval width **0.830661** on a
severity range of [0, 1]. The nominal coverage is reached by being
uninformative, not by being calibrated. Stated the same way in MODEL_CARD.md.

**Top impurity-based feature importances** (last fitted forest): `channel:u`
0.3415, `param0` 0.1135, `duration_frac` 0.1059, `start_frac` 0.0786,
`kind:numerical_nan` 0.0783, `kind:numerical_overflow` 0.0687.

## 6. Wrapper transparency and taxonomy distinctness

| Check | Result |
|---|---|
| 6a. Bit-transparency vs an independently written closed loop, 50 seeds | **0 mismatches**, 10 800 bytes per trace |
| 6b. A target written with no knowledge of this package, wrapped and injected | identical to the bare target before the window, different inside it, called exactly 40 times |
| 6c. `bus_reorder` produces out-of-order deliveries | 14 at `swap_prob = 0.5` |
| 6c. `bus_reorder` at `swap_prob = 1.0` | **0 out of order** — see below |
| 6c. `bus_reorder` with no swap fired | delivery is exactly a one-step delay |
| 6d. `bus_delay` after its window closes | still **4 steps stale**, permanently |
| 6d. `timing_late_sample` after its window closes | **0 steps stale**, immediately fresh |

**Non-monotonicity found and documented.** The out-of-order rate of
`bus_reorder` is not monotone in `swap_prob`. An exchange requires a swap
followed by a non-swap, so at `swap_prob = 1` every step swaps, the held frame
is never released, and nothing arrives out of order:

| swap_prob | swaps | out-of-order deliveries |
|---|---|---|
| 0.10 | 4 | 4 |
| 0.25 | 17 | 13 |
| 0.50 | 27 | **14** |
| 0.75 | 46 | 11 |
| 1.00 | 59 | **0** |

**Model correction made during validation.** `bus_delay` was first documented as
draining its backlog after the window. It does not and cannot: a FIFO with
matched producer and consumer rates never drains, so the added latency is
permanent. The docstring, the model and this check now say so, and that
permanence is the real behavioural difference from `timing_late_sample`.

---

## Checks that are not claimed

- No comparison against an external fault-injection tool. `pytorchfi` targets
  PyTorch tensors and PyTorch is not installed here; there is no shared
  interface to compare against, and none is faked.
- No hardware, no real flight software, no real telemetry. The target is a
  textbook double integrator chosen because it has a closed form.
- The severity function is validated for internal consistency (bounds, labels,
  overrides, hand-computed component values) but **not** against any external
  notion of criticality, because none exists for this target.
- Coverage is cell coverage over a declared taxonomy. It says a region of the
  fault space was visited. It says nothing about faults outside the taxonomy.
