# Validation evidence — photoncount 0.1.0

**Validation level 3, hardware-pending.** Every number below was produced by a
script in this directory, run in this build container on 2026-10-06, with its
raw stdout committed beside it. Nothing here is a measurement of a
photon-counting detector, and nothing here is a measurement on a
flight-representative board. What that costs is set out in
[What is missing](#what-is-missing-for-level-4).

Reproduce everything:

```bash
cd validation
python3 validate_webb_limits.py
python3 validate_deadtime.py
python3 validate_afterpulse.py
python3 validate_ppm.py
python3 validate_capacity.py
python3 validate_simulator.py
python3 validate_hal_contract.py
python3 validate_correction.py
python3 worked_example.py
```

Total runtime about 2 minutes on two shared cores. Individual timings vary with
container load; each script prints its own.

---

## 1. Summary table

| Check | Reference | Result | Tolerance | Source |
|---|---|---|---|---|
| Webb closed-form moments vs quadrature of the density | derived here from Webb, McIntyre & Conradi (1974) form | mean residual ≤ 2.8e-16, variance ≤ 1.9e-16, third central ≤ 1.1e-13 relative, over four (m, F) pairs | 1e-5 / 1e-4 / 5e-3 | `validate_webb_limits_output.txt` §1 |
| Webb at F = 1 vs `N(m, m)` | exact Gaussian limit | sup-norm 1.04e-17 | 1e-15 | `validate_webb_limits_output.txt` §2 |
| Webb → Gaussian as F → 1 | monotone convergence | sup-norm 7.15e-4 at F = 2 falling to 1.38e-6 at F = 1.001, monotone | monotone, < 1e-5 at F = 1.001 | `validate_webb_limits_output.txt` §2 |
| Webb → Poisson, integer-binned | the Gaussian-to-Poisson rate `m^-1/2` | TV × √m = 0.125929, 0.125841, 0.125835 at m = 100, 1000, 10000; successive ratios 3.1645 and 3.1624 vs √10 = 3.1623 | each ratio within 8 % of √10 | `validate_webb_limits_output.txt` §3 |
| Webb inverse-CDF sampling vs closed-form moments | n = 400000 | mean z = +1.77, variance within 0.52 %, skewness 0.21562 vs 0.21213 | \|z\| < 4, variance 2 % | `validate_webb_limits_output.txt` §4 |
| Non-paralyzable dead time at τ = 1 µs, n = 1e5/s | Knoll ch. 4 | m = 90909.0909090909 vs hand 90909.0909090909, rel 0.00e+00 | 1e-14 | `validate_deadtime_output.txt` §1 |
| Paralyzable dead time at the same point | Knoll ch. 4 | m = 90483.7418035960 vs hand 90483.7418035960, rel 0.00e+00 | 1e-14 | `validate_deadtime_output.txt` §1 |
| Paralyzable maximum | `n = 1/τ`, `m = 1/(eτ)` | closed form 1.000000e+06 / 367879.4411714424; dense sweep 1.000001e+06 / 367879.4411712472; central-difference dm/dn = 0.000e+00 | sweep within 0.2 %, derivative < 1e-6 | `validate_deadtime_output.txt` §2 |
| Both paralyzable inverse branches | Lambert W0 and W−1 | round-trip through the forward map ≤ 1.98e-16 relative at five fractions of the maximum | 1e-9 | `validate_deadtime_output.txt` §3 |
| Conditioning of the paralyzable inverse | d ln n / d ln m | 1.04 at m = 0.1 m_max rising to 22.70 at 0.999 m_max, monotone | monotone increasing | `validate_deadtime_output.txt` §4 |
| Non-paralyzable round trip over five decades | exact inverse | worst relative error 2.166e-15 | 1e-12 | `validate_deadtime_output.txt` §7 |
| Afterpulse cluster moments by direct sampling, n = 2e6 | compound-Poisson identity | cascading mean 2.498112 vs 2.500000 at p = 0.6; first-order variance 0.240010 vs 0.240000 | 5 s.e. on the mean, 2 % on the variance | `validate_afterpulse_output.txt` §1–2 |
| Afterpulse Fano factor vs `(1+p)/(1-p)` and `(1+3p)/(1+p)` | derived here | worst relative deviation 0.00979 over six cases | 5 % | `validate_afterpulse_output.txt` §3 |
| Effective afterpulse probability `p e^{-τ/t_ap}` | exponential survival | \|z\| ≤ 1.17 against the binomial s.e. over five τ/t_ap | \|z\| < 5 | `validate_afterpulse_output.txt` §5 |
| PPM exact symbol error vs `exp(-n_s)(M-1)/M` | background-free closed form | worst absolute deviation 6.13e-15 over six (M, n_s) | 1e-12 | `validate_ppm_output.txt` §1 |
| PPM exact symbol error vs Monte Carlo with background | 400000 symbols per point | \|z\| ≤ 2.79 over six configurations | \|z\| < 4 | `validate_ppm_output.txt` §2 |
| Poisson PPM soft metric (P3) vs the full product likelihood | derived in the module docstring | worst deviation in the log posterior 1.776e-14 nats over 2000 × 16 | 1e-10 nats | `validate_ppm_output.txt` §3 |
| Bit LLRs vs symbol-posterior marginals | derived here | worst deviation 2.220e-16; all-zero symbol gives exactly 0 | 1e-10 | `validate_ppm_output.txt` §5 |
| Erasure capacity `(1 - e^{-n_s}) log2 M` vs independent Monte Carlo | derived here | \|z\| ≤ 1.77 over four (M, n_s) at 400000 symbols | \|z\| < 4 | `validate_capacity_output.txt` §1 |
| Photons per bit → `1/log2 M` | derived here | excess over the limit 5.000e-06 at n_s = 1e-5, strictly above and decreasing for every M tested | converged to 1e-4, strictly above | `validate_capacity_output.txt` §2 |
| Soft-decision rate ≥ hard-decision capacity | data-processing inequality | gap +0.005703 to +0.522786 bits/symbol over six configurations, never negative | ≥ −4 s.e. | `validate_capacity_output.txt` §4 |
| Simulator vs the non-paralyzable rate law | (D1) | \|z\| ≤ 1.14 over n τ ∈ {0.01 … 3} at ≈400000 counts per point | \|z\| < 5 | `validate_simulator_output.txt` §2 |
| Simulator vs the paralyzable rate law | (D3) | \|z\| ≤ 1.99 over the same sweep; observed rate rises to n τ = 1 and falls beyond | \|z\| < 5 plus the turnover | `validate_simulator_output.txt` §3 |
| Simulator vs the cascading afterpulse rate law | (A2) | \|z\| ≤ 1.58 over p ∈ {0.02 … 0.4} | \|z\| < 5 | `validate_simulator_output.txt` §4 |
| Fano factor sign | dead time below 1, afterpulsing above 1 | 0.11109 with dead time only at n τ = 2; 1.18 to 1.91 with afterpulsing, within 5.7 % of the closed form | < 1, and within 15 % | `validate_simulator_output.txt` §5 |
| Dead time and afterpulsing do not compose as a product | this repository's claim | the naive product deviates from the simulation by up to 25.98 % | deviation must be non-zero | `validate_simulator_output.txt` §7 |
| HAL contract, both backends | one shared suite | 4 of 4 checks pass; the device backend refuses every hardware operation by name | all | `validate_hal_contract_output.txt` |
| Preflight blocks a short window and a saturated rate | executable procedure | `window_exceeds_dead_time` fails at 10 dead times; `rate_below_saturation` fails at 2e7 counts/s against 1/τ = 1e7 | both must fail | `validate_hal_contract_output.txt` §4 |
| Backout of a half-finished run | executable procedure | one in-progress journal converted to an aborted record, second call returns `[]`, completed record untouched | idempotent, nothing deleted | `validate_hal_contract_output.txt` §5 |
| No absolute path in any filename or environment string | this repository's own rule | `[]` | empty | `validate_hal_contract_output.txt` §6 |
| Learned correction interval coverage | nominal 0.90 | **0.8995** over 2000 held-out rows, z = −0.07, median relative width 0.2217 | measured, not assumed | `validate_correction_output.txt` §4 |

---

## 2. The learned correction against the closed forms

Held-out set: 2000 simulated acquisitions, seed 99001, independent of the 6000
training rows (seed 20261006). Median total counts per row 4812, so the
**counting-noise floor** on the observed rate is 1/√4812 = 0.01442 relative, and
the smallest achievable median absolute relative error is about **0.00972**. Every
number below should be read against that floor.

Overall (`validate_correction_output.txt` §3):

| Estimator | Defined | Median abs rel error | p90 | Median signed error |
|---|---:|---:|---:|---:|
| non-paralyzable closed form (D2) | 1.000 | 0.0418 | 0.1874 | +0.0237 |
| paralyzable lower-branch closed form (D5) | 0.863 | 0.0469 | 0.1692 | +0.0361 |
| model-matched closed form | 0.989 | 0.0371 | 0.1307 | +0.0288 |
| composed closed form (A2 then D2/D5) | 0.998 | 0.0454 | 0.2450 | −0.0454 |
| **learned, quantile GBM** | 1.000 | **0.0234** | **0.0901** | −0.0002 |

**The overall table is not the result.** The regimes are
(`validate_correction_output.txt` §5):

| Regime | n | matched closed form | composed closed form | learned | Who wins |
|---|---:|---:|---:|---:|---|
| `p <= 0.02` (afterpulsing effectively absent) | 268 | **0.0143** | **0.0136** | 0.0243 | **the closed forms**, at the noise floor |
| `p > 0.02` | 1732 | 0.0434 | 0.0530 | **0.0233** | the learned model |
| `n τ < 0.1` | 1094 | 0.0391 | 0.0279 | **0.0170** | the learned model |
| `0.1 <= n τ < 0.7` | 510 | **0.0278** | 0.0582 | 0.0289 | a tie between matched and learned |
| `0.7 <= n τ <= 1.5` | 208 | 0.0343 (defined on 88.9 %) | 0.1680 | 0.0411 (defined on 100 %) | matched on error, learned on coverage |
| `n τ > 1.5`, non-paralyzable | 97 | 0.0275 | 0.2168 | **0.0191** | the learned model |
| `n τ > 1.5`, **paralyzable** | 91 | 0.8336 | 0.8552 | 0.8399 | **nobody** |

Read plainly:

- **Where the textbook assumptions hold, the textbook wins.** With `p <= 0.02`
  the matched and composed closed forms reach 0.0143 and 0.0136 median relative
  error, within 50 % of the counting-noise floor, and the learned model is worse
  at 0.0243. The closed forms are also a line of algebra with no training set, no
  seed and no 196 KiB of trees. Where afterpulsing is negligible, use them.
- **Where afterpulsing is present, the learned model wins, and the reason is
  structural.** Neither (D2) nor (D5) can accept an afterpulse probability at
  all; the composed form can, but it inverts the two effects in sequence when
  they do not commute — `validate_simulator_output.txt` §7 measures the
  non-commutation at up to 25.98 %. At `p > 0.02` the learned model halves the
  median error, 0.0233 against 0.0434.
- **Past the paralyzable maximum nothing works.** At `n τ > 1.5` on a paralyzable
  detector every estimator, learned and closed-form, has a median relative error
  between 0.83 and 0.86. The forward map is two-valued there and the lower branch
  is the wrong root; the learned model does not recover it. This is a limit of the
  measurement, not of the method, and it is the reason `photoncount.ops`
  preflight **fails** a run planned at or above `1/τ`.
- **The interval is honest.** Nominal 0.90, measured 0.8995 over 2000 rows
  (z = −0.07), median relative width 0.2217.

### Why there is no label leak

An earlier version of the generator exposed the mean count per window as a sixth
feature. Because the window length is `events_per_window / n`, that feature
equals `events_per_window × m/n` and therefore contains the answer. With it
present the learned model reached the counting-noise floor in **every** regime,
including the regimes where the closed forms are exactly right — the signature of
a leak, not of a good model. The feature was removed and
`tests/test_correction.py::test_dataset_has_no_mean_count_feature` keeps it out.

A weaker residual coupling remains: the window length still depends on the true
rate, so the Fano-factor estimate is weakly rate-dependent. The ablation in
`validate_correction_output.txt` §7 retrains with the Fano column flattened to a
constant. The result is **0.0208** median error against **0.0234** with the
feature — flattening it makes the model slightly *better*. None of the learned
margin comes from the Fano feature, so none of it comes from the residual
coupling, and the Fano factor carries no usable information at 5000 counts per
row. The dead-time/afterpulsing signature it carries in principle is below the
counting noise of these acquisitions.

Feature importances (median gain across the three quantile models) say the same
thing: `log10_observed_x` 0.857, `afterpulse_probability` 0.063,
`log10_delay_ratio` 0.045, `is_paralyzable` 0.009, `fano_factor` 0.006.

---

## 3. What was checked and found wanting

| Finding | Where |
|---|---|
| The learned model is **worse** than both closed forms where afterpulsing is negligible: 0.0243 against 0.0143 and 0.0136. | `validate_correction_output.txt` §5, regime `p<=0.02` |
| Past the paralyzable maximum every estimator fails, median relative error 0.83 to 0.86. The learned model adds nothing there. | `validate_correction_output.txt` §5, regime `paralyzable & x>1.5` |
| The composed closed form is **worse** than the matched one overall (0.0454 against 0.0371) and much worse at `0.1 <= n τ < 0.7` (0.0582 against 0.0278), because removing the afterpulse inflation first over-corrects once dead time is also removing counts. Removing a known effect does not always help. | `validate_correction_output.txt` §3, §5 |
| The paralyzable lower-branch inversion is undefined on 13.7 % of the held-out rows, and on 66.3 % of rows in `0.7 <= n τ <= 1.5`. | `validate_correction_output.txt` §3, §5 |
| The Fano factor, which in principle separates dead time from afterpulsing, contributes nothing at these counting statistics. | `validate_correction_output.txt` §7 |
| The Webb distribution cannot reach the Poisson pmf faster than `m^-1/2`, and its third central moment at F = 1 is 0 where the Poisson's is m. | `validate_webb_limits_output.txt` §3 |
| The simulated backend's observed rate can be **above** the incident rate when afterpulsing outweighs dead time, which breaks the common assumption that dead time only removes counts. | `tests/test_hal_contract.py::test_simulated_backend_records_the_true_rate` |

---

## 4. What is missing for Level 4

Level 4 requires measured timing and resource use from a Jetson Orin Nano.
Nothing in this repository is that measurement. Specifically:

1. **`benchmark/run_benchmark.py` run on the board**, with its raw output file
   kept. The numbers in `benchmark/benchmark_results.md` are from a shared
   two-core cloud container running four other build jobs, and that file says so
   in its first line.
2. **A measured dead time, dead-time model, afterpulse probability and delay
   distribution, dark-count rate, timestamp resolution and jitter** from an
   actual detector, replacing the *declared* values that
   `photoncount.hal.DeviceBackend` carries.
3. **A `DeviceBackend` implementation** of `open`, `self_test` and `acquire`
   against that hardware. The stub raises `NotImplementedError` naming what is
   missing, and `validate_hal_contract.py` §1 shows it doing so.

No simulated backend, no extrapolation from this container and no vendor
datasheet substitutes for any of the three. This product is **Level 3,
hardware-pending**, and must not be labelled Level 4 until all three exist.

---

## 5. Files

| Script | Raw output | What it establishes |
|---|---|---|
| `validate_webb_limits.py` | `validate_webb_limits_output.txt` | Webb moments, Gaussian limit, Poisson limit rate, sampling, excess-noise relation |
| `validate_deadtime.py` | `validate_deadtime_output.txt` | Both dead-time laws, the maximum, both inverse branches, conditioning, model ordering |
| `validate_afterpulse.py` | `validate_afterpulse_output.txt` | Cluster moments, Fano factors, exact inverses, the effective-probability relation |
| `validate_ppm.py` | `validate_ppm_output.txt` | Exact symbol error probability, the soft metric derivation, bit LLRs, truncation |
| `validate_capacity.py` | `validate_capacity_output.txt` | Erasure capacity, photons-per-bit limit, data-processing inequality, monotonicity |
| `validate_simulator.py` | `validate_simulator_output.txt` | The simulator against every rate law it generalises, and the non-commutation |
| `validate_hal_contract.py` | `validate_hal_contract_output.txt` | The HAL contract on both backends, preflight, capture, backout, no absolute paths |
| `validate_correction.py` | `validate_correction_output.txt` | Baselines first, learned model second, coverage, regimes, ablation |
| `worked_example.py` | `worked_example_output.txt` | The README's worked example end to end |
| — | `example_*_output.txt` | Printed output of the four scripts in `examples/`, which also produce the screenshots |
| — | `rate_corrector.joblib` | The fitted corrector, 196 KiB, regenerated deterministically by `validate_correction.py` |
