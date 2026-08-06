# BeamTwin — Validation Evidence

Version 0.1.0 · Validation Level 3 (Professional, at v0.1 MVP depth) · Status: TESTING

Every number in this document was produced by running the scripts in this
directory during the build session of **2026-08-06** on the target machine
(Linux 6.18.5 x86_64, Python 3.11.15, numpy 2.4.4, 2 CPU cores). Raw script
output is saved next to each script as a `.txt` file and is the authoritative
record.

| Script | Raw output | Scope | Result |
|---|---|---|---|
| `v1_budget_handcheck.py` | `v1_budget_handcheck.txt` | Deterministic budget arithmetic vs longhand hand-calculation | **PASS** |
| `v2_limit_cases.py` | `v2_limit_cases.txt` | MC vs analytic limits; combined-case monotonicity | **PASS** |
| `v3_performance.py` | `v3_performance.txt` | Throughput and runtime bounds | **PASS** |
| `v4_uncertainty.py` | `v4_uncertainty.txt` | Uncertainty analysis (MC, resolution, sensitivity, surrogate) | **PASS** |
| `../scripts/train_surrogate.py` | `surrogate_benchmark.txt` | Surrogate vs analytic baseline vs MC truth | **PASS** (with caveats, §5) |

Reproduce everything with:

```bash
python validation/v1_budget_handcheck.py
python validation/v2_limit_cases.py
python validation/v3_performance.py
python validation/v4_uncertainty.py
python scripts/generate_dataset.py && python scripts/train_surrogate.py
```

---

## 1. Deterministic link budget — hand-check (V1)

Reference case: 1550 nm, `w0` = 2 cm, 10 km range, 5 cm aperture radius,
20 dBm transmit, 0.8/0.8 optics efficiencies, 2 µrad static bias,
2.5 dB/km attenuation, −30 dBm sensitivity.

Every term was recomputed longhand (arithmetic shown step by step in
`v1_budget_handcheck.txt`) and compared against `compute_budget`:

| Quantity | Hand calculation | Code | Deviation |
|---|---|---|---|
| Rayleigh range `z_R = pi w0^2/lambda` | 810.7336 m | — | — |
| Beam radius `w(10 km)` | 0.247500 m | 0.247500 m | 0.0 |
| Geometric loss `-10log10(1-e^{-2a^2/w^2})` | 11.057829 dB | 11.057829 dB | 0.0 |
| Static pointing loss | 0.056719 dB | 0.056719 dB | 0.0 |
| Atmospheric loss `alpha*L` | 25.000000 dB | 25.000000 dB | 0.0 |
| Optics losses `-10log10(0.8)` | 0.969100 dB ×2 | 0.969100 dB ×2 | 0.0 |
| **Received power** | **−18.052748 dBm** | **−18.052748 dBm** | **0.0** |
| **Margin** | **+11.947252 dB** | **+11.947252 dB** | **0.0** |
| Kim attenuation, V=7 km, 1550 nm | 0.630817 dB/km | 0.630817 dB/km | 0.0 |

**Max deviation across all terms: 0.0** (tolerance 1e-9). The agreement is
exact because both paths evaluate the same closed-form expressions in IEEE
double precision; this check verifies the *implementation*, not the physics
model itself.

---

## 2. Scintillation-only limit vs the analytic lognormal baseline (V2a)

With jitter and bias set to zero the combined model must reduce to the
closed-form lognormal fade probability
`P = Phi((ln 10^{-M/10} + sigma_ln^2/2)/sigma_ln)` (Andrews & Phillips 2005,
Ch. 11). Case: Cn² = 5e-16 m^(−2/3), 10 km, 1550 nm → σ_R² = 0.6782,
σ_ln = 0.7195. 400 000 samples, seed 2024.

| Margin [dB] | P (Monte Carlo) | MC 95 % CI | P (analytic) | Analytic inside CI |
|---|---|---|---|---|
| 2.0 | 3.8899e-01 | [3.8748e-01, 3.9050e-01] | 3.8964e-01 | yes |
| 4.0 | 1.7906e-01 | [1.7788e-01, 1.8025e-01] | 1.7871e-01 | yes |
| 6.0 | 5.9443e-02 | [5.8714e-02, 6.0179e-02] | 5.9345e-02 | yes |
| 8.0 | 1.4185e-02 | [1.3823e-02, 1.4556e-02] | 1.3892e-02 | yes |
| 10.0 | 2.2700e-03 | [2.1272e-03, 2.4223e-03] | 2.2533e-03 | yes |

**5 of 5 cases agree within the Monte Carlo 95 % confidence interval.** This
is the strongest available check on the stochastic core: it confirms the
lognormal sampling, the mean-normalisation (`E[X]=1`), and the fade-counting
logic simultaneously.

## 3. Jitter-only limit vs closed-form mean pointing loss (V2b)

With Cn² = 0 the mean pointing-loss factor must equal
`E[L_p] = 1/(1 + 4 sigma_d^2/w^2)` (Gaussian integral; consistent with Farid
& Hranilovic 2007 in the point-receiver limit). Tolerance is 4× the Monte
Carlo standard error of the mean.

| Jitter [µrad] | σ_d/w | E[L_p] Monte Carlo | E[L_p] closed form | Relative error | Tolerance |
|---|---|---|---|---|---|
| 2.0 | 0.0808 | 0.974463 | 0.974545 | 8.38e-05 | 1.62e-04 |
| 5.0 | 0.2020 | 0.859288 | 0.859661 | 4.34e-04 | 9.00e-04 |
| 10.0 | 0.4040 | 0.604351 | 0.604962 | 1.01e-03 | 2.72e-03 |
| 20.0 | 0.8081 | 0.276612 | 0.276856 | 8.79e-04 | 6.63e-03 |

**4 of 4 cases pass.** Relative errors are 2–5× below tolerance and do not
grow systematically with jitter, indicating no bias in the displacement
sampling.

## 4. Combined-case sanity — monotonicity (V2c)

No closed form exists for combined jitter + scintillation, so the combined
case is checked against qualitative physics. All three monotonicity
properties hold (400 000 samples per point):

- **Fade probability increases with Cn²**: 2.50e-06 → 9.68e-04 → 1.08e-02 → 1.38e-01 → 2.21e-01 for Cn² = 1e-16 … 1e-14. Monotone.
- **Fade probability increases with jitter**: 2.65e-04 → 3.18e-04 → 9.68e-04 → 3.97e-02 → 3.99e-01 for 0 … 20 µrad. Monotone.
- **Fade probability decreases with margin**: 4.77e-01 → 2.48e-01 → 9.72e-02 → 2.81e-02 → 9.68e-04 for 2 … 12 dB. Monotone.

A useful cross-check appears in the 10 km reference case: the combined fade
probability (1.08e-03) exceeds the scintillation-only analytic baseline
(2.67e-04) by a factor of **4.0**. This gap is exactly the modelling territory
the ML surrogate exists to cover (§5).

---

## 5. Surrogate benchmark

Held-out test set of 800 scenarios (20 % of 4000, split seed 123), target
`log10 P_fade` with floor 1e-4. The baseline is the analytic lognormal
(scintillation-only) prediction evaluated on the same held-out rows.

| Subset | n | MAE surrogate | MAE baseline | RMSE surrogate | RMSE baseline |
|---|---|---|---|---|---|
| All test | 800 | **0.270** | 0.404 | **0.377** | 0.843 |
| Low jitter (ratio < 0.05) | 80 | 0.348 | **0.005** | 0.485 | **0.012** |
| High jitter (ratio ≥ 0.05) | 720 | **0.261** | 0.448 | **0.363** | 0.889 |

**Honest reading of this table:**

- Where jitter is negligible, the analytic baseline is *far* better (MAE 0.0049 vs 0.348) — as it must be, since that is the regime where it is exact. **The surrogate does not beat the baseline there and should not be used there.**
- Where jitter is significant — 90 % of the sampled domain and the regime the baseline cannot represent — the surrogate reduces MAE by 42 % (0.261 vs 0.448) and RMSE by 59 %.
- The surrogate's value proposition is therefore **speed in the combined jitter + scintillation regime**, not accuracy in general. Measured cost per fade-probability query: **7.60 µs (surrogate) vs 6004 µs (1e5-sample Monte Carlo) — a 790× speed-up** (`v3_performance.txt`).
- Recommended use: analytic baseline when `jitter_ratio < 0.05`; surrogate for fast sweeps in the combined regime; full Monte Carlo whenever a number is used for a decision.

---

## 6. Uncertainty analysis

### 6.1 Monte Carlo sampling uncertainty (V4/U1)

Fade probability is a binomial proportion; the reported interval is the
Wilson score interval. Verified by running 30 independent seeds and comparing
the empirical scatter against binomial theory `sqrt(p(1-p)/n)`:

| n samples | Mean P | Empirical std | Binomial std | Ratio (emp/theory) | Wilson half-width |
|---|---|---|---|---|---|
| 10 000 | 7.833e-04 | 2.534e-04 | 2.798e-04 | 0.906 | 5.863e-04 |
| 100 000 | 8.680e-04 | 8.977e-05 | 9.313e-05 | 0.964 | 1.858e-04 |
| 400 000 | 8.878e-04 | 4.872e-05 | 4.709e-05 | 1.035 | 9.215e-05 |

Ratios lie within 0.91–1.04 of unity, so **the reported confidence interval is
a faithful description of sampling uncertainty**. Note the mean estimate drifts
from 7.83e-04 to 8.88e-04 as n increases — small-n runs of rare events are
biased low in practice because many runs observe zero fades.

### 6.2 Rare-event resolution (V4/U2)

Resolving a fade probability requires observing fades. Using ≥10 fades for
~32 % relative uncertainty:

| n samples | Smallest resolvable P | Relative std at that P |
|---|---|---|
| 10 000 | 1e-03 | 31.6 % |
| 100 000 | 1e-04 | 31.6 % |
| 1 000 000 | 1e-05 | 31.6 % |
| 10 000 000 | 1e-06 | 31.6 % |

**Consequence:** the default 1e5-sample configuration cannot resolve fade
probabilities below ~1e-4. Availability requirements such as 99.999 %
(P_fade = 1e-5) need ≥1e6 samples. The surrogate inherits this floor —
a prediction at 1e-4 means "below 1e-4", not a calibrated small number.

### 6.3 Input sensitivity (V4/U3)

Elasticities of the fade probability at the 10 km reference case
(`d ln P_fade` per +1 % change in the input, 400 000 samples, seed 7):

| Input | d ln P per +1 % | Interpretation |
|---|---|---|
| Range | **+0.420** | Dominant — 1 % range error moves P_fade by ~42 % |
| Attenuation [dB/km] | **+0.248** | Second — and the least well-known input in practice |
| Transmit power [dBm] | −0.175 | |
| Receive aperture radius | −0.077 | |
| Cn² | +0.042 | |
| Pointing jitter | +0.037 | |

**Practical implication:** fade probability is a tail statistic and is
extremely sensitive to inputs. A 1 % attenuation error (far better than any
real atmospheric forecast) already moves the answer ~25 %. Atmospheric
attenuation uncertainty, not model form, dominates the total uncertainty
budget for operational prediction. These elasticities are local to this
reference case.

### 6.4 Surrogate uncertainty output (V4/U4)

Bootstrap ensemble spread on 800 held-out scenarios:

| Metric | Value | Ideal |
|---|---|---|
| Mean absolute error (log10 P) | 0.2696 | — |
| Mean ensemble std (log10 P) | 0.0906 | ≈ MAE if calibrated |
| Coverage of ±1σ band | 21.1 % | 68.3 % |
| Coverage of ±2σ band | 39.9 % | 95.4 % |
| Spearman corr(σ, abs error) | **+0.388** | > 0 |

**HONEST FINDING — the uncertainty band is under-dispersed.** The ensemble
std is ~3× smaller than the actual error, so ±2σ covers 40 % of cases rather
than 95 %. It captures model and data-sampling variance but not Monte Carlo
label noise or systematic bias.

Bootstrap resampling was adopted precisely because the naive ensemble (all
members trained on identical data) was worse still: coverage 21.6 % at ±2σ.
The improvement to 39.9 % is real but insufficient for calibration.

**The band must therefore be read as a relative confidence ranking — higher
spread means less trustworthy — and not as a probability interval.** The
positive rank correlation (+0.388) with actual error is what makes it useful
for that purpose. Proper calibration (conformal prediction or quantile
regression) is deferred to v0.2; see README Roadmap.

### 6.5 Model-form uncertainty (not quantified)

The uncertainties above are all *statistical*. The following *systematic*
model-form uncertainties are **not** quantified anywhere in this release and
are expected to dominate in real deployments:

| Source | Direction of error | Notes |
|---|---|---|
| Lognormal weak-fluctuation assumption | Underestimates deep fades when σ_R² ≳ 1 | Flagged at runtime via `weak_regime_valid` |
| Point-receiver pointing model (a ≪ w) | Overestimates fade depth when the aperture is comparable to the beam | At the reference case a/w = 0.20; error grows beyond a/w ≈ 0.3 |
| No aperture averaging of scintillation | **Overestimates** scintillation for large apertures | Real receivers average speckle, reducing σ_I² |
| No turbulence-induced beam spreading/wander | Underestimates loss | Only diffractive spreading is modelled |
| Independence of scintillation and jitter | Unknown sign | Assumed, not verified |
| Homogeneous Cn² and attenuation along the path | Unknown sign | Real paths are stratified |

---

## 7. What was NOT validated

Stated plainly, because this is an MVP:

1. **No comparison against measured FSO link data.** Every check here is internal consistency or agreement with closed-form theory. No experimental or field dataset was used. This is the single largest gap and the reason the product is not fit for operational prediction.
2. **No validation of the Kim model itself** — only that it is implemented as published. The underlying empirical fit was not re-derived or checked against visibility/attenuation measurements.
3. **No strong-turbulence validation.** The saturation regime (σ_R² ≳ 1) is detected and flagged but not modelled; no gamma-gamma or other strong-fluctuation distribution is implemented.
4. **No temporal correlation.** All samples are independent draws. Fade *duration*, fade rate, and burst-error statistics — which drive coding and interleaver design — cannot be obtained from this model.
5. **No hardware effects.** Detector noise, thermal drift, tracking-loop dynamics, and modulation/coding are out of scope; "sensitivity" is a single scalar threshold.
6. **Surrogate uncertainty is not calibrated** (§6.4).

---

## 8. Test suite summary

`python -m pytest tests/ -q` from the product directory:

```
251 passed in 11.75s
```

| File | Focus | Count |
|---|---|---|
| `test_budget.py` | Unit, known-answer, input validation for the deterministic budget | 55 |
| `test_channel.py` | Scintillation, jitter, Monte Carlo behaviour | 40 |
| `test_scenario.py` | Scenario loading, configuration variants, failure modes | 43 |
| `test_surrogate.py` | Features, dataset generation, fitting, persistence, committed model | 35 |
| `test_stats.py` | Fade probability, analytic baseline, margin statistics | 24 |
| `test_properties.py` | Hypothesis property-based algebraic identities | 19 |
| `test_cli.py` | End-to-end CLI integration (run, sweep, module invocation) | 17 |
| `test_regression.py` | Pinned seeded outputs, reproducibility | 13 |
| `test_performance.py` | Runtime and throughput bounds | 5 |
| **Total** | | **251** |

0 failed, 0 skipped, 0 xfail. No tolerance was widened and no test was
disabled to obtain this result. The two failures encountered during
development were both defective *tests* (a float-equality assertion on a
value analytically zero but 2.2e-19 in floating point, and a seed-comparison
test that used a 31 dB-margin link where both seeds correctly gave zero
fades); both were corrected to test the intended property.
