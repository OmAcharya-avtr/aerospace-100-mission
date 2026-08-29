# LinkSwitch

**Status:** TESTING · **Class:** medium · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

LinkSwitch simulates a hybrid RF/free-space-optical (FSO) link — an optical
channel with simulated atmospheric-turbulence fading alongside a lower-rate,
more stable RF channel with simulated rain fade — and benchmarks three
policies for deciding, at every time step, which channel to use:

1. **Fixed-threshold baseline** (implemented first): switch to RF whenever
   current irradiance drops below a single threshold.
2. **Hysteresis baseline**: two thresholds to prevent chatter near the
   switching point.
3. **Learned predictive policy**: a scikit-learn `RandomForestClassifier`
   predicting imminent optical outage from recent rolling telemetry
   features, with a confidence output, triggering a *proactive* switch.

**Headline, honest result:** the learned policy does **not** beat the
classical hysteresis baseline. In a mild-turbulence scenario it statistically
ties hysteresis on throughput (95% confidence intervals overlap almost
entirely) while hysteresis still wins on outage time and switch count; in a
moderate-turbulence scenario, hysteresis wins outright on every metric and
the learned policy underperforms even the naive fixed-threshold baseline.
Both numbers are reported as measured — see
[AI model details](#ai-model-details) and `validation/VALIDATION.md`.

**All fading and rain data in this package are SIMULATED. No field-measured
turbulence, scintillometer, rain-gauge, or RF-link telemetry is used
anywhere.**

## Aerospace problem

Hybrid RF/FSO terminals (ground-to-ground, ground-to-air, and
satellite-ground links) carry most of their traffic over a high-rate optical
channel but need a lower-rate RF channel as fallback, because optical
scintillation and cloud/fog attenuation cause deep, fast fades that RF is
largely immune to (while RF itself suffers rain fade). Handover between the
two channels has a real cost — beam re-acquisition, RF re-lock — so
switching too late loses data during outage, and switching too early or too
often wastes capacity and burns switch-cost budget on chatter. LinkSwitch is
a research vehicle for studying that trade-off honestly, including whether a
learned predictive trigger is actually worth its complexity relative to
simple, interpretable classical policies (Kaushal & Kaddoum 2017 survey the
broader hybrid RF/FSO switching literature this package sits alongside).

## Intended users

- Researchers studying hybrid RF/FSO link-switching policies who need a
  reproducible, seeded simulation with an honest classical-vs-learned
  comparison.
- Students and instructors covering lognormal fading, level-crossing theory,
  and Monte Carlo policy evaluation with worked, tested code.
- Engineers prototyping switching-policy trade studies before any
  measured-data validation effort.

Not intended for operational link switching, availability guarantees, or any
flight or mission-critical decision.

## Engineering theory

### Optical channel — lognormal irradiance, simulated

Mean-normalised irradiance `E[I]=1`: `ln I ~ N(mu_z, sigma_z^2)`,
`mu_z = -sigma_z^2/2`, `sigma_z^2 = ln(1 + sigma_I^2)`. Source:
L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press, 2005 (Ch. 8-9). Validity: weak-fluctuation
regime, `sigma_I^2 < ~1` (a `UserWarning` fires above that).

Temporal correlation is added via a discrete AR(1) process on the
standardised log-irradiance, `rho = exp(-1/coherence_steps)` — an
**engineering approximation**, not derived from a measured turbulence
temporal spectrum. Full derivation and units in `src/linkswitch/optical.py`.

A gamma-gamma i.i.d. sampler is also provided (Al-Habash, Andrews &
Phillips, *Optical Engineering* 40(8), 1554, 2001; `sigma_I^2 = 1/a + 1/b +
1/(ab)`), for moderate/strong-turbulence use outside the temporal telemetry
pipeline.

Physical outage threshold from a dB link margin: `tau = 10^(-margin_db/10)`
(definitional).

### RF channel — rain fade, simulated

Two-state Markov rain-occurrence process (stationary probability + mean
event duration, purely synthetic), lognormal rain rate when raining,
ITU-R P.838-form specific attenuation `gamma_R = k * R^alpha` dB/km
(functional form cited; the shipped `(k, alpha)` are **illustrative
defaults, not verified against current ITU-R tables**), a simplified
effective-path-length reduction factor (concept from ITU-R P.618, not its
exact procedure), and a fixed-margin availability rule. Full derivation,
units, and honesty caveats in `src/linkswitch/rf.py`.

### Optimal fixed switching threshold — closed form

The standardised log-irradiance is a stationary AR(1) Gaussian process, so
the per-step level-crossing (switch) probability at any threshold has an
exact closed form via the bivariate normal CDF (cf. S. O. Rice, *Bell Syst.
Tech. J.* 23-24, 1944-45; H. Cramér & M. R. Leadbetter, *Stationary and
Related Stochastic Processes*, Wiley, 1967, Ch. 10). Combined with the
channel rates and switch-downtime cost, this gives a closed-form expected
throughput objective `J(z_th)`, maximised by bounded scalar optimisation.
**Provable result:** in the frictionless (zero switch-cost) limit, the
optimum is exactly the physical outage threshold — verified to
`1.3e-8` in `validation/VALIDATION.md` §V1. Full derivation in
`src/linkswitch/analytic.py`.

## Architecture

```
src/linkswitch/
├── optical.py     lognormal + gamma-gamma irradiance, AR(1) temporal model
├── rf.py          rain Markov chain, ITU-R P.838-form attenuation, availability
├── scenario.py     ScenarioConfig, Telemetry, generate_telemetry (seeded)
├── policies.py    FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy
├── features.py    causal rolling features + imminent-outage labels
├── learn.py       OutagePredictor (RandomForest pipeline), training-set builder
├── simulate.py     switch-cost-aware per-episode scoring, Monte Carlo loop
├── metrics.py      confidence-interval aggregation, paired policy comparison
├── analytic.py     closed-form optimal-threshold model
└── cli.py, __main__.py   python -m linkswitch {threshold,simulate,compare}
```

Dependency direction: `policies` → `features`/`learn`; `analytic` →
`optical`/`rf`; `simulate`/`metrics` → `scenario`/`policies`. No
cross-product imports — this package is fully self-contained (see
`products/P006` LinkBudgetX, `products/P010` BERBench, and `products/P003`
ScintiNet for related prior work in this mission, cited here rather than
imported).

## Installation

Requires Python 3.11+, numpy, scipy, scikit-learn (matplotlib and pandas for
examples/features; pandas is used in `features.py` for rolling statistics).

```bash
cd products/P015
pip install -e .            # or: pip install -e ".[examples]"
```

No installation is strictly necessary — the tests and scripts add `src/` to
`sys.path` themselves.

## Quick start

```python
from linkswitch import ScenarioConfig, generate_telemetry
from linkswitch import FixedThresholdPolicy, HysteresisPolicy
from linkswitch import simulate_policy, compare_policies

cfg = ScenarioConfig()  # mild-turbulence defaults
tel = generate_telemetry(cfg, n_steps=2000, seed=0)

fixed = FixedThresholdPolicy(tau=cfg.optical.tau_phys)
metrics = simulate_policy(tel, fixed.select_channels(tel), cfg)
print(metrics)  # RunMetrics(throughput_mbps=..., outage_fraction=..., switch_count=...)

results = compare_policies(
    cfg,
    {"fixed": lambda: FixedThresholdPolicy(tau=cfg.optical.tau_phys),
     "hysteresis": lambda: HysteresisPolicy(tau_low=cfg.optical.tau_phys * 0.85,
                                            tau_high=cfg.optical.tau_phys * 1.15)},
    n_steps=2000, n_reps=100, seed0=0,
)
for name, agg in results.items():
    t = agg["throughput_mbps"]
    print(f"{name}: {t.mean:.2f} [{t.ci_low:.2f}, {t.ci_high:.2f}] Mb/s")
```

CLI:

```bash
python -m linkswitch threshold                          # analytic optimal threshold
python -m linkswitch simulate --policy hysteresis        # one episode
python -m linkswitch compare --n-reps 100 --horizon 5    # full 3-policy comparison
```

## Configuration

`OpticalParams` (all SI / dimensionless as noted): `sigma_i2` (scintillation
index, default 0.25), `coherence_steps` (AR(1) coherence time in steps,
default 5.0), `margin_db` (link margin, default 6.0), `rate_mbps` (default
1000.0), `fading_model` (`"lognormal"`, the only model wired into telemetry
generation).

`RFParams`: `p_rain` (default 0.04), `mean_event_steps` (default 20.0),
`r_med_mm_hr` / `rate_sigma` (rain-rate-when-raining, defaults 8.0 / 0.7),
`k` / `alpha` (ITU-R P.838 form, illustrative defaults 0.07 / 1.10),
`path_length_km` / `reduction_length_km` (defaults 5.0 / 20.0),
`snr_clear_db` / `snr_min_db` (defaults 25.0 / 6.0), `rate_mbps` (default
150.0).

`SwitchCost`: `downtime_steps` (steps of zero throughput per switch,
default 1).

Two scenarios are used throughout this package's own examples and
validation: the **mild** package-default scenario above, and a **moderate**
scenario (`sigma_i2=0.4, coherence_steps=4.0, margin_db=4.0`) used wherever
switching decisions need to matter more — see `DATASET_CARD.md` for the
measured outage-fraction difference between them.

## Examples

Both scripts run standalone and write PNGs to `screenshots/`.

```bash
python examples/telemetry_and_switching.py   # ~2 s
python examples/policy_comparison.py         # ~4 s
```

- **`screenshots/telemetry_and_switching.png`** — one 400-step simulated
  episode: the irradiance trace with the physical outage threshold marked,
  and a green/red strip per policy showing which channel each one selected,
  with total switch counts.
- **`screenshots/policy_comparison.png`** — bar charts with 95% confidence
  intervals for throughput, outage fraction, and switch count across all
  three policies (moderate scenario, 150 paired Monte Carlo episodes);
  hysteresis wins on throughput in this run.

## Validation

Full evidence, criteria, and raw script outputs: **`validation/VALIDATION.md`**,
with `analytic_threshold_check_output.txt`, `policy_comparison_ci_output.txt`
and `horizon_sensitivity_output.txt` committed alongside their scripts.
Every number below came from running those scripts in this build session.

| ID | Check | Result |
|---|---|---|
| V1a | Closed-form optimal threshold (zero switch cost) vs. grid search vs. `z_phys` | PASS — matches `z_phys` to 1.3e-8 |
| V1b | Independent Monte Carlo argmax cross-check (zero switch cost) | PASS — MC argmax tau exactly matches `tau_phys` |
| V1c | Realistic switch cost: closed-form-predicted direction vs. MC sweep | PASS — MC throughput highest at smallest tau tested, decreasing monotonically toward `tau_phys` |
| V2 | Policy comparison with 95% CIs, seeded Monte Carlo, two scenarios | Learned policy loses to hysteresis in both — reported as measured |
| V3 | Learned-policy sensitivity to prediction horizon (H=1..20) | PASS as a sensitivity measurement — throughput falls monotonically H=1→12, then degenerates at H=20 |

**Overall: the AI/learned policy does not win.** No tolerance was loosened
and no scenario retuned to change this; see `validation/VALIDATION.md` for
full tables and `MODEL_CARD.md` for the failure analysis.

Test suite: **201 passed, 0 failed, 0 skipped**
(`python -m pytest tests/ -q`, ~25 s), including Hypothesis property tests,
hand-calculated known-answer tests, an end-to-end integration test, CLI
subprocess tests, and pinned-seed benchmark/regression tests.

## Benchmark results

Moderate scenario (`sigma_i2=0.4, margin_db=4.0, coherence_steps=4.0`),
200 paired Monte Carlo episodes × 2000 steps (`validation/policy_comparison_ci.py`):

| Policy | Throughput [Mb/s] (95% CI) | Outage fraction (95% CI) | Switches/episode |
|---|---|---|---|
| fixed_threshold | 867.045 [864.702, 869.387] | 0.0891 [0.0877, 0.0905] | 178.16 |
| **hysteresis** | **876.905** [874.568, 879.242] | **0.0770** [0.0757, 0.0782] | **99.34** |
| learned | 856.796 [854.327, 859.265] | 0.0871 [0.0858, 0.0885] | 158.65 |

**Hysteresis wins every metric here, and the learned policy is strictly
worse than the naive fixed-threshold baseline.** This is the reported,
un-retuned outcome. Runtime budget (2 CPU cores): each validation script
under 9 s; full test suite ~25 s; examples ~6 s combined — all well inside
the per-script compute budget.

## AI model details

Full detail in **`MODEL_CARD.md`** and **`DATASET_CARD.md`**.

- **Baselines (implemented first):** `FixedThresholdPolicy`,
  `HysteresisPolicy`, both validated against the closed-form analytic model
  (`analytic.py`) before the learned policy was benchmarked against them.
- **Architecture:** `sklearn.ensemble.RandomForestClassifier`
  (`n_estimators=40, max_depth=4`) in a `StandardScaler` pipeline over 5
  causal rolling log-irradiance features.
- **Dataset:** entirely synthetic, generated on demand
  (`DATASET_CARD.md`) — 10-15 episodes × 500 steps per trained model in
  this build's validation runs.
- **Training:** no held-out hyperparameter search; architecture fixed a
  priori. Evaluated on fresh seeded telemetry disjoint from training seeds.
- **Metrics:** see [Benchmark results](#benchmark-results) and
  `validation/VALIDATION.md` §V2/§V3. **The learned policy does not beat
  hysteresis in either tested scenario.**
- **Uncertainty output:** `LearnedPolicy.outage_confidence(telemetry)` /
  `OutagePredictor.predict_proba` returns the RandomForest's class-1 vote
  fraction — a raw ensemble confidence, **not formally calibrated**.
- **Failure cases:** over-triggering under higher scintillation
  (more switches than even the naive baseline, in the moderate scenario);
  degenerates to an almost-always-RF policy at long prediction horizons
  (H=20, throughput collapses to near the RF-only floor); no extrapolation
  guard; no calibration. Full list in `MODEL_CARD.md`.
- **Reproducibility:** exact commands and seeds in `MODEL_CARD.md`;
  identical `(data, random_state)` gives bit-identical predictions.

**This model is not certified for operational flight use.**

## Hardware requirements

- CPU only. Developed and validated on 2 cores; no GPU, no PyTorch;
  `n_jobs=1` everywhere.
- Peak memory well under 100 MB (small arrays, `n_steps` in the low
  thousands, RandomForest with `max_depth=4`).
- Disk: no committed dataset (all telemetry is generated on demand); PNGs
  in `screenshots/` total ~220 KB.
- Python 3.11 with numpy, scipy, scikit-learn; pandas (rolling features);
  matplotlib for examples only.

## Limitations

1. **The learned policy loses to hysteresis** in every scenario tested —
   see [AI model details](#ai-model-details). Reported plainly, not hidden.
2. **AR(1) temporal fading is an engineering approximation**, not derived
   from a measured or published turbulence temporal spectrum.
3. **Gamma-gamma irradiance is i.i.d.-only**, not wired into the temporal
   telemetry/switching pipeline (no moderate/strong-turbulence time series).
4. **RF attenuation coefficients are illustrative**, not verified against
   current ITU-R P.838 tables; the path reduction factor is a simplified
   concept-only model, not the exact ITU-R P.618-13 procedure.
5. **No spatial or Cn²(h) profile modelling** — single point-to-point
   horizontal link only; no slant-path or satellite-downlink capability.
6. **Confidence output is uncalibrated** (raw RandomForest vote fraction).
7. **Small training sets** (10-15 episodes × 500 steps) in the shipped
   validation runs; no training-set-size convergence study performed.
8. **No experimental validation.** Nothing here has been compared against
   measured link telemetry. Level 2 (Research) evidence is analytic,
   self-consistency, and Monte Carlo cross-check only.
9. **No temporal frozen-flow / wind-speed model** — the AR(1) coherence
   time is a free configuration knob, not derived from wind physics.
10. **Switch-cost model is simplified**: a fixed `downtime_steps` per switch
    regardless of direction; the analytic model further approximates the
    expected cost as `downtime_steps * R_opt` (documented in `analytic.py`).

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

- Wire gamma-gamma (or a von Kármán-spectrum-informed) temporal model into
  telemetry generation for moderate/strong-turbulence scenarios.
- Calibrate the learned policy's confidence output (Platt scaling or
  isotonic regression) and re-run the horizon sensitivity study with a
  calibration-aware trigger.
- A group-aware, larger training campaign to test whether the learned
  policy's loss to hysteresis is a data-scale artifact or a structural
  disadvantage of a per-step-reactive-return design.
- Verified ITU-R P.838 coefficient tables (frequency/polarization-indexed)
  and the exact ITU-R P.618-13 effective-path-length procedure.
- Cn²(h) altitude profiles for slant-path and satellite-downlink scenarios,
  citing this mission's Batch 01/02 turbulence-profile products
  (`products/P019` CnCast, `products/P020` AtmoProfile) as related work
  rather than duplicating them.
- A cost-sensitive or class-balanced training objective aimed specifically
  at closing the over-triggering gap identified in `MODEL_CARD.md`.

## License

Apache-2.0. See `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

Theory references: L. C. Andrews and R. L. Phillips, *Laser Beam Propagation
through Random Media*, 2nd ed., SPIE Press, 2005; M. A. Al-Habash,
L. C. Andrews and R. L. Phillips, *Optical Engineering* 40(8), 1554 (2001);
ITU-R Recommendations P.618, P.837, P.838; S. O. Rice, *Bell Syst. Tech. J.*
23-24 (1944-45); H. Cramér and M. R. Leadbetter, *Stationary and Related
Stochastic Processes*, Wiley, 1967; H. Kaushal and G. Kaddoum, "Optical
Communication in Space: Challenges and Mitigation Techniques," *IEEE
Communications Surveys & Tutorials* 19(1), 57-96, 2017 (hybrid RF/FSO
switching survey, cited for context, not reproduced).

## Citation

```bibtex
@software{linkswitch_2026,
  title   = {LinkSwitch: hybrid RF/FSO link switching policies benchmarked
             on simulated dual-channel telemetry},
  author  = {{OPTIMA Organisation}},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  note    = {Research-grade software; not certified for operational aerospace use}
}
```
