# BeamTwin

**Status:** TESTING · **Class:** flagship · **Validation level:** 3 · **AI:** yes

A digital twin for free-space optical (FSO) communication links: a
deterministic link budget, a stochastic atmospheric channel, Monte Carlo fade
statistics, and a machine-learning fade-probability surrogate — with every
equation cited and every validation number measured.

---

## Executive overview

An FSO link's viability is not decided by its average received power but by
how often that power dips below the receiver's sensitivity. Two atmospheric
effects drive those dips: **scintillation** (turbulence-induced intensity
fluctuation) and **pointing jitter** (residual tracking error). Each has a
textbook treatment in isolation. Together they do not.

BeamTwin does three things:

1. **Computes the deterministic budget** — every loss term from transmitter to detector, hand-verified to machine precision.
2. **Simulates the stochastic channel** — vectorised Monte Carlo combining lognormal scintillation and Gaussian jitter, at 1.3e7 samples/s, giving fade probability with a proper confidence interval.
3. **Learns a fast surrogate** — a gradient-boosting ensemble trained on the twin's own Monte Carlo output, answering fade-probability queries **790× faster** than simulation in exactly the regime where the closed-form solution does not exist.

The honest summary: the closed-form lognormal baseline is *exact* when jitter
is negligible, and BeamTwin says so and defers to it there. The surrogate
earns its place only in the combined jitter + scintillation regime — where it
cuts error 42 % versus that baseline — and only when speed matters more than
a factor-of-two in probability.

This is a v0.1 MVP. It has never been compared against a measurement from a
real optical link.

## Aerospace problem

Free-space optical links carry high data rates without spectrum licensing —
inter-satellite crosslinks, satellite-to-ground downlinks, HAPS backhaul,
terrestrial last-mile. Their weakness is the atmosphere. A 1550 nm link that
closes with 12 dB of margin on paper can drop below threshold for
milliseconds at a time when turbulence and tracking error conspire, and it is
the *statistics* of those drops — not the mean — that determine whether the
link meets an availability requirement.

Link designers need to answer: given range, turbulence strength, visibility,
transmit power, aperture, and tracking accuracy — **what fraction of the time
is this link down?** Answering it well requires Monte Carlo. Answering it
thousands of times, inside a design sweep or an optimiser, requires something
faster. BeamTwin provides both, and is explicit about which to trust when.

## Intended users

- **FSO link engineers** sizing margin, aperture, or transmit power for a terrestrial or space link.
- **Systems engineers** trading availability against SWaP during early design.
- **Researchers and students** in optical communications and atmospheric propagation who need a transparent, citable implementation.
- **ML practitioners** studying physics-surrogate modelling, uncertainty quantification, and the honest baseline-versus-model comparison.

Not intended for: operational link management, availability certification, or
any decision affecting real hardware or flight safety.

## Engineering theory

Losses are stored as non-negative dB and subtracted from transmit power.

### Gaussian beam propagation

| Quantity | Equation | Units | Source |
|---|---|---|---|
| Rayleigh range | `z_R = pi w0^2 / lambda` | m | Saleh & Teich 2007, Eq. (3.1-11) |
| Beam radius | `w(z) = w0 sqrt(1 + (z/z_R)^2)` | m | Saleh & Teich 2007, Eq. (3.1-8) |
| Divergence half-angle | `theta = lambda / (pi w0)` | rad | Saleh & Teich 2007, Eq. (3.1-21) |

`w0` is the 1/e² intensity radius at the waist.
**Assumptions:** diffraction-limited TEM00 beam, paraxial propagation.
**Validity:** `w0 >> lambda`, `theta << 1 rad`. Turbulent beam spreading and
wander are *not* included.

### Geometric capture

```
eta_geo = 1 - exp(-2 a^2 / w^2)
```

Integration of `I(r) = I0 exp(-2r^2/w^2)` over a disc of radius `a`
(Saleh & Teich 2007, Sec. 3.1). Dimensionless, in (0, 1).
**Assumptions:** beam centred on aperture; far field.
**Limits:** `-> 2a^2/w^2` for `a << w`; `-> 1` for `a >> w`.

### Pointing loss

```
L_p = exp(-2 d^2 / w^2)
```

On-axis intensity ratio for transverse displacement `d` — the **point-receiver
approximation** (`a << w`). See Farid & Hranilovic, *J. Lightwave Technol.*
25(7), 2007, Eq. (8) with unit aperture-averaging factor; Majumdar & Ricklin
2008, Ch. 3.
**Validity:** accurate to `a/w ≈ 0.3`; at the 10 km reference case `a/w = 0.20`.

Under zero-bias Gaussian jitter with per-axis RMS displacement `sigma_d`:

```
E[L_p] = 1 / (1 + 4 sigma_d^2 / w^2)
```

(Gaussian integral `E[exp(-c x^2)] = (1 + 2 c sigma^2)^(-1/2)` applied per
axis.) Verified against Monte Carlo to <0.11 % relative error — see
Validation.

### Atmospheric attenuation

Beer-Lambert along the path: `L_atm[dB] = alpha[dB/km] * L[km]`.

Optionally derived from visibility with the **Kim model** (Kim et al. 2001,
Proc. SPIE 4214):

```
beta(lambda) = (3.91 / V) * (lambda / 550 nm)^(-q)     [1/km]
alpha[dB/km] = 4.343 * beta
```

with `q = 1.6` (V > 50 km), `1.3` (6 < V ≤ 50), `0.16V + 0.34` (1 < V ≤ 6),
`V - 0.5` (0.5 < V ≤ 1), `0` (V ≤ 0.5).
**Assumptions:** homogeneous haze/fog; visibility defined at 550 nm, 2 %
contrast threshold. **Validity:** empirical fit; `q = 0` below 0.5 km makes
dense fog wavelength-independent — a real and important physical result.

### Scintillation

Plane-wave Rytov variance (Andrews & Phillips 2005, Ch. 8):

```
sigma_R^2 = 1.23 Cn2 k^(7/6) L^(11/6),    k = 2 pi / lambda
```

In the weak regime `sigma_I^2 ≈ sigma_R^2`, and irradiance is lognormal with
`sigma_ln^2 = ln(1 + sigma_I^2)`. Samples use
`Z ~ N(-sigma_ln^2/2, sigma_ln^2)` so that `E[exp(Z)] = 1` — mean irradiance
is preserved (verified to 1 % at 4e5 samples).
**Units:** `Cn2` in m^(−2/3); typical 1e-16 (calm night) to 1e-13 (strong daytime).
**Validity: `sigma_R^2 < 1`.** Beyond that the model underestimates deep
fades; BeamTwin sets `weak_regime_valid = False` and prints a warning rather
than silently extrapolating.

### Fade probability

Monte Carlo: `P_fade = #(P_rx < sensitivity) / N`, reported with a 95 %
**Wilson score interval** (Agresti & Coull 1998), which stays well-behaved at
zero observed fades.

Closed-form baseline for the scintillation-only case (Andrews & Phillips
2005, Ch. 11) — a fade occurs when `X < 10^(-M/10)`, so:

```
P_fade = Phi( (ln 10^(-M/10) + sigma_ln^2 / 2) / sigma_ln )
```

**Validity:** no jitter, weak fluctuation. This is the baseline the ML model
must beat or match.

## Architecture

```
                          scenario.yaml
                               │
                    ┌──────────▼───────────┐
                    │  beamtwin.scenario   │  strict YAML validation
                    │  Scenario + report   │  ScenarioError names bad key
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼─────────┐ ┌─────────▼──────────┐ ┌────────▼─────────┐
│ beamtwin.budget  │ │ beamtwin.channel   │ │beamtwin.surrogate│
│ ──────────────── │ │ ────────────────── │ │ ──────────────── │
│ Gaussian beam    │ │ Rytov variance     │ │ 5x GradientBoost │
│ geometric capture│─▶│ lognormal scint.  │ │ bootstrap ensembl│
│ pointing (bias)  │ │ Gaussian jitter    │ │ log10 P_fade     │
│ Kim attenuation  │ │ vectorised MC      │ │ + spread + extrap│
│ → margin_db      │ │ → samples [dBm]    │ │   flag           │
└────────┬─────────┘ └─────────┬──────────┘ └────────▲─────────┘
         │                     │                     │
         │                     ▼                     │ trained on
         │           ┌────────────────────┐          │ MC output
         │           │  beamtwin.stats    │          │ (seeded)
         │           │ ────────────────── │          │
         └──────────▶│ P_fade + Wilson CI │──────────┘
                     │ margin percentiles │
                     │ ANALYTIC BASELINE  │◀── benchmark reference
                     │ (lognormal, exact  │    for the surrogate
                     │  when jitter = 0)  │
                     └─────────┬──────────┘
                               │
                    ┌──────────▼───────────┐
                    │  report: text + JSON │
                    │  python -m beamtwin  │
                    │     run  |  sweep    │
                    └──────────────────────┘

Data flow for the AI component:
  budget+channel ──MC (4000 scenarios x 5e4 samples, seed 42)──▶ dataset
  dataset ──train (split seed 123, model seed 7)──▶ models/surrogate.joblib
  surrogate ──benchmarked against──▶ analytic baseline AND held-out MC truth
```

**Design rule:** the analytic baseline is computed on every run alongside the
Monte Carlo and the surrogate, so a user always sees all three numbers and can
judge for themselves. The twin never hides its ML prediction behind a single
authoritative-looking figure.

## Installation

Python 3.11+. Dependencies: numpy, scipy, matplotlib, scikit-learn, pyyaml, joblib.

```bash
cd products/P001
pip install -e .
```

Development extras (pytest, hypothesis):

```bash
pip install -e ".[test]"
```

## Quick start

```bash
# Full twin report for the shipped 10 km terrestrial scenario
python -m beamtwin run examples/link_10km.yaml

# Same, plus a machine-readable report
python -m beamtwin run examples/link_10km.yaml --json report.json

# Fade probability vs range, with plot
python -m beamtwin sweep examples/link_10km.yaml --param range_km \
    --start 1 --stop 15 --steps 15 --output screenshots/sweep.png
```

Library use:

```python
from beamtwin import LinkParams, ChannelParams, compute_budget, sample_received_power_dbm
from beamtwin.stats import fade_probability

link = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0)
channel = ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6)

print(compute_budget(link).margin_db)                    # +11.947 dB
mc = sample_received_power_dbm(link, channel, n_samples=200_000, seed=1)
print(fade_probability(mc.samples_dbm, link.rx_sensitivity_dbm))
# FadeEstimate(probability=0.001075, ci_low=0.000941, ci_high=0.001229, ...)
```

## Configuration

Scenarios are YAML. Unknown keys are **rejected**, so a typo can never
silently select a default.

```yaml
name: terrestrial-10km
link:
  wavelength_nm: 1550
  tx_power_dbm: 20.0
  tx_efficiency: 0.8
  rx_efficiency: 0.8
  beam_waist_radius_m: 0.02
  rx_aperture_radius_m: 0.05
  range_km: 10.0
  pointing_bias_urad: 2.0
  attenuation_db_per_km: 2.5   # OR visibility_km: 7.0 (Kim model) — not both
  rx_sensitivity_dbm: -30.0
channel:
  cn2: 5.0e-16                 # m^-2/3
  pointing_jitter_urad: 5.0    # per-axis RMS
monte_carlo:
  n_samples: 200000
  seed: 1
```

Units are in the key names (`_km`, `_urad`, `_nm`, `_dbm`) to make unit errors
hard. Sweepable CLI parameters: `range_km`, `tx_power_dbm`,
`rx_sensitivity_dbm`, `attenuation_db_per_km`, `cn2`, `pointing_jitter_urad`.

## Examples

All three are runnable and produced the committed PNGs in `screenshots/`.

| Script | Output | What it shows |
|---|---|---|
| `examples/run_full_report.py` | `screenshots/budget_waterfall_10km.png` | Full twin report for a 10 km terrestrial link + loss waterfall |
| `examples/fade_vs_range.py` | `screenshots/fade_vs_range.png` | Fade probability vs range: Monte Carlo (with CI band), analytic baseline, and surrogate on one axis |
| `examples/margin_histogram.py` | `screenshots/margin_histogram.png` | Margin distribution decomposed into scintillation-only, jitter-only, and combined |

The 10 km reference report:

```
  Tx power                 +20.00 dBm
  Tx optics loss        -    0.97 dB
  Geometric loss        -   11.06 dB
  Pointing loss (bias)  -    0.06 dB
  Atmospheric loss      -   25.00 dB
  Rx optics loss        -    0.97 dB
  Received power           -18.05 dBm
  Margin                   +11.95 dB
  Fade probability      1.0750e-03   [95% CI 9.406e-04, 1.229e-03]
  Analytic baseline     2.6661e-04   (lognormal, scintillation-only)
  Surrogate             1.1908e-03   [spread 8.639e-04, 1.641e-03]
```

Note the 4× gap between the combined Monte Carlo result (1.08e-3) and the
scintillation-only baseline (2.67e-4). That gap is the whole reason the
surrogate exists.

## Validation

Full evidence in [`validation/VALIDATION.md`](validation/VALIDATION.md); raw
script output in `validation/*.txt`. All numbers measured on 2026-08-06,
Linux x86_64, Python 3.11.15, numpy 2.4.4, 2 cores.

**1. Deterministic budget hand-check (V1)** — every term recomputed longhand
with arithmetic shown. Beam radius 0.247500 m, geometric loss 11.057829 dB,
received power −18.052748 dBm, margin +11.947252 dB, Kim attenuation
0.630817 dB/km at V = 7 km. **Max deviation from code: 0.0** (tolerance 1e-9).

**2. Scintillation-only limit vs closed form (V2a)** — the strongest check on
the stochastic core. 400 000 samples, seed 2024, σ_ln = 0.7195:

| Margin | P (Monte Carlo) | MC 95 % CI | P (analytic) | Inside CI |
|---|---|---|---|---|
| 2 dB | 3.8899e-01 | [3.8748e-01, 3.9050e-01] | 3.8964e-01 | yes |
| 6 dB | 5.9443e-02 | [5.8714e-02, 6.0179e-02] | 5.9345e-02 | yes |
| 10 dB | 2.2700e-03 | [2.1272e-03, 2.4223e-03] | 2.2533e-03 | yes |

**5 of 5 margins agree within the MC confidence interval.**

**3. Jitter-only limit vs closed form (V2b)** — `E[L_p]` versus
`1/(1 + 4σ_d²/w²)` at four jitter levels: relative errors **8.4e-05 to
1.0e-03**, every case inside its 4σ Monte Carlo tolerance, with no systematic
growth in jitter.

**4. Combined-case monotonicity (V2c)** — fade probability is monotone
increasing in Cn² (2.5e-06 → 2.21e-01), monotone increasing in jitter
(2.65e-04 → 3.99e-01), and monotone decreasing in margin (4.77e-01 →
9.68e-04). All three hold.

**5. Monte Carlo uncertainty (V4/U1)** — across 30 independent seeds, the
empirical scatter matches binomial theory with ratio **0.906 / 0.964 / 1.035**
at n = 1e4 / 1e5 / 4e5, confirming the reported Wilson interval is faithful.

**Uncertainty analysis** (`VALIDATION.md` §6) additionally covers rare-event
resolution limits, input sensitivity (range dominates: `d ln P/+1 % = +0.42`;
attenuation second at +0.25), surrogate calibration, and an explicit table of
**unquantified model-form uncertainties**.

**What was NOT validated** (`VALIDATION.md` §7): no comparison against
measured FSO data; the Kim model is implemented-as-published but not
independently checked; no strong-turbulence validation; no temporal
correlation or fade-duration statistics; no hardware effects; surrogate
uncertainty uncalibrated.

## Benchmark results

Measured, `validation/v3_performance.txt` (2 cores, best of 3):

| Operation | Measured |
|---|---|
| Monte Carlo throughput (peak) | **1.31e7 samples/s** |
| Monte Carlo, 1e6 samples | 0.080 s |
| End-to-end report, 2e5 samples | **0.0169 s** (requirement: < 5 s) |
| Surrogate query (batched) | **7.60 µs** |
| Monte Carlo query (1e5 samples) | 6004 µs |
| **Surrogate speed-up** | **790×** |
| Dataset generation (4000 × 5e4) | 12.3 s |
| Surrogate training | 5.8 s |
| Test suite (251 tests) | 11.75 s |

## AI model details

Full card: [`MODEL_CARD.md`](MODEL_CARD.md) · Data: [`DATASET_CARD.md`](DATASET_CARD.md)

**This model is not certified for operational flight use.**

**Baseline (implemented first).** The closed-form lognormal fade probability
in `beamtwin.stats`. Exact for scintillation-only fading; validated against
Monte Carlo in 5/5 cases.

**Model.** 5 × `GradientBoostingRegressor` (300 trees, depth 3, lr 0.05), each
on an independent bootstrap resample, predicting `log10 P_fade` from 5
features: `log10_range_m`, `log10_cn2`, `jitter_ratio` (σ_jitter/θ_div),
`attenuation_db_per_km`, `margin_db`.

**Dataset.** 4000 scenarios from the twin's own seeded Monte Carlo (5e4
samples each), master seed 42, committed at 193 KB. 23.8 % of labels sit at
the 1e-4 probability floor.

**Test split.** Random 80/20, split seed 123 → 3200/800. Valid as held-out
because every scenario is an independent draw; it does not measure
out-of-domain generalisation.

**Metrics** (held-out, error in log10 P_fade):

| Subset | n | MAE surrogate | MAE baseline | RMSE surrogate | RMSE baseline |
|---|---|---|---|---|---|
| All test | 800 | **0.270** | 0.404 | **0.377** | 0.843 |
| Low jitter (< 0.05) | 80 | 0.348 | **0.005** | 0.485 | **0.012** |
| High jitter (≥ 0.05) | 720 | **0.261** | 0.448 | **0.363** | 0.889 |

**The value proposition, stated exactly:** the surrogate is worth using *only*
where pointing jitter and scintillation combine — the regime in which the
analytic baseline is scintillation-only and therefore structurally wrong —
and *only* where a 790× speed-up outweighs factor-of-two accuracy. Where
jitter is negligible the baseline is 71× more accurate and **must** be
preferred. BeamTwin prints both so the choice is always visible.

**Uncertainty output.** Every prediction returns `probability`, `p_low`,
`p_high`, `log10_std`, `extrapolating`. **Measured calibration is poor:** ±2σ
covers **39.9 %** of held-out cases against a 95 % ideal, and mean σ (0.091)
is ~3× smaller than mean error (0.270). Bootstrap resampling improved this
from 21.6 % but did not fix it. Read `log10_std` as a *relative confidence
ranking* (Spearman +0.388 with actual error), **never as a probability
interval**. Calibration is deferred to v0.2.

**Failure cases.** (1) Scintillation-only regime — 71× worse than baseline.
(2) Cannot resolve P below 1e-4. (3) Largest absolute errors where P → 1
(15 km: predicts 0.668 vs truth 0.944). (4) Flat extrapolation outside the
training domain, flagged but unbounded. (5) 1550 nm only. (6) Inherits every
model-form assumption of its teacher simulation.

**Reproducibility.**

```bash
python scripts/generate_dataset.py    # seed 42,  12.3 s
python scripts/train_surrogate.py     # split 123, model seed 7,  5.8 s
```

Deterministic; total 18.1 s on 2 cores, no GPU.

## Hardware requirements

- **CPU:** any x86_64/ARM64; developed and measured on 2 cores.
- **RAM:** < 500 MB for default workloads. A 1e6-sample run peaks near 200 MB; `n_samples` is capped at 2e7 to prevent multi-GB allocations.
- **Disk:** ~5 MB (2.5 MB model + 193 KB dataset + code).
- **GPU:** not used and not supported.
- **OS:** Linux/macOS/Windows; Python 3.11+.

## Limitations

Ordered by how much they should worry you.

1. **Never validated against real link data.** Every check is internal consistency or agreement with closed-form theory. This is the single largest gap and the reason BeamTwin cannot support operational decisions.
2. **Surrogate uncertainty is not calibrated.** ±2σ covers 39.9 %, not 95 %. Usable only as a relative ranking.
3. **Weak-fluctuation limit.** The lognormal model is valid only for `sigma_R^2 < 1`. Beyond it, deep fades are underestimated; BeamTwin flags the condition but does not model the saturation regime (no gamma-gamma distribution).
4. **Rare fades unresolvable.** ~1e-4 at default sample counts; 99.999 % availability analysis needs ≥1e6 samples and is beyond the surrogate's floor entirely.
5. **Point-receiver pointing model.** Assumes `a << w`; error grows past `a/w ≈ 0.3` (reference case: 0.20).
6. **No aperture averaging.** Real receivers average speckle, reducing effective scintillation — so BeamTwin is **pessimistic** for large apertures.
7. **No turbulence-induced beam spreading or wander.** Only diffractive spreading; this makes it **optimistic** in the opposite direction. The two omissions do not cancel in any controlled way.
8. **Independent samples only.** No temporal correlation, so no fade duration, fade rate, or burst-error statistics — precisely what coding and interleaver design need.
9. **Homogeneous path.** Constant Cn² and attenuation; no slant paths, altitude profiles (Hufnagel-Valley), or stratification. Terrestrial horizontal links only.
10. **Scintillation and jitter assumed independent.** Assumed, not verified.
11. **Single scalar sensitivity.** No detector noise model, thermal drift, modulation, or coding.
12. **Surrogate is 1550 nm only**, with fixed aperture and transmitter geometry in training.

No deviations from the build guide were required.

## Safety statement

This software is **research-grade MVP**. It is not flight-qualified, not
certified, and not approved for operational aerospace use.

## Roadmap

- **v0.2** — Calibrated surrogate uncertainty (conformal prediction / quantile regression); gamma-gamma distribution for the strong-turbulence regime; aperture averaging.
- **v0.3** — Temporally correlated channel (Kolmogorov phase screens) enabling fade duration and rate statistics; slant paths with Hufnagel-Valley Cn² profiles.
- **v0.4** — Detector noise models and BER/coding integration; multi-wavelength surrogate.
- **Ongoing** — Comparison against any publicly available measured FSO campaign data. This is the highest-value open item and the precondition for raising the validation level.

## License

AGPL-3.0-only. Full text in [`LICENSE`](LICENSE). Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

Built on numpy, scipy, scikit-learn, matplotlib, and PyYAML. Physics from the
references below.

## Citation

```bibtex
@software{beamtwin_2026,
  title        = {BeamTwin: A Free-Space Optical Link Digital Twin with
                  Monte Carlo Fade Statistics and an ML Surrogate},
  author       = {{OPTIMA Organisation}},
  year         = {2026},
  version      = {0.1.0},
  license      = {AGPL-3.0-only},
  note         = {Research-grade MVP; not flight-qualified or certified}
}
```

**References**

- L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random Media*, 2nd ed., SPIE Press, 2005.
- B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics*, 2nd ed., Wiley, 2007.
- I. I. Kim, B. McArthur, E. Korevaar, "Comparison of laser beam propagation at 785 nm and 1550 nm in fog and haze for optical wireless communications", Proc. SPIE **4214**, 2001.
- A. K. Majumdar and J. C. Ricklin (eds.), *Free-Space Laser Communications: Principles and Advances*, Springer, 2008.
- A. A. Farid and S. Hranilovic, "Outage capacity optimization for free-space optical links with pointing errors", *J. Lightwave Technol.* **25**(7), 2007.
- A. Agresti and B. A. Coull, "Approximate is better than 'exact' for interval estimation of binomial proportions", *The American Statistician* **52**(2), 1998.
