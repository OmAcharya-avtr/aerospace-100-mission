# TurbScope

**Status:** TESTING · **Class:** medium · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

TurbScope estimates the **path-averaged optical turbulence strength Cn2_path**
(m⁻²/³) — the number that sizes free-space-optical link margins,
adaptive-optics correction budgets and beam-wander predictions — from what a
scintillometer and a Differential Image Motion Monitor (DIMM) actually
measure along a horizontal or slant path.

It ships two things, in this order:

1. **Classical closed-form inversion, implemented and validated first**: the
   weak-fluctuation (Rytov) scintillometer inversion (Tatarski 1961; Andrews
   & Phillips 2005; Wang, Ochs & Lawrence 1978) and the DIMM differential-
   motion inversion (Sarazin & Roddier 1990), each with propagated
   uncertainty and each round-trip-validated against its own forward model.
   This includes a **documented, quantified failure mode**: at strong
   turbulence, scintillation saturates and the scintillometer-only inversion
   becomes genuinely multi-valued — demonstrated concretely in
   `validation/saturation_regime.py`.
2. **A learned multi-sensor model**: three quantile-regression trees mapping
   scintillometer + DIMM readings + path length to Cn2_path, with a
   split-conformal-calibrated prediction interval, benchmarked against the
   closed-form single-sensor baselines on the same held-out synthetic data.

The learned model is trained entirely on **synthetic** data generated from
this product's own forward models plus a hand-chosen noise model. Read
`DATASET_CARD.md` before using any accuracy figure: measured accuracy is
fidelity to a generative process, not agreement with a real instrument.

## Aerospace problem

Path-averaged Cn2 is rarely measured directly; it is inferred from
scintillation, differential image motion, or meteorological proxies, each
with its own bias. A scintillometer alone has a well-known, textbook failure
mode — scintillation saturates at strong turbulence or long paths, and the
simple weak-theory inversion silently returns a wrong, single-valued answer
where the true relationship is multi-valued. A DIMM alone avoids that
specific failure but has its own real-world sensitivity limits at very poor
seeing (not modelled in this release — see Limitations). The gap this
product addresses: combining both instruments' readings, with a model that
is honest about when it is extrapolating and how wide its uncertainty really
is — and honest about when the fusion helps and when a single good sensor
is already enough.

## Intended users

* Free-space-optical-link and adaptive-optics engineers doing early sizing
  who need Cn2_path with an uncertainty, not a single number.
* Researchers who need a clean, cited implementation of the Rytov
  scintillometer and DIMM forward/inverse models, with a concrete,
  reproducible demonstration of the scintillation saturation failure mode.
* Anyone building a Cn2-sensing pipeline who needs an honest baseline to
  beat and a worked example of benchmarking and calibrating a multi-sensor
  learned model against classical single-sensor inversions.

## Engineering theory

All Cn2 in m⁻²/³, path length in metres, wavelength in metres, angles in
degrees at public APIs. Full derivations, units, assumptions and validity
ranges are in the module docstrings of `src/turbscope/`; summarised here.

### Scintillometer: Rytov variance (weak-fluctuation theory)

```
sigma_R^2 = C_w * Cn2_path * k^(7/6) * L^(11/6)
```

`k = 2*pi/lambda`; `C_w` = 1.23 (plane wave) or 0.50 (spherical wave).
Source: Tatarski (1961); Andrews & Phillips (2005) Ch. 1, 5; scintillometer
form as Wang, Ochs & Lawrence (1978). In the weak-fluctuation limit
`sigma_I^2 ~= sigma_R^2`. Valid for `sigma_R^2 <= WEAK_REGIME_MAX_SIGMA_R2 =
0.3` (conservative; `validation/round_trip_recovery.py` §1.1 quantifies how
fast the error grows toward that boundary — ~10% at σ_R²=0.3).

### Scintillation saturation (heuristic, this product)

Beyond the weak regime, `turbscope.scintillometer.scintillation_index_full`
models the well-documented qualitative behaviour of real scintillation
(linear Rytov growth, a "focusing" overshoot above an order-unity asymptote,
then saturation) with a bridging function **built for this product**, not a
literature curve fit — see that module's docstring "Honesty note" before
treating any of its numbers as a specific published result. It correctly
reproduces the *existence* of a multi-valued σ_I² band
(`validation/saturation_regime.py` §2): **[0.999, 1.126]**, in which a single
measurement has two consistent Cn2 candidates 2.00× apart.

### DIMM: differential image motion (Sarazin & Roddier 1990)

```
sigma_l,t^2 = 0.358 (lambda/D)^2 (D/r0)^(5/3) [1 - c_l,t (d/D)^(-1/3)]
```

`c_l` = 0.541 (longitudinal), `c_t` = 0.798 (transverse); `D` subaperture
diameter, `d` centre-to-centre separation. Source: Sarazin & Roddier (1990);
reproduced in Tokovinin (2002). This is the standard long-baseline,
diffraction-neglected approximation (requires `d > D`) — see Limitations.
`r0` relates to Cn2_path by the Fried-parameter integral (Fried 1965, 1966):
`r0 = [0.423 k^2 sec(zeta) Cn2_path L]^(-3/5)`.

### Multi-sensor closed-form fusion

`turbscope.inversion.multi_sensor_closed_form_estimate` combines the three
single-sensor closed-form estimates by inverse-variance weighting (Bevington
& Robinson 2003, Ch. 4) — a classical, non-learned multi-sensor estimator,
distinct from both the mandated single-sensor baseline and the learned model,
reported for context in `validation/VALIDATION.md`.

## Architecture

```
src/turbscope/
├── constants.py        physical constants, unit conventions, validity ranges
├── scintillometer.py    Rytov forward model, weak inversion, saturation model, multi-root inversion
├── dimm.py               DIMM forward model, Fried-parameter conversions, inversion
├── synthetic.py          seeded synthetic scenario + measurement generator
├── dataset.py             feature-table construction, scenario-grouped splits
├── inversion.py           classical closed-form inversion WITH uncertainty, multi-sensor fusion
├── model.py                learned quantile-GBR model + all baseline comparators
├── __main__.py             python -m turbscope CLI (forward / invert / predict)
└── __init__.py
```

## Installation

```bash
cd products/P013
python -m pip install -e ".[test]"
```

Python 3.11+. Dependencies: numpy, scipy, scikit-learn, matplotlib (Agg
backend only). No PyTorch.

## Quick start

```python
from turbscope.scintillometer import rytov_variance, invert_cn2_weak
from turbscope.dimm import differential_variance, invert_cn2_from_variance
from turbscope.model import train_default_model

# classical forward + inverse round trip
sigma_r2 = rytov_variance(1e-15, 500.0, 880e-9, "spherical")
cn2_back = invert_cn2_weak(float(sigma_r2), 500.0, 880e-9, "spherical")

# learned multi-sensor model with a prediction interval
model, _ = train_default_model()
pred = model.predict(sigma_i2_scint=0.05, var_long_dimm=1e-12, var_trans_dimm=8e-13, path_length_m=500.0)
print(pred.cn2_path, pred.cn2_lower, pred.cn2_upper)
```

## Configuration

Fixed instrument geometry used throughout `turbscope.synthetic` (documented,
not hidden): scintillometer 880 nm, spherical wave; DIMM 500 nm, D=0.14 m,
d=0.20 m (the classic ESO DIMM values). Ground-truth draw ranges: target
Rytov variance log-uniform over `[10^-3.5, 10^1.85]` (~3e-4 to ~71), path
length uniform over [150, 2500] m. Sensor noise: 8% (scintillometer) / 10%
(DIMM) relative, hand-chosen and documented in `DATASET_CARD.md`.

## Examples

```bash
python examples/saturation_curve.py          # -> screenshots/saturation_curve.png
python examples/prediction_vs_baselines.py   # -> screenshots/prediction_vs_baselines.png
```

`saturation_curve.png` plots the weak-theory line against the full
saturation curve, marking the multi-valued band and a concrete two-root
example. `prediction_vs_baselines.png` sweeps true Cn2 from weak to
saturated turbulence and plots the learned model's median + 90% interval
against both classical baselines and the ground truth — the scintillometer
baseline visibly plateaus (fails) past σ_R² ≈ 1 while the learned model and
the DIMM baseline continue tracking truth.

## Validation

Level 2 (Research): closed-form implementations validated against their own
cited mathematics via known-answer tests and round-trip recovery; the
learned model benchmarked against classical baselines on held-out synthetic
data. Full evidence with real executed-script numbers in
`validation/VALIDATION.md`. Headline results:

* **Weak-regime round trip**: multi-sensor closed-form fusion recovers a
  known Cn2_path to **median 3.9% relative error** with realistic 8-10%
  sensor noise, across 1639 independent weak-regime scenarios.
* **Saturation failure, quantified**: applying the classical weak-theory
  inversion outside its validity range gives a **median 89.5% relative
  error** (vs 5.9% in the weak regime) — the required documented failure
  mode, demonstrated and quantified, not hidden.
* **Multi-valued inversion, demonstrated**: a σ_I² measurement in
  [0.999, 1.126] has two consistent Cn2 candidates 2.00× apart, with no way
  to distinguish them from the scintillometer reading alone.
* **Prediction-interval coverage**: nominal 90%, achieved **87.7%**
  (conformally calibrated, held-out test data), within the ±2-3 point
  resolution of the 225-scenario test set.

## Benchmark results

Held-out test set, 675 rows from 225 unseen scenarios, errors in dex
(decades of Cn2):

| predictor | RMSE | 
|---|---:|
| **TurbScope learned model** | 0.0714 |
| Scintillometer weak baseline (mission-mandated) | 0.6577 |
| DIMM-only closed-form baseline | **0.0318** |
| Training mean (learned-nothing floor) | 1.6877 |

**The learned model beats the mission-mandated single-sensor baseline
decisively (89% RMSE reduction). It does NOT beat the DIMM-only closed-form
baseline**, which wins outright because this product's synthetic generator
gives DIMM zero saturation-related model-form error at any turbulence
strength. Both facts are reported plainly — see `MODEL_CARD.md` §7 for the
full honest discussion, including why, and what the mission takeaway is.

## AI model details

See `MODEL_CARD.md` for the complete card (baseline, architecture, dataset
source, training procedure, metrics, test-split strategy, uncertainty
output, failure cases, reproducibility). **This model is not certified for
operational flight use.**

## Hardware requirements

No GPU. Runs on 2 CPU cores; the full model fit + conformal calibration
takes ~3-4 seconds; the entire test suite runs in well under a minute.

## Limitations

* **All ground truth is synthetic** — see `DATASET_CARD.md`. No real
  scintillometer or DIMM measurement appears anywhere in this product.
* **The scintillation saturation model is a heuristic built for this
  product**, correct in its qualitative shape but not a literature curve
  fit — see `turbscope.scintillometer` module docstring.
* **DIMM's own real-world sensitivity limits are not modelled.** The
  long-baseline, diffraction-neglected DIMM formula (requiring `d > D`) is a
  documented simplification (Tokovinin 2002 gives a more exact,
  diffraction-corrected form, not implemented here); real DIMM performance
  also degrades at very poor seeing, a regime this generator does not
  simulate.
* **A single fixed instrument geometry** (one scintillometer wavelength/wave
  type, one DIMM wavelength/aperture/separation) is used throughout; a
  differently configured deployment is outside the training domain.
* **8%/10% sensor noise levels are hand-chosen illustrative values**, not
  vendor specifications.
* **No anisoplanatism, beam wander, non-Kolmogorov turbulence spectra, or
  time dependence** are modelled.
* **The learned model loses to the strongest available closed-form
  single-sensor baseline (DIMM-only)** in this synthetic design — see
  Benchmark results above and `MODEL_CARD.md` §7 for the full honest
  discussion.
* **Prediction-interval coverage is marginal, not conditional**, and its
  exchangeability guarantee holds only because calibration and test data
  share one synthetic generator.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

* A "smart switching" baseline (trust DIMM, treat scintillometer saturation
  as a detector rather than an estimator) as an additional comparator —
  `MODEL_CARD.md` §7 flags this as the most informative next experiment.
* A diffraction-corrected DIMM forward model (Tokovinin 2002 full form).
* Heterogeneous instrument geometries in the synthetic generator (varying
  wavelength, aperture, separation per scenario).
* A DIMM-specific saturation/sensitivity-limit model at very poor seeing.

## License

Apache-2.0. See `LICENSE`.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). TurbScope: path-averaged Cn2 estimation from
scintillometer and DIMM measurements with classical closed-form inversion
and a learned multi-sensor regressor. Version 0.1.0.
```

Underlying physics cited throughout the code and `validation/VALIDATION.md`:
Tatarski (1961); Andrews & Phillips (2005); Wang, Ochs & Lawrence (1978);
Sarazin & Roddier (1990); Fried (1965, 1966); Tokovinin (2002); Romano,
Patterson & Candès (2019); Bevington & Robinson (2003).
