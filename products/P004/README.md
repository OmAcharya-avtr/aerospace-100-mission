# PassPlanner

**Status:** TESTING · **Class:** medium · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

PassPlanner turns a set of TLEs and a set of optical ground stations into a
contact plan. It propagates each satellite with SGP4, finds the visibility
windows above each station's elevation mask, weights every window by the
probability that the sky will actually be clear, and then selects the subset
of contacts that maximises expected delivered data subject to the physical
constraint that a station can track one satellite at a time and a satellite
can talk to one station at a time.

Two schedulers are provided: a greedy baseline and an exact integer-linear
program (PuLP/CBC). The availability weighting can come from climatological
monthly priors, from a user-supplied forecast, or from a machine-learning
model that outputs a calibrated pass-success probability together with an
uncertainty estimate.

Everything ships with the evidence: `validation/` contains the scripts and
their raw output, including a rise/set cross-check against a dense-grid
recomputation, ILP-vs-hand-solved-optimum tables and a calibration curve.

## Aerospace problem

Free-space optical downlinks offer orders-of-magnitude more capacity than RF,
but they are blocked by cloud. A single optical ground station may be usable
only 30–80 % of the time depending on site climate, so operators build
networks of stations and must decide, ahead of time, which contacts to
commit to. The decision is a scheduling problem with three coupled parts:

1. **Where and when are the passes?** — orbital geometry over an elevation
   mask set by atmospheric path length, not just horizon visibility.
2. **How likely is each pass to work?** — site climatology plus, ideally, a
   short-range forecast.
3. **Which subset should we actually book?** — passes compete: two stations
   cannot use the same satellite simultaneously, and a station cannot track
   two satellites at once. Slew and acquisition time makes near-adjacent
   passes mutually exclusive too.

Planning by "take the highest-elevation pass" leaves data on the table when
the highest pass is over the cloudiest site. PassPlanner makes the trade
explicit and solves it exactly on instances of realistic size.

## Intended users

* Optical ground-segment engineers doing network sizing and site trade
  studies.
* Mission-operations analysts building contact plans and downlink budgets for
  study or training purposes.
* Researchers benchmarking contact-scheduling heuristics against an exact
  optimum.
* Students learning pass geometry, SGP4 usage and scheduling formulations.

It is **not** an operational planning tool (see Safety statement).

## Engineering theory

### Propagation

Satellite state comes from SGP4/SDP4 as implemented in the `sgp4` package
(Vallado, Crawford, Hujsak & Kelso 2006, AIAA 2006-6753, *Revisiting
Spacetrack Report #3*). Positions are returned in the TEME frame in km.
Validity: SGP4 is designed for TLE-derived mean elements; accuracy degrades
roughly kilometres per day away from epoch, so a TLE more than a few days old
is unsuitable for anything but demonstration.

### TEME → ECEF

    r_ECEF = R3(θ_GMST) · r_TEME

with GMST from the IAU 1982 polynomial (Aoki et al. 1982, *Astron.
Astrophys.* 105, 359; given in this exact form by Meeus 1998, *Astronomical
Algorithms*, 2nd ed., Eq. 12.4; see also Vallado 2013, *Fundamentals of
Astrodynamics and Applications*, 4th ed., Ch. 3):

    GMST[deg] = 280.46061837 + 360.98564736629·d + 0.000387933·T² − T³/38 710 000
    d = JD_UT1 − 2451545.0,  T = d/36525

**Documented simplification and accuracy class.** This is the "GMST-only"
reduction. It neglects polar motion (< ~1 arcsec, sub-metre at the ground),
the UT1−UTC difference (bounded by 0.9 s, i.e. ≤ 0.00375° of Earth rotation
≈ 0.45 km at the equator) and equation-of-the-equinoxes / TEME-vs-PEF terms
(arcsecond class). The resulting error on LEO rise/set *times* is far below
one second — negligible against tens-of-seconds scan steps and minutes-long
passes — so the model is adequate for **scheduling**, and inadequate for
precision pointing or orbit determination.

Units: km, radians internally, degrees at the API. Validity: the GMST
polynomial is used near the J2000 era (the `sgp4` Julian-date routine is
valid 1900–2100).

### Site position and topocentric look angles

WGS-84 ellipsoid (NIMA TR8350.2, 3rd ed., 2000): a = 6378.137 km,
f = 1/298.257223563.

    N = a / sqrt(1 − e²·sin²φ)
    r_ECEF = [(N+h)cosφ·cosλ, (N+h)cosφ·sinλ, (N(1−e²)+h)·sinφ]

The site→satellite vector is rotated into the SEZ (south-east-zenith) frame
using the geodetic latitude (the "RAZEL" algorithm, Vallado 2013 Ch. 4):

    el = atan2(ρ_Z, hypot(ρ_S, ρ_E))
    az = atan2(ρ_E, −ρ_S)          (from north, clockwise)
    range = |ρ|

The two-argument elevation form is used because `asin(ρ_Z/|ρ|)` is
ill-conditioned at the zenith (≈ 1e-6° error in double precision versus
≈ 1e-13° for `atan2`). **Assumption: no atmospheric refraction.** Refraction
raises apparent elevation by up to roughly half a degree at the horizon
(Vallado 2013 Ch. 4; Meeus 1998 Ch. 16); at a 20° optical mask the induced rise/set-time bias is small, at a
0–5° mask it is not.

### Pass finding

`e(t) − mask` is sampled on a coarse grid (default 30 s); sign changes bracket
rise and set events, which are refined by bisection to 0.05 s. Culmination is
found by a bounded scalar maximisation inside the pass. Passes shorter than
the coarse step can be missed, so `coarse_step_s` must be at most half the
shortest expected pass (LEO passes above a 5–30° mask last several minutes).

### Closed-form check case

For a spherical Earth of radius Re and a circular orbit of radius r, the
elevation at central angle ψ from the sub-point is (Wertz & Larson (eds.),
*Space Mission Analysis and Design*, 3rd ed., Ch. 5 "Space Mission Geometry"):

    tan(el) = (cos ψ − Re/r)/sin ψ    ⇒    ψ₀ = arccos((Re/r)·cos el₀) − el₀

so a zenith pass runs from t_c − ψ₀/n to t_c + ψ₀/n with n = sqrt(μ/r³),
μ = 398 600.4418 km³/s² (WGS-84). This is used as a TLE-free known-answer
test (see Validation §4).

### Expected delivered data

    E[data] (Gbit) = data_rate_gbps × duration_s × p_clear(station, t_culmination)

**Assumptions:** all-or-nothing cloud outcome evaluated once per pass;
constant data rate above the mask. Real optical links vary rate with range
and turbulence, and cloud can clear mid-pass. This is the first-order
site-availability treatment common in optical ground-network studies
(methodology as in Fuchs & Moll 2015, *Journal of Optical Communications and
Networking*, "Ground station network optimization for space-to-ground optical
communication links"; no numerical values are taken from that work).

### Scheduling formulation

Maximum-weight independent set on the pass conflict graph:

    maximise   Σ vᵢ·xᵢ
    subject to xᵢ + xⱼ ≤ 1   for every conflicting pair (i, j)
               xᵢ ∈ {0, 1}

with vᵢ the expected delivered data and a conflict defined as *sharing a
station or a satellite* **and** overlapping in time after both intervals are
padded by `setup_time_s` (slew + acquisition). The ILP is solved exactly by
CBC through PuLP. The greedy baseline takes passes in descending value and
keeps each if compatible; it has no optimality guarantee, and its measured gap
is reported in `validation/VALIDATION.md`.

## Architecture

```
src/passplanner/
├── frames.py        # time, GMST, TEME→ECEF, geodetic→ECEF, az/el/range
├── passes.py        # TLE dataclass, Pass dataclass, find_passes (+ generic form)
├── fixtures.py      # two checksummed historic public TLEs for tests/examples
├── stations.py      # Station dataclass + YAML loader
├── availability.py  # climatology, forecast override, expected_data
├── synthdata.py     # seeded synthetic weather dataset generator
├── mlmodel.py       # climatology baseline + bagged-GBM model with uncertainty
├── scheduler.py     # conflict graph, greedy baseline, exact ILP (PuLP/CBC)
├── cli.py           # `plan` subcommand, scenario YAML parsing
└── __main__.py      # python -m passplanner
```

Data flow: `TLE + Station → find_passes → Pass[] → expected_data(availability)
→ schedule_greedy / schedule_ilp → ScheduleResult`.

There are no cross-product imports; the package is self-contained.

## Installation

Python 3.11+. Dependencies: numpy, scipy, matplotlib, scikit-learn, pyyaml,
sgp4, pulp (CBC ships with PuLP).

```bash
cd products/P004
pip install -e .          # or: export PYTHONPATH=src
python -m pytest tests/ -q
```

## Quick start

```python
from datetime import datetime, timedelta, timezone
from passplanner import (ClimatologyAvailability, Station, find_passes,
                         load_stations, schedule_greedy, schedule_ilp)
from passplanner.fixtures import ISS_2008

stations = load_stations("examples/stations_example.yaml")
t0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)

passes = []
for st in stations:
    passes.extend(find_passes(ISS_2008, st, t0, t0 + timedelta(days=1)))

availability = ClimatologyAvailability.from_stations(stations)
plan = schedule_ilp(passes, availability, setup_time_s=1800.0)

for p, v in zip(plan.selected, plan.values):
    print(f"{p.station.name:<20} {p.t_rise:%H:%M:%S} "
          f"{p.duration_s:5.0f}s  max_el {p.max_elevation_deg:4.1f}°  {v:8.1f} Gbit")
print(f"total expected data: {plan.total_value:,.1f} Gbit")
```

Command line:

```bash
# if not pip-installed, put the package on the path first:
export PYTHONPATH=src
python -m passplanner plan --config examples/scenario_example.yaml
```

produces the candidate/selected table and, when `scheduler: both`, the greedy
gap:

```
window : 2008-09-20T12:00:00+00:00 .. 2008-09-21T12:00:00+00:00
inputs : 1 satellite(s), 3 station(s), 9 candidate pass(es), setup_time_s=60
weight : climatological priors + forecast overrides

schedule (ilp): 9/9 passes selected, total expected data 7,501.3 Gbit
satellite      station                rise (UTC)           set (UTC)             dur[s]  maxEl  p_clr   E[Gbit]
ISS (ZARYA)    Karoo Vlakte OGS       2008-09-20 13:20:56  2008-09-20 13:24:40      224   59.1   0.74    1656.3
...
summary: greedy achieves 100.00% of ILP optimum (gap 0.00%)
```

### Public API

| Function | Purpose |
|---|---|
| `find_passes(tle, station, t0, t1, min_elev_deg=None, ...)` | visibility windows above the mask |
| `find_passes_from_position_fn(fn, station, t0, t1, ...)` | same, for any ECEF ephemeris (analytic tests) |
| `expected_data(pass, availability)` | Gbit expected from one pass |
| `schedule_greedy(passes, availability=None, setup_time_s=0.0)` | greedy baseline |
| `schedule_ilp(passes, availability=None, setup_time_s=0.0, time_limit_s=60.0)` | exact optimum via CBC |
| `PassSuccessModel.predict_with_uncertainty(x)` | `(probability, ensemble spread)` |

## Configuration

**Stations** (`examples/stations_example.yaml`, `stations_network_example.yaml`):

```yaml
stations:
  - name: Cerro Ficticio OGS
    lat_deg: -24.50
    lon_deg: -70.20
    alt_km: 2.60
    min_elevation_deg: 20.0
    data_rate_gbps: 10.0
    monthly_clear_prob: [0.88, 0.86, ...]   # 12 values, Jan..Dec
```

> **The shipped station files are fictional-but-plausible illustrative
> placeholders.** Names, coordinates and clear-sky probabilities are invented
> for demonstration. They are **not real sites** and **not real site
> statistics**. Replace them with your own surveyed values before drawing any
> conclusion about a real network.

**Scenario** (`examples/scenario_example.yaml`): planning window, satellites
with inline TLE lines, `stations_file`, `scheduler: greedy|ilp|both`,
`setup_time_s`, and optional `forecast` override intervals
(`{station, start, end, p_clear}`). Invalid input (inverted window, bad TLE
checksum, duplicate station names, probabilities outside [0, 1], non-positive
data rate) raises `ValueError` with an actionable message.

## Examples

Both scripts write PNGs to `screenshots/` and were run to produce the images
committed here.

```bash
python examples/example_schedule_gantt.py
python examples/example_weighted_vs_unweighted.py
```

**`screenshots/pass_schedule_gantt.png`** — 24 h of ISS passes over the three
example stations with a 30 min setup time. Top: per-station timeline, marker
fill = climatological clear-sky probability, red outline = selected by the
ILP, labels give peak elevation and duration. Bottom: elevation profiles of
the selected contacts against time from rise. Run output: 9 candidates,
8 contacts selected, 9 987.8 Gbit (greedy identical here).

**`screenshots/weighted_vs_unweighted.png`** — the same candidates over the
8-station example network, planned once ignoring cloud and once with
availability weighting, both scored with the availability model:

| Setup [min] | Unweighted plan [Gbit] | Weighted ILP plan [Gbit] | Gain |
|---:|---:|---:|---:|
| 0 | 18 403.6 | 18 754.7 | +1.91 % |
| 15 | 15 792.5 | 16 143.5 | +2.22 % |
| 30 | 14 375.9 | 14 727.0 | +2.44 % |
| 45 | 13 512.7 | 13 863.8 | +2.60 % |
| 60 | 12 743.6 | 13 294.7 | +4.32 % |
| 90 | 10 025.9 | 10 564.0 | +5.37 % |

The right panel shows the mechanism: as contention rises the weighted plan
swaps long passes over cloudy sites for shorter passes over clear ones (at a
60 min setup time the two plans keep 11 passes each but differ on 6 of them).

## Validation

Full evidence, method descriptions and raw script output:
[`validation/VALIDATION.md`](validation/VALIDATION.md). Level 2 (Research)
means analytic cases and internal cross-checks, **not** comparison against
flight data or a certified reference tool.

1. **Rise/set times vs an independent dense-grid recomputation**
   (`validate_passes.py`). Reference: brute-force elevation sampling at 1.0 s
   for detection plus a 0.002 s local sweep for each crossing. Three
   satellite/station cases (ISS at 20° and 10° masks, NOAA 14 at 5°),
   11 passes, 22 events. Pass counts agree 5/5, 2/2, 4/4. **Worst |Δ| on any
   rise/set time: 0.0146 s** against a 0.05 s bisection tolerance → PASS.
   *Scope:* this validates the root-finding and pass bookkeeping only — SGP4
   and the frame code are shared by both paths. **No external service or tool
   was consulted and no such comparison is claimed.**
2. **ILP vs known optima** (`validate_scheduler.py`). Four hand-solved
   instances with the arithmetic written out (see VALIDATION §2a); the ILP
   reproduces every hand optimum exactly. On the adversarial "greedy trap"
   instance the hand optimum is 1300 Gbit = {B, C} and greedy returns
   1000 Gbit — a **23.077 %** gap. Additionally, ILP equals exhaustive
   enumeration on **20/20** seeded random instances.
3. **Greedy-vs-ILP gap from actual runs.** Random instances: mean gap
   **3.650 %**, max **19.040 %**, non-zero on 7/20. Realistic 27-pass
   instance over 8 stations: gap 0 % up to a 30 min setup time, **6.981 %** at
   60 min (33 conflicting pairs); ILP solves in 0.01 s throughout.
4. **Closed-form pass geometry.** The analytic circular-orbit case reproduces
   rise/set to within 0.05 s and culmination elevation to within 0.01° of the
   exact 90°, for masks of 0°, 5°, 10°, 20°, 40°. Hand values: ψ₀ = 0.4246999
   rad, n = 1.0780076e-3 rad/s, half-width 393.967 s.
5. **Availability-model calibration** (`validate_availability_model.py`,
   figure `validation/calibration_curve.png`). See AI model details.

## Benchmark results

Measured in this build session on 2 CPU cores (Python 3.11.15):

| Operation | Size | Time |
|---|---|---|
| Pass finding | 24 h, 1 satellite × 3 stations | 0.21 s |
| Pass finding (dense-grid reference) | same, 1 s grid | 2.1–2.6 s |
| ILP solve | 27 passes, 33 conflict pairs | 0.01 s |
| ILP solve | 27-pass instance incl. model build | 0.58 s |
| Greedy solve | 27 passes | < 0.01 s |
| ML training | 8000 samples, 5 GBM members | 5.9 s |
| Full test suite | 106 tests | 7.9 s |

Regression guards live in `tests/test_integration.py`: pass finding must stay
under 10 s and the ILP under 30 s.

## AI model details

Full card: [`MODEL_CARD.md`](MODEL_CARD.md). Data card:
[`DATASET_CARD.md`](DATASET_CARD.md).

* **Baseline first.** `ClimatologyBaselineModel` predicts the station's
  climatological monthly prior and ignores weather features. It is trained and
  scored on exactly the same splits as the ML model.
* **Model.** `PassSuccessModel`: 5 bootstrap-bagged
  `GradientBoostingClassifier` members (150 estimators, depth 2, lr 0.1).
  Mean of member probabilities is the prediction.
* **Dataset.** 100 % synthetic, generated by the committed script
  `src/passplanner/synthdata.py` from a fixed seed; no data files are
  committed. Seven features (climatological prior, humidity, IR cloud
  fraction, pressure anomaly, wind, month sin/cos) generated from a latent
  synoptic state that also drives the label.
* **Test split.** Train `generate_dataset(8000, seed=20260301)`, test
  `generate_dataset(4000, seed=20260302)` — independent i.i.d. draws, no
  shared samples, no leakage, no hyperparameter search against the test set.
* **Metrics (held-out, n = 4000).**

  | Model | Brier ↓ | Log loss ↓ | ROC AUC ↑ | ECE ↓ |
  |---|---:|---:|---:|---:|
  | Climatology baseline | 0.2328 | 0.6585 | 0.6358 | 0.0493 |
  | **Bagged GBM** | **0.1687** | **0.5086** | **0.8216** | **0.0179** |
  | Oracle `p_true` (floor) | 0.1634 | 0.4961 | 0.8325 | 0.0180 |

  Brier improves **27.51 %** over the baseline and lands within 0.0053 of the
  generative-process floor. Calibration curve:
  `validation/calibration_curve.png`.
* **Uncertainty output.** `predict_with_uncertainty` returns the ensemble
  standard deviation alongside the probability (mean 0.0321, p95 0.0670, max
  0.2190 on the test split; correlation with realised error +0.2584). It is a
  triage flag for epistemic disagreement, **not** a calibrated error bar, and
  it does not represent the irreducible randomness of the weather.
* **Failure cases.** Real weather features are out of distribution; extreme
  probability bins are sparsely populated; consecutive passes are predicted
  independently so correlated cloud persistence is not represented; the model
  is not a forecast system.
* **Reproducibility.** `python validation/validate_availability_model.py`
  regenerates every number above; 5.9 s training on 2 cores.

**This model is not certified for operational flight use.**

## Hardware requirements

CPU only; 2 cores are sufficient and were used for every number quoted here.
Peak memory well under 1 GB (largest array is the 12 000 × 7 synthetic
dataset). No GPU, no network access at runtime. Disk footprint of the package
plus artefacts is a few MB. The heaviest operations are ML training (5.9 s)
and the dense-grid validation reference (≈ 2.5 s per case).

## Limitations

1. **No atmospheric refraction.** Rise/set at low masks is biased; use masks
   ≥ 10–20° for meaningful results (optical stations normally do anyway).
2. **GMST-only Earth rotation.** Polar motion, UT1−UTC and
   equation-of-the-equinoxes terms are neglected — fine for scheduling, not
   for pointing or OD.
3. **SGP4 accuracy away from epoch.** The shipped fixtures are historic
   (2008 and 1997) and are demonstration/test data only.
4. **All-or-nothing cloud model.** One Bernoulli outcome per pass sampled at
   culmination; partial-pass obscuration and cloud persistence between passes
   are not modelled, so schedule risk is understated.
5. **Constant data rate over a pass.** No elevation- or range-dependent rate,
   no turbulence-driven fading, no link budget.
6. **Fictional station data.** All shipped priors and coordinates are
   invented placeholders, clearly labelled in the YAML files.
7. **ML trained only on synthetic data.** Metrics characterise the synthetic
   generator; they say nothing about real availability prediction.
8. **Resource model is simple.** One telescope per station, one terminal per
   satellite, a single scalar setup time. No energy, thermal, buffer,
   keep-out or crew constraints; no priorities or deadlines per satellite.
9. **ILP scale.** Pairwise conflict constraints are O(n²); the tested range is
   tens of passes (solve times ~0.01 s). Very large networks were not tested
   and would need a stronger formulation (clique constraints) or
   decomposition.
10. **Pass-finder resolution.** Passes shorter than `coarse_step_s` can be
    missed; the 30 s default is conservative for LEO but must be reduced for
    very high masks or very high orbits' grazing passes.
11. **PuLP deprecation warnings.** The installed PuLP emits
    `DeprecationWarning`s about the upcoming 4.0 API (`LpVariable`
    construction and `PULP_CBC_CMD`). They are surfaced, not suppressed; the
    218 warnings in the test run are all of this kind.
12. **No time-zone handling beyond UTC.** Naive datetimes are interpreted as
    UTC by design.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

* Refraction-corrected elevation and an elevation-dependent link/data-rate
  model.
* Correlated cloud modelling (persistence between consecutive passes at a
  site, spatial correlation across nearby sites) and risk-aware objectives
  (CVaR rather than expectation).
* Clique-based ILP constraints and a rolling-horizon decomposition for
  large multi-satellite networks.
* Per-satellite data-buffer and priority/deadline constraints.
* Ingestion of real forecast products through the existing
  `ForecastAvailability` interface, and validation of the ML model against
  real observations.

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

TLE fixtures are public historic element sets: the ISS example TLE
distributed with the `sgp4` Python package documentation (originating from
the Vallado et al. 2006 SGP4 verification material) and the NOAA 14 example
from T.S. Kelso's Celestrak "FAQ: Two-Line Element Set Format".

## Citation

```bibtex
@software{passplanner_2026,
  title        = {PassPlanner: optical ground-station contact planner with
                  cloud-availability weighting and schedule optimization},
  author       = {{OPTIMA Organisation}},
  version      = {0.1.0},
  year         = {2026},
  license      = {Apache-2.0},
  note         = {Research-grade software; not flight-qualified}
}
```
