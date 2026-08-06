# PassPlanner — Validation evidence (Level 2, Research)

All numbers below were produced by running the scripts in this directory in
the build session on 2026-08-06 (Python 3.11.15, 2 CPU cores). Raw script
output is committed alongside each script:

| Check | Script | Raw output | Verdict |
|---|---|---|---|
| Rise/set times vs dense-grid recomputation | `validate_passes.py` | `validate_passes_output.txt` | PASS |
| ILP vs hand/exhaustive optima; greedy gap | `validate_scheduler.py` | `validate_scheduler_output.txt` | PASS |
| Availability model vs climatology baseline + calibration | `validate_availability_model.py` | `validate_availability_model_output.txt`, `calibration_curve.png` | PASS |
| Closed-form circular-orbit pass geometry | `../tests/test_passes.py` | pytest run | PASS |

Rerun everything with:

```bash
python validation/validate_passes.py
python validation/validate_scheduler.py
python validation/validate_availability_model.py
python -m pytest tests/ -q
```

---

## 1. Rise/set times vs an independent dense-grid recomputation

**What is being validated.** The coarse-scan (30 s) + bisection (0.05 s
tolerance) root finder and the pass-assembly logic in
`passplanner.passes.find_passes`.

**Reference method.** A brute-force recomputation inside this repository:
the elevation function is sampled on a 1.0 s grid across the whole window to
detect and count passes, and each mask crossing is then located by a local
0.002 s linear sweep (± 3 s around the coarse bracket) with linear
interpolation between the two bracketing samples. The reference is therefore
accurate to well under 0.002 s in its own right.

**Scope limitation — stated explicitly.** The reference shares the SGP4
propagator (`sgp4` package) and the frame code (`passplanner.frames`) with
the code under test. It validates the numerical root finding and the pass
bookkeeping, **not** the underlying SGP4 propagation or the TEME→ECEF
reduction. **No comparison against any external service or tool (STK, GMAT,
Heavens-Above, Celestrak SatVis, …) was performed, and none is claimed.**
Independent validation of the geometry is provided by the closed-form case in
§4.

**Cases and results** (from `validate_passes_output.txt`):

| Satellite (fixture) | Station (fictional) | Mask | Window | Passes found (test / reference) |
|---|---|---|---|---|
| ISS (ZARYA), epoch 2008-09-20 | Alpengipfel OGS (47.10 N, 10.90 E, 2.00 km) | 20° | 2008-09-20 12:00 UTC + 1 d | 5 / 5 |
| ISS (ZARYA), epoch 2008-09-20 | Cerro Ficticio OGS (24.50 S, 70.20 W, 2.60 km) | 10° | 2008-09-20 12:00 UTC + 1 d | 2 / 2 |
| NOAA 14, epoch 1997-11-16 | Karoo Vlakte OGS (31.50 S, 21.00 E, 1.60 km) | 5° | 1997-11-17 00:00 UTC + 1 d | 4 / 4 |

Pass counts agree in every case. Per-event differences (11 passes, 22 rise/set
events):

* **worst |Δ| on any rise or set time: 0.0146 s**
* tolerance requested from the solver (`refine_tol_s`): 0.05 s → **PASS**
* worst |Δ| on pass duration: 0.0202 s
* peak-elevation agreement: ≤ 0.012° (largest single difference 0.0118° on the
  ISS/Cerro Ficticio 79.5° pass, where the culmination is sharpest)

Runtime for the same work: 0.06–0.08 s (code under test) vs 2.1–2.6 s
(dense-grid reference), i.e. the production path is ~30× cheaper for the same
answer to 15 ms.

## 2. Scheduler: ILP vs known optima, and the greedy gap

### 2a. Hand-solved instances

Rate is 1 Gbit/s everywhere, so value in Gbit equals duration in seconds and
the arithmetic is checkable by inspection. The same instances are asserted in
`tests/test_scheduler.py`.

| Instance | Passes (station, [start, end] s) | Hand optimum | Exhaustive | ILP | Greedy | Greedy gap |
|---|---|---|---|---|---|---|
| A — greedy trap | A: S1 [0,1000]; B: S1 [0,600]; C: S1 [700,1400]; one satellite | **1300** = {B,C} | 1300 | 1300 | 1000 | **23.077 %** |
| B — satellite constraint | D: S1 [0,500]; E: S2 [100,900]; same satellite | **800** = {E} | 800 | 800 | 800 | 0 % |
| C — mixed constraints | F: SAT-A@S1 [0,400]; G: SAT-B@S1 [200,900]; H: SAT-A@S2 [0,300] | **1000** = {G,H} | 1000 | 1000 | 1000 | 0 % |
| D — setup time 120 s | I: S1 [0,200]; J: S1 [260,500] | **240** = {J} | 240 | 240 | 240 | 0 % |

Hand solutions (reproduced in the raw output):

* **A**: the only feasible sets are {}, {A}=1000, {B}=600, {C}=700 and
  {B,C}=1300 (B sets at 600 s, C rises at 700 s, so they are disjoint). A
  overlaps both B and C, so it can only appear alone. Optimum = **1300**.
  Greedy takes the single largest value first (A = 1000) and is then blocked
  by both B and C → 1000, a gap of (1300−1000)/1300 = 23.0769 %.
* **B**: D and E share the satellite and overlap on [100, 500] s, so at most
  one is usable: max(500, 800) = **800**.
* **C**: conflicts are F–G (same station S1, overlap 200–400 s) and F–H (same
  satellite SAT-A, overlap 0–300 s). G and H share neither station nor
  satellite, so {G,H} = 700 + 300 = **1000** > {F} = 400.
* **D**: padding both intervals by the 120 s setup time gives [0,320] and
  [260,620], which overlap, so only one may be scheduled: max(200, 240) =
  **240**. Without the setup time both fit (440).

**ILP reproduces the hand optimum exactly in all four cases.**

### 2b. Randomized instances vs exhaustive enumeration

20 seeded random instances (seed 20260306, n = 5…12 passes, 2 satellites,
2 stations) solved by enumerating every feasible subset:

* **ILP == exhaustive optimum on 20/20 instances** (agreement to < 1e-6 Gbit)
* greedy gap: **mean 3.650 %, max 19.040 %**, non-zero on 7/20 instances

### 2c. Realistic instances (ISS fixture over an 8-station synthetic network)

27 candidate passes in 24 h; setup time swept to vary contention:

| Setup [s] | Conflict pairs | Greedy [Gbit] | ILP [Gbit] | Gap [%] | ILP solve [s] |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 45 568.6 | 45 568.6 | 0.000 | 0.01 |
| 300 | 4 | 40 717.5 | 40 717.5 | 0.000 | 0.01 |
| 600 | 6 | 39 312.5 | 39 312.5 | 0.000 | 0.01 |
| 1200 | 10 | 34 417.8 | 34 417.8 | 0.000 | 0.01 |
| 1800 | 12 | 32 552.6 | 32 552.6 | 0.000 | 0.01 |
| 3600 | 33 | 22 075.8 | 23 732.6 | **6.981** | 0.01 |

Interpretation: with a single satellite and a well-spread network the conflict
graph is sparse and greedy is optimal; the gap appears only once contention is
high (33 conflicting pairs at a 1 h setup time). The adversarial instance in
§2a shows the worst case is far larger than the typical case.

## 3. Availability model vs climatology baseline (the AI element)

Data is **synthetic** (`passplanner.synthdata`; see `../DATASET_CARD.md`).
Train n = 8000 (seed 20260301), test n = 4000 (seed 20260302) — independent
draws from the same generative process, no sample sharing. Baseline is the
climatological monthly prior (implemented first, `ClimatologyBaselineModel`).
Model is a 5-member bagged gradient-boosting ensemble.
Training time: **5.9 s** on 2 CPU cores.

| Model | Brier ↓ | Log loss ↓ | ROC AUC ↑ | ECE (10 bins) ↓ |
|---|---:|---:|---:|---:|
| Climatology baseline | 0.2328 | 0.6585 | 0.6358 | 0.0493 |
| **ML (bagged GBM)** | **0.1687** | **0.5086** | **0.8216** | **0.0179** |
| Oracle `p_true` (irreducible floor) | 0.1634 | 0.4961 | 0.8325 | 0.0180 |

* Brier improvement of the ML model over the baseline: **27.51 %**
  (0.2328 → 0.1687).
* The model is within **0.0053 Brier** of the generative-process oracle, i.e.
  it recovers most of the learnable signal in this synthetic problem. That
  gap is a property of the synthetic generator, not evidence about real
  weather.
* Calibration: ECE = **0.0179**; worst bin deviation 0.036 (bin 0.40–0.50,
  n = 367, predicted 0.4519 vs observed 0.4877). Acceptance threshold was
  ECE < 0.05 → PASS. Reliability diagram: `calibration_curve.png`.
* Uncertainty output (ensemble std): mean 0.0321, median 0.0289, p95 0.0670,
  max 0.2190. Correlation with the actual error |p_pred − p_true| is
  **+0.2584** — the spread is a weak but positively-signed error indicator,
  and must not be read as a calibrated error bar.

## 4. Closed-form pass geometry (independent of SGP4)

`tests/test_passes.py` builds a TLE-free ephemeris: a circular orbit of radius
r = 7000 km in the Earth-fixed equatorial plane passing directly over a
station at (0° N, 0° E, 0 km), where |r_site| = a_WGS84 = 6378.137 km exactly,
so the spherical-Earth relation applies without approximation:

    tan(el) = (cos ψ − Re/r) / sin ψ      (Wertz & Larson, SMAD 3rd ed., Ch. 5)
    ψ0 = arccos((Re/r)·cos el0) − el0
    rise/set = t_culmination ∓ ψ0/n,   n = sqrt(μ/r³)

Hand values for a 0° mask: ψ0 = arccos(0.91116243) = 0.4246999 rad,
n = 1.0780076e-3 rad/s, half-width = **393.967 s**.
`find_passes_from_position_fn` reproduces the closed-form rise and set times
to **within 0.05 s** for masks of 0°, 5°, 10°, 20° and 40°, and the
culmination elevation to within 0.01° of the exact 90°.

Elevation known-answers also checked: satellite on the geodetic local vertical
→ exactly 90.0° (to 1e-9°, verified over a Hypothesis sweep of latitudes and
longitudes); due-north and due-east horizon targets → 0° elevation at azimuth
0° and 90°; sub-station point → −90°. WGS-84 polar radius recovered as
6356.752314 km. GMST at JD 2451545.0 = 280.46061837° (the IAU 1982 constant
term).

## 5. Test suite

`python -m pytest tests/ -q` from `products/P004/`: **106 passed, 0 failed,
0 skipped** (runtime 7.9 s; 218 warnings, all `DeprecationWarning` from PuLP
about the upcoming PuLP 4.0 API — see README Limitations). Includes the
Medium-class integration test
(TLE → passes → availability → greedy and ILP schedules → CLI) and two
benchmark/regression tests (24 h of passes over 3 stations < 10 s, measured
0.21 s; ILP on a 27-pass instance < 30 s, measured 0.58 s).

## 6. Known gaps in this validation

* SGP4 itself and the TEME→ECEF reduction are **not** independently validated
  here; they are used as supplied by the `sgp4` package. The GMST-only
  rotation neglects polar motion, UT1−UTC (≤ 0.9 s ⇒ ≤ 0.00375° of Earth
  rotation) and equation-of-the-equinoxes terms — see
  `src/passplanner/frames.py`. No refraction model is applied, which biases
  low-elevation rise/set times by up to a few tenths of a degree of
  elevation (Vallado 2013 Ch. 4); at a 20° optical mask this is small, at
  a 0–5° mask it is not.
* The cloud-availability model is validated only against its own synthetic
  generator. **There is no validation against real meteorological data or
  real optical-link statistics**, and the shipped station priors are
  invented placeholders.
* The all-or-nothing per-pass cloud model (one Bernoulli draw at culmination)
  is an assumption, not a validated result; partial-pass cloud transit is not
  modelled.
* Scheduling assumes one telescope per station and one terminal per
  satellite, constant data rate over a pass, and no energy/thermal/buffer
  constraints.
