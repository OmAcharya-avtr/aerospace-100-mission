# Changelog

All notable changes to FDIScope are recorded here.

## 0.1.0 — 2026-10-02

Initial release.

- **A GNC fault-injection loop.** Single-axis attitude plant (`J = 12 kg m²`,
  `dt = 0.1 s`) with a PD controller, a wheel torque clip, a Kalman filter written to
  expose its innovation, and seven injectable fault modes: sensor bias, drift, stuck and
  dropout; actuator loss of effectiveness, stuck and runaway. The residual is normalised
  to `N(0, I)` by the Cholesky factor of the steady-state innovation covariance, and that
  claim is checked before anything is built on it — mean NIS **2.003549** over 342 000
  fault-free samples against an expected 2, inside [1.993303, 2.006708], with every
  autocorrelation lag under its 4σ band (`validation/VALIDATION.md`, V1-A1).
- **Thresholds from formulae, then measured.** `chi2_threshold`,
  `cusum_threshold_for_arl0`, `cusum_delay_wald`, `cusum_delay_siegmund`,
  `cusum_arl0_siegmund` and `cusum_delay_mean_path`, each with its source in the docstring
  (Bar-Shalom, Rong Li & Kirubarajan 2001 §5.4; Mehra & Peschon 1971; Page 1954;
  Basseville & Nikiforov 1993; Siegmund 1985). The chi-squared test's measured false-alarm
  rate contains its design `alpha` in **6 of 6** Wilson intervals across two decades and
  two window lengths (V1-A2); the CUSUM's measured `ARL0` matches Siegmund at ratios
  **0.9945, 1.0759, 0.8528** (V1-A4); and the closed-form round trip
  `chi2_false_alarm_rate(chi2_threshold(a, k), k)` is exact to **7.806e-15** (V1-A3b).
- **Detection delay predicted and checked, including where the prediction fails.** In the
  exact change-point model the measured CUSUM run length matches Siegmund in **9 of 9**
  cells within 10 %, worst disagreement **2.94 %** (V2-B1a), and the Wald approximation's
  sign flip at `mu = 1/1.1652 = 0.858` is reproduced (V2-B1b). In the closed loop the
  mean-path prediction **FAILS** at a 1σ bias, ratio **1.4706** against a 25 % tolerance
  fixed before the run (V2-B2a), and the step-change Siegmund expression runs 1.40 to 1.88
  high because the estimator absorbs part of the bias before the residual mean settles
  (V2-B2b). Both are reported as findings; no tolerance was widened.
- **Five methods benchmarked at a matched operating point.** Two chi-squared windows, a
  channel CUSUM bank, a classical GLR signature bank (Willsky 1976) and a random-forest
  classifier, all calibrated to a 10 % per-run false-alarm probability on a fault-free
  block and then scored once on 240 held-out scenarios. **The classical tests win
  detection**: the CUSUM has the shortest mean delay (**54.33 samples**) and misses
  nothing, the GLR bank has the highest AUC (**0.9751**), and the learned classifier is
  slower (**56.58**) at a *higher* measured false-alarm rate (**0.1500 against 0.1389**)
  with a lower AUC (0.9695). The classifier is never fastest on any of the seven fault
  classes (V3-C2 to V3-C4).
- **The false-alarm calibration transfer is reported as a FAILURE** for the two methods
  that need data to set a threshold: `glr` at 0.0389 and `learned` at 0.1500 against a
  0.10 target, while the three methods with a closed-form threshold all landed inside
  their Wilson interval (V3-C1 **FAIL**, 2 of 5).
- **Isolation reported in full, as the specification requires.** Both 8×8 confusion
  matrices are printed complete rather than summarised. The learned classifier reaches
  **0.6958** [0.6349, 0.7506] against the classical bank's **0.4667** [0.4046, 0.5298] and
  wins six of the seven fault classes — but **the classical bank wins actuator loss of
  effectiveness, recall 0.4667 against 0.1000** (V4-D2). The structural reason for the
  bank's actuator failures is in the signature Gram matrix: `|cos|` between
  actuator-stuck and actuator-runaway is **0.9966**, so those hypotheses are inseparable
  by a matched-filter bank at any sample size (V4-D1).
- **Both confidence outputs measured rather than asserted.** The GLR posterior is badly
  over-confident — 137 of 182 declarations in the top bucket with mean confidence 0.9973
  and accuracy 0.4599, a gap of **+0.5375**. The forest's vote fraction tracks accuracy to
  within ±0.20 but is still not a calibrated probability, and no recalibration was fitted
  (V4-D4).
- **Sensitivity to the one idealisation both methods share.** Isolation assumes a known
  onset; shifting the window 50 samples early costs the classifier 49 % of its accuracy
  and the GLR bank 54 % (V4-D3).
- **Level-2 validation with saved raw output** (`validation/`): four rerunnable scripts
  totalling ≈ 227 s on two cores, each writing its stdout to a committed `*_output.txt`.
  The summary table holds 17 entries — 7 PASS, **2 FAIL**, 8 reported measurements.
- **Entirely synthetic data, regenerable byte for byte.** 240 training scenarios (seeds
  1000–1239), 240 held out (5000–5239), 150 fault-free calibration runs (9000–9149) and
  150 held-out fault-free runs (12000–12149), with a SHA-256 manifest. No flight
  telemetry, no measured innovation record, no on-orbit fault log. See `DATASET_CARD.md`
  for the eleven things the dataset does not model.
- **435 pytest tests** — unit, input-validation, known-answer with the hand arithmetic in
  the comments, Hypothesis property tests, an integration test, and pinned benchmark
  regression values. Ruff-clean at line length 100.
- **CLI**: `python -m fdiscope design | simulate | signatures | benchmark` (also installed
  as `fdiscope`). `benchmark` runs a deliberately reduced campaign and prints a warning on
  every run that its numbers are not the published ones.
- Four runnable examples writing PNGs to `screenshots/` with the Agg backend.
- `MODEL_CARD.md` and `DATASET_CARD.md`, both carrying the statement that this model is
  not certified for operational flight use.

### Known defects and gaps, not fixed in 0.1.0

- The filter model is exactly the plant model, so the fault-free innovation is exactly
  white. A mismatched filter's innovation is not, which invalidates the distributional
  assumption every method here rests on — the classical ones included. Nothing in this
  release measures that gap, and it is the largest one.
- Averaging each fault signature over eight onset phases destroys the dropout signature:
  the classical bank's `sensor_dropout` recall is 0.0333.
- The classifier has no threshold formula and no out-of-distribution guard; outside its
  training distribution it still returns a class and a confidence.
- One axis, one inertia, one gain set, one reference manoeuvre, exactly one fault at a
  time.
