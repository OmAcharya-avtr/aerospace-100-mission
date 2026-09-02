# Model card — momentummgr learned desaturation scheduler v0.1.0

**This model is not certified for operational flight use.**

**Every episode, orbit, magnetic-field value, wheel state and decision label used to train
or evaluate this model is SIMULATED — see `DATASET_CARD.md`. No flight telemetry, no
measured magnetometer record, no wheel tachometer log and no on-orbit desaturation record
is used anywhere in this package.**

## Problem

Decide, once per 600 s window over about nine hours, whether to run a spacecraft's
magnetorquers to unload its reaction wheels. Running them costs magnetorquer duty (coil
on-time, power and thermal load); not running them lets the wheels fill toward saturation,
where control authority is lost. Two metrics are reported for every policy:

* **`duty_fraction`** = ∫|m| dt / (m_max · T), dimensionless;
* **`near_saturation_fraction`** = fraction of the episode with |h_wheel| above 0.8 of the
  wheel array's conservative body envelope.

The learned component predicts, from quantities a spacecraft actually has onboard, whether
this window is a good one to dump, together with a confidence.

## The physical fact the model is trying to exploit

A magnetorquer produces `T = m × B`, so it removes momentum at a rate proportional to
`|B| sin θ` with θ the angle between the wheel momentum and the field, and it can never
remove the component along B at all (`validation/magnetic_controllability.py` quantifies
this). A fixed-threshold rule ignores the geometry entirely: it dumps when the wheels are
full, whatever the field is doing. A scheduler that can wait a few windows for a better
angle should buy the same saturation margin for less duty. Whether it does is the question
the benchmark answers, and the answer is "partly, and at a price".

## Classical baseline, implemented and tuned first

`momentummgr.policies.FixedThresholdScheduler` — dump when |h|/h_env rises to
`on_fraction`, keep dumping until it falls below `off_fraction`. Threshold unloading with
hysteresis is the standard logic (Wertz, *Spacecraft Attitude Determination and Control*;
Sidi, *Spacecraft Dynamics and Control*) and is what most smallsats fly.

It is **tuned on the same 85 non-held-out episodes the learned model gets**, by grid
search over 28 (on, off) pairs, minimising the same cost. Chosen: `on = 0.60`,
`off = 0.48`, training mean cost 0.074266. The best five grid points differ by 0.011 in
mean cost, so the baseline is not sensitive to the grid resolution. Two trivial references
are also implemented and reported: always-on and never (safety override only).

## Where the labels come from

There is no analytic optimum for "which windows should the magnetorquers run in", so
labels come from an **offline search** per fitting episode
(`momentummgr.learned.search_best_mask`): seeded candidates (all-off, all-on, nine
fixed-threshold schedules, 160 Bernoulli masks at rates drawn in [0.05, 0.95]), then
coordinate descent on single-window flips until no flip improves the episode cost, at most
six rounds. About 513 schedules are simulated per episode, vectorised.

**The search is not a policy and could never fly**: it sees the whole episode. The
classifier trained on its output *is* causal — see the feature list. Behaviour cloning of
a non-causal search is a documented weakness, and the gap it leaves is measured: the
learned scheduler captures 52.4 % of the headroom between the tuned baseline and the
search.

## Architecture

* `sklearn.ensemble.GradientBoostingClassifier(n_estimators=150, max_depth=3,
  min_samples_leaf=8, learning_rate=0.1, random_state=0)`
  (`momentummgr.learned.LearnedScheduler`). No PyTorch, no GPU.
* Gradient-boosted stumps were chosen over a random forest for **prediction latency**: the
  scheduler is called once per window inside thousands of closed-loop rollouts, and a
  300-tree forest costs 37 ms per single-row `predict_proba` against 0.29 ms here. Training
  accuracy of the two was within a percent.
* **Inputs, eleven features** (`momentummgr.episodes.FEATURE_NAMES`), with their measured
  importances from the run in `validation/learned_vs_fixed_ci_output.txt`:

  | Feature | Onboard source | Importance |
  |---|---|---|
  | `coast_h_fraction_3` | wheel tachometers + modelled disturbance torque, propagated 3 windows with the torquers off | 0.3080 |
  | `coast_h_fraction_6` | same, 6 windows | 0.1320 |
  | `merit_now` = \|B\| sin θ scaled | magnetometer + tachometers | 0.1099 |
  | `h_fraction` = \|h\|/h_env | tachometers | 0.0941 |
  | `windows_since_dump` capped at 20 | the controller's own record | 0.0931 |
  | `b_norm_now` | magnetometer | 0.0851 |
  | `sin_theta_now` | magnetometer + tachometers | 0.0553 |
  | `dh_last_window` | tachometers, two samples | 0.0464 |
  | `b_norm_next` | onboard field model + propagator | 0.0280 |
  | `best_merit_next_3` | onboard field model + propagator | 0.0274 |
  | `sin_theta_next` | onboard field model + propagator | 0.0208 |

  Nothing uses the realised future disturbance, the future wheel state under the policy's
  own future actions, or any quantity a spacecraft would not have at the instant of the
  decision. The two most important features are the coast predictions, which are the causal
  substitute for the foreknowledge the offline search gets for free.
* **Target**: the searched schedule's action in that window, 0 or 1. Windows where the
  safety override had already fired are dropped, because the override is applied
  identically to every policy at run time.
* **Output**: `(actuate, confidence)`. `confidence = p` when it actuates and `1 − p` when
  it does not, with `p` the model's class probability.

## Decision knobs, and where they are tuned

`decision_threshold` (probability above which it actuates) and `min_confidence` (below
which it defers to the classical fallback) are tuned by closed-loop grid search on **25
episodes the classifier was not fitted on**. Tuning them on the fitting episodes instead
was tried and rejected: it overfitted them and cost about 15 % on held-out mean episode
cost. Chosen: threshold 0.05, deferral band 0.70.

A **safety override** applies to every policy in this package, learned or classical: above
0.95 of the envelope, dump regardless. It is not part of the model and does not
distinguish the two.

## Dataset and its limitations

See `DATASET_CARD.md`. In one line: 60 fitting, 25 knob-tuning and 80 held-out simulated
episodes drawn from a documented parameter envelope, 3413 feature rows, positive label
rate 0.0847, entirely synthetic. The environment models are a Vallado exponential
atmosphere with no solar-activity dependence and a **centred tilted rotating dipole**
geomagnetic field, whose *direction* errs against IGRF by tens of degrees — and direction
is exactly what this model is reasoning about.

## Metrics, held out, 80 episodes

Full table, protocol and confidence intervals: `validation/VALIDATION.md` section 5 and
`validation/learned_vs_fixed_ci_output.txt`. Differences are paired by episode with a
95 % percentile bootstrap interval over 10 000 resamples.

| Metric | baseline | learned | difference | 95 % CI | verdict |
|---|---|---|---|---|---|
| Magnetorquer duty | 0.070837 | 0.056150 | −0.014687 | [−0.01882, −0.01072] | learned better |
| Time near saturation | 0.000044 | 0.003586 | +0.003542 | [+0.00035, +0.00883] | baseline better |
| Dipole cost [A m² s] | 3665.18 | 2842.37 | −822.81 | [−1044.79, −605.68] | learned better |
| Combined cost (weight 1.0) | 0.070880 | 0.059736 | −0.011145 | [−0.01639, −0.00501] | learned better |
| Peak \|h\|/envelope | 0.653456 | 0.685692 | +0.032236 | [+0.01965, +0.04561] | baseline better |

Envelope exceedances: 0 of 80 for both.

**The honest summary.** The learned scheduler buys 20.7 % less magnetorquer duty by
spending saturation margin. At a saturation weight of 2.0 or more the combined-cost
difference falls inside its confidence interval and the two are **indistinguishable**;
that row is in the validation table and is not omitted.

## Uncertainty and its calibration

On 1426 held-out classification rows (positive rate 0.0428):

* Brier score **0.039862** against **0.040947** for a constant base-rate predictor, i.e.
  a skill of **+0.0265**. Real, small.
* Reliability: well calibrated below 0.05 (gap −0.0059 over 1072 rows) and **overconfident
  above it**, by +0.0365, +0.0327 and **+0.1120** in the [0.05, 0.10), [0.10, 0.20) and
  [0.20, 0.40) bins. Above 0.40 there are fewer than 20 rows per bin and no calibration is
  claimed.

The confidence is a **decision score with modest skill, not a calibrated posterior**. Do
not read `confidence = 0.3` as "30 % chance this is the right action".

## Failure cases

1. **It runs the wheels closer to saturation than the baseline**, by 0.032 of the envelope
   on average and by more on individual episodes. On the 80 held-out episodes neither
   policy exceeded the envelope, but the margin the learned scheduler leaves is smaller,
   and the safety override at 0.95 is doing real work.
2. **It needs data.** Measured during development on 30 held-out episodes: with 24
   fitting and 10 knob-tuning episodes the learned scheduler *loses* to the tuned baseline
   on combined cost, 0.0907 against 0.0597; with 44 and 18 (the configuration
   `examples/scheduler_comparison.py` now uses) it wins, 0.0440 against 0.0597. The
   reported advantage is not robust to a small training budget, and the crossover lies
   inside the range a user might plausibly choose.
3. **Overconfidence above p = 0.05**, above.
4. **Nothing outside the episode distribution is validated.** Vehicles, orbits, wheel sizes
   or magnetorquers outside `DATASET_CARD.md`'s ranges are extrapolation.
5. **The field model is a dipole.** The whole benefit is geometric, and the geometry it
   learns is that of a centred dipole. On the real field the learned advantage may shrink
   or vanish; nothing here measures that.
6. **Episodes where full-duty dumping still saturates are excluded** at sampling. About
   8 % of raw draws are like that. They are a sizing failure rather than a scheduling
   problem, but a user whose vehicle is one of them gets no guidance from this benchmark.

## Reproducibility

```bash
cd products/P029
python3 data/generate_dataset.py           # rebuilds data/training_features.csv, ~62 s
cd validation && python3 learned_vs_fixed_ci.py    # the full benchmark, ~116 s
```

Seeds: fitting 1000–1059, knob tuning 2000–2024, held out 5000–5079, offline search seed
0, model `random_state=0`, bootstrap RNG seed 20260902. Episode draws use
`numpy.random.default_rng([seed, attempt])` and are deterministic.

## Compute used

Two CPU cores, no GPU. Training (offline search on 60 episodes, fit, knob tuning on 25
episodes) takes **31.3 s**; the whole benchmark including 80 held-out episodes evaluated
twice, the calibration search and the integrator-sensitivity rerun takes **116.2 s**.
Dataset regeneration takes 62.4 s. Peak memory is a few hundred MB.

## Ethical and safety limits

* This model schedules a spacecraft actuator. It is **not certified for operational flight
  use**, has never been run against real telemetry, and must not be placed in a control
  loop that matters.
* It is validated only inside a synthetic distribution. Its advantage over the classical
  baseline is a 20.7 % reduction in one metric bought with a measured loss in another; it
  is not a general improvement and must not be presented as one.
* Anything safety-relevant here is the safety override and the classical fallback, both of
  which are simple, inspectable and independent of the model. If the model is removed the
  system degrades to the tuned fixed-threshold scheduler, which is the correct failure
  mode.
