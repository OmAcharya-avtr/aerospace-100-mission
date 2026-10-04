# Dataset card — HilForge synthetic latency traces

**Product:** P031 HilForge 0.1.0 · **Status:** TESTING
**Validation level:** 3, hardware-pending · **Licence:** AGPL-3.0-or-later
**Date:** 2026-10-04

**These traces are model output, not measurement.** No trace in this
repository was recorded from a real control loop, on hardware or on a
workstation. They exist so that the ground-truth overrun labels are known
exactly and a predictor's false-alarm and missed-overrun rates can be measured
rather than estimated. Read §7 before using any number derived from them.

---

## 1. What is in the dataset

Per-iteration execution latencies of a four-stage periodic control loop.

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `stage_s` | `(n, 4)` | s | durations of `sense`, `estimate`, `control`, `actuate` |
| `totals_s` | `(n,)` | s | `stage_s.sum(axis=1)`, the iteration duration `d[i]` |
| `regime` | `(n,)` | — | hidden state, 0 calm and 1 busy. Ground truth; no predictor is given it |
| `interrupted` | `(n,)` | — | whether an interrupt spike was added to that iteration |

The label a predictor is trained on is derived, not stored:
`y[i] = 1` iff `max(d[i+1], …, d[i+H]) > D` for horizon `H` and deadline `D`.

**Nothing is committed.** The traces are regenerated deterministically from a
seed by `hilforge.predict.data.generate_trace`, so there is no data file in the
repository at all. A 40 000-iteration trace is about 1.4 MB as float64 and
takes under a second to generate.

---

## 2. Provenance and generating process

`hilforge/predict/data.py`. Three components:

**1. A two-state Markov regime chain.** `z[k] ∈ {calm, busy}` with transition
probabilities `p_calm_to_burst` and `p_burst_to_calm`. Markov-modulated models
are the standard way to represent correlated bursts in a service process; see
Fischer & Meier-Hellstern, "The Markov-modulated Poisson process (MMPP)
cookbook", *Performance Evaluation* 18(2):149-171, 1993, §2, for the
construction. Stationary busy probability `p/(p+q)`, mean burst length `1/q`
(Ross 2014, §4.4).

**2. Gamma stage durations, scaled by regime.** Each stage `j` draws from
`Gamma(shape=k_j, scale=(mean_j / k_j) × m)` where `m = burst_scale` in the
busy regime and 1 in the calm one. Gamma rather than exponential because
measured execution times are positive, right-skewed, and have a tail heavier
than exponential at small shape; see Harchol-Balter, *Performance Modeling and
Design of Computer Systems*, Cambridge University Press 2013, Chapter 20. The
gamma coefficient of variation is `1/sqrt(k)`, so `k` is the knob that sets how
heavy the per-stage tail is.

**3. A Bernoulli interrupt spike.** With probability `interrupt_prob` an
exponential spike of mean `interrupt_mean_s` is added to one uniformly chosen
stage, modelling a pre-emption the loop did not schedule.

All randomness comes from `numpy.random.Generator(PCG64(seed))`. NumPy
documents PCG64 streams as reproducible across platforms and versions for a
given seed, which is what makes the dataset a function of the seed rather than
of the machine.

---

## 3. The two families, and why there are exactly two

| Parameter | `jittery` | `bursty` |
|---|---|---|
| `period_s` (= deadline) | 0.010 | 0.010 |
| `stage_mean_s` (calm) | (0.0020, 0.0014, 0.0008, 0.0022) | same |
| `stage_shape` `k` | (2.4, 1.5, 3.0, 1.2) | (9.0, 6.0, 12.0, 5.0) |
| `burst_scale` | 1.35 | 2.6 |
| `p_calm_to_burst` | 0.16 | 0.010 |
| `p_burst_to_calm` | 0.62 | 0.055 |
| `interrupt_prob` | 0.035 | 0.004 |
| `interrupt_mean_s` | 0.0040 | 0.0030 |
| **mean burst length** [iterations] | **1.6** | **18.2** |
| stationary busy probability | 0.205 | 0.154 |

Measured on 20 000-iteration traces, seed 11:

| Property | `jittery` | `bursty` |
|---|---|---|
| utilisation `E[d]/T` | 0.696 | 0.793 |
| per-iteration overrun rate | 0.146 | 0.157 |
| **lag-1 autocorrelation of the overrun indicator** | **0.0215** | **0.8502** |

That last row is the entire point. The two families have nearly the same
per-iteration overrun rate and similar utilisation, and differ by a factor of
40 in how much the recent past tells you about the next iteration. A difference
in predictor performance between them is therefore attributable to
autocorrelation and not to base rate.

`tests/test_predict.py::test_bursty_overruns_are_autocorrelated_and_jittery_ones_are_not`
asserts `|acf1| < 0.10` for jittery and `acf1 > 0.60` for bursty, so the
distinction cannot silently disappear if the presets are edited.

The split into "a family where prediction should work" and "a family where it
should not" was made **before** any predictor was fitted, which is why the
jittery result in `MODEL_CARD.md` §7a is reported as a finding rather than
explained away.

---

## 4. Splits

Chronological, never random:

```python
train, test = split_trace(trace, train_fraction=0.6)
```

A random split would put iteration `i` in training and `i+1` in test, leaking
exactly the autocorrelation the predictors exist to exploit and flattering the
learned model. `build_dataset` then drops the first 32 rows of each part (the
longest feature window) and the last `H` (labels not yet determined).

For a 40 000-iteration trace at `H = 3`:

| Family | train rows | test rows | train positive rate | test positive rate |
|---|---|---|---|---|
| `jittery` | 23 965 | 15 965 | 0.3878 | 0.3913 |
| `bursty` | 23 965 | 15 965 | 0.1895 | 0.1684 |

The jittery positive rate is high because `1 − (1 − 0.146)³ ≈ 0.38`: with a
15 % per-iteration overrun rate and a 3-iteration window, more than a third of
windows contain an overrun. That is why the F1-maximising operating point on
that family degenerates to flagging everything, as `MODEL_CARD.md` §7a reports.

Three seeds — 4242, 909090, 31337 — give three independent realisations per
family, and every benchmark number is the mean over those three.

---

## 5. Validity range

The generating model is credible as a *shape* of execution-time distribution,
within these bounds:

- **Calm-regime mean total must be below the period.** `TraceConfig` rejects a
  configuration where it is not, with the message naming both numbers, because
  otherwise every iteration overruns and the dataset is degenerate.
- **Positive, right-skewed, finite-variance stage durations.** Gamma has all
  three. A real loop with a hard floor and a bimodal tail (cache hit vs miss,
  for instance) is not gamma, and the model does not pretend to be.
- **Two regimes.** A real system has as many regimes as it has interfering
  activities. Two is the smallest number that produces autocorrelation.
- **Stage independence given the regime.** The four stages are drawn
  independently within an iteration. In a real loop they share a cache and a
  memory bus, so they are not independent, and the model's per-iteration
  variance is therefore an underestimate of a real one's.
- **Stationary parameters.** No thermal drift, no workload phase change, no
  slow degradation. A trace from a long real run would not be stationary.
- **No coupling to the control law.** The durations are generated
  independently of what the loop is computing. In a real loop a saturated
  actuator, a filter reset or a mode change costs time, so duration and
  controller state are correlated. Nothing here captures that.

---

## 6. Regeneration

```bash
cd products/P031

# one trace, from Python
PYTHONPATH=src python3 -c "
from hilforge.predict import TraceConfig, generate_trace
cfg = TraceConfig.preset('bursty', n_iterations=40000)
t = generate_trace(cfg, seed=4242)
print(t.stage_s.shape, t.utilisation, t.overruns().mean())
"

# overrun accounting for a generated trace, from the CLI
PYTHONPATH=src python3 -m hilforge overruns --preset bursty --iterations 5000 --seed 99

# the full predictor benchmark over both families and three seeds
PYTHONPATH=src python3 validation/predictor_benchmark.py
```

`tests/test_regression.py::test_pinned_trace_statistics` pins, for seed 99 and
5000 iterations:

| Family | mean duration [s] | direct overruns | cascade overruns | max consecutive cascade |
|---|---|---|---|---|
| `jittery` | 7.0087972905e-03 | 764 | 1109 | 10 |
| `bursty` | 7.7282903634e-03 | 684 | 1854 | 413 |

to 14 significant figures on the mean, so a change in the generator surfaces as
a test failure rather than as a quietly different benchmark.

A second, separate trace is generated by
`validation/overrun_handcount.py` for the independent cross-check with P036
RtClock: 1200 iterations, PCG64 seed 36031, `gamma(5, 0.0062/5)` plus
`Bernoulli(0.09) × exponential(0.0075)`, rounded to whole nanoseconds and
written out in full in `validation/crosscheck_overruns.json` so P036 does not
have to run this code to reproduce it.

---

## 7. Limitations, stated before any result derived from this data

1. **Synthetic.** Nothing here is a measurement. The strongest claim any
   result on this data supports is "under this model of latency, this
   predictor behaves like this".
2. **Not from hardware.** No Jetson, no board, no embedded target. This is the
   reason the product is Level 3, hardware-pending rather than Level 4.
3. **Not from a real loop at all**, not even a workstation one. The obvious
   next step — record `stage_s` from an actual run of
   `HilLoop(SimulatedBackend(...))` with wall-clock timing and benchmark the
   predictors on *that* — was not taken, because on a shared contended core the
   recorded latencies would be dominated by other agents' scheduling and would
   be a trace of the container rather than of the loop. That is a real gap and
   it is the first thing to do on a quiet machine.
4. **The model's own structure is the thing being exploited.** The learned
   model's single dominant feature (`total_max8_s`, `MODEL_CARD.md` §7d) is a
   rolling maximum, which is the most direct possible read-out of the regime
   variable the generator uses. A predictor doing well here is partly
   recovering the generator, not only learning a transferable pattern.
5. **Two families is not a survey.** There is no family with periodic
   interference, no family with a heavy-tailed regime dwell time, no family
   with non-stationary parameters. The conclusions are about these two.
6. **The interrupt spike is added to a uniformly random stage**, which is not
   how a real pre-emption lands — it hits whatever is running, which is biased
   toward the longest stage. The effect on the totals is the same; the effect
   on the per-stage features is not, and the per-stage features are among the
   least important ones (§7d of the model card), so this is a small error in a
   place that barely matters. It is listed because it is a known
   simplification, not because it was measured to be harmless.

---

## 8. References

- Fischer, W. and Meier-Hellstern, K. (1993). "The Markov-modulated Poisson
  process (MMPP) cookbook." *Performance Evaluation* 18(2):149-171.
- Harchol-Balter, M. (2013). *Performance Modeling and Design of Computer
  Systems.* Cambridge University Press.
- Ross, S. M. (2014). *Introduction to Probability Models*, 11th ed. Academic
  Press.

---

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
