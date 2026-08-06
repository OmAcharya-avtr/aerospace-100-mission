# TrackForge

**Status:** TESTING · **Class:** flagship · **Validation level:** 3 · **AI:** yes

Pointing-acquisition-tracking (PAT) simulation suite for optical links:
spiral/raster acquisition scans over a Gaussian uncertainty cone, a two-axis
gimbal with jitter and sensor models, PID and LQR pointing loops with a
benchmark harness, and a reinforcement-learned reacquisition policy
benchmarked against two scripted baselines.

---

## Executive overview

Free-space optical links must first *find* the far terminal inside a pointing
uncertainty cone, then *hold* it against platform jitter, and — when a
disturbance breaks the lock — *find it again*, fast. TrackForge simulates that
whole chain end to end and makes each stage measurable:

- **Acquire.** Archimedean-spiral and serpentine-raster scan generators whose
  track spacing is tied to the beam-overlap factor, with Monte Carlo coverage
  verification and an analytic expected-acquisition-time model.
- **Track.** Two-axis gimbal as second-order plants with documented torque,
  rate and acceleration limits; platform jitter synthesised from a target PSD
  by spectral factorisation; a noise-equivalent-angle sensor with quantisation
  and dropout.
- **Control.** PID with derivative-on-measurement and anti-windup, and LQR via
  the algebraic Riccati equation, compared on step response, disturbance
  rejection and measured closed-loop bandwidth — all from actual runs.
- **Reacquire.** A tabular Q-learning policy that chooses between local
  restart, full-cone restart and an expanding-ring search, benchmarked against
  always-local and always-full baselines on identical seeded episodes.

Everything is seeded and bitwise reproducible; 295 tests and seven validation
scripts back the numbers in this README.

## Aerospace problem

An optical inter-satellite or space-to-ground link has a beam divergence of
tens of microradians but an initial pointing knowledge of hundreds of
microradians. The link cannot close until the beam footprint and the far
terminal coincide. The engineering questions are:

1. **How long does acquisition take?** It depends on the scan pattern, the
   track spacing relative to beam width, the dwell time and the detection
   probability. Getting the spacing wrong leaves coverage gaps; getting it too
   tight multiplies the scan time.
2. **How well can the loop hold the line of sight?** Platform micro-vibration
   feeds through the gimbal; loop bandwidth, actuator limits and sensor noise
   set the residual pointing error.
3. **What do you do when you lose lock?** A full-cone re-scan always works but
   is slow. A local re-scan is fast but fails when the disturbance threw the
   line of sight far. The right answer depends on state — which is a
   sequential decision problem, not a fixed rule.

TrackForge lets these be traded quantitatively, before any hardware exists.

## Intended users

- **GNC and pointing engineers** sizing acquisition scans and control-loop
  bandwidth for an optical terminal.
- **Optical communications system engineers** budgeting acquisition and
  reacquisition time into link availability.
- **Researchers** benchmarking learned versus scripted PAT logic on a
  reproducible, fully open testbed.
- **Educators** teaching second-order control, spectral disturbance synthesis
  and tabular reinforcement learning on a physically motivated problem.

This is a simulation tool. It is not a flight-software component, and its
default parameters are illustrative, not design data.

## Engineering theory

All angles in radians, times in seconds, torques in N m, frequencies in Hz.

### Uncertainty region and containment

The target angular offset is an isotropic 2-D Gaussian with per-axis standard
deviation σ, so the radial offset is Rayleigh-distributed (standard result,
e.g. Papoulis & Pillai 2002, *Probability, Random Variables and Stochastic
Processes*):

```
P(r <= R) = 1 - exp(-R^2 / (2 sigma^2))            r(p) = sigma * sqrt(-2 ln(1-p))
```

*Assumptions:* small-angle tangent-plane geometry, isotropic errors.
*Validity:* σ ≲ 10⁻² rad. Verified: `P(r ≤ 3σ) = 0.98889100346` reproduced to
10⁻⁹ in `tests/test_scan.py`.

### Scan track spacing

```
s = 2 * R_beam * (1 - overlap)                     overlap in [0, 1)          (S1)
```

Adjacent tracks leave no radial gap iff `s ≤ 2·R_beam`. The overlap factor is
design margin against along-track motion and jitter — standard PAT scan design
practice (Kaymak et al. 2018, *IEEE Communications Surveys & Tutorials* 20(2),
acquisition section; Hemmati (ed.) 2006, *Deep Space Optical Communications*).
*Measured consequence:* at `overlap = 0` (tangent tracks) the covered mass
falls 1.04 % short of the design containment because of spiral curvature —
see Validation §1.

### Archimedean spiral

```
r(phi) = a * phi ,  a = s / (2 pi)                 radial pitch per turn = s   (S2)
ds     = sqrt(r^2 + a^2) dphi                      arc-length element
```

Dwell points are placed at constant arc-length steps `dl = step_fraction ·
R_beam`. *Assumption:* constant along-track speed. *Validity:* r ≫ a for the
uniform-coverage approximations below.

### Expected acquisition time (uniform-coverage approximation)

A spiral of spacing `s` at along-track speed `v` sweeps area at rate `s·v`, so
reaching radius `r` costs `t(r) ≈ πr²/(s v)`. Averaging over the truncated
Rayleigh density:

```
E[T | r <= r_max] = (pi / (s v)) * E[r^2 | r <= r_max]                        (S3)
E[T]              -> 2 pi sigma^2 / (s v)          as r_max -> infinity
```

**Source:** internal derivation (documented in `scan.py`), validated against
Monte Carlo, not taken from a specific publication. Comparable spiral-scan
acquisition statistics appear in the laser-comm PAT literature cited above; no
page-specific formula is claimed. *Agreement:* −0.65 % vs 20 000-target Monte
Carlo (Validation §2). *Caveat:* the per-pass detection probability in (S3) is
a **per-crossing** probability; using the per-dwell probability naively
overestimates by 38 % — see Validation §2.

### Gimbal axis

Rotation about a fixed axis with viscous damping (Meirovitch 2001,
*Fundamentals of Vibrations*, ch. 1):

```
J * theta_ddot + b * theta_dot = tau                                          (D1)
x = [theta, theta_dot]^T,  A = [[0,1],[0,-b/J]],  B = [0, 1/J]^T              (D2)
```

*Units:* J [kg m²], b [N m s/rad], τ [N m]. *Assumptions:* rigid body, no
cross-axis coupling, no structural modes, viscous (not Coulomb) friction.
*Validity:* small excursions about boresight; loop bandwidth well below the
first structural mode. *Limits applied:* |τ| ≤ τ_max, |θ̇| ≤ rate_max, and an
optional |θ̈| ≤ accel_max. Integration is fixed-step RK4 with zero-order-hold
torque. Verified against the exact solutions `θ = ½(τ/J)t²` (b = 0) and
`θ̇(t) = (τ/b)(1 − e^{−bt/J})` to 10⁻⁶ relative.

### Jitter synthesis

Random-phase spectral factorisation of a target one-sided PSD (spectral
representation of stationary Gaussian processes: Shinozuka & Deodatis 1991,
*Applied Mechanics Reviews* 44(4); Percival & Walden 1993, *Spectral Analysis
for Physical Applications*):

```
X_k  = sqrt(S(f_k) * fs * N / 2) * exp(i phi_k),   phi_k ~ U[0, 2 pi)         (D3)
x[n] = irfft(X)[n]                                 DC and Nyquist bins zeroed
```

so that the periodogram `2|rfft(x)_k|²/(fs N)` has expectation `S(f_k)`.
The shipped parametric shape is

```
S(f) = S0 / (1 + (f/f_c)^2)^(order/2)   [rad^2/Hz]                            (D4)
```

a flat plateau rolling off above `f_c`. *Source:* (D4) is a generic empirical
shape for platform micro-vibration; the specific parameters are user-supplied
and **not** taken from any mission. *Verified:* band-median PSD ratio within
1.7 % and variance within 0.081 % of `∫S df` (Validation §4).

### Sensor

Additive white Gaussian noise of standard deviation NEA (noise-equivalent
angle), optional LSB quantisation, optional dropout returning the last valid
sample flagged invalid. NEA is the conventional figure of merit for optical
tracking sensors (Hemmati (ed.) 2006).

### PID

Discrete PID with derivative-on-measurement and conditional-integration
anti-windup (Åström & Hägglund 2006, *Advanced PID Control*, ch. 3):

```
u = clip( Kp e + I + (-Kd (y[k]-y[k-1])/dt) , -u_max, u_max)                  (C1)
I updated only if the update does not push a saturated output further out
```

Tuning rule used throughout (treating the plant as `1/(J s²)`, valid when
`b/J ≪ ω_n`):

```
Kp = J wn^2 ,  Kd = 2 zeta J wn ,  Ki = alpha wn Kp   (alpha ~ 0.1)           (C2)
```

Because the derivative acts on the measurement, PD control gives a closed loop
with **no numerator zero**, i.e. exactly the canonical second-order system:

```
wn = sqrt(Kp/J) ,  zeta_eff = (Kd + b) / (2 sqrt(Kp J))
Mp = exp(-pi zeta / sqrt(1-zeta^2))        tp = pi / (wn sqrt(1-zeta^2))      (C3)
```

(Ogata 2010, *Modern Control Engineering*, 5th ed., ch. 5). *Verified* to
≤1.43 % relative on Mp/t_p/t_r and 0.12 % of the step pointwise
(Validation §3).

### LQR

Infinite-horizon LQR on the exact linear model (D2) with cost
`∫ (xᵀQx + uᵀRu) dt`, gain `K = R⁻¹BᵀP` where P solves the continuous-time
algebraic Riccati equation (Anderson & Moore 1990, *Optimal Control: Linear
Quadratic Methods*), via `scipy.linalg.solve_continuous_are` (or
`solve_discrete_are` with a zero-order-hold discretisation):

```
A^T P + P A - P B R^-1 B^T P + Q = 0                                          (C4)
```

**Linearisation statement:** the plant (D2) is already linear and
time-invariant — there is nothing to linearise. The only nonlinearities in the
simulated system are the torque/rate/acceleration saturations, which the LQR
design *ignores*; the controller is applied with output clipping. This is the
documented modelling deviation. For the double integrator with `Q = diag(q,0)`
the closed loop follows the Butterworth root-square locus, giving the weight
rule

```
r = q / (J^2 wn^4)   =>   |poles| = wn ,  zeta = sqrt(2)/2                    (C5)
```

*Verified* to 2.6·10⁻¹² relative pole error (Validation §3C).

### Reacquisition MDP

State `(t since loss, |last-known offset|, σ(t), r_searched)`; actions
`{LOCAL, FULL, RING}`; reward = −(attempt duration); tabular Q-learning
(Watkins & Dayan 1992; Sutton & Barto 2018, sec. 6.5):

```
Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]               (C6)
```

The loss-of-lock statistics are a **modelling choice of this simulator**, fully
described in `DATASET_CARD.md`, not a published model.

## Architecture

```
                     ┌──────────────────────────────────────────────────┐
                     │             python -m trackforge                 │
                     │      run <scenario.yaml> | benchmark | reacq     │
                     └───────────────────────┬──────────────────────────┘
                                             │ argparse CLI
                                             v
  scenario.yaml ───────────>┌───────────────────────────────────┐
   (validated, unknown      │        trackforge.sim             │
    keys rejected)          │  Scenario · run_episode           │
                            │  run_monte_carlo · steps/sec      │
                            └───┬───────┬───────────┬───────┬───┘
              ┌─────────────────┘       │           │       └────────────────┐
              v                         v           v                        v
  ┌────────────────────┐  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │ trackforge.scan    │  │trackforge.dynamics│ │trackforge.control│ │ trackforge.reacq │
  │                    │  │                  │ │                  │ │      (AI)        │
  │ GaussianUncertainty│  │ GimbalAxis       │ │ PIDController    │ │ ReacqEnv         │
  │ spiral_scan        │  │ TwoAxisGimbal    │ │  (anti-windup)   │ │ AlwaysFullPolicy │
  │ raster_scan        │  │ JitterPSD        │ │ LQRController    │ │ AlwaysLocalPolicy│
  │ coverage_fraction  │  │ synthesize_jitter│ │  (CARE / DARE)   │ │  ^ baselines     │
  │ simulate_acquisit. │  │ welch_psd        │ │ step_response    │ │ QLearningPolicy  │
  │ E[T] analytic      │  │ AngleSensor      │ │ dist_rejection   │ │  + confidence    │
  │                    │  │  (NEA/quant/drop)│ │ bandwidth_est.   │ │ train_q_learning │
  └────────────────────┘  └──────────────────┘ └──────────────────┘ │ evaluate_policy  │
                                                                     └──────────────────┘

  ── episode data flow ─────────────────────────────────────────────────────────
     ①  ACQUIRE      scan pattern  ×  Gaussian target  ×  per-dwell detection
                                    │  t_acquire
                                    v
     ②  TRACK        gimbal ⊕ jitter(PSD) ─> sensor(NEA) ─> controller ─> torque
                                    │  LOS error time series, RMS, peak
                                    v
     ③  LOSE LOCK    disturbance spike drives |error| > threshold for hold time
                                    │  t_loss
                                    v
     ④  REACQUIRE    policy.act(state) ─> {LOCAL | FULL | RING} ─> success?
                                    │  t_reacquire, attempts, success
                                    v
                              metrics / EpisodeResult
```

Package layout:

```
products/P002/
├── src/trackforge/     scan.py  dynamics.py  control.py  reacq.py  sim.py  __main__.py
├── tests/              10 modules, 295 tests
├── examples/           3 runnable scripts + 2 YAML scenarios
├── screenshots/        3 PNGs, produced by running the examples
├── validation/         7 rerunnable scripts + raw outputs + VALIDATION.md
├── docs/REQUIREMENTS.md
├── MODEL_CARD.md  DATASET_CARD.md  CHANGELOG.md  LICENSE  pyproject.toml
```

## Installation

Python 3.11+. Dependencies: numpy, scipy, matplotlib, pyyaml (plus pytest and
hypothesis for the test suite). No PyTorch, no network access, no GPU.

```bash
cd products/P002
pip install -e .          # or: pip install -e ".[test]"
```

Or run in place without installing:

```bash
export PYTHONPATH=src
python -m trackforge --version
```

## Quick start

```python
from trackforge.sim import Scenario, run_episode

res = run_episode(Scenario(name="demo", controller="lqr"), seed=99)
print(res.summary())
# acquisition_time_s 2.742 s · track_rms_rad 1.41e-06 · lock_lost True at 1.184 s
# reacq_time_s 2.229 s in 4 attempts · total_time_s 6.155 s
```

Command line:

```bash
python -m trackforge run examples/scenario_leo_downlink.yaml
python -m trackforge run examples/scenario_high_jitter.yaml --json
python -m trackforge benchmark
python -m trackforge reacq --episodes 20000 --seed 12345
```

## Configuration

Scenarios are YAML files validated against the `Scenario` dataclass; **unknown
keys raise an error** rather than being silently ignored, and out-of-range
values raise `ValueError` with an actionable message. Two fully commented
examples ship in `examples/`:

| File | Purpose |
|------|---------|
| `examples/scenario_leo_downlink.yaml` | LEO downlink coarse-pointing stage, PID loop, always-local reacquisition |
| `examples/scenario_high_jitter.yaml` | High-jitter platform, LQR loop, sensor dropout and quantisation, full-cone reacquisition |

Key groups: simulation timing (`dt`, `track_duration`), acquisition
(`sigma_uncertainty`, `beam_radius`, `overlap`, `containment`, `dwell_time`,
`p_dwell`, `pattern`), gimbal (`inertia`, `damping`, `torque_max`, `rate_max`,
`accel_max`), control (`controller`, `bandwidth_hz`, `damping_ratio`,
`integral_alpha`, LQR weights), disturbance and sensor (`jitter_*`, `spike_*`,
`nea`, `sensor_dropout`, `quantization`), lock logic (`track_threshold`,
`loss_hold_s`) and reacquisition (`reacq_policy`, `reacq` overrides).

## Examples

All three scripts were executed to produce the committed PNGs.

| Script | Output | Shows |
|--------|--------|-------|
| `examples/ex01_scan_patterns.py` | `screenshots/ex01_scan_patterns.png` | Spiral (9 987 dwells, 9.99 s) and raster (13 199 dwells, 13.20 s) over a σ = 300 µrad cone, with 1/2/3-σ contours, plus coverage and dwell cost vs overlap |
| `examples/ex02_tracking_error.py` | `screenshots/ex02_tracking_error.png` | LOS error under jitter for PID and LQR, the loss-of-lock spike at 1.2 s, commanded torque (peak 62.8 mN m = 3.14 % of the limit), and target-vs-realised PSDs showing ≈2 decades of rejection at 0.6 Hz |
| `examples/ex03_reacq_comparison.py` | `screenshots/ex03_reacq_comparison.png` | Mean time-to-reacquire with 95 % CIs, survival curves, and action mix for both baselines and the learned policy |

```bash
python examples/ex01_scan_patterns.py
python examples/ex02_tracking_error.py
python examples/ex03_reacq_comparison.py
```

## Validation

Full evidence, method descriptions, tolerances and uncertainty analysis:
[`validation/VALIDATION.md`](validation/VALIDATION.md). Raw script output is
committed alongside each script. Every number below came from running these
scripts in the build session.

| # | Check | Reference | Result |
|---|-------|-----------|--------|
| 1 | Spiral coverage vs geometric track-spacing argument | geometric derivation (s ≤ 2R ⇒ no radial gap) | Covered mass **0.99485–0.99529** vs design 0.995 for overlap ≥ 0.10; negative control (gapped pattern) **0.49623**. **Reported shortfall:** at overlap = 0 coverage is **0.98460**, 1.04 % below design, due to spiral-curvature gaps between tangent tracks — PASS |
| 2 | Acquisition time, Monte Carlo vs analytic | internal derivation (S3), 20 000 targets | MC **1.8226 s** vs analytic **1.8344 s** → **−0.65 %**. With p_dwell = 0.9 and the corrected per-crossing probability: −0.56 %. **Reported finding:** using the per-dwell probability naively gives **−38 %** disagreement — PASS |
| 3 | PD/PID/LQR step response vs analytic second-order | Ogata 2010 ch. 5; Anderson & Moore 1990 | Worst relative deviation on Mp/t_p/t_r **1.43 %**; pointwise trajectory error **0.1155 % of the step**; LQR pole placement error **2.6·10⁻¹²** — PASS |
| 4 | Jitter synthesis PSD vs target | Shinozuka & Deodatis 1991; Welch estimator | Band-median ratio within **1.66 %**; variance vs ∫S df within **0.081 %**; ensemble mean converges to 1 as realisations are added — PASS |
| 5 | Learned reacquisition policy vs both baselines | scripted baselines, common random numbers, 2 000 held-out episodes | Learned **4.148 s** [3.775, 4.521] vs always-local **6.440 s** [5.961, 6.918] and always-full **8.649 s** [8.293, 9.004]; **33.8 %** and **50.7 %** reductions, CIs disjoint, across 3 training seeds — PASS |
| 6 | Performance / compute budget | mission 180 s limit | **36 284 steps/s** (7.3× realtime); training **2.97 s** (1.6 % of budget); MC evaluation **0.73 s** — PASS |
| 7 | Regression baseline | 28 pinned seeded values | All 28 reproduce; tolerances 10⁻⁹ relative (10⁻⁶ for the LQR gain) — PASS |

No validation check failed. Two checks required the *analysis* to be corrected
rather than the tolerance to be loosened (items 1 and 2); both corrections are
documented in `validation/VALIDATION.md` and reflected in the source
docstrings.

**Uncertainty analysis** (VALIDATION.md §8) covers Monte Carlo sampling error,
time discretisation, spectral-estimator variance, model-form error, training
stochasticity, floating-point reproducibility and censoring. What is *not*
quantified is model **validity** error: nothing here has been compared against
hardware or mission telemetry.

## Benchmark results

Controller comparison, from `python -m trackforge benchmark` (J = 0.05 kg m²,
b = 0.02 N m s/rad, τ_max = 2 N m, 5 Hz design bandwidth, dt = 2·10⁻⁴ s,
open-loop disturbance RMS 2.112·10⁻⁶ rad):

| controller | rise time [s] | overshoot | settling time [s] | closed-loop RMS [rad] | rejection factor | −3 dB BW [Hz] |
|------------|---------------|-----------|-------------------|------------------------|------------------|---------------|
| PID | 0.0590 | 0.1761 | 0.6450 | 1.347·10⁻⁶ | 1.567 | 4.955 |
| PD (Ki = 0) | 0.0688 | 0.0402 | 0.1884 | 1.310·10⁻⁶ | 1.612 | 4.921 |
| LQR | 0.0682 | 0.0426 | 0.1890 | 1.315·10⁻⁶ | 1.605 | 4.962 |

Reading: the integral term removes steady-state error at the cost of 4× the
overshoot and 3.4× the settling time. LQR at the same design bandwidth is
practically indistinguishable from well-tuned PD on this second-order plant —
expected, since both span the same pole locations. The rejection factor is
modest (≈1.6) because most jitter power in this PSD sits *above* the 5 Hz loop
bandwidth; below it, rejection is ≈2 decades (see
`screenshots/ex02_tracking_error.png`, panel 3).

Reacquisition benchmark (2 000 held-out episodes, common random numbers):

| policy | mean [s] | 95 % CI [s] | median [s] | p90 [s] | success | attempts |
|--------|----------|-------------|------------|---------|---------|----------|
| baseline always-full | 8.6485 | [8.2926, 9.0044] | 5.2360 | 30.000 | 0.877 | 1.685 |
| baseline always-local | 6.4395 | [5.9612, 6.9179] | 0.5762 | 30.000 | 0.838 | 4.716 |
| **learned (Q-learning)** | **4.1480** | **[3.7752, 4.5209]** | 0.5001 | 8.814 | **0.909** | 3.427 |

Simulation throughput: **36 284 closed-loop steps/s**, 7.3× realtime at
dt = 2·10⁻⁴ s, on 2 CPU cores.

Test suite: **295 tests, 295 passed, 0 failed, 0 skipped** in ≈70 s
(`python -m pytest tests/ -q`), spanning unit, input-validation,
known-answer, property-based (Hypothesis), integration, regression,
performance, failure-mode, configuration and reproducibility categories.

## AI model details

Full details: [`MODEL_CARD.md`](MODEL_CARD.md) and
[`DATASET_CARD.md`](DATASET_CARD.md).

- **Baseline first.** `AlwaysFullPolicy` and `AlwaysLocalPolicy` are
  implemented, evaluated and reported *before* the learned policy, on
  identical seeded episodes.
- **Dataset.** There is no collected dataset — the environment *is* the
  simulator. Episodes are generated deterministically from seeds; nothing is
  stored on disk. The loss-of-lock statistics are a documented modelling
  choice, not measured data (`DATASET_CARD.md` §3).
- **Architecture.** Tabular Q-learning in numpy; 320 discrete states × 3
  actions (≈15 kB). PyTorch is unavailable in this environment, so no deep-RL
  variant exists — see Limitations.
- **Training.** 20 000 episodes, γ = 0.99, α 0.30→0.02, ε 1.00→0.05, reward =
  −(attempt duration), seed 12345, **2.97 s** wall time.
- **Test split.** By seed: evaluation episodes are re-seeded to
  `999 + i` (i = 0…1999), disjoint from the training stream and identical
  across all compared policies. Three training seeds (12345, 20260, 777) are
  evaluated on the same held-out episodes.
- **Metrics.** Mean time-to-reacquire **4.148 s** [3.775, 4.521] vs 6.440 s
  (always-local) and 8.649 s (always-full): **33.8 %** and **50.7 %**
  reductions with disjoint 95 % CIs; success rate 0.909 vs 0.838 / 0.877.
- **Uncertainty output.** `act_with_confidence(state) -> (action, confidence)`
  returns a margin × support score in [0, 1]; unvisited states fall back to the
  `FULL` baseline action. Measured: 64/320 states visited, mean confidence
  0.204 on visited states. **This is a heuristic score, not a calibrated
  probability.**
- **Failure cases.** Distribution shift outside the trained `ReacqConfig`;
  state aliasing at bin boundaries; 80 % of the tabulated states unvisited;
  censored objective; ±3.4 % training-seed variance; no safety envelope.
  Detailed in `MODEL_CARD.md` §9.
- **Reproducibility.** `python -m trackforge reacq --episodes 20000 --seed
  12345 --eval-episodes 2000 --eval-seed 999` reproduces the table; identical
  seeds give bitwise identical Q-tables.

**This model is not certified for operational flight use.**

## Hardware requirements

- CPU: any x86-64 or arm64; the reference figures were measured on **2 cores**.
- RAM: < 500 MB for every shipped workflow (a 2²⁰-sample jitter series is
  8 MB; the Q-table is 15 kB).
- Disk: < 5 MB installed, plus ~0.7 MB of committed PNGs.
- No GPU, no network access, no special instructions required.
- Full test suite ≈70 s; all validation scripts together ≈2 minutes.

## Limitations

**Modelling**

1. **Coarse stage only.** No fine-steering-mirror / two-stage coarse-fine
   architecture; a real terminal would close a fast inner loop on an FSM.
2. **Decoupled axes.** Azimuth and elevation are independent second-order
   systems. No gimbal kinematics, gimbal-lock geometry, Coriolis or
   cross-coupling terms.
3. **Rigid body, viscous damping only.** No structural flexible modes, no
   Coulomb friction, no cogging, no backlash, no motor electrical dynamics.
4. **No optical physics.** Detection is an abstract per-dwell Bernoulli trial:
   no link budget, detector noise, background light, atmospheric turbulence or
   scintillation. TrackForge deliberately does not model channel effects.
5. **Illustrative parameters.** Every default (inertia, PSD level, beam
   radius, drift rate) is an order-of-magnitude placeholder, not design data
   for any terminal. Nothing has been compared against hardware or telemetry.
6. **Jitter PSD shape assumed.** The flat-then-roll-off model (D4) is generic;
   real micro-vibration spectra contain discrete reaction-wheel and cryocooler
   lines that this model does not reproduce.
7. **Loss-of-lock statistics assumed.** Eqs. (10)–(12) in `reacq.py` are a
   modelling choice designed to create a realistic decision trade-off, not a
   published or measured model. See `DATASET_CARD.md` §3 and §6.
8. **Uniform-coverage timing.** Reacquisition attempt durations use the same
   uniform-coverage approximation shown in Validation §2 to be ≈0.6 %
   optimistic; the inner-turn regime is where it is weakest.
9. **Mid-point coverage test.** During a reacquisition attempt the target
   position is evaluated once, at the attempt mid-time, not continuously.
10. **Timeout convention.** An attempt in progress is never aborted, so
    `EpisodeResult.reacq_time_s` can exceed `ReacqConfig.max_time`.
    `evaluate_policy` censors at `max_time`; compare policies on the censored
    statistic.
11. **LQR ignores saturation.** The Riccati design uses the unconstrained
    linear model; limits are imposed by clipping the output. No
    constrained/MPC formulation.

**Machine learning**

12. **PyTorch unavailable.** The build environment has no PyTorch, so the RL
    component is a numpy tabular Q-learner rather than any deep-RL method.
    This is a deliberate, documented deviation from any assumption that an RL
    library would be used. It also has an upside: the tabular method is
    bitwise reproducible.
13. **Small discrete state space.** 320 bins with fixed edges tuned for the
    default configuration; not adaptive, and 256 of them are never visited in
    a 20 000-episode run.
14. **Confidence is uncalibrated.** The margin × support score has not been
    checked against realised outcome frequencies.
15. **Single-configuration training.** A policy trained on one `ReacqConfig`
    should not be applied to another without retraining.

**Software**

16. **No parallelism.** Everything is single-threaded pure Python/numpy;
    throughput is ≈36 k steps/s, which bounds practical Monte Carlo sizes.
17. **No persistence format.** Learned policies are in-memory objects; there
    is no serialisation format in v0.1 (retraining takes 3 s, so this is a
    deliberate omission).
18. **No 3-D geometry.** All scan and pointing geometry is in a 2-D tangent
    plane; valid only for small uncertainty cones (≲ 0.05 rad).

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

- **0.2** — two-stage coarse/fine architecture with a fine-steering mirror;
  coupled two-axis gimbal kinematics; policy serialisation.
- **0.3** — structural flexible modes and reaction-wheel spectral lines in the
  jitter model; constrained control (saturation-aware LQR / MPC).
- **0.4** — calibrated uncertainty for the reacquisition policy (conformal or
  bootstrap intervals on time-to-reacquire); function-approximation RL if a
  suitable library becomes available.
- **0.5** — validation against published PAT acquisition-time measurements,
  replacing the currently internal analytic references.

## License

AGPL-3.0-only. Full text in [`LICENSE`](LICENSE). Copyright © 2026 OPTIMA
Organisation.

## Credits

Developed by the OPTIMA Organisation aerospace software portfolio programme.
This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```bibtex
@software{trackforge_2026,
  title        = {TrackForge: a pointing-acquisition-tracking simulation suite
                  for optical links},
  author       = {{OPTIMA Organisation}},
  version      = {0.1.0},
  year         = {2026},
  license      = {AGPL-3.0-only},
  note         = {Product P002 of the OPTIMA aerospace software portfolio.
                  Research-grade; not flight-qualified.}
}
```

Key references used in this package:

- Kaymak, Y. et al. (2018). "A Survey on Acquisition, Tracking, and Pointing
  Mechanisms for Mobile Free-Space Optical Communications."
  *IEEE Communications Surveys & Tutorials*, 20(2).
- Hemmati, H. (ed.) (2006). *Deep Space Optical Communications*. Wiley.
- Ogata, K. (2010). *Modern Control Engineering*, 5th ed. Prentice Hall.
- Åström, K. J. and Hägglund, T. (2006). *Advanced PID Control*. ISA.
- Anderson, B. D. O. and Moore, J. B. (1990). *Optimal Control: Linear
  Quadratic Methods*. Prentice Hall.
- Franklin, G. F., Powell, J. D. and Workman, M. (1998). *Digital Control of
  Dynamic Systems*, 3rd ed. Addison-Wesley.
- Meirovitch, L. (2001). *Fundamentals of Vibrations*. McGraw-Hill.
- Shinozuka, M. and Deodatis, G. (1991). "Simulation of Stochastic Processes
  by Spectral Representation." *Applied Mechanics Reviews*, 44(4).
- Percival, D. B. and Walden, A. T. (1993). *Spectral Analysis for Physical
  Applications*. Cambridge University Press.
- Papoulis, A. and Pillai, S. U. (2002). *Probability, Random Variables and
  Stochastic Processes*, 4th ed. McGraw-Hill.
- Watkins, C. J. C. H. and Dayan, P. (1992). "Q-learning."
  *Machine Learning*, 8(3–4).
- Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction*, 2nd ed. MIT Press.
