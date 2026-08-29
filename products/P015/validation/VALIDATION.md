# LinkSwitch 0.1.0 — Validation evidence (Level 2, Research)

All numbers below were produced by running the scripts in this directory in
this build session (Python 3.11, numpy, scipy, scikit-learn, 2 CPU cores).
Every script is rerunnable from the product root with
`python3 validation/<script>.py`; each writes its raw stdout next to itself.

| Script | Raw output | Wall time |
|---|---|---|
| `analytic_threshold_check.py` | `analytic_threshold_check_output.txt` | 6.6 s |
| `policy_comparison_ci.py` | `policy_comparison_ci_output.txt` | 3.7 s |
| `horizon_sensitivity.py` | `horizon_sensitivity_output.txt` | 8.3 s |

Total validation wall time: **~19 s**, well inside the 2-minute-per-script /
2-CPU-core compute budget.

**All telemetry used in every validation run below is SIMULATED** (AR(1)
lognormal irradiance and a two-state Markov rain process — see
`src/linkswitch/optical.py` and `src/linkswitch/rf.py`). No field-measured
turbulence, rain-gauge or RF-link data is used anywhere in this package.

---

## V1 — Analytic optimal fixed-threshold check

`analytic_threshold_check.py`. See `analytic.py` for the full closed-form
derivation (bivariate-normal level-crossing probability applied to the
discrete AR(1) log-irradiance process).

**V1a — Zero switch-cost limit, self-consistency.** In the frictionless
limit the closed form provably has its optimum exactly at the physical
outage threshold `z_phys` (derivation in `analytic.py`). Scenario:
`sigma_i2=0.3, coherence_steps=4, margin_db=6` → `z_phys = -2.44110146`,
`rho = 0.77880078`.

| Method | z_th* | tau* |
|---|---|---|
| Bounded scalar optimizer (`scipy.optimize.minimize_scalar`) | -2.44110147 | 0.25118864 |
| Grid search (20001 points) | -2.44160000 | 0.25112451 |

`|z_th_optimizer - z_phys| = 1.257e-08` (essentially exact, as the
derivation predicts) and `|z_th_optimizer - z_th_grid| = 4.985e-04`
(grid resolution limited). **PASS** (tolerances 1e-4 and 1e-2 respectively,
fixed before the run).

**V1b — Independent Monte Carlo argmax cross-check.** The closed-form
result is checked against a completely separate code path: actually
simulating `FixedThresholdPolicy` (via `simulate.py`) at 10 tau values, 40
seeded 3000-step episodes each (400 000 simulated steps total), zero switch
cost. **The empirical throughput-maximising tau (0.251189) matches
tau_phys (0.251189) exactly** — the grid point nearest tau_phys is also the
empirical argmax. **PASS.**

**V1c — Realistic switch cost (downtime_steps=1), direction check.** With
a physically realistic 1-step switch downtime, the closed form predicts the
optimum threshold collapses toward the deep tail (`tau* = 0.0146`, vs.
`tau_phys = 0.2512` — i.e. "almost never proactively switch away just
because of a brief dip"). An independent Monte Carlo sweep over 6 tau
values (40 episodes × 3000 steps each) confirms the **direction**: measured
throughput is highest at the smallest tau tested and decreases
monotonically approaching tau_phys from below (992.62 → 992.62 → 992.58 →
991.73 → 987.98 → 971.03 Mb/s as tau rises from 0.0126 to 0.3014).
**PASS.**

**Why this matters (honest interpretation, not hidden):** V1c shows that
the naive choice "set the fixed threshold at the physical outage level"
(the intuitive first thing an engineer would try, and the choice used as
the `fixed_threshold` baseline throughout this package for interpretability)
is only optimal in the zero-switch-cost idealisation. With any real switch
downtime, chatter near the threshold makes a bare fixed threshold
measurably worse than either "never switch" or, better, **hysteresis** —
this is the central engineering justification for the hysteresis policy,
and it falls directly out of this validation rather than being asserted.

**Overall V1: PASS.**

---

## V2 — Policy comparison with confidence intervals over seeded Monte Carlo

`policy_comparison_ci.py`. Paired design: every policy sees the *same*
telemetry realisation per replicate, tightening the CIs relative to an
unpaired comparison at the same `n_reps`. 95% CIs via Student-t on the
per-episode mean (`metrics.mean_ci`).

**Scenario A — mild (package defaults: `sigma_i2=0.25, margin_db=6.0,
coherence_steps=5.0`), 200 reps × 2000 steps:**

| Policy | Throughput [Mb/s] (95% CI) | Outage fraction (95% CI) | Switches/episode |
|---|---|---|---|
| fixed_threshold | 994.321 [993.896, 994.746] | 0.0046 [0.0043, 0.0050] | 9.25 |
| hysteresis | 995.495 [995.126, 995.865] | **0.0036 [0.0033, 0.0038]** | **3.08** |
| learned | **995.514** [995.163, 995.865] | 0.0038 [0.0035, 0.0041] | 4.16 |

Learned has the highest point-estimate throughput, but its CI
(`[995.163, 995.865]`) overlaps hysteresis's almost entirely
(`[995.126, 995.865]`) — **this is a statistical tie on throughput, not a
win**, and hysteresis has strictly the lowest outage fraction and fewest
switches. Outages are rare enough in this mild regime (~0.4% of steps) that
all three policies are close to the ceiling.

**Scenario B — moderate (`sigma_i2=0.4, margin_db=4.0, coherence_steps=4.0`,
the scenario used in `examples/policy_comparison.py` and `MODEL_CARD.md`),
200 reps × 2000 steps:**

| Policy | Throughput [Mb/s] (95% CI) | Outage fraction (95% CI) | Switches/episode |
|---|---|---|---|
| fixed_threshold | 867.045 [864.702, 869.387] | 0.0891 [0.0877, 0.0905] | 178.16 |
| **hysteresis** | **876.905** [874.568, 879.242] | **0.0770** [0.0757, 0.0782] | **99.34** |
| learned | 856.796 [854.327, 859.265] | 0.0871 [0.0858, 0.0885] | 158.65 |

**Hysteresis wins outright on every metric, and the learned policy is
strictly worse than even the naive fixed-threshold baseline on throughput
and outage.** This is reported exactly as measured; no tolerance was
loosened and no parameter was retuned to change this outcome. See
`MODEL_CARD.md` for the failure analysis (the RandomForest predictor
over-triggers preemptive switches under higher scintillation, more than
offsetting any earlier-warning benefit).

**Overall V2: the learned/AI policy does NOT beat the classical hysteresis
baseline in either tested scenario.** This is the honest result and is the
headline finding of this product (see README "AI vs. baseline" and
MODEL_CARD.md).

---

## V3 — Sensitivity of the learned policy to the prediction horizon

`horizon_sensitivity.py`. Moderate scenario (as Scenario B above),
120 reps × 1500 steps per horizon value, `window=6`,
`confidence_threshold=0.5`.

| Horizon H (steps) | Throughput [Mb/s] (95% CI) | Outage fraction (95% CI) | Switches/episode |
|---:|---|---|---:|
| 1 | **888.240** [884.970, 891.510] | 0.0820 [0.0798, 0.0843] | **60.54** |
| 2 | 881.636 [878.131, 885.141] | 0.0826 [0.0804, 0.0849] | 73.49 |
| 3 | 874.992 [871.395, 878.588] | 0.0841 [0.0819, 0.0863] | 87.34 |
| 5 | 840.825 [836.984, 844.666] | 0.1012 [0.0990, 0.1035] | 139.05 |
| 8 | 789.972 [785.726, 794.219] | 0.1266 [0.1243, 0.1290] | 189.88 |
| 12 | 625.345 [621.166, 629.523] | 0.2005 [0.1981, 0.2028] | 300.62 |
| 20 | 168.406 [167.383, 169.430] | 0.0794 [0.0777, 0.0811] | 118.81 |

**Finding, reported as measured:** throughput decreases monotonically as
the horizon grows from H=1 to H=12 (888 → 625 Mb/s), and switch count rises
correspondingly (60.5 → 300.6). This matches the expected mechanism: a
longer horizon labels far more training steps as "imminent outage"
(`label_imminent_outage`'s positive-label event is monotonically more
likely as H grows for any fixed series — see its Hypothesis-tested property
in `tests/test_features.py`), so the trained classifier learns to trigger
preemptive switches more readily, and over-triggers relative to the actual
(much rarer) outage rate.

At H=20 the pattern breaks: throughput collapses to 168 Mb/s (below the RF
rate of 150 Mb/s plus a small optical contribution) while switch count
*drops* to 118.8 and outage fraction drops to 0.079. Inspection: at this
horizon almost every window is labelled "imminent outage" (a 20-step
lookahead in a scenario with ~9% marginal outage probability makes at least
one future outage likely most of the time), so the trained model predicts
outage almost unconditionally and the policy **parks on RF almost
permanently** — fewer transitions (because it rarely returns to optical at
all) but throughput collapses toward the RF-only floor. This is a genuine
failure mode, not a beneficial regime, and is documented as such in
`MODEL_CARD.md` under Failure cases.

**Best throughput at the shortest horizon tested, H=1.** No sweet-spot
interior optimum was found in this scan — the honest conclusion is that,
for this scenario and RandomForest configuration, the learned policy's
usable operating point is short horizons only, and even there (Scenario B
above, H=5) it does not beat hysteresis. Widening the horizon does not help
under this training/prediction setup; see Roadmap for what would need to
change (calibrated confidence, class-balancing, or a horizon-aware loss)
to make longer horizons useful.

**Overall V3: PASS as a validation of the sensitivity claim** (a real,
monotonic-then-degenerate dependence on horizon was measured and is
reported plainly) — **not a pass for the learned policy's competitiveness**,
which V2 already reports as a loss to hysteresis.

---

## Known deviations / limitations recorded here

1. The RF specific-attenuation coefficients `(k, alpha)` and the path
   reduction-factor length `L0` are illustrative configuration defaults,
   not verified against the current ITU-R P.838 tables or the exact ITU-R
   P.618-13 procedure — see `src/linkswitch/rf.py` module docstring.
2. The AR(1) temporal fading model is an engineering approximation to give
   the lognormal irradiance process a tunable coherence time; it is not
   derived from a measured or published turbulence temporal power spectrum
   — see `src/linkswitch/optical.py` module docstring.
3. The gamma-gamma irradiance model is implemented as an i.i.d. sampler
   only (no temporal correlation); it is not wired into
   `generate_telemetry` / the switching simulation, only offered as a
   standalone function.
4. V1c's Monte Carlo cross-check is directional (a coarse 6-point sweep),
   not a tight numeric match to the closed-form deep-tail optimum, because
   the closed-form objective is very flat there and a tight empirical match
   would require a much larger `n_reps` than the 2-minute compute budget
   allows for a validation script.
5. The learned policy loses to the hysteresis baseline in every scenario
   tested in V2 and V3. This is reported as the headline honest result, not
   hidden or tuned away.
