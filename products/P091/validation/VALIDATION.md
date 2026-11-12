# Validation — thermalbudget

**Validation level: not claimed here.** The registry records
`validation_level: 4` for this product's slot as a TARGET. The achieved level is
the mission coordinator's decision at Phase 9, from the evidence below.
`thermalbudget.VALIDATION_LEVEL` and the README badge still read
"hardware-measurement pending at build time" and have deliberately **not** been
updated by this session, because setting the level is not this package's call.
The build-session wording is stale; this file is the current state.

## What changed, and when

The hardware campaign finished after the package was built. There were two
distinct states and both are on the record:

- **At build time** `data/traces/` was empty. Every measured validation script
  refused by name and exited non-zero rather than falling back to the synthetic
  generator. No figure in the repository was measured on the board.
- **Now** `data/traces/` holds all 17 CSV traces plus `manifest.json`,
  transferred from the board with a verified sha256 at both ends. Every script
  below has been re-executed against them.

## What is and is not hardware-backed, by name

| Claim | State |
|---|---|
| The headline overstatement, the sustained and burst throughputs, the steady-state die temperature, every tau, the clock-versus-temperature relation, the contamination census, the equilibrium verdicts, the learned-model scores | **MEASURED** on an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (`manifest.json` `model` matches the declared target), power mode MAXN_SUPER, one uninstrumented ambient, CPU-only payload, soaks of at most 720 s, one board with its own heatsink and fan. |
| "The GPU was idle" | **PARTLY MEASURED, and the gap is named.** The devfreq frequency was recorded and sat at its 306 MHz floor in 4954 of 4975 samples. **GR3D utilisation is not a column in the trace schema and was never sampled**, so GPU idleness is an inference from the devfreq floor, not a measurement. |
| Any figure at a power mode other than `MAXN_SUPER` — by name 10W, 15W, 25W, MAXN, any custom `nvpmodel` profile | **UNMEASURED.** `nvpmodel` needs root and the campaign account has no passwordless sudo. Refused by `thermalbudget.limits.refuse_unmeasured("power_mode_other")`. |
| Any figure at a second ambient | **UNMEASURED.** Refused by name. Ambient was never instrumented in degrees; the idle-run die temperature, 44.489 degC, is its only proxy. |
| A held-out-AMBIENT score for the learned predictor | **STRUCTURALLY UNMEASURABLE.** One ambient, no second point to hold out. Not approximated by the duty-cycle split. |
| Any figure under GPU load, for another payload, for other hardware, for a different fan curve, or for a soak longer than 720 s | **UNMEASURED.** Each refused by name; run `python -m thermalbudget limits`. |
| Every figure in `validate_loader_contract`, `validate_equilibrium`, `validate_independent_bound`, `validate_negative_controls`, `worked_example` | **SYNTHETIC.** These are checks of the estimators and of the code, computed from `tests/fixtures/synthetic_trace.py`, and are labelled SYNTHETIC in their own output. They establish nothing about the board. |

## How to reproduce everything below

```bash
python -m pytest tests/ -q -p no:cacheprovider --junit-xml=junit.xml
python validation/validate_measured_campaign.py
python validation/validate_measured_learned.py
python validation/validate_clock_confound.py
python validation/validate_loader_contract.py
python validation/validate_equilibrium.py
python validation/validate_independent_bound.py
python validation/validate_negative_controls.py
python validation/validate_approx_audit.py
python validation/validate_property_hunt.py
python validation/worked_example.py
python validation/validate_tree_hygiene.py
python validation/probe_working_directories.py
python examples/thermal_rise.py
python examples/overstatement.py
python examples/clock_relation.py
python examples/learned_vs_rc.py
```

Each script writes its raw output beside itself as
`validation/<script-name>_output.txt`. Every number in this file and in
`README.md` comes from one of those files and the file is named beside the
number.

## The measured headline

From `validate_measured_campaign_output.txt`:

```
  burst      26160.398 iters/s
  sustained  25741.404 iters/s
  sustained as a fraction of burst  0.983984 [0.955584, 1.010990]
  cold-burst overstatement          +1.6277 per cent [-1.0870, +4.6481] per cent
  VERDICT: INDISTINGUISHABLE
```

Steady-state die temperature **56.7085 [56.4813, 56.9959] degC**, half-width
0.2573. Sustained throughput half-width 146.382, standard error 75.178, bias
-1.252, over 27 distinct run-level resample multisets from three runs. The
independent Student-t interval over the per-run means is
[25553.428, 25929.380] iters/s and the bootstrap point estimate lies inside it.

## Why the headline came out that way: the board did not throttle

From `validate_measured_campaign_output.txt` and
`validate_clock_confound_output.txt`:

- **0 of 9654** per-core clock samples across the three duty-1.0 soaks sit below
  `cpuinfo_max_freq` 1728000 kHz, over a die rise of **10.94 degC** (46.34 to
  57.28).
- The pooled clock channel varies by **0.0000 per cent** of its maximum over
  45.875–57.281 degC, across **one** distinct level, so
  `clock_temperature_relation` reports **DEGENERATE**.
- The governor's resolution is not the limiting factor: the campaign used **14
  distinct clock levels** from 729600 to 1728000 kHz and the smallest gap
  between adjacent levels is **76800 kHz**, which is **8.9 times** the 0.5 per
  cent identifiability floor. A real throttle step could not hide under it.

**Consequence: baseline 2 reduces to baseline 1**, because the
clock-to-throughput chain has no thermal signal to act on. That reduction is a
measured property of this board under this payload, **not** an implementation
defect.

## The clock-versus-temperature relation is confounded by duty cycle

`examples/clock_relation.py` fits a pooled slope near +46000 kHz/degC across the
whole campaign. **Read alone that invites exactly the wrong conclusion** — that
the clock was measured falling as the die warmed. It was not. Duty cycle drives
both the clock, through the `schedutil` governor responding to load, and the die
temperature, through dissipated power, so a pooled fit crosses a common cause.

`validate_clock_confound.py` decomposes it (3 checks, 0 failed):

| group | n | die range, degC | clock range, kHz | slope, kHz/degC |
|---|---:|---|---|---|
| duty 1.00 | 1644 | 46.34 – 57.28 | 1728000 – 1728000 | **EXACTLY ZERO** |
| duty 0.75 | 468 | 46.25 – 54.41 | 1676800 – 1728000 | -41.6 |
| duty 0.50 | 468 | 46.03 – 50.38 | 1011200 – 1728000 | +5293.3 |
| duty 0.25 | 406 | 45.78 – 46.44 | 729600 – 1574400 | +28201.2 |
| **pooled** | 2986 | | | **+46110.5** |

The pooled figure is generated **entirely between duty groups**. The slope that
bears on throttling is the within-duty one, and at the only duty cycle where
throttling could act it is exactly zero.

`examples/clock_relation.py` now colours its scatter by duty, draws the
within-duty fits, and labels the pooled line CONFOUNDED. It applies the same
2 s skip as this script so the two report identical within-duty slopes. The
pooled figures differ between the two — +45794.3 kHz/degC in the example, which
pools all samples through the package API, against +46110.5 here, which pools
the skip-applied samples — and both are confounded, so neither is a throttling
measurement.

## Equilibrium, tested on every run

Criterion: |fitted slope| <= 0.005 degC/s over the final 120 s, at least 30
finite samples. Measured final-window slopes, degC/s:

| run | status | slope | AR(1) 95 per cent interval |
|---|---|---:|---|
| soak_1.00_r1 | EQUILIBRATED | +0.000353 | [-0.000149, +0.000855] |
| soak_1.00_r2 | EQUILIBRATED | +0.000980 | [+0.000394, +0.001566] |
| soak_1.00_r3 | EQUILIBRATED | -0.000251 | [-0.000795, +0.000294] |
| soak_0.25 | EQUILIBRATED | -0.000345 | [-0.000821, +0.000132] |
| soak_0.50 | EQUILIBRATED | -0.000756 | [-0.002030, +0.000517] |
| soak_0.75 | EQUILIBRATED | +0.001371 | [+0.000837, +0.001905] |
| burst_1.00_a, burst_1.00_b | INSUFFICIENT_SAMPLES | — | 20 samples, below the 30 minimum; explicitly not a pass |

The declared 0.005 degC/s threshold was **guessed before the data existed** and
turned out not to be marginal: all three duty-1.0 soaks stay EQUILIBRATED at
every rung of the ladder, down to the tightest, 0.001 degC/s.

**One nuance worth stating.** Three cooldown runs — `cool_a` (-0.004626),
`cool_b` (-0.004895) and `cool_d25` (-0.004770) — also pass the criterion, while
`cool_d50`, `cool_d75`, `cool_r1`, `cool_r2` and `cool_r3` do not. The test is a
statement about the slope and nothing else: a cooldown that has nearly settled
satisfies it. No sustained figure in this product is computed from a cooldown
run, so this changes no result, but a reader should not take EQUILIBRATED on a
`cool_*` row to mean the board is at steady state under load.

## Baselines against the measured ground truth

| | prediction, iters/s | error vs ground truth | |error| / half-width | verdict |
|---|---:|---:|---:|---|
| Baseline 1, cold burst | 26160.398 | +418.994 (+1.6277 per cent) | **2.862** | OUTSIDE the interval |
| Baseline 2, RC network | 26160.398 | +418.994 (+1.6277 per cent) | **2.862** | OUTSIDE the interval |
| Baseline 3, measured | 25741.404 [25593.028, 25885.792] | — | — | ground truth |

Baseline 2's **die-temperature** prediction is good on its own terms:
**56.649 degC** predicted against **56.7085** measured, error **-0.0597 degC**
against a half-width of **0.2573** — inside the interval.

**The headline is INDISTINGUISHABLE while baseline 1's error is 2.862
half-widths outside the ground-truth interval, and both are correct.** The
ground-truth half-width describes the spread of the sustained figure alone; the
headline interval carries the spread of the burst figure as well, and there are
only two burst runs, giving four distinct run-level resample multisets. The
burst side dominates the headline's width.

## The first-order model is inadequate for this board, and two independent bounds say so

From section 8b of `validate_measured_campaign_output.txt`:

- **Monotonicity.** A monotone first-order rise approaches its asymptote from
  below, so the fitted asymptote should exceed the final-window mean. Measured:
  asymptote **56.765 degC** against a final-window mean of **56.999 degC**, a
  shortfall of **+0.233 degC**. The bound is NOT satisfied on this board.
- **The local-slope identity** `T_inf = T + tau dT/dt`, with tau taken from the
  **cooldown** traces (79.70 to 153.43 s — different runs, the opposite
  direction of heat flow, a different fit) and the final-window slope at 4
  sigma, gives the bracket **[56.803, 57.117] degC**. The fitted asymptote lies
  **outside it, below, by 0.038 degC**.
- **The corroborating evidence is in the fit's own residuals:** lag-1
  autocorrelation **+0.732**, rms residual **0.2100 degC** over 8.09 tau. That is
  the signature of a second, slower time constant which a single exponential
  cannot represent — a die, a heat spreader and a fan-cooled heatsink should
  produce exactly that.

**These two outcomes are reported as FINDINGS, not as failed checks, and the
distinction is principled rather than convenient.** On synthetic fixtures the
generating model *is* first order, so the same bound is a correctness check on
the estimator and a miss is a defect. On the board the bound tests whether the
physics is first order, and the answer is no. What *would* be a defect is a
disagreement larger than the fit's own residual scale, and that is the check
that is run: **0.0378 degC against an rms residual of 0.2100 degC, a ratio of
0.180 — PASS.** Two controls guard it, must-fire 1/1 and must-NOT-fire silent
1/1.

The practical consequence: **the RC fit is biased low in a known direction**,
and a longer soak would widen that bias. The measured die was still rising at
+0.000353 degC/s at the end of the 720 s soak; it had slowed to within the
equilibrium criterion, not stopped.

## Tau, three routes, and one refusal

| route | value | note |
|---|---:|---|
| Separable least squares, rise, per run | 88.85, 82.17, 71.96 s | mean **80.99 s**; rms residuals 0.2100, 0.1317, 0.1111 degC; rho +0.732, +0.299, +0.189 |
| Separable least squares, cooldown, 8 runs | 79.70 – 153.43 s | mean **110.69 s** |
| Area rule on the primary rise | **81.76 s** | tail correction 1.000 over 8.79 tau |
| Log-difference slope | **REFUSED** | "no part of the differenced trace clears the 3-sigma signal floor of 0.1996 degC" |

**Ratio cooldown/rise = 1.367**, where a true single-pole system gives 1.000.
Neither value is preferred and they are not averaged into one number. The
refusal is reported by name rather than silently dropped: on a measured trace
this noisy, differencing has no signal to work with.

## The learned model lost, decisively

From `validate_measured_learned_output.txt`. Dataset: **36 rows from 6 runs x 6
warm-up windows (60 to 150 s), 4 duty cycles, 11 features, 0 runs excluded**.
Splits grouped by run, evaluation grouped by duty. Ridge alpha 1, declared and
not tuned. The RC baseline is fitted to **exactly the same warm-up window**.

| model | pooled MAE, degC | RC MAE, degC | margin, degC | margin / run-to-run half-width | verdict |
|---|---:|---:|---:|---:|---|
| ridge (primary) | **4.9205** | 0.5120 | **-4.4084** | **17.13** | **BASELINE_BETTER** |
| forest (comparison) | 2.9492 | 0.5120 | -2.4372 | 9.47 | **BASELINE_BETTER** |

Per-duty ridge MAE, degC: **0.25 → 2.3870**, **0.50 → 0.3496**, **0.75 →
0.9794**, **1.00 → 8.6022**. The failure mode is the predicted one: the model
cannot extrapolate past its training duty cycles, and duty 1.00 — the hottest,
held out from a training set that saw only 0.25, 0.50 and 0.75 — is where it
breaks.

**One honest counter-detail: at duty 0.50 the ridge beat the RC baseline**,
0.3496 degC against 0.6054, a margin of +0.2558 degC. That single split is
reported because the rule cuts both ways; it is one of four and does not change
the pooled verdict.

**The uncertainty output is badly calibrated and is reported as such.**
Two-sigma empirical coverage is **0.6111** for the ridge and **0.8333** for the
forest, against a calibrated 0.95. Neither band should be relied on.

## Held-out ambient: STRUCTURALLY UNMEASURABLE

There is one ambient in this campaign and no way to vary it without physical
access to the board, so there is no second point to hold out. This axis has **no
score**. It is not dropped, and it is **not** approximated by the duty-cycle
split: a model can pass one and fail the other, which is why the registry
summary asks for both separately. Reported with its reason in
`validate_measured_learned_output.txt`, `README.md`, `MODEL_CARD.md` and
`thermalbudget.limits.structurally_unmeasurable_axes()`.

## Data quality of the campaign itself

- **Missing data: zero.** Not one sentinel-coded or unparseable sample in the
  entire campaign, across 17 runs.
- **Contamination: measured, not assumed absent.** Pooled foreign CPU **17.05
  per cent** over 4975 samples; per-run means from **14.93** (`cool_r3`) to
  **20.62** (`idle_pre`) per cent, per-sample maxima up to 28.9 per cent, mostly
  `avahi-daemon`. Every throughput figure is a figure for this payload on a
  board carrying that co-resident load.
- **Achieved duty, measured and not assumed.** soak_1.00_r1/r2/r3 achieved
  0.9998 / 0.9999 / 0.9999 (per-sample range 0.9494 to 1.2041); the 0.25, 0.50
  and 0.75 soaks achieved 0.2527, 0.5030 and 0.7507. Achieved duty can exceed
  the commanded value because the per-worker control window is 50 ms and an
  iteration that starts inside it runs past its end.
- **Start-up transient, measured for all eight load runs.** Every non-zero-duty
  run reports a first one-second sample below its own post-skip mean: 25.4 and
  26.6 per cent for the two bursts, 4.2 / 17.3 / 24.2 per cent for duty
  0.25 / 0.50 / 0.75 and 25.6 / 26.1 / 24.3 per cent for the three duty-1.0
  soaks. The declared 2 s skip is justified in every case. The deficit is **not**
  constant across duty cycles, and the table in
  `thermalbudget.baselines`'s module docstring is the figure to quote.
- **GPU devfreq.** 306 MHz floor in 4954 of 4975 samples (99.578 per cent); 21
  samples (0.422 per cent) at 1020 MHz, scattered across 13 runs including idle
  and cooldown runs, so they are not caused by the payload. **GR3D utilisation
  was not recorded.**

## Results of the code and estimator checks

Computed from SYNTHETIC fixtures or from this repository's own code, and
labelled so in every output file.

| Check | Script | Result |
|---|---|---|
| Suite, from junit XML | `pytest --junit-xml` | **221 declared, 221 `testcase` counted, 0 failed, 0 errored, 0 skipped** |
| Measured campaign | `validate_measured_campaign_output.txt` | **10 checks, 0 failed; 2 controls, must-fire 1/1, must-NOT-fire silent 1/1; 1 structurally unmeasurable** |
| Measured learned predictor | `validate_measured_learned_output.txt` | **3 checks, 0 failed; 1 structurally unmeasurable** |
| Clock confound | `validate_clock_confound_output.txt` | **3 checks, 0 failed** |
| Loader contract | `validate_loader_contract_output.txt` | 29 checks, 0 failed; 15 controls, must-fire 14/14, must-NOT-fire silent 1/1 |
| Equilibrium estimator | `validate_equilibrium_output.txt` | 16 checks, 0 failed; 9 controls, must-fire 7/7, must-NOT-fire silent 2/2 |
| Four independent bounds on the estimators | `validate_independent_bound_output.txt` | 16 checks, 0 failed; 6 controls, must-fire 4/4, must-NOT-fire silent 2/2 |
| Cross-module negative controls | `validate_negative_controls_output.txt` | 41 checks, 0 failed; 40 controls, must-fire 32/32, must-NOT-fire silent 8/8 |
| `pytest.approx` hidden-default audit | `validate_approx_audit_output.txt` | 51 `rel=` sites, 51 with an explicit `abs=`, 0 bare; suite re-run with `abs=0.0` forced, 221 tests, 0 failed |
| Property hunt | `validate_property_hunt_output.txt` | 10 properties x 2000 examples, about 20000 generated cases, 0 falsified |
| Worked example, both fixture regimes | `worked_example_output.txt` | 4 checks, 0 failed |
| Working-directory probe | `probe_working_directories_output.txt` | every script giving the same exit status from four directories |
| Tree hygiene | `validate_tree_hygiene_output.txt` | see below |

### Tree hygiene

- **git tracks no artifact-shaped file** under `products/P091/`.
- **14 paths this product must ship** checked with `git check-ignore -v`, none
  excluded; **8 cache and build paths** checked, all excluded.
- **The measured traces are not excluded by `.gitignore`** — now checkable,
  because they exist. At build time this check reported STRUCTURALLY
  UNMEASURABLE with the reason that there was nothing to check.
- **This product's own `.gitignore` anchors every scratch pattern**, in
  particular `/junit.xml` and not a bare `junit.xml`.
- **INHERITED FINDING, reported and not failed:** the repository-root
  `.gitignore` carries `**/junit.xml`, which would exclude a nested `junit.xml`
  fixture in any product that shipped one. P091 ships none.
- **Running the script adds no cache directory and no bytecode file.**
  Artifacts already present at entry are listed as inherited and do not fail the
  check, because release-gate check 2 writes `.ruff_cache` into the product root
  one check before check 6 re-executes this script.

## Corrections made after the traces arrived

Each is recorded rather than quietly applied, because every one is an instance
of the same error class: a confident sentence about data that had not been seen.

1. **`thermalbudget.baselines` module docstring** stated that the first row of
   the measured cold-burst trace carries `iters_cum = 0`. It does not: the
   measured values are 19468 (`burst_1.00_a`) and 19272 (`burst_1.00_b`).
   Corrected by the coordinating session.
2. **The replacement text for (1)** then said the start-up deficit was
   "consistently near 25 per cent" across "all six load runs". There are
   **eight** non-zero-duty runs, not six, and the deficit is **4.2 per cent** at
   duty 0.25, which is not near 25 per cent. Replaced with the measured table
   for all eight runs.
3. **`thermalbudget.clock_throughput` module docstring** asserted that the real
   cold-burst trace shows all six cores pinned at 1728000 kHz "for the entire
   20 s run". That was written from the five rows quoted in `docs/SPEC.md`, not
   from the whole run. It has since been checked and is true — all 120 per-core
   samples of each burst run sit at 1728000 kHz — but the sentence asserted more
   than its evidence, so it is replaced rather than merely confirmed.
4. **`thermalbudget.clock_throughput`** asserted that the board's governor steps
   are "far larger" than the 0.5 per cent identifiability floor, without having
   measured one. Measured: 14 distinct levels, smallest gap 76800 kHz, 8.9 times
   the floor.
5. **`tests/conftest.py` and `tests/fixtures/synthetic_trace.py`** described the
   `no_throttle` regime as "what the one real cold-burst trace available at
   build time shows". Updated to the measured campaign-wide result, with the
   provenance of the original claim retained.
6. **`tests/test_trace.py`** carried the comment "All six cores pinned at
   cpuinfo_max_freq for the whole run" beside an assertion over five rows.
   Narrowed to those five rows, with a pointer to where the whole-run
   measurement lives.
7. **The coordinator's hand-off** stated that the GPU "sat at 306 MHz and GR3D 0
   per cent for the entire campaign". Measured: 306 MHz in **4954 of 4975**
   samples, with 21 transient samples at 1020 MHz, and **GR3D is not a column in
   the trace schema and was never sampled at all**. Both parts are corrected
   wherever the GPU scope limit appears.

## Known gaps in this validation

1. **The dependency ranges were not exercised at both ends.** The build
   environment is a shared pre-pinned interpreter this product is forbidden to
   run `pip` against. Only `ruff` was measured at both ends, 0.16.9 and 0.17.4,
   clean over `src/ tests/ examples/ validation/`.
2. **The cluster bootstrap is granular.** Three soak runs give at most 27
   distinct run-level resample multisets and two burst runs give 4. More runs is
   the only fix and this campaign has what it has.
3. **Ambient was never instrumented**, so even the campaign's own ambient has no
   value in degrees.
4. **GR3D utilisation was never sampled**, so "CPU-only" rests on the devfreq
   floor rather than on an engine-utilisation measurement.
5. **No cold-clone `pip install -e ".[dev]"` was executed**, for the same reason
   as (1).
6. **`validate_independent_bound.py` still runs on fixtures only.** The measured
   equivalent is section 8b of `validate_measured_campaign.py`, added in this
   session; the fixture script was left as the estimator-correctness check it is.

---

Research-grade. This software is not flight-qualified, not certified and not
approved for operational aerospace use.
