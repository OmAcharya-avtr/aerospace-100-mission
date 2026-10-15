# Validation evidence — safemodetree 0.1.0

Validation level 3. Every number in this file was produced by a script in this
directory, run in this session on 2026-10-15, and the raw output of each run is
committed beside the script as `validation/<script-name>_output.txt`. Where a
check went against the product, it is kept and marked.

Environment: Python 3.13.16 CPython, Linux 6.18.44 x86_64, two contended CPU
cores, numpy 2.5.3, scikit-learn 1.9.1, PyYAML 6.0.3, matplotlib 3.11.2, pytest
9.1.1, hypothesis 6.169.3. Full record in
[validate_environment_output.txt](validate_environment_output.txt).

Requirement identifiers (R-x.y) refer to [../docs/REQUIREMENTS.md](../docs/REQUIREMENTS.md).

| script | what it establishes | raw output |
|---|---|---|
| `validate_environment.py` | the versions every number was measured at, and that no alternative library is installed | [txt](validate_environment_output.txt) |
| `validate_known_answers.py` | 35 hand-derived arithmetic results | [txt](validate_known_answers_output.txt) |
| `validate_reachability.py` | reachability and cheapest-path against exhaustive enumeration, 2000 declarations | [txt](validate_reachability_output.txt) |
| `validate_planted_case.py` | two planted defects detected and localised | [txt](validate_planted_case_output.txt) |
| `validate_calibration.py` | learned against threshold logic at an equal false-alarm rate, with ECE | [txt](validate_calibration_output.txt) |
| `validate_conservatism.py` | the measured cost of routing on confidence | [txt](validate_conservatism_output.txt) |
| `validate_cli.py` | 23 CLI invocations and their exit codes | [txt](validate_cli_output.txt) |
| `worked_example.py` | the README's worked example | [txt](worked_example_output.txt) |
| — | the test suite | [pytest_output.txt](pytest_output.txt) |

## 1. Test suite

262 tests, 0 failures, 0 errors, 0 skipped, counted from the `testsuite`
element of `junit.xml` and cross-checked against a count of 262 `testcase`
elements. The committed run took 27.57 s on two contended cores; the same
command took 11.24 s, 19.75 s and 18.50 s on three other runs in this session,
the last of those at the dependency floor, so the counts are the reproducible
part and the seconds are not.

```
python -m pytest tests/ -q --junitxml=junit.xml
```

`junit.xml` is a per-run artifact and is gitignored with a leading slash
(`/junit.xml`), so the pattern cannot match a committed file at another depth.
`git check-ignore -v` was run against every shipped file — both packaged
declarations, all six screenshots, `pyproject.toml`, `LICENSE`, the workflow,
and every file under `validation/`, `examples/` and `tests/` — and none is
ignored.

## 2. Reachability against exhaustive enumeration (R-2.4, R-4.1)

`validate_reachability.py` builds 2000 random declarations (seeds 0 to 1999, 4
modes, 3 faults, 6 responses, 3 channels; `safemodetree.synth`), computes
safe-mode reachability with the backward breadth-first fixpoint and
cheapest-path cost with multi-source Dijkstra, and recomputes both by
enumerating every simple path.

| check | reference | result | tolerance |
|---|---|---|---|
| reachable-state sets agree | exhaustive simple-path enumeration | 2000 of 2000 agree, 0 disagreements over 32 000 states and 32 581 transitions | exact (sets of discrete states) |
| cheapest-cost supports agree | same | 2000 of 2000 agree | exact |
| worst cheapest-cost deviation | same | **0.000e+00 min** | 1e-9 relative, 1e-9 absolute |
| worst witness-path sum deviation | sum of the declared `cost_min` along the reported path | **0.000e+00 min** | 1e-9 relative, 1e-9 absolute |

1218 of the 2000 random declarations contain at least one fault state that
cannot reach a declared safe mode, so the cross-check is not exercising the
trivial all-reachable case.

Both comparisons carry a tolerance on both sides. No boolean produced by a
strict threshold comparison is asserted equal anywhere in the suite; the only
exact comparisons are over sets of discrete automaton states, which carry no
floating-point quantity.

## 3. Planted cases (R-2.5, R-4.2)

### 3.1 The shipped planted declaration

`src/safemodetree/specs/planted_unreachable.yaml` plants two defects:
`sensor_fail`'s only response enters `degraded`, declared absorbing and not
safe; and `orphan_safe` is declared safe and targeted by no response.

| check | result |
|---|---|
| `(ops, sensor_fail)` reported unable to reach a safe mode | detected |
| its reported cost | `inf` |
| `(ops, actuator_fail)` still reachable | yes, **11.000000 min** in 1 hop via `actuator_safe_hold` |
| `orphan_safe` reported as an unreachable declared safe mode | detected |
| number of fault states reported unreachable | exactly 1 of 2 |
| fixpoint against exhaustive enumeration on this declaration | agree |

### 3.2 The shipped declaration with `safe_mode_entry` removed

| check | result |
|---|---|
| intact declaration: every fault state reaches a safe mode | yes, worst **53.0 min** |
| stripped declaration: every fault state still reaches one | yes, worst **480.0 min** |
| surviving witness paths | exactly `('bat_cell_bypass',)` and `('survival_entry',)` |
| 14 of 16 fault states fall back to `survival_entry` at 480.0 min | yes |
| declared safe modes made unreachable | **none** |

**The first draft of this script predicted the wrong answer here.** It asserted
that removing `safe_mode_entry` would strand `sun_safe`. It does not:
`bat_cell_bypass` declares `to_mode: sun_safe` and clears its own fault, so one
route into `sun_safe` survives and `bat_cell_short` keeps a 60.0 min route in
both operational modes. The script was corrected to the measured answer and the
wrong prediction is recorded here rather than removed.

## 4. Known answers (R-4.5)

35 of 35 checks pass, each with its derivation printed in
`validate_known_answers_output.txt`. Tolerance is 1e-12 absolute and relative
on every comparison.

| quantity | derivation from the declaration | measured |
|---|---|---|
| state count | 4 modes x (8 faults + 1 no-fault) | 36 |
| fault-state count | faults declared to occur in 2 modes, 8 x 2 | 16 |
| absorbing states with no outgoing transition | `survival` x 9 | 9 |
| cheapest route from `coarse_pointing` | `min(45.0, 480.0)` | 45.000000000000 min, 1 hop |
| cheapest route from `fine_pointing` | `min(8.0 + 45.0, 480.0)` | 53.000000000000 min, 2 hops |
| `st_wait_and_reacquire` on `st_blinding` (right) | 6.0, fault cleared | 6.000000000000 min |
| `st_power_cycle` on `st_latchup` (right) | 25.0, fault cleared | 25.000000000000 min |
| `st_wait_and_reacquire` on `st_latchup` (wrong) | 6.0 + ladder 53.0 | 59.000000000000 min |
| `st_power_cycle` on `st_blinding` (wrong) | 25.0 + ladder 45.0 | 70.000000000000 min |
| star-tracker class cost of conservatism | 53.0 - (6.0 + 25.0)/2 | 37.500000000000 min |
| power class cost of conservatism | 53.0 - (15.0 + 35.0)/2 | 28.000000000000 min |
| `bat_cell_bypass` specific route | declared | 60.0 min, **dearer than the 53.0 min conservative ladder** |
| threshold logic, unique signature | rule 3 | confidence 1.000000000000 |
| threshold logic, ambiguity class of 2 | rule 4, `1/m` | confidence 0.500000000000 |
| threshold logic, Hamming-1 tie of 3 | rule 5, `1/(3 x (1+1))` | confidence 0.166666666667 |
| `wheel_torque_error` margin at 4.20 mN*m | `(4.20 - 3.00)/0.60` | 2.000000000000 sigma |
| `array_current` margin at 0.84 A | `(1.20 - 0.84)/0.18` | 2.000000000000 sigma |
| `bus_voltage` margin at 25.60 V | `(26.00 - 25.60)/0.40` | 1.000000000000 sigma |
| ECE, 5 bins, 4 predictions | 0.200 + 0.100 + 0.050 | 0.350000000000 |
| MCE, same | largest bin gap | 0.400000000000 |
| multi-class Brier, one row, three classes | 0.04 + 0.25 + 0.09 | 0.380000000000 |
| latency budget boundary | `latency <= budget`, 30.0 s against 30.0 s and 29.9 s | admits, refuses |

The detection latency budget is a real knob and not decoration: at a 60 s
budget the two faults with a declared 120 s latency (`sa_string_open`,
`pcdu_shunt_stuck`) lose their fault-specific responses, the conflicting
ambiguity class count falls from **4 to 2**, and reachability still holds from
every fault state.

## 5. Findings on the shipped declaration (R-2.6 to R-2.9)

`python -m safemodetree check` on `reaction_wheel_fmea`, unbounded latency
budget:

| finding | count |
|---|---|
| fault states with no reachable safe mode | 0 |
| declared safe modes no fault state can reach | 0 |
| fault states with no enabled response | 0 |
| faults with no declared response | 0 |
| responses whose precondition can never hold | **2** |
| conflicting ambiguity classes | **4** |
| ambiguity classes with no conservative response | 0 |

The two vacuous responses are `rw_swap_to_spare`, which requires
`redundant_wheel_available`, declared `always_false` because the spare wheel
slot is not populated on this build; and `st_power_cycle_in_safe`, which
requires `tracker_powered` from `sun_safe`, where the tracker is declared
unpowered. Both are the specification defect the check exists to find, planted
in the shipped declaration deliberately so the check has something to report.

The four conflicting ambiguity classes are the two declared indistinguishable
pairs, each analysed in both operational modes:

| class | mode | preferred responses | conservative route | cost of conservatism |
|---|---|---|---|---|
| `pcdu_shunt_stuck` + `sa_string_open` | `fine_pointing` | `pcdu_reset` (35.0) vs `array_reconfigure` (15.0) | `coarse_downmode` then `safe_mode_entry`, 53.000 min | **+28.000 min** |
| `pcdu_shunt_stuck` + `sa_string_open` | `coarse_pointing` | same | `safe_mode_entry`, 45.000 min | **+20.000 min** |
| `st_blinding` + `st_latchup` | `fine_pointing` | `st_wait_and_reacquire` (6.0) vs `st_power_cycle` (25.0) | `coarse_downmode` then `safe_mode_entry`, 53.000 min | **+37.500 min** |
| `st_blinding` + `st_latchup` | `coarse_pointing` | none vs `st_power_cycle` (25.0) | `safe_mode_entry`, 45.000 min | **+20.000 min** |

## 6. Classifier against the declared threshold logic, at an equal false-alarm rate (R-3.3, R-4.3)

`validate_calibration.py`, seed 63001, 400 samples per fault, nominal class
half the data, split 50 / 25 / 25 into train, calibration and test (3200 / 1600
/ 1600 rows). The baseline's false-alarm rate is measured on the calibration
split at its declared operating point; the learned classifier's
declare-a-fault threshold is then tuned on the same split to match it; both are
scored on the test split neither was tuned on.

| noise scale | target FAR | baseline FAR | learned FAR | baseline ident. acc. | learned ident. acc. | delta | winner |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.00000 | 0.00000 | 0.00368 | **0.74650** | 0.72739 | -0.01911 | threshold logic |
| 1.5 | 0.02439 | 0.02209 | 0.01595 | **0.74522** | 0.74013 | -0.00510 | threshold logic |
| 2.0 | 0.13736 | 0.12147 | 0.07117 | 0.73503 | 0.73503 | +0.00000 | tie |
| 2.5 | 0.30167 | 0.27239 | 0.26135 | 0.71592 | **0.71720** | +0.00127 | learned |
| 3.0 | 0.45186 | 0.40859 | 0.44172 | 0.68153 | **0.70064** | +0.01911 | learned |

**FINDING, kept: the learned classifier does not beat the declared threshold
logic on identification accuracy.** Over five noise scales it wins twice, loses
twice and ties once, and the largest win is +0.019 of accuracy at the noisiest
setting where both detectors are already firing on 40 per cent of no-fault
samples. The baseline is correctly specified for this generator — each class is
an isotropic Gaussian with a shared covariance, so the declared per-channel
thresholds are close to the Bayes boundary — and a reader should not take the
learned model as the better detector here. Its value in this product is the
calibrated posterior, not the point identification.

Accuracy ceiling imposed by the declared observability model: **0.874062**.
Four of eight faults sit in declared ambiguity classes of size two, so
`(800 + 400 + 200) / 1600 = 0.875` is the most any rule can reach on this test
split, and no detector exceeded it.

### Calibration (R-3.4)

ECE at three bin counts, because binned ECE is a biased estimator whose value
depends on the binning:

| noise scale | detector | ECE@5 | ECE@10 | ECE@20 |
|---|---|---|---|---|
| 1.0 | threshold logic | **0.00156** | **0.00156** | **0.00156** |
| 1.0 | learned, matched FAR | 0.01478 | 0.01567 | 0.01611 |
| 1.5 | threshold logic | 0.00778 | **0.00778** | **0.00778** |
| 1.5 | learned, matched FAR | **0.00588** | 0.00816 | 0.01162 |
| 2.0 | threshold logic | 0.02689 | 0.02783 | 0.02783 |
| 2.0 | learned, matched FAR | **0.02448** | **0.02448** | 0.02901 |
| 2.5 | threshold logic | 0.05521 | 0.05521 | 0.05521 |
| 2.5 | learned, matched FAR | **0.04972** | **0.04972** | **0.05256** |
| 3.0 | threshold logic | 0.07371 | 0.07371 | 0.07576 |
| 3.0 | learned, matched FAR | **0.06431** | **0.06463** | **0.06634** |

The learned classifier has the lower ECE at 10 bins at **3 of 5** scales. The
pattern is the interesting part: the declared threshold logic is extremely well
calibrated when the thresholds sit far from the noise (ECE 0.00156 at scale
1.0, better than the learned model) and becomes steadily overconfident as the
noise grows (0.00156 to 0.07371), because its confidence is a lookup on a
latched symptom word and carries no information about how close the measurement
was to the threshold. The learned posterior degrades more slowly (0.01567 to
0.06463).

Maximum calibration error is reported beside ECE and is noisy: the baseline's
MCE at scale 2.0 is 0.66667, from a bin holding **2** predictions. The bin
occupancy table is printed in `validate_calibration_output.txt` section 4 and
in the right panel of `screenshots/calibration_reliability.png` precisely so
that a single sparse bin cannot be read as a result.

Multi-class Brier score of the learned posterior on the test split: 0.12924,
0.13893, 0.16237, 0.19523, 0.23383 at the five scales. The baseline has no
posterior over all classes and so has no Brier score; that is reported as `-`
rather than filled in.

### A recorded artefact of matching a zero rate

At noise scale 1.0 the baseline's calibration-split false-alarm rate is exactly
0, so the matched threshold is the one just above the largest no-fault fault
posterior on the calibration split. Three test-split rows exceed it, giving a
test-split rate of 0.00368 against the baseline's 0.00000, while the plain
argmax happens to give 0.00000. At that one scale the match is therefore **not**
closer to the baseline than the argmax. This is a finite-sample artefact of
matching a zero rate, it is recorded as its own check in
`validate_calibration_output.txt` section 5, and the monotone check is
restricted to the four scales with a non-zero target rate.

## 7. The cost of conservatism (R-3.5, R-3.6)

`validate_conservatism.py`, same split, routing threshold 0.70. Cost is the
declared resolution cost in minutes of mission outage per **commanded** fault
event. Missed detections are counted and never costed, because any number put
there would be invented.

| detector | mode | policy | commanded | missed | mean cost [min] | mis-routed | unsafe | conservative fraction |
|---|---|---|---|---|---|---|---|---|
| learned | fine_pointing | oracle | 785 | 0 | **23.4535** | 0 | 0 | 0.0000 |
| learned | fine_pointing | always specific | 783 | 2 | 35.4866 | **206** | 0 | 0.0000 |
| learned | fine_pointing | confidence routed | 783 | 2 | 44.6322 | **6** | 0 | 0.6181 |
| learned | fine_pointing | always conservative | 783 | 2 | 53.0000 | 0 | 0 | 1.0000 |
| learned | coarse_pointing | oracle | 785 | 0 | 26.9860 | 0 | 0 | 0.0000 |
| learned | coarse_pointing | always specific | 783 | 2 | 37.4087 | 138 | 0 | 0.1673 |
| learned | coarse_pointing | confidence routed | 783 | 2 | 39.6871 | 6 | 0 | 0.6181 |
| learned | coarse_pointing | always conservative | 783 | 2 | 45.0000 | 0 | 0 | 1.0000 |
| baseline | fine_pointing | always specific | 784 | 1 | 36.8750 | 207 | 0 | 0.0000 |
| baseline | fine_pointing | confidence routed | 784 | 1 | 41.5727 | 10 | 0 | 0.5485 |
| baseline | coarse_pointing | always specific | 784 | 1 | 39.5128 | 104 | 0 | 0.2462 |
| baseline | coarse_pointing | confidence routed | 784 | 1 | 37.1849 | 10 | 0 | 0.5485 |

| detector | mode | cost of conservatism vs oracle [min] | vs always-specific [min] | mis-routes avoided | min per avoided mis-route |
|---|---|---|---|---|---|
| learned | fine_pointing | +21.1787 | **+9.1456** | 200 | 35.8050 |
| learned | coarse_pointing | +12.7011 | +2.2784 | 132 | 13.5152 |
| baseline | fine_pointing | +18.1192 | +4.6977 | 197 | 18.6954 |
| baseline | coarse_pointing | +10.1990 | **-2.3278** | 94 | **-19.4149** |

**FINDING, kept: the sign of the cost of conservatism is not the same in every
configuration.** With the learned detector in `fine_pointing`, routing on
confidence costs **+9.1456 min per commanded event** against the always-specific
policy and eliminates 200 of 206 mis-routes, which is 35.8 min of outage per
avoided mis-route. With the declared threshold logic in `coarse_pointing` the
cost is **-2.3278 min**: routing conservatively is both cheaper and safer there,
because the baseline's identification is wrong often enough that the mis-route
penalty exceeds the conservative route's cost. Both numbers come from the same
test split and both are reported.

Routing-threshold sweep, learned detector, `fine_pointing`:

| threshold | mean cost [min] | mis-routed | conservative fraction |
|---|---|---|---|
| 0.00 | 35.4866 | 206 | 0.0000 |
| 0.20 | 35.4687 | 206 | 0.0026 |
| 0.30 | 35.5275 | 206 | 0.0051 |
| 0.40 | 35.5824 | 194 | 0.0307 |
| 0.50 | 36.3653 | 165 | 0.1060 |
| 0.60 | 43.2427 | 9 | 0.5683 |
| 0.70 | 44.6322 | 6 | 0.6181 |
| 0.80 | 45.2363 | 1 | 0.6488 |
| 0.90 | 47.0741 | 0 | 0.7088 |
| 1.00 | 49.8633 | 0 | 0.8263 |
| 2.00 | 53.0000 | 0 | 1.0000 |

The curve is not smooth and the knee matters: almost all of the mis-route
reduction happens between thresholds 0.50 and 0.60, where the cost rises by
6.88 min per event. A threshold of 0.90 removes the last mis-route at a cost of
**11.59 min per event** over the always-specific policy, and going beyond it to
2.00 — always conservative — costs another 5.93 min per event and removes
nothing further. A threshold of exactly 1.00 does not route an identification
whose confidence is exactly 1, which declared threshold logic emits on an exact
signature match, which is why the always-conservative arm uses 2.00.

Under a 60 s detection latency budget the always-specific policy can no longer
command the two power faults' responses and falls back to the conservative
route for them, so its cost rises to 36.8914 min and its mis-route count falls
to 104 with no change to the detector at all.

## 8. CLI (R-1.6)

23 of 23 invocations behaved as declared; full transcript in
`validate_cli_output.txt`. Exit codes: **0** success, **1** a finding with
`--fail-on-findings`, **2** a refused request.

| invocation | exit |
|---|---|
| `check` on `reaction_wheel_fmea` | 0 |
| `check --fail-on-findings` on `reaction_wheel_fmea` | 1 (6 findings) |
| `check --spec planted_unreachable --fail-on-findings` | 1 (2 findings) |
| `reach --spec planted_unreachable --fail-on-findings` | 1 |
| `check --spec no_such_declaration` | 2 |
| `reach --spec /nonexistent/path.yaml` | 2 |
| `policy --mode nowhere` | 2 |

## 9. Dependency ranges, both ends measured (R-4.7)

Every declared range was measured at both ends. A floor virtual environment was
built with the lowest version of each dependency, the **whole suite** and
**every validation script** were run in it, and the committed validation output
was diffed against the ceiling run line by line (ignoring the `elapsed` lines,
which are wall-clock).

| dependency | declared range | floor measured | ceiling measured |
|---|---|---|---|
| numpy | `>=2.1.3,<2.6` | 2.1.3 | 2.5.3 |
| scikit-learn | `>=1.5.2,<1.10` | 1.5.2 | 1.9.1 |
| pyyaml | `>=6.0.2,<6.1` | 6.0.2 | 6.0.3 |
| matplotlib (plot) | `>=3.9.2,<3.12` | 3.9.2 | 3.11.2 |
| pytest (test) | `>=8.3.3,<9.2` | 8.3.3 | 9.1.1 |
| hypothesis (test) | `>=6.112.1,<6.170` | 6.112.1 | 6.169.3 |
| ruff (test) | `>=0.6.9,<0.18` | 0.6.9 | 0.17.0 (and 0.16.8) |

Results:

| check | result |
|---|---|
| test suite at the floor | **262 passed**, 0 failures, 0 errors, 0 skipped, 18.50 s |
| test suite at the ceiling | **262 passed**, 0 failures, 0 errors, 0 skipped, 11.24 s to 27.57 s over three runs |
| `validate_known_answers.py` output, floor against ceiling | **identical** |
| `validate_reachability.py` output | **identical** |
| `validate_planted_case.py` output | **identical** |
| `validate_calibration.py` output | **identical**, including every classifier number |
| `validate_conservatism.py` output | **identical** |
| `worked_example.py` output | **identical** |
| `ruff check src/ tests/ examples/ validation/` at 0.6.9, 0.16.8 and 0.17.0 | clean at all three |

The classifier numbers being bit-identical across scikit-learn 1.5.2 and 1.9.1
is the check that matters: a suite that passes while its validation numbers
move is the defect class this requirement exists for, and the diff is empty.

`ruff` 0.6.9 enforces `UP038`, which newer ruff does not; the five
`isinstance(x, (int, float))` calls in the source were rewritten as
`isinstance(x, int | float)` so that all three ruff versions are clean. That
change was made because the floor run failed, not because a newer ruff asked
for it.

**This product commits no pickled or otherwise serialised model.** The
classifier is fitted at run time from a seeded generator, so the
serialisation-format hazard that broke P001 does not apply here. The
scikit-learn bound is nonetheless pinned to the minor version, because the
numbers in sections 6 and 7 are the outputs of `LogisticRegression` and
`CalibratedClassifierCV` and a minor release may change either.

**The cost of these pins is real and is not resolved by removing them.**
`hypothesis>=6.112.1,<6.170` in particular is a narrow window, and this package
cannot take a hypothesis release above 6.169.x without re-running the property
suite. That is recorded in the README under Limitations rather than widened by
guessing.

## 10. Property-based tests (R-4.6)

`tests/test_properties.py`, 19 tests, 40 to 120 Hypothesis examples each.
Properties checked over random declarations: fixpoint reachability equals brute
force; Dijkstra cost equals brute force within 1e-9; a witness path's declared
costs sum to its reported cost; safe states are reachable from themselves;
relaxing the detection assumption never removes a reachable state; a larger
latency budget never removes a transition; a reported conservative route keeps
every class member safe-reachable; finding counts are non-negative and sum to
the total. Over calibration data: ECE is bounded in [0, 1] and MCE is at least
ECE; bin occupancy sums to the sample count; Brier is in [0, 2].

**Hypothesis found no falsifying example for any property in this file during
this session.** `.hypothesis/` is gitignored, so the boundary cases the
strategies can draw — an empty precondition, a single prediction with one bin, a
confidence of exactly 0.0, a latency budget of exactly 0.0 s — are pinned at the
bottom of that file as four named regression tests with literal inputs, so they
are checked on every run rather than only when Hypothesis happens to draw them.
If a future run does falsify a property, its example goes in that section.

Two monotonicity properties deserve a note on what they do and do not claim.
`test_symptom_bits_are_monotone_in_the_margin` asserts that raising the margin
can only turn a symptom bit off, which is monotonicity of `margin(value) > m`
in `m` for a **fixed** floating-point value and holds exactly. It is not a claim
that the bit is invariant under a rescaling of the channel; that would be a
strict threshold comparison on a recomputed quantity and is asserted nowhere in
this repository. `ObservationChannel.asserted` says so in its own docstring.

## 11. Alternatives, how they were checked

Checked on 2026-10-15 against the live PyPI index and, for the two installable
ones, by downloading the wheel and reading the unpacked source. Existence is
not equivalence, so the capability claims in the README's alternatives table
come from the source, not from the project description.

| package | probe | result |
|---|---|---|
| `transitions` | `pip index versions transitions` | 0.9.3 latest, MIT, pure Python, requires `six` |
| `python-statemachine` | `pip index versions python-statemachine` | 3.2.2 latest, MIT |
| `pynusmv` | `pip index versions pynusmv` | **"No matching distribution found"**; the project exists on PyPI with releases 1.0rc3 to 1.0rc8 and ships only CPython 3.4 / 3.5 / 3.6 wheels plus an sdist, so pip finds nothing installable on Python 3.13 and the pre-release-only versioning hides it from a default `pip index` |
| `pysmt`, `sismic`, `automata-lib` | PyPI JSON endpoint | all three exist (HTTP 200), noted as adjacent primitives |
| `nusmv`, `py-nusmv` | PyPI JSON endpoint | HTTP 404, no such PyPI project |

Capability findings read from the unpacked source:

- **`transitions` 0.9.3** has no reachability check at any depth. `grep -niE
  "reachab|unreachable|bfs|dijkstra"` over `transitions/` returns nothing. It
  has `extensions/markup.py` and `extensions/diagrams_*.py` for serialising and
  drawing a machine, hierarchical and parallel states, locking and asyncio
  variants. It is a state-machine runtime, not an analyser.
- **`python-statemachine` 3.2.2 does check reachability.**
  `statemachine/graph.py` provides `visit_connected_states`,
  `disconnected_states` and `states_without_path_to_final_states`, and
  `statemachine/factory.py` calls them at class-definition time, raising
  `InvalidDefinition` for unreachable states and for states with no path to a
  final state. **That is a genuine overlap and the README says so.** What it
  does not do: its traversal follows every transition regardless of guards
  (`visit_connected_states` reads `state.transitions` and ignores conditions),
  so it cannot answer "reachable under these detection assumptions and only
  through responses whose preconditions can hold"; and it has no notion of an
  observability model, so it cannot report an ambiguity class.
- **`pynusmv`** could not be installed or inspected here. It is cited from the
  literature, not from its source.

`validate_environment.py` asserts that `transitions`, `statemachine`,
`pynusmv`, `nusmv`, `pysmt` and `sismic` are all absent from the container every
number in this file was measured in, so no claim here can have come from
importing one.

## 12. References

Every reference below was used for a definition or a vocabulary and each is a
real, locatable publication. **No reference in this repository was verified
against a publisher page in this session**: the container's egress proxy reached
`pypi.org` and nothing else, so the PyPI checks in section 11 were performed and
executed while the academic and standards references are pointers to look up.

1. US Department of Defense, **MIL-STD-1629A**, "Procedures for Performing a
   Failure Mode, Effects and Criticality Analysis", 24 November 1980; cancelled
   1998. Used for the FMEA worksheet columns and the severity classification
   Category I to IV of Task 101. Cited as vocabulary, not as an active
   requirement.
2. ESA Requirements and Standards Division, **ECSS-Q-ST-30-02C**, "Space product
   assurance — Failure modes, effects (and criticality) analysis (FMEA/FMECA)",
   6 March 2009. Used for the FMEA/FMECA process vocabulary.
3. SAE International, **ARP4761**, "Guidelines and Methods for Conducting the
   Safety Assessment Process on Civil Airborne Systems and Equipment", 1996.
   Cited as the civil-aviation counterpart of the same worksheet practice.
4. M. Sampath, R. Sengupta, S. Lafortune, K. Sinnamohideen and D. Teneketzis,
   "Diagnosability of Discrete-Event Systems", **IEEE Transactions on Automatic
   Control**, vol. 40, no. 9, pp. 1555-1575, September 1995. The source of the
   indistinguishability relation. This package implements the **static special
   case** — equality of one declared observation vector — which is strictly
   weaker than trace diagnosability. Stated as a limitation, not claimed.
5. E. W. Dijkstra, "A Note on Two Problems in Connexion with Graphs",
   **Numerische Mathematik**, vol. 1, pp. 269-271, 1959. The cheapest-path
   algorithm; valid here because the loader enforces non-negative declared
   costs.
6. E. M. Clarke, O. Grumberg and D. A. Peled, **Model Checking**, MIT Press,
   1999. The standard reference for the reachability problem this product
   solves by brute force on small automata and a model checker solves properly
   at scale.
7. A. Cimatti, E. M. Clarke, E. Giunchiglia, F. Giunchiglia, M. Pistore, M.
   Roveri, R. Sebastiani and A. Tacchella, "NuSMV 2: An OpenSource Tool for
   Symbolic Model Checking", **Computer Aided Verification (CAV) 2002**, LNCS
   2404, pp. 359-364, Springer.
8. R. Cavada, A. Cimatti, M. Dorigatti, A. Griggio, A. Mariotti, A. Micheli, S.
   Mover, M. Roveri and S. Tonetta, "The nuXmv Symbolic Model Checker",
   **Computer Aided Verification (CAV) 2014**, LNCS 8559, pp. 334-342, Springer.
9. S. Busard and C. Pecheur, "PyNuSMV: NuSMV as a Python Library", **NASA
   Formal Methods (NFM) 2013**, LNCS 7871, pp. 453-458, Springer.
10. M. P. Naeini, G. F. Cooper and M. Hauskrecht, "Obtaining Well Calibrated
    Probabilities Using Bayesian Binning", **AAAI 2015**. The origin of the
    binned calibration error used here.
11. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern
    Neural Networks", **ICML 2017**, PMLR vol. 70, pp. 1321-1330. The top-label
    ECE in equation 3 is the form implemented in `calibration.py`.
12. G. W. Brier, "Verification of Forecasts Expressed in Terms of Probability",
    **Monthly Weather Review**, vol. 78, no. 1, pp. 1-3, 1950. The Brier score.
13. J. C. Platt, "Probabilistic Outputs for Support Vector Machines and
    Comparisons to Regularized Likelihood Methods", in **Advances in Large
    Margin Classifiers**, MIT Press, 1999. The sigmoid calibration option.
14. B. Zadrozny and C. Elkan, "Transforming Classifier Scores into Accurate
    Multiclass Probability Estimates", **KDD 2002**. The isotonic calibration
    option, which is the default here.
15. A. Niculescu-Mizil and R. Caruana, "Predicting Good Probabilities With
    Supervised Learning", **ICML 2005**. Why a calibration wrapper is needed at
    all.

## 13. Reproducing every number in this file

From the repository root, with the package importable:

```bash
pip install -e ".[test,plot]"
python -m pytest tests/ -q --junitxml=junit.xml
ruff check src/ tests/ examples/ validation/
PYTHONPATH=src python validation/validate_environment.py
PYTHONPATH=src python validation/validate_known_answers.py
PYTHONPATH=src python validation/validate_reachability.py
PYTHONPATH=src python validation/validate_planted_case.py
PYTHONPATH=src python validation/validate_calibration.py
PYTHONPATH=src python validation/validate_conservatism.py
PYTHONPATH=src python validation/validate_cli.py
PYTHONPATH=src python validation/worked_example.py
PYTHONPATH=src MPLBACKEND=Agg python examples/reachability_map.py
PYTHONPATH=src MPLBACKEND=Agg python examples/ambiguity_costs.py
PYTHONPATH=src MPLBACKEND=Agg python examples/detector_vs_threshold.py
PYTHONPATH=src MPLBACKEND=Agg python examples/calibration_reliability.py
PYTHONPATH=src MPLBACKEND=Agg python examples/conservatism_tradeoff.py
PYTHONPATH=src MPLBACKEND=Agg python examples/planted_case.py
```

Seeds: every random quantity in this file comes from seed **63001** except the
2000 random declarations of section 2, which use seeds **0 to 1999**, and the
plotting tests, which use seed 63002 to keep them independent of the benchmark.
Every validation script exits 0 by design, including the ones whose finding
goes against the product: the finding is printed as a `FINDING` line, the
measured value is asserted against the expectation documented beside the
assertion, and the raw output is committed.

Wall-clock on two contended cores: the whole set above runs in about 90 s. The
slowest single script is `validate_calibration.py` at 11.32 s; the test suite
took 11.24 s, 19.75 s and 27.57 s on three runs at the dependency ceiling and
18.50 s at the floor. Timings move by a factor of 2.5 between runs of the same
command on contended cores, so quote the counts and the ratios rather than the
seconds.
