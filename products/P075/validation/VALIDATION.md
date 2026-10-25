# Validation record

Validation level 2. Every number in this document comes from a script in this
directory that was executed in the build session dated 2026-10-25, and every
script writes its raw output beside itself as `<script>_output.txt`. Nothing
here is carried, estimated or placeholder.

## Summary

| | |
|---|---|
| tests, from junit XML `testsuite` attributes | 454 tests, 0 failures, 0 errors, 0 skipped |
| junit cross-check, literal `testcase` element count | 454, agrees with the declared attribute |
| `ruff check src/ tests/` | exit 0 |
| validation scripts | 14 |
| validation checks | **135 declared: 134 pass, 1 fails** |
| the failing check | the cumulative-exposure ageing model against calendar-time hazard integration; see below |
| blocking failures | 0 |
| negative controls | 8, all 8 verified to fire |
| `pytest.approx` default-tolerance audit | 0 of 454 assertions depended on the default |
| working-directory probe | 87 subprocess invocations, 0 non-zero exits |
| absolute paths in committed files | 0 matches over 83 text files, 5 patterns |
| interpreter measured on | Python 3.13.17, numpy 2.5.3, scipy 1.18.1, pyyaml 6.0.3 |
| `reliability`, `lifelines` | **not installed**; see deviation 1 |

The test count comes from the junit XML `testsuite` element and is
cross-checked against a literal count of `testcase` elements, never from
pytest stdout. A suite that collects zero tests prints success and exits 0;
that is how a product of this mission shipped publicly with a recorded 251
tests passing and a configuration that collected none.

Reproduce:

```bash
PYTHONPATH=src python -m pytest tests/ -q --junit-xml=junit.xml
python -c "import xml.etree.ElementTree as ET; r=ET.parse('junit.xml').getroot(); s=r if r.tag=='testsuite' else r.find('testsuite'); print(s.attrib); print(len(s.findall('.//testcase')))"
```

## Scripts and what each one establishes

| script | checks | establishes |
|---|---|---|
| `validate_structures.py` | 16 pass | every closed form against an independent reference |
| `validate_precision.py` | 8 pass | what reliability space costs, measured |
| `validate_cutsets.py` | 10 pass | cut sets against exhaustive state enumeration |
| `validate_importance.py` | 11 pass | importance against finite-difference perturbation |
| `validate_bridge.py` | 10 pass | the bridge polynomial, five independent routes |
| `validate_commoncause.py` | 12 pass | the headline, and the model's non-conservatism map |
| `validate_mission.py` | 6 pass, **1 fail** | the ageing model, including the check it fails |
| `validate_montecarlo.py` | 2 pass | simulation against the closed forms, budgets derived |
| `validate_negative_controls.py` | 20 pass | 8 injected violations, each verified to fire |
| `validate_cli.py` | 5 pass | every documented invocation, and `AI_ENABLED is False` |
| `regenerate_declarations.py` | 21 pass | declarations round-trip; 8 refusals exercised |
| `worked_example.py` | 4 pass | every figure the README quotes |
| `audit_approx_tolerances.py` | 4 pass | no assertion relies on `approx`'s absolute default |
| `probe_working_directories.py` | 5 pass | every script from five working directories |

## 1. Closed forms against independent references

`validate_structures_output.txt`. No reference below is a second evaluation of
the expression under test: `scipy` is a separate implementation by other
people, the enumeration walks the structure function, and the ODE solution
never forms the closed form at all.

| structure | reference | cases | worst relative difference | declared tolerance |
|---|---|---|---|---|
| series `R` | product of block reliabilities | 75 | `7.544748e-16` | `1e-14` |
| series `Q` | Bonferroni orders 1 and 2 (a bound, not an expression) | 75 | 0 outside the bracket | bracket + propagated slack |
| parallel `Q` | exact product in `Q` space | 18 | `1.836710e-16` | `1e-14` |
| parallel | exhaustive `2**n` state enumeration | 3 | `0.000000e+00` | `1e-14` |
| k-of-n `R` | `scipy.stats.binom.sf` | 300 | `4.281257e-16` | `1e-13` |
| k-of-n at `k = n` | the series chain | 4 | `3.017899e-16` | `1e-13` |
| k-of-n at `k = 1` | the parallel group | 4 | `1.147943e-16` | `1e-13` |
| k-of-n, non-identical | exhaustive state enumeration | 14 | `3.990254e-16` | `1e-13` |
| standby | `scipy.integrate.quad` on the convolution integral | 81 | `2.211217e-16` | `1e-10` |
| `m` cold standby units | the Erlang survival sum | 16 | `2.220446e-16` | `1e-13` |
| Weibull standby pair | `quad` on the convolution integral | 3 | `0.000000e+00` | `1e-10` |
| shared load | `solve_ivp`, DOP853, rtol 1e-12, atol 1e-15 | 108 | `4.038264e-14` | `1e-9` |
| shared load at degeneracy | the analytic limit `(1 + x) exp(-x)` | 9 | `2.220447e-16` | `1e-14` |
| shared load at `s = 1` | a parallel pair | 4 | `1.120369e-16` | `1e-12` |
| network | the derived bridge closed form | 7 | `9.673179e-16` | `1e-13` |
| network | exhaustive state enumeration | 4 | `2.269281e-16` | `1e-13` |

No tolerance was adjusted to make any of these pass. The one tolerance that is
loose relative to the measured agreement is the shared-load ODE at `1e-9`
against a measured `4.038264e-14`; it is loose because the reference is a
numerical ODE solution whose own accuracy is bounded by its rtol, and a
tolerance tighter than the reference's own error would not be a measurement.

## 2. Minimal cut sets against exhaustive state enumeration

`validate_cutsets_output.txt`. The structural recursion and the `2**n` walk
share no code path. Agreement is required on the **sets**, not on their count.

| | |
|---|---|
| diagrams compared | 13 |
| states enumerated | 147712 |
| disagreements | 0 |
| cut sets that did not fail the system | 0 |
| minimal cut sets not restored by one event | 0 |
| reported cut sets containing another | 0 |
| path sets checked against the structure function | 95 |
| path sets that did not succeed | 0 |
| path sets not minimal | 0 |

The largest diagram enumerated is the smallsat bus with common cause applied:
17 basic events, 131072 states, 130847 of them failures, 16 minimal cut sets,
identical by both routes.

What a declared common cause does to the structure, measured:

| diagram | cut sets (independent) | order 1 | cut sets (with cc) | order 1 |
|---|---|---|---|---|
| `redundant_pair` | 1 | 0 | 2 | 1 |
| `smallsat_bus` | 13 | 3 | 16 | 6 |
| `voting_triple` | 4 | 1 | 5 | 2 |

Each declared beta factor adds exactly one cut set of order one, by
construction, and that is checked rather than asserted.

## 3. Importance measures against finite-difference perturbation

`validate_importance_output.txt`.

| check | reference | result | tolerance |
|---|---|---|---|
| Birnbaum | central finite difference, 135 comparisons over 6 diagrams and 5 decades of step | worst relative `3.490329e-09` | `1e-6` |
| Birnbaum | whole-interval pivot (same relation as the implementation; consistency only) | `0.000000e+00` | `1e-14` |
| criticality identity | `(Q_sys - Q_sys(q_i=0)) / Q_sys`, 27 rows | worst relative `1.789122e-15` | `1e-10` |
| `I_FV >= I_C` | the definitions | 0 violations in 27 rows | exact |
| `I_FV` in `[0, 1]` | the definition | 0 violations | exact |
| Birnbaum non-negative | coherence | 0 violations | exact |
| `Q_sys` | exhaustive state enumeration, 6 diagrams | worst relative `6.492712e-16` | `1e-12` |

The finite-difference error grows as the step shrinks, exactly as the
cancellation in the numerator predicts: on `mixed_rates`, `2.170840e-13` at a
step of `1e-3` rising to `3.490329e-09` at `1e-7`. `Q_sys` is multilinear in
the `q_i`, so the difference is exact for any step in exact arithmetic and the
only error is that cancellation. Measuring across five decades rather than at
one step is what makes the agreement informative.

**The rare-event approximation, measured rather than assumed.** Fussell-Vesely
is computed exactly here, by enumerating the states of the events in the
relevant cut sets. The rare-event approximation -- the sum of cut-set
probabilities, which is what most tools report -- overestimates by up to
`1.268384e-01` absolute on these diagrams. Worst cases: `three_of_four`, every
member, exact `0.467479953` against rare-event `0.594318391`; `mixed_rates`,
every `q` member, `9.439e-02` excess.

**Refusals, exercised.** Importance measures are refused for
`load_sharing_pair`, `warm_standby` and `smallsat_bus`, each with the offending
node kind named in the message.

## 4. The bridge network

`validate_bridge_output.txt`. **The sourcing matters more than the numbers, so
it is stated first and the same statement is in the script's own output.**

**Derived in this repository, in full**: the closed form
`R = 2p^5 - 5p^4 + 2p^3 + 2p^2`, by pivotal decomposition on the bridge edge,
and the value `0.97848` at `p = 0.9`. The derivation is written out in
`reliablock.bridge`'s module docstring, reproduced in `docs/METHOD.md` section
8, and printed in the script's output.

**Cited for the method, not for a number**: Barlow, R. E. and Proschan, F.
(1975), *Statistical Theory of Reliability and Life Testing*, Holt, Rinehart
and Winston, chapter 1, for the pivotal decomposition of a coherent structure;
Billinton, R. and Allan, R. N., *Reliability Evaluation of Engineering
Systems*, 2nd edition, Plenum Press, 1992, for the bridge network and the
conditional-probability method applied to it.

**Not cited**: no page number and no published figure. This build had no access
to either volume, and a page number that cannot be checked is worse than none.
The product specification for P075 permits a derived-and-stated result in place
of a citation that cannot be verified, and that is what this is. See deviation
2 below.

**What stands in place of a citation**: five independent routes.

| route | at `p = 0.9` | worst relative difference against the derived form |
|---|---|---|
| derived closed form, Horner's rule | `0.978480000000000` | reference |
| non-identical pivotal form at equal `p` | `0.978480000000000` | `9.673179e-16` over 10 values of `p` |
| inclusion-exclusion over 4 minimal paths, 16 signed terms | `0.978480000000000` | included above |
| `reliablock.quantify` over the declared `network` node | `0.978480000000000` | included above |
| exhaustive enumeration of all 32 states | `0.978480000000000` | `1.128538e-14` over 6 values of `p` |

Further properties checked, none of which the implementation was written to
satisfy:

- **Self-duality.** `R(p) + R(1-p) = 1`, worst absolute residual
  `0.000000e+00` over 8 values of `p`. The bridge's minimal cut sets and path
  sets have the same shape -- two of size 2 and two of size 3 -- so this is a
  property of the structure.
- **Exactly half the states fail**: 16 of 32.
- **Not series-parallel.** The two decompositions a reader might reach for
  bracket the true answer: at `p = 0.9`, two-disjoint-paths
  `0.963900000000000`, exact `0.978480000000000`, merged-vertices
  `0.980100000000000`.
- **Esary-Proschan bracket has non-trivial width**, because the bridge's cut
  sets share components: `[9.781407801000e-01, 9.973487799000e-01]` at
  `p = 0.9`, width `1.920800e-02`, narrowest over three values of `p`
  `1.920800e-02`.
- **Monte Carlo**: 436663 trials at seed 20261025, sized for a half-width of 2
  percent of `Q`, achieved `4.328969e-04`, closed form inside the interval at
  `1.1511` standard errors.
- **Non-identical edges** `(0.91, 0.87, 0.95, 0.80, 0.99)`: all four routes
  agree to `3.380975e-16` on `0.985120910000000`.

## 5. The headline: with and without the declared common cause

`validate_commoncause_output.txt`. The diagram is
`examples/diagrams/smallsat_bus.yaml`, 13 blocks, 14 basic events, over 26298
hours. Declared beta factors: `power = 0.12`, `reaction_wheels = 0.08`,
`receivers = 0.05`. **Every beta is a declared input chosen to span a range.
None is data and none is taken from any source.**

| | independence assumed | declared common cause |
|---|---|---|
| system reliability | `0.969504265371` | `0.962289516309` |
| system unreliability | `3.049573463e-02` | `3.771048369e-02` |
| cumulative hazard | `3.097040e-02` | `3.843992e-02` |
| minimal cut sets | 13 | 16 |
| single points of failure | 3 | 6 |

- unreliability ratio: **`1.236582`**
- unreliability attributable to the independence assumption:
  **`7.214749063e-03`** absolute
- new single points of failure: `power::common_cause`,
  `reaction_wheels::common_cause`, `receivers::common_cause`

Per group:

| group | kind | beta | shape | `Q` independent | `Q` with cc | ratio | conservative |
|---|---|---|---|---|---|---|---|
| `power` | parallel | 0.12 | 1-of-2 | `9.650253e-04` | `4.527052e-03` | `4.6911` | yes |
| `reaction_wheels` | k_of_n | 0.08 | 3-of-4 | `3.903519e-03` | `5.414720e-03` | `1.3871` | yes |
| `receivers` | parallel | 0.05 | 1-of-2 | `2.625206e-03` | `4.995540e-03` | `1.9029` | yes |

**Both halves simulated independently.** The gap between the two columns is
confirmed by simulation and not only by arithmetic: closed-form gap in `R`
`7.214749063e-03`, simulated gap `6.968945176e-03`, difference
`2.458038863e-04` against a combined interval half-width of `6.835218608e-04`.

**Exactness at both limits.** At `beta = 0` the rewrite reproduces the
independent figure bit for bit, for `n = 2..5`, because it creates no
common-cause block at all. At `beta = 1` the group reduces to a single member,
worst relative difference `0.000000e+00`.

**The beta split preserves a member's total hazard.** With `beta = 0.17`,
`H_common = 0.17 H_total` and `H_member = 0.83 H_total`, summing back to
`H_total` to `1e-15` relative.

## 6. Where the beta-factor model stops being conservative

`validate_commoncause_output.txt`. This is a limitation of the model, measured
rather than caveated. The beta-factor model has exactly one common-cause mode,
the one that fails every member; a k-of-n group fails once `n - k + 1` members
have failed, which for `k >= 2` is fewer. Crossing located by bisection at
`beta = 0.1`:

| group | failures needed | ratio at `H = 0.01` | ratio at `H = 3` | crossing `H` | verdict |
|---|---|---|---|---|---|
| 1-of-2 | 2 | `10.905368` | `1.000960` | none | conservative everywhere |
| 1-of-3 | 3 | `1015.322358` | `1.002914` | none | conservative everywhere |
| 2-of-3 | 2 | `4.198132` | `0.997584` | `0.302944` | crosses |
| 1-of-4 | 4 | `101968.098219` | `1.005896` | none | conservative everywhere |
| 2-of-4 | 3 | `256.285271` | `0.995519` | `0.653729` | crosses |
| 3-of-4 | 2 | `2.516174` | `0.999621` | `0.124016` | crosses |
| 1-of-5 | 5 | `10247813.431300` | `1.009942` | none | conservative everywhere |
| 2-of-5 | 4 | `20557.783027` | `0.993070` | `0.924649` | crosses |
| 3-of-5 | 3 | `103.720793` | `0.999115` | `0.340012` | crosses |
| 4-of-5 | 2 | `1.841492` | `0.999958` | `0.067949` | crosses |
| 1-of-6 | 6 | `1029913730.774888` | `1.015093` | none | conservative everywhere |
| 2-of-6 | 5 | `1722249.954759` | `0.990350` | `1.141930` | crosses |
| 3-of-6 | 4 | `6907.998017` | `0.998346` | `0.533470` | crosses |
| 4-of-6 | 3 | `52.612956` | `0.999882` | `0.217911` | crosses |
| 5-of-6 | 2 | `1.503309` | `0.999996` | `0.042978` | crosses |

10 of 15 `(k, n)` pairs cross. Every `k = 1` pair is conservative at every
hazard measured.

**A product change this measurement caused.** `k = n` is absent from the table
because `Diagram.validate` now **refuses** a beta factor there. An earlier run
of the same sweep measured `k = n` as non-conservative at every hazard -- a
ratio of `0.918939` at `k = n = 6` -- which is correct arithmetic for a series
group and meaningless as a redundancy model. The refusal message quotes that
measured ratio, and `tests/test_diagram.py` pins it.

`reliablock.commoncause.group_beta_effects` reports the effect per group and
`reliablock.report.common_cause_comparison` prints a warning naming the
multiple-Greek-letter and alpha-factor models when any group crosses; neither
is implemented here. `validation/validate_cli_output.txt` and
`tests/test_cli.py` exercise that warning path.

## 7. Monte Carlo, with the budget derived from a target

`validate_montecarlo_output.txt`. The closed form is the reference throughout.
Budgets come from `n = z^2 Q (1 - Q) / h^2` at a pilot estimate of `Q`, for a
target half-width of **1 percent of the system unreliability**, so a closed
form wrong by 5 percent of `Q` would sit five interval half-widths away and
could not be missed. Seed 20261025; pilot 50000 trials.

| case | trials | achieved half-width | half-width / `Q` | standard errors | inside interval |
|---|---|---|---|---|---|
| `load_sharing_pair` | 70010 | `3.545572e-03` | `0.009999` | `0.7233` | yes |
| `redundant_pair` | 14403160 | `2.644009e-05` | `0.010053` | `0.2045` | yes |
| `smallsat_bus` | 1221257 | `3.062810e-04` | `0.009978` | `1.7495` | yes |
| `voting_triple` | 1275318 | `2.937978e-04` | `0.010017` | `1.9129` | yes |
| `warm_standby` | 120413 | `2.426241e-03` | `0.010006` | `1.8208` | yes |
| bus, independence assumed | 1221257 | `3.062810e-04` | `0.009978` | `1.7495` | yes |
| bus, declared common cause | 980257 | `3.772409e-04` | `0.010003` | `0.1433` | yes |
| bus, LEO phased profile | 1485940 | `2.525369e-04` | `0.009993` | `0.8434` | yes |

8 runs, 0 outside the interval, worst distance `1.9129` standard errors,
20777612 trials in total. A 95 percent interval is expected to miss about 1 run in 20 by
construction, so the pass criterion declared here is 4 standard errors and
interval membership is reported alongside.

The trial counts are not round because they were not chosen. They are the
ceiling of a formula evaluated at a measured pilot.

## 8. Numerical precision: what reliability space costs

`validate_precision_output.txt`. Machine epsilon `2.220446e-16`.

**Parallel: the complementary route loses the whole answer.** `Q = prod Q_i` is
exact; `Q = 1 - prod(1 - Q_i)` returns exactly `0.0` in **10 of 18** measured
cases where the answer is positive.

| `n` | `q` each | `Q` in `Q` space | `Q` in `R` space | relative error |
|---|---|---|---|---|
| 2 | `1.0e-06` | `1.000000e-12` | `9.999779e-13` | `2.212e-05` |
| 2 | `1.0e-08` | `1.000000e-16` | `1.110223e-16` | `1.102e-01` |
| 2 | `1.0e-09` | `1.000000e-18` | `0.000000e+00` | all digits lost |
| 3 | `1.0e-06` | `1.000000e-18` | `0.000000e+00` | all digits lost |
| 4 | `1.0e-06` | `1.000000e-24` | `0.000000e+00` | all digits lost |

Through the package's own quantifier, at `q = 1e-9` per block the parallel
pair's reliability is exactly `1.00000000000000000` and `1 - R` is exactly
`0.000000e+00`, while the `Q` column reads `1.000000e-18`.

**Series: the naive expression falls outside an independently derived bound.**
This is the check class that matters, because two implementations of one
expression agree when the expression is wrong. The Bonferroni bounds are
derived by truncating inclusion-exclusion and both sides are sums of
non-negative terms.

- hazard-space figures outside the bracket: **0 of 20**
- naive figures outside the bracket: **13 of 20**

The worked case the README quotes, ten blocks at `q = 1e-12`:

| | |
|---|---|
| Bonferroni lower bound | `9.999999999954999e-12` |
| Bonferroni upper bound | `9.999999999999999e-12` |
| bracket width | `4.500056e-23` |
| bracket slack allowed | `1.221245e-25` |
| hazard-space `Q` (this package) | `9.999999999955000e-12` |
| naive `1 - prod(1 - q_i)` | `9.999778782798785e-12` |
| naive relative error | `2.212172e-05` |
| significant figures lost | `4.66` |
| naive inside the bracket | **False** |

A property test asserting the implementation against the same expression would
have passed. That is why `reliablock.bounds` exists.

**k-of-n: two spaces, neither derived from the other.** Worst `R + Q - 1` over
27 cases: `2.220446e-16`. The `Q`-space route against the leading-order
expansion `C(n, n-k+1) q^(n-k+1)` at `q = 1e-9`: ratios `0.999999999` to
`0.999999998`, where a `Q` computed as `1 - R` would be zero in the whole
table.

**Bridge: `Q = 2q^2` kept to the subnormal range.**

| `q` each | `Q` computed | `2 q^2` | ratio | `R` | `1 - R` |
|---|---|---|---|---|---|
| `1.0e-05` | `2.000020e-10` | `2.000000e-10` | `1.000010000` | `0.99999999979999799` | `2.000e-10` |
| `1.0e-07` | `2.000000e-14` | `2.000000e-14` | `1.000000100` | `0.99999999999998002` | `1.998e-14` |
| `1.0e-09` | `2.000000e-18` | `2.000000e-18` | `1.000000001` | `1.00000000000000000` | `0.000e+00` |
| `1.0e-11` | `2.000000e-22` | `2.000000e-22` | `1.000000000` | `1.00000000000000000` | `0.000e+00` |

**Hazard round trip.** `H -> Q -> H` keeps full relative precision for every
`H` measured from `1e-18` to `1`, worst relative error `0.000000e+00`.
`H -> R -> H` returned exactly `0.0` for 1 of the 7 hazards measured, the one
at `H = 1e-18`, where the reliability is exactly `1.0`.

**Does this package compute in unreliability space?** Yes, for series,
parallel, k-of-n and network -- which is every structure whose quantification
is the independent-event arithmetic over its Boolean structure function.
Standby and shared load are computed in reliability space, because their
closed forms are sums of exponentials and an Erlang polynomial for which no
unreliability-space form exists; `Quantity.notes` says so on every such result
and the README repeats it under Limitations.

## 9. Negative controls, each verified to fire

`validate_negative_controls_output.txt`. **Offsetting a value by its tolerance
does not guarantee a violation**: the reference may already sit up to one
tolerance away on the other side, so only an offset strictly greater than
twice the tolerance guarantees failure. Every control uses a factor of 10,
anchors on the true value or on the bracket's own width, and reports whether it
fired as its own check.

| control | injected violation | real check | fired |
|---|---|---|---|
| 1 | k-of-n reliability offset by 10x the `1e-13` tolerance | passes at `1.536224e-16` | yes, perturbed error `1.000236e-12` |
| 2 | Bonferroni bracket exceeded by 10x (width + slack) | passes | yes |
| 2b | the naive series expression, a real defect and no injection | -- | yes, outside the bracket |
| 3 | a proper subset of a minimal cut set | the real cut set fails the system and is minimal | yes, the subset does not fail the system |
| 3b | a superset of a minimal cut set | -- | yes, the superset is not minimal |
| 4 | Birnbaum offset by 10x the `1e-6` tolerance | passes at `6.248409e-14` | yes, `1.000000e-05` |
| 5 | closed form offset by 10x the Monte Carlo interval width | passes at `2.0114` standard errors | yes, `41.2106` standard errors |
| 6 | the wrong parallel expression, a real mistake | passes at `0.000000e+00` | yes, `5.716592e+00` relative |
| 7 | the two series-parallel decompositions of the bridge | passes at `1.134640e-16` | yes, both rejected |
| 8 | Esary-Proschan bracket exceeded by 10x (width + slack) | passes | yes |

20 checks, all 20 pass. 8 of 8 controls fire.

**Control 3 found a defect in itself.** The first version asked whether the
system *still failed* when one event was restored, which is the negation of
minimality, and the control therefore reported a superset as minimal. The
inverted predicate is easy to write and the control firing is what caught it;
the corrected form is in the script and noted in its comments.

**Control 5's criterion was corrected rather than its seed.** The real check
was first written as membership of the 95 percent Wilson interval, and the
seeded run sits at `2.0114` standard errors -- just outside, which is an
expected 1-in-20 event. The criterion is now the 4-standard-error one this
repository declares everywhere else, with interval membership reported
alongside. Changing the seed would have been tuning.

## 10. The `pytest.approx` default-absolute-tolerance audit

`audit_approx_tolerances_output.txt`. `pytest.approx(expected, rel=r)` carries
a **default absolute tolerance of 1e-12** and passes on **either** bound. For
this package that is acute rather than theoretical: the unreliability figures
here are routinely of order `1e-5` and go down to `1e-18`, so any relative
tolerance tighter than about `2e-8` on a `1e-5` quantity is silently replaced
by the absolute default.

The audit re-runs the whole suite with `pytest.approx` monkey-patched so that
any call passing `rel=` without an explicit `abs=` **raises**, which makes the
omission itself an error rather than testing whether a tightened tolerance
still passes.

| | |
|---|---|
| direct `pytest.approx` calls with `rel=` and no `abs=` in `tests/test_*.py` | **0** |
| calls through the `rel()` / `abs_()` helpers | 124 |
| run 1, the suite as committed | 454 tests, 0 failures, 0 errors, exit 0 |
| run 2, with the omission made an error | 454 tests, 0 failures, 0 errors, exit 0 |
| **assertions that depended on the default** | **0 of 454** |

`tests/conftest.py` provides `rel(expected, t)` returning
`pytest.approx(expected, rel=t, abs=0.0)` and `abs_(expected, t)` returning
`pytest.approx(expected, rel=0.0, abs=t)`, and every comparison in the suite
goes through one of them.

**The audit found 5 real cases on its first run** --
`test_density_uses_its_own_survival_not_the_base_s`,
`test_the_model_is_an_approximation_for_a_weibull_block`,
`test_the_achieved_half_width_is_close_to_the_target`,
`test_exponential_lifetimes_have_the_right_mean` and
`test_scaled_hazard_lifetimes_are_longer_by_the_factor`, all of which called
`pytest.approx` directly with `rel=` and no `abs=`. All five were routed
through the helpers.

## 11. The working-directory probe

`probe_working_directories_output.txt`. Four products in Batch 07 passed every
check in their own build and then failed the release gate, which runs each
validation script with its working directory set to the script's own directory
rather than to the product root. The build agent, its self-report and the
coordinator's re-measurement all ran from the product root -- three independent
verifications that shared an assumption instead of a method.

| | |
|---|---|
| scripts run from all five directories | 17 |
| scripts run from two | 1 (`validate_cli.py`) |
| subprocess invocations | 87 |
| non-zero exits | **0** |

Working directories: the filesystem root, the temporary directory, the product
root, `validation/` and `src/`. `python -I` is deliberately not used: it strips
the script's own directory from `sys.path` and makes `import _bootstrap` fail,
which is a false failure indistinguishable from the defect being probed for.

`audit_approx_tolerances.py` is excluded (31 s measured, runs the whole suite
twice) and `validate_cli.py` runs from two directories (22 s measured, spawns
22 subprocesses with `cwd` set to the repository root explicitly). Both
exclusions are stated in the script's own output with the measured cost.

The probe takes about 4 minutes wall clock, against the release gate's 600 s
per-script limit.

Also in this script:

- **absolute paths in committed files**: 83 text files scanned against 5
  patterns, **0 matches**. The patterns are assembled from fragments in the
  source so that the script does not match its own text; two consecutive
  sessions of this mission introduced this defect in the act of writing about
  it.
- **every example wrote its screenshot**: 6 of 6, all above 60 kB.
- **`.gitignore` does not shadow a deliverable**: 13 patterns, 1 unanchored
  non-glob pattern (`__pycache__/`), 0 negation patterns, 89 deliverables
  scanned, **0 shadowed**. `git` is not run here -- this build is forbidden to
  run it -- so the patterns are matched directly with the anchoring rule
  applied explicitly, and the output records that a pattern beginning with `!`
  is a negation that re-includes a file, so a check reading only
  `git check-ignore`'s exit status would report a deliberately re-included file
  as ignored.

## 12. THE CHECK THAT FAILS

`validate_mission_output.txt` declares and **fails**:

> the cumulative-exposure model agrees with calendar-time hazard rate
> integration to within 1 percent for a non-exponential block

| | |
|---|---|
| cases | 20 |
| cases agreeing to within 1 percent | **4**, all of them the exponential ones |
| worst relative difference | **`1.852733e+01`**, that is `1852.73` percent |

| Weibull shape | stress factor | cumulative exposure `H` | calendar `H` | relative difference |
|---|---|---|---|---|
| 0.60 | 2.0 | `4.855933748e-01` | `6.319194309e-01` | `2.316e-01` |
| 0.60 | 4.0 | `6.597539554e-01` | `1.134296717e+00` | `4.184e-01` |
| 1.00 | 2.0 | `3.000000000e-01` | `3.000000000e-01` | `1.850e-16` |
| 1.00 | 4.0 | `5.000000000e-01` | `5.000000000e-01` | `0.000e+00` |
| 1.80 | 4.0 | `2.871745887e-01` | `1.027359822e-01` | `1.795e+00` |
| 2.50 | 4.0 | `1.767766953e-01` | `2.737537680e-02` | `5.458e+00` |
| 3.50 | 4.0 | `8.838834765e-02` | `4.526392062e-03` | `1.853e+01` |

**Why it is published rather than removed.** The two models answer different
physical questions. Cumulative exposure -- the Nelson model, which this package
implements -- says a stress factor advances the block along its own ageing
curve. Calendar-time integration says the stress multiplies the block's
instantaneous hazard rate at its current calendar age. For a wear-out process
the first is usually the intended one, which is why it is the one implemented.
But a reader who sees `stress_factors: {obc: 3.0}` may well assume the second,
and the magnitude of the difference -- a factor of nearly twenty at shape 3.5
-- is why the check is declared and its failure carried into the README,
`docs/METHOD.md` and `CHANGELOG.md` rather than removed.

The same divergence is measured for a lognormal block, reported without a
declared tolerance because the two models are not meant to agree and declaring
one would imply that they are.

The companion checks in the same script pass: the model **is** exact for an
exponential block (worst relative `0.000000e+00` over four rates against a
`quad` integration of the piecewise-constant stress profile), a one-shot block
is refused by both a duty cycle and a stress factor, and the per-phase
breakdown sums to the reported effective age for every block to `1e-14`.

## 13. Defects found during this build

Three were found by Hypothesis and not by the hand-written suite. All are
fixed, and all are pinned as named regression tests with literal inputs in
`tests/test_regressions.py`, because `.hypothesis/` is gitignored and a
counterexample found by one run is lost on a fresh clone.

| # | found by | defect | symptom |
|---|---|---|---|
| 1 | `test_k_equals_n_is_series` | the Poisson-binomial tail is a sum of non-negative float64 terms and can land above 1 | returned `1.0000000000000002` at `rates = [0.0078125, 5e-05, 5e-05, 1.0000000000000003e-09, 1e-09]` over 229415.5 h; `hazard_from_unreliability` then **raised** on a valid diagram |
| 2 | `test_esary_proschan_bounds_bracket_a_nested_structure` | the bracket slack was a fixed `8 eps`, then an operation count; neither follows the conditioning | at `rates = [0.0078125, 5.960464477539063e-08]`, `hours = 762213`, the exact reliability sits `2.7e-17` below a lower bound whose slack allowed `3.4e-18` |
| 3 | `test_the_naive_expression_falls_outside_the_bound` | the bracket slack was floored at an absolute constant | a slack of `1.8e-15` on a bracket of width `4.500056e-23` made the check vacuous and admitted the naive expression it exists to catch |
| 4 | hand-written `test_degenerate_limit_is_continuous` | the shared-load closed form switched to an analytic limit below a threshold, which was a discontinuity | the threshold would have had to be tuned; replaced with the `expm1` form, which is exact at degeneracy and has no constant |
| 5 | hand-written `test_edge_on_no_path_is_rejected` | the dangling-edge check tested reachability *with* the edge present | an RBD network is undirected, so in a connected graph the test passes for a stub whose reliability can never enter the answer |
| 6 | smoke test | `ages_with_time` was a dataclass field on the base class | it occupied the first positional slot, so `Exponential(1e-5)` set the flag and left the rate at 0 -- a block that never fails, reported without complaint |
| 7 | smoke test | the Poisson-binomial recurrence updated its array in place through a numpy view | the row being written was partially read back as the previous row; it produced a plausible number that disagreed with `scipy.stats.binom.sf` |
| 8 | `validate_commoncause.py` sweep | a beta factor on a `k = n` group was accepted | measured non-conservative at every hazard, ratio `0.918939` at `k = n = 6`; now refused, with the measured ratio in the message |
| 9 | `audit_approx_tolerances.py` | 5 assertions called `pytest.approx` with `rel=` and no `abs=` | the relative tolerance was silently replaced by the 1e-12 absolute default |
| 10 | negative control 3 | the control's own minimality predicate was inverted | it reported a superset of a minimal cut set as minimal |

**Which were found by Hypothesis rather than by the hand-written tests**:
defects 1, 2 and 3. Defect 1 is a crash on a valid input that no hand-written
case reached, because it needs a five-block `k = n` group at a hazard large
enough that `Q` saturates. Defects 2 and 3 are both the same mistake in a
tolerance, in opposite directions, and neither would have been found by a
hand-chosen input: defect 2 needed a two-cut-set structure at a specific
conditioning, and defect 3 needed a bracket four million times narrower than
the slack.

## 14. Deviations from the product specification

**Deviation 1: no cross-check against `reliability` or `lifelines`.** The
product specification names both in the alternatives table, and both are named
and recommended in the README. **Neither is installed in the environment these
numbers were measured in**, and the build was forbidden to install anything.
No number in this repository came from either of them and none claims to have
been cross-checked against them. Every output file records their absence in
its header, verbatim:

```
  reliability      : not installed
  lifelines        : not installed
```

What replaced the cross-check: every structure is scored against an
independent reference that *is* available -- `scipy.stats.binom.sf`,
`scipy.integrate.quad`, `scipy.integrate.solve_ivp`, exhaustive state
enumeration over the structure function, and independently derived
Esary-Proschan and Bonferroni bounds. Those are listed in section 1.

**Deviation 2: no page number or published figure for the textbook bridge
network.** The specification asks for "a known textbook bridge network
reproduced with its published answer cited", and permits a derived-and-stated
result where a specific source cannot be verified. This build had no access to
either volume cited for the method, so no page number and no published figure
is quoted. The closed form and the value `0.97848` at `p = 0.9` are **derived
in full** in `reliablock.bridge`'s module docstring, reproduced in
`docs/METHOD.md` section 8 and in `validate_bridge_output.txt`, and verified
five independent ways. See section 4.

**Not a deviation, but worth stating: see section 15 for exactly how many
points in each declared dependency range were executed.**

## 15. What was executed in each declared dependency range

`pyproject.toml` bounds every dependency on both sides. How much evidence
stands behind each bound differs, and the difference is worth stating rather
than leaving a reader to assume.

**The three runtime dependencies were measured at exactly one point each.**
This build ran inside a pre-pinned interpreter and was forbidden to install
anything, so `numpy>=2.5,<2.6`, `scipy>=1.18,<1.19` and `pyyaml>=6.0,<6.1`
record the minor version the numbers come from and are **not** evidence that
both ends of each range were exercised. Pinning below a future release is a
real cost, not a free safety measure: the product cannot take that release
without a port.

| runtime dependency | declared range | points executed |
|---|---|---|
| `numpy` | `>=2.5,<2.6` | 1 (2.5.3) |
| `scipy` | `>=1.18,<1.19` | 1 (1.18.1) |
| `pyyaml` | `>=6.0,<6.1` | 1 (6.0.3) |

**The four tool-only packages were measured at two points each.** The build
environment carried two independent installations with **identical** numpy,
scipy and pyyaml and different tooling, and the whole suite, every validation
script and every example pass under both. Each ceiling is set above the higher
measured point and no further.

| tool-only package | declared range | points executed |
|---|---|---|
| `pytest` | `>=8.4,<9.2` | 2 (8.4.2 and 9.1.1) |
| `hypothesis` | `>=6.140,<6.171` | 2 (6.140.3 and 6.170.9) |
| `matplotlib` | `>=3.10,<3.12` | 2 (3.10.8 and 3.11.2) |
| `ruff` | `>=0.14,<0.18` | 2 (0.14.2 and 0.17.2) |

What was run on each:

| | pinned interpreter | second interpreter |
|---|---|---|
| `python -m pytest tests/ -q --junit-xml=...` | 454 tests, 0 failures, 0 errors, 0 skipped | 454 tests, 0 failures, 0 errors, 0 skipped |
| junit `testsuite` attribute against literal `testcase` count | 454 = 454 | 454 = 454 |
| `ruff check src/ tests/` | exit 0 | exit 0 |
| `python -c "import reliablock"` with only `PYTHONPATH=src` | exit 0 | exit 0 |
| `python -m reliablock --help` | exit 0 | exit 0 |
| every script in `validation/` and `examples/`, run with `cwd` set to its own directory | 20 of 20 exit 0 | 20 of 20 exit 0 |

**The committed output files were written by the pinned interpreter**, so every
header in this directory records `matplotlib 3.10.8`, `pytest 8.4.2`,
`hypothesis 6.140.3`, `ruff 0.14.2`. The second interpreter's run produced the
same verdicts and, because the three runtime dependencies are identical, the
same numbers; it is recorded here as a second point in the tool ranges and not
as a second set of figures.

## 16. Reproducing this document

Every script, in the order the CI workflow runs them:

```bash
PYTHONPATH=src python validation/validate_structures.py
PYTHONPATH=src python validation/validate_precision.py
PYTHONPATH=src python validation/validate_cutsets.py
PYTHONPATH=src python validation/validate_importance.py
PYTHONPATH=src python validation/validate_bridge.py
PYTHONPATH=src python validation/validate_commoncause.py
PYTHONPATH=src python validation/validate_mission.py
PYTHONPATH=src python validation/validate_montecarlo.py
PYTHONPATH=src python validation/validate_negative_controls.py
PYTHONPATH=src python validation/validate_cli.py
PYTHONPATH=src python validation/regenerate_declarations.py
PYTHONPATH=src python validation/worked_example.py
PYTHONPATH=src python validation/audit_approx_tolerances.py
MPLBACKEND=Agg PYTHONPATH=src python validation/probe_working_directories.py
```

Each writes `<script>_output.txt` beside itself. Every script is seeded. A
failed check is reported in line, again in the `FAILED CHECKS` block at the end
of the output file, and carried into this document; it does **not** change the
script's exit code, because the release gate uses the exit code to decide
whether the script *ran*, and a script that exited non-zero over one tolerance
would be indistinguishable from one that crashed. Only a blocking failure --
a file that could not be written, a declaration that will not round-trip --
sets a non-zero exit. The convention is stated in `_bootstrap.py` and printed
at the foot of every output file.

Nothing in this repository tunes a tolerance to make a check pass.
