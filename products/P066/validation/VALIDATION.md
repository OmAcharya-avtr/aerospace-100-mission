# Validation evidence — budgetroll 0.1.0

**Validation level 1 (educational grade).** Every number in this document was
produced by running the script named beside it, in this container, on
2026-10-16, and the raw output of each run is committed next to the script as
`validation/<script-name>_output.txt`. No number here was typed from memory or
carried over from an earlier run.

**What a clean run means.** The declared budget is arithmetically consistent
and the combination rules it declares agree with the correlation structure it
declares. It is not a statement that any estimate is good, that any margin is
adequate, or that any system works. This software is educational-grade. It is
**not flight-qualified, not certified, and not approved for operational
aerospace use.**

**What is NOT validated here.** No input number in any shipped example case is
measured. The loads, losses, masses, antenna sizes and noise temperature in
`examples/cases/` are illustrative figures chosen for the examples, and the
three margin policies in those files are declared by the author of those files
rather than reproduced from any standard. What is validated is the arithmetic,
the published constants of the link equation, and the behaviour of the checks.

## Method references

- **JCGM 100:2008**, *Evaluation of measurement data — Guide to the expression
  of uncertainty in measurement* (the GUM), Joint Committee for Guides in
  Metrology, BIPM. Section 5.1.2 gives the law of propagation of uncertainty
  for uncorrelated inputs and section 5.2.2 the correlated case. For a sum
  with unit sensitivity coefficients these reduce to `u = sqrt(w^T R w)`,
  which is what `src/budgetroll/correlation.py` implements.
- **Friis, H. T.**, "A Note on a Simple Transmission Formula", *Proceedings of
  the I.R.E.*, 34(5), May 1946, pp. 254–256. The free-space transmission
  formula.
- **Wertz, J. R. and Larson, W. J., eds.**, *Space Mission Analysis and
  Design*, 3rd edition, Microcosm Press and Kluwer Academic Publishers, 1999,
  chapter 13, "Communications Architecture". Cited **by chapter title only**
  for the decibel form of the link equation. **No equation number and no page
  number is quoted anywhere in this repository**, because that edition was not
  available in the container the validation ran in and the build guide for
  this portfolio forbids inventing one. Nothing in section 4 below rests on
  that citation: every constant the link equation uses is reproduced from
  first principles and compared with its published rounded value.
- **ANSI/AIAA S-120A-2015**, *Mass Properties Control for Space Systems*, and
  **NASA/SP-2016-6105 Rev 2**, *NASA Systems Engineering Handbook*,
  section 6.5. Cited as the published sources of maturity-keyed contingency
  practice. **No table from either is reproduced here.** The package ships no
  margin schedule of its own and asserts nothing about the correctness of any
  schedule declared through it.
- SI defining constants used directly: the speed of light
  c = 299792458 m/s exactly, and the Boltzmann constant
  k = 1.380649e-23 J/K exactly (2019 SI redefinition).

## How to reproduce every number

From the repository root, with `pip install -e ".[test]"`:

```bash
python -m pytest tests/ -q --junitxml=junit.xml
cd validation
python validate_environment.py
python validate_known_answers.py
python validate_identities.py
python validate_link_budget.py
python validate_power_budget.py
python validate_checks.py
python validate_parser_rejections.py
python validate_cli.py
cd ..
python validation/worked_example.py
cd examples && python combination_comparison.py && python margin_waterfall.py \
  && python sensitivity_tornado.py && python assumption_check.py
```

`worked_example.py` is run from the repository root because the code block it
contains — the one quoted in README.md — uses repository-relative paths on
purpose. Each script exits 0 when every check in it passed and 1 otherwise.

## 1. Test suite

`validation/validate_environment.py`

Counted from the junit XML, never from pytest's summary line. A suite that
collects zero tests prints success and exits 0; that defect shipped elsewhere
in this portfolio, and the count below is parsed from the `testsuite` element
and cross-checked against a count of `testcase` elements.

| quantity | value |
|---|---|
| `testsuite/@tests` | 264 |
| `testcase` elements counted | 264 |
| cross-check | MATCH |
| `testsuite/@failures` | 0 |
| `testsuite/@errors` | 0 |
| `testsuite/@skipped` | 0 |
| pytest exit status | 0 |
| wall clock | 9.59 s on 2 contended cores; varies between runs |

No test is marked xfail and no test is skipped.

## 2. Hand-calculated known answers

`validation/validate_known_answers.py`

Every expected value was computed by hand from the YAML before the code was
run on it. The arithmetic is printed in the script's output and repeated in
the comments of `tests/test_known_answers.py`. Tolerance 1e-9 absolute on both
sides.

### `examples/cases/clean_mass.yaml`

| quantity | hand arithmetic | expected | computed |
|---|---|---|---|
| nominal | 2.00 + 1.50 + 0.80 + 1.20 | 5.500 kg | 5.500000000000 |
| contingency allowance | 5 % of each of four weighed items | 0.275 kg | 0.275000000000 |
| predicted | 5.500 + 0.275 | 5.775 kg | 5.775000000000 |
| 1 sigma | sqrt(0.0100+0.0064+0.0025+0.0036) = sqrt(0.0225) | 0.150 kg | 0.150000000000 |
| margin | 10.000 - 5.775 | 4.225 kg | 4.225000000000 |
| margin at 1 sigma | 10.000 - 5.925 | 4.075 kg | 4.075000000000 |
| findings | — | none | 0 error, 0 warning, 0 info |

### `examples/cases/planted_findings.yaml`

| quantity | hand arithmetic | expected | computed |
|---|---|---|---|
| structure nominal | 6.0 + 3.0 + 2.0 | 11.00 kg | 11.000000000000 |
| structure allowance | 0.30 + 0.30 + 0.40 | 1.00 kg | 1.000000000000 |
| structure margin | 11.50 - 12.00 | -0.50 kg | -0.500000000000 |
| structure declared 1 sigma | sqrt(4 + 1 + 9) = sqrt(14) | 3.741657 kg | 3.741657386774 |
| structure exact 1 sigma | sqrt(14 + 3.6 + 10.8) = sqrt(28.4) | 5.329165 kg | 5.329165037789 |
| root nominal | 11.0 + 3.5 + 1.8 + 1.8 | 18.10 kg | 18.100000000000 |
| root allowance | 1.00 + 0.60 + 0.18 + 0.13 | 1.91 kg | 1.910000000000 |
| root declared 1 sigma (linear) | 3.741657 + 0.5 + 0.4 + 0.5 | 5.141657 kg | 5.141657386774 |
| root exact 1 sigma | sqrt(28.94) | 5.379591 kg | 5.379590318044 |

### `examples/cases/cubesat_power.yaml`

| quantity | expected | computed |
|---|---|---|
| nominal | 7.1500 W | 7.150000000000 |
| contingency allowance | 0.7525 W | 0.752500000000 |
| predicted | 7.9025 W | 7.902500000000 |
| margin against the 8.4000 W allocation | 0.4975 W | 0.497500000000 |

### `examples/cases/cubesat_link.yaml`

| quantity | expected | computed |
|---|---|---|
| nominal | 6.3000 dB | 6.300000000000 |
| contingency allowance | 0.5650 dB | 0.565000000000 |
| predicted | 6.8650 dB | 6.865000000000 |
| margin against the 17.6655 dB allocation | 10.8005 dB | 10.800500000000 |

## 3. The ten checks

`validation/validate_checks.py`

Each of BR001 to BR010 is raised by the smallest declaration that triggers it
and then silenced by a minimally altered declaration. A check that cannot be
made silent reports nothing.

| case | error | warning | info | blocking |
|---|---:|---:|---:|---|
| `clean_mass.yaml` | 0 | 0 | 0 | no |
| `planted_findings.yaml` | 6 | 4 | 2 | yes |
| `cubesat_power.yaml` | 4 | 2 | 3 | yes |
| `cubesat_link.yaml` | 0 | 0 | 2 | no |

Per-code counts on `planted_findings.yaml`, asserted in
`tests/test_known_answers.py::test_planted_finding_counts_per_code`:

| code | count | severity raised |
|---|---:|---|
| BR001 RSS node with a declared correlation among its descendants | 1 | error |
| BR002 hierarchy diverges from the exact correlated form | 3 | 2 error, 1 info |
| BR003 allocation exceeded | 1 | error |
| BR004 margin lost at k sigma | 1 | warning |
| BR005 contingency override diverges from the policy | 1 | warning |
| BR006 zero uncertainty at an immature level | 1 | warning |
| BR007 combination rule on a single child | 1 | info |
| BR008 declared correlation matrix is not positive semi-definite | 1 | error |
| BR009 `correlated` rule with no correlation to apply | 1 | error |
| BR010 shared common driver with no declared correlation | 1 | warning |

BR002 raises three findings on this case rather than one, and the reason is
recorded rather than tidied away: the rule is graded by direction. A
hierarchy that **understates** the exact correlated form is an error, and the
`structure` node and the root both do; a hierarchy that **overstates** it is
information, which is what a deliberately conservative `linear` node does, and
the `thermal` node is one.

## 4. Published constants of the link budget

`validation/validate_link_budget.py`

Each constant is computed from its definition and compared with the published
rounded value that appears in the engineering form of the equation.

| constant | computed | published | difference | tolerance | verdict |
|---|---|---|---|---|---|
| free-space path loss constant, d in km and f in GHz | 92.447783 dB | 92.45 dB | -0.002217 dB | 0.005 dB | PASS |
| Boltzmann constant | -228.599167 dBW/K/Hz | -228.6 dBW/K/Hz | +0.000833 dB | 0.005 dB | PASS |
| aperture gain constant at 55 % efficiency, D in m and f in GHz | 17.810210 dB | 17.8 dB | +0.010210 dB | 0.02 dB | PASS |
| uncoded coherent BPSK requirement at BER 1e-5 | 9.587858 dB | 9.6 dB | -0.012142 dB | 0.02 dB | PASS |

The BPSK figure is obtained by solving `0.5 erfc(sqrt(Eb/N0)) = 1e-5`
numerically with `math.erfc`; it is the standard tabulated 9.6 dB and is
reproduced here rather than asserted.

Free-space path loss at 1000 km and 2.2 GHz, computed two independent ways:

| route | result |
|---|---|
| from metres and hertz, `20 log10(d) + 20 log10(f) + 20 log10(4 pi / c)` | 159.296236838 dB |
| from the published 92.45 dB engineering form | 159.296236838 dB |
| difference | +2.842e-14 dB |

The assembled downlink, with all spacecraft and ground station parameters
illustrative:

| line | value |
|---|---|
| transmit power 2.0 W | +3.0103 dBW |
| transmit line loss | -1.0000 dB |
| spacecraft patch antenna | +6.0000 dBi |
| EIRP | +8.0103 dBW |
| free-space path loss at 1000 km | -159.2962 dB |
| other propagation losses | -2.5000 dB |
| ground station 3.0 m dish at 55 % efficiency | +34.2011 dBi |
| G/T at 150 K system noise temperature | +12.4402 dB/K |
| Boltzmann constant | +228.5992 dBW/K/Hz |
| data rate 1.0 Mbit/s | -60.0000 dB |
| **received Eb/N0** | **+27.2534 dB** |
| required Eb/N0, BPSK at BER 1e-5 | -9.5879 dB |
| **gross link margin** | **+17.6655 dB** |

That gross margin is the allocation declared in
`examples/cases/cubesat_link.yaml`; the two agree to 4.8e-5 dB, which is the
rounding of the four-decimal value written into the YAML. Rolling the
degradation budget up against it gives 6.8650 dB predicted, 10.8005 dB
remaining, 0.839643 dB of declared 1 sigma and 9.9609 dB of margin at one
standard deviation, with no error and no warning raised.

## 5. Orbit geometry behind the power budget

`validation/validate_power_budget.py`

| quantity | expression | result |
|---|---|---|
| orbital radius at 500 km | 6378.137 + 500 | 6878.137 km |
| period | `2 pi sqrt(r^3 / mu)`, mu = 398600.4418 km³/s² | 5676.978 s = 94.6163 min |
| eclipse fraction at beta = 0 | `arcsin(R_E / r) / pi`, cylindrical shadow | 0.377882 |
| eclipse duration | — | 35.7490 min |
| sunlit duration | — | 58.8673 min |

The eclipse expression is the cylindrical-shadow result with the Sun in the
orbital plane, which is the worst case; it ignores the penumbra and does not
apply at high beta angle, where the orbit can be fully sunlit. The period
ignores J2, which changes the nodal period at this altitude by about
0.1 per cent. Both are stated in the script docstring with their assumptions.
Neither is used to derive the 8.40 W allocation, which is an illustrative
figure and is not claimed to be derived from anything.

## 6. The power budget rollup and its four combination rules

`validation/validate_power_budget.py`

| quantity | result |
|---|---|
| nominal | 7.1500 W |
| contingency allowance | 0.7525 W |
| predicted | 7.9025 W |
| allocation | 8.4000 W |
| margin | 0.4975 W (5.9226 %) |
| margin at k = 2 sigma | **-0.158244 W** |

| combination rule at the root | 1 sigma |
|---|---|
| linear, fully correlated upper bound | 0.790000 W |
| root-sum-square over all nine leaves, independent | 0.322335 W |
| the hierarchy the YAML declares | 0.327872 W |
| exact form with the declared correlations | 0.351198 W |

The declared RSS **understates the exact correlated form by 6.6418 per cent**,
and that is the finding the example exists to produce: the budget declares
`rss` at the payload node and at the root while also declaring a correlation
of 0.6 between the two payload items, which is check BR001.

Sensitivity, +10 per cent on each leaf in turn, top three:

| leaf | predicted | value share | variance share | change in root margin |
|---|---:|---:|---:|---:|
| Transmitter, duty cycled | 1.7600 W | 22.27 % | 32.43 % | -0.1760 W |
| Detector and front end | 1.4400 W | 18.22 % | 34.15 % | -0.1440 W |
| Reaction wheel assembly | 1.1550 W | 14.62 % | 2.92 % | -0.1155 W |

The leaf that costs the most margin is not the leaf that carries the most
uncertainty, which is why the package reports the two separately.

## 7. Algebraic identities

`validation/validate_identities.py`

5258 cases: 258 exhaustive over the grid (0, 0.25, 1, 3.5, 12, 100) at lengths
1 to 3, and 5000 random vectors of 2 to 12 uncertainties drawn uniformly from
[0, 1000] with seed 20261016. All five identities are exact in exact
arithmetic; the numbers below are the measured floating-point disagreement.

| identity | worst disagreement | tolerance | verdict |
|---|---|---|---|
| RSS ≤ linear | smallest (linear - RSS) seen 0.000e+00; 0 violations | — | PASS |
| associativity of independent aggregation | 3.076e-16 relative | 1e-12 | PASS |
| `R = I` reduces the quadratic form to RSS | 2.602e-16 relative | 1e-12 | PASS |
| `R = J` reduces the quadratic form to the linear sum | 2.428e-16 relative | 1e-12 | PASS |
| variance shares sum to 1 | 4.441e-16 absolute | 1e-9 | PASS |

Two-sided reference points for the correlated form with u = (3, 4):

| rho | expected | computed |
|---|---|---|
| +1.0 | 7 (the linear sum) | 7.000000000000 |
| +0.5 | sqrt(37) = 6.082763 | 6.082762530298 |
| 0.0 | 5 (the RSS) | 5.000000000000 |
| -1.0 | 1 (the absolute difference) | 1.000000000000 |

### Counterexamples found and pinned

`.hypothesis/` is gitignored, so a falsifier found by one run is lost on a
fresh clone. Every one found while building this release is pinned in
`tests/test_regression.py` with its literal inputs.

| what was found | where it is pinned |
|---|---|
| With four weights at exactly DBL_MIN = 2.2250738585072014e-308, `sqrt(w^T J w)` returns 0.0 while the arithmetic sum is 8.900295434028806e-308. The relative disagreement is 1.0 and no relative tolerance rescues it. | `test_regression_quadratic_form_underflows_at_subnormal_weights` |
| The same loss in the hierarchical RSS. A 1 sigma below about 1.49e-154 has a subnormal square and loses bits; below about 2.22e-162 the square underflows to zero and the term disappears. | `test_regression_rss_underflows_at_the_same_threshold` |
| Worst associativity disagreement found in 40000 draws: u = [1.9, 45.331874064628245, 1.9] split after the second term, 45.4114391559144 against 45.41143915591439, a relative 3.13e-16. | `test_regression_worst_associativity_disagreement_found` |
| Worst variance-share drift found: twelve uncertainties spanning 152 down to 2.2e-313 sum to 1.0000000000000004. | `test_regression_variance_shares_sum_drift_with_mixed_magnitudes` |
| Three items of 0.1 kg against an allocation of 0.3 kg give a margin of -5.551115123125783e-17, because 0.1+0.1+0.1 is 0.30000000000000004. A strict `margin < 0` test calls that an overrun. | `test_regression_margin_sign_at_the_rounding_boundary` |

The last of these changed the implementation rather than only the test. BR003
and BR004 now carry a rounding band of `1e-12 * max(|allocation|, 1)`: inside
it the finding is graded a warning and worded as closed exactly, because the
sign of a difference that small is a property of the summation order and not
of the budget. `test_regression_a_genuine_overrun_is_still_an_error` holds the
band to a real overrun of engineering size.

## 8. Dependency ranges, measured at both ends

Every declared range was installed at both ends and the whole suite plus every
validation script was run at each end. The two configurations:

| package | low end, measured | high end, measured | declared range |
|---|---|---|---|
| numpy | 2.3.4 | 2.5.3, and 2.5.4 in the cold-clone run | `>=2.3.4,<2.6` |
| pyyaml | 6.0.1 | 6.0.3 | `>=6.0.1,<6.1` |
| matplotlib (`plot`, `test` extras) | 3.10.0 | 3.11.2 | `>=3.10.0,<3.12` |
| pytest (`test` extra) | 8.3.5 | 9.1.1 | `>=8.3.5,<9.2` |
| hypothesis (`test` extra) | 6.100.0 | 6.170.0 | `>=6.100.0,<6.171` |
| ruff (`test` extra) | 0.16.8 | 0.16.10 in the cold-clone run | `>=0.16.8,<0.17` |

Python 3.13.16 in every configuration.

A third configuration was measured: a clean copy of the repository in a fresh
virtual environment with `pip install -e ".[test]"`, letting pip resolve
inside the declared ranges. It resolved numpy 2.5.4 and ruff 0.16.10, which
are inside the ranges but were not the container's versions, and it is the
configuration a new user actually gets. 264 of 264 tests passed,
`ruff check src/ tests/ examples/ validation/` was clean, and all eight
validation output files were byte-identical to the committed ones. The
cold-clone run also confirmed that `/junit.xml` in this product's
`.gitignore` excludes the root scratch file and nothing else.

| measurement | low end | high end |
|---|---|---|
| `testsuite/@tests` from junit XML | 264 | 264 |
| `testcase` elements | 264 | 264 |
| failures, errors, skipped | 0, 0, 0 | 0, 0, 0 |
| pytest exit status | 0 | 0 |

**Diff of the committed validation output between the two configurations.**
Not the test verdict alone: a suite that passes while its numbers move is a
defect this portfolio has shipped before. All eight validation output files
plus `worked_example_output.txt` were regenerated at the low end and compared
byte for byte with the high-end versions:

| file | result |
|---|---|
| `validate_checks_output.txt` | identical |
| `validate_cli_output.txt` | identical |
| `validate_identities_output.txt` | identical |
| `validate_known_answers_output.txt` | identical |
| `validate_link_budget_output.txt` | identical |
| `validate_parser_rejections_output.txt` | identical |
| `validate_power_budget_output.txt` | identical |
| `worked_example_output.txt` | identical |
| `validate_environment_output.txt` | differs only in the recorded library versions and the wall clock; the counted test numbers are identical |

The four examples were also run at the low end. They complete and write their
PNGs, but **the PNG files are not byte-identical between matplotlib 3.10.0 and
3.11.2**: the renderer changes between minor versions. The numbers plotted are
identical, and they are the numbers in the tables above. This is recorded here
rather than worked around, because a figure that is not byte-reproducible is a
limitation of the figure, not of the arithmetic.

**The ceilings are a cost.** This package cannot take NumPy 2.6, PyYAML 6.1,
matplotlib 3.12, pytest 9.2 or hypothesis 6.171 without a port and a
re-measurement. That is recorded in README.md under Limitations and is not
resolved by removing the ceiling.

## 9. Parser rejection paths

`validation/validate_parser_rejections.py`

| quantity | value |
|---|---|
| schema rejection paths exercised | 36 |
| rejected with `DeclarationError` | 36 of 36 |
| malformed documents accepted | 0 |
| rejection messages that quote an identifier or name a schema field | 36 of 36 |
| input and output paths exercised (missing file, directory, invalid UTF-8) | 3 of 3 rejected |

## 10. CLI exit-status contract

`validation/validate_cli.py`

| quantity | value |
|---|---|
| real `python -m budgetroll` subprocesses started | 32 |
| exit-status mismatches against the documented contract | 0 |
| reporting commands that print the not-flight-qualified statement | 5 of 5 |

The contract: 0 when the command ran and raised no `error` finding, 1 when it
ran and raised at least one, 2 when it could not run.

## 11. What failed, and what is not covered

Nothing in the final run failed. Two things are recorded because they were
true at some point during the build and remain true as limitations:

1. **`validate_parser_rejections.py` failed on its first run**, reporting that
   five rejection messages named nothing. Four of the five did name the
   offending object (`node 'root'`, `item 'a'`, `correlation 'a'-'c'`,
   `k_sigma`) and the criterion was too narrow; the fifth, "the document is
   empty", has nothing to name. The criterion was replaced with a precise one
   — the message must quote an identifier in single quotes or name a schema
   field by its YAML name — and is stated in the script.
2. **The margin-sign boundary was a real defect in the first implementation.**
   BR003 compared `margin < 0` strictly. Section 7 records the input that
   falsified it and the rounding band that replaced it.

Not covered by any evidence in this document:

- Whether any input number in any example is correct. None is measured.
- Whether any margin policy is appropriate. The package ships none.
- Any claim about a real spacecraft, mission or programme.
- Unit correctness. The package does no unit conversion and no dimensional
  analysis; a budget that mixes kg and lb is summed without complaint.
- Any distributional statement. No distribution is assumed anywhere, so a
  k-sigma figure is a multiple of a standard deviation and not a confidence
  level.
