# Validation — P098 EvidenceStale

Validation level **2**. Status `TESTING`.

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced
by a script in this directory, run in the session dated 2026-11-19, and the raw
output of each run is committed beside the script as
`validation/<script-name>_output.txt`. Where a number appears below, the script
that produced it is named.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.

## Contents

| script | checks | failures | controls | what it establishes |
|---|---|---|---|---|
| `validate_approx_audit.py` | 4 | 0 | 2 of 2 fired | no assertion in the suite rests on `pytest.approx`'s hidden 1e-12 default |
| `validate_baselines.py` | 18 | 0 | — | both classical baselines against the by-construction ground truth |
| `validate_bound_bracket.py` | 17 | 0 | 3 of 3 fired, 1 of 1 silent | the stale fraction lies inside a bound derived from mathematics the implementation does not use |
| `validate_corpus_pin.py` | 14 | 0 | — | the committed corpora, their digests and their declared structure |
| `validate_learned_model.py` | 14 | 0 | — | the learned model against the validated baselines, on both splits |
| `validate_negative_controls.py` | 16 | 0 | 8 of 8 fired, 6 of 6 silent | injected staleness of a known rate is recovered, and each control fired |
| `validate_tree_hygiene.py` | 38 | 0 | — | what a cold clone carries, and that no run adds a cache artifact |
| `probe_working_directories.py` | 6 | 0 | — | every script runs from four working directories |

Order matters and is enforced by the file names: `validate_baselines.py` sorts
before `validate_learned_model.py`, so the gate's own ordering runs the
baselines' validation before the model's benchmark. No script in this directory
regenerates data that another pins; the corpora are generated once by
`data/generate_corpus.py`, which is deliberately not here.

## 1. The headline output

`validate_baselines.py` section 1, over `data/corpus_layered.json`: 480
evidence items about 240 artifacts, audited at revision 2000.

| quantity | value |
|---|---|
| truly stale items | 194 of 480 |
| **stale fraction, ground truth (baseline 3)** | **0.404167** |
| stale fraction, closure comparison (baseline 2) | 0.404167 |
| goal-weighted stale fraction, over 120 GSN leaf goals | 0.641667 |
| 95 % bootstrap, unit `cluster`, 240 subject artifacts, 2000 resamples | [0.345833, 0.466667], half width 0.060417 |
| 95 % bootstrap, unit `item`, 480 items, 2000 resamples | [0.360417, 0.447917], half width 0.043750 |

**The resampling unit is declared and asserted.**
`bootstrap_stale_fraction` has no default for `unit`; calling it with an
unimplemented unit is refused, which is check 18 of
`validate_baselines.py`. The cluster unit is primary because the two evidence
items about one artifact go stale together, so resampling items treats
correlated observations as independent draws: measured, the cluster interval is
0.060417 half-width against the item interval's 0.043750, so the item interval
understates the variability by about 28 per cent on this corpus. The percentile
method is used, after Efron and Tibshirani (1993); the bias-corrected and
accelerated interval is not implemented.

The intervals are statements about resampling variability of this synthetic
corpus. They are not statements about any population of real assurance cases.

## 2. Baseline 1 — the thing being refuted, and its two error directions

`validate_baselines.py` section 2. The two directions are different failures
with different consequences for an audit and are never summed.

| method | missed staleness | rate | false alarms | rate | agreement |
|---|---|---|---|---|---|
| baseline 1, subject modification revision | 115 of 194 | 0.592784 | 143 of 286 | 0.500000 | 0.462500 |
| baseline 1b, subject and direct dependencies | 26 of 194 | 0.134021 | 226 of 286 | 0.790210 | 0.475000 |

- **Missed staleness** cites evidence that no longer describes the system. This
  is the direction an audit cannot tolerate.
- **A false alarm** sends an auditor to re-run work that did not need it, and
  teaches them to discount the tool.

Baseline 1 was implemented at full strength and not weakened to lose. The
comparison is strict (`mtime > produced_at`), the modification revision is the
real last-event revision from the replayed history, and the strongest honest
variant of the same idea — reading one level of the declared graph — is
implemented and reported beside it. That variant reduces missed staleness from
115 to 26 and raises false alarms from 143 to 226. It removes neither, because
a change at depth 2 or more is invisible to a modification-revision comparison
however many levels short of the whole closure it reads.

**The aggregate hides this.** Baseline 1 reports a stale fraction of 0.462500
against a true 0.404167 — an aggregate error of +0.058333 — while agreeing with
the truth on only 46.25 per cent of items, because its missed staleness and its
false alarms partly cancel. On the hand-worked two-item example in
`tests/test_baselines.py` the cancellation is total: baseline 1 reports 0.5
against a true 0.5 with an agreement of 0.0. That is the whole argument for
reporting both directions.

## 3. Baseline 2 — the exactness claim, and what it is conditional on

`validate_baselines.py` section 3.

| quantity | value |
|---|---|
| items compared | 480 |
| missed staleness | 0 |
| false alarms | 0 |
| agreement | 1.000000 |
| stale fraction, closure | 0.40416666666666667 |
| stale fraction, ground truth | 0.40416666666666667 |
| absolute difference | 0.000000e+00 |
| comparison slack, from the conditioning of one division | 8.881784e-16 |

**"Exactly" is a claim, so here is precisely what is claimed.** The comparison
is over 64-character hexadecimal digests and integer counts, so there is no
rounding inside it; the only floating-point quantity is the stale fraction
itself, one division of two exact integers, compared against the ground truth's
same division to a slack of four units in the last place. The agreement is
exact in exact arithmetic and the two fractions agree to within rounding —
measured difference 0.000000e+00.

The same claim holds on `data/corpus_cyclic.json`, which is **not** a directed
acyclic graph: it contains the cycle `('c00', 'c01', 'c02')`, the closure of
each cycle member is the whole component, and the closure comparison agrees
with the ground truth on all 12 items.

**It is not a tautology, and it is not independent either.** The closure
comparison and the ground truth implement the same definition of staleness.
They differ in representation: one compares SHA-256 digests of contents
reconstructed from the event log at two revisions, the other compares the
content values themselves, and the committed labels were written by the
generator before this package read them. So the agreement tests the closure
search, the replay and the hashing against each other and against the
generator; it does not test the definition, and no measurement could. The
independent check on the definition is section 5.

### The three conditions, each measured

1. **Complete dependency declarations.** With 75 of 249 true edges withheld
   (`validate_baselines.py` section 4): the reported stale fraction falls to
   0.281250, missed staleness is 59 of 194 (rate 0.304124), false alarms are 0
   (rate 0.000000), agreement 0.877083. An undeclared edge can only hide a
   change, never invent one, so the error is one-directional by construction.
2. **No digest collision.** `validate_corpus_pin.py` section 4: 273 distinct
   contents over the whole layered history, 37128 pairs, pairwise union bound
   3.206437e-73, log10 −72.493977. This is a bound on an idealised
   random-oracle model of SHA-256 as specified in FIPS PUB 180-4 (NIST, 2015),
   not a proof about SHA-256, and it says nothing about contents chosen to
   collide.
3. **The closure depth limit not binding.** `validate_corpus_pin.py`
   section 2: the deepest closure in the layered corpus is 2 levels against a
   default limit of 64; in the cyclic corpus it is 5 against a declared bound
   of 6. The limit never bound, so no closure was truncated.

## 4. The corpus, pinned and structurally checked

`validate_corpus_pin.py`. This script reads and hashes; it never writes to
`data/`.

| file | SHA-256 | bytes | artifacts | true edges | declared | events | items | stale |
|---|---|---|---|---|---|---|---|---|
| `data/corpus_layered.json` | `19309221e6161e425870fcd3df2e4725c1554b80e4a5670873734f7c43f04ac2` | 167319 | 240 | 249 | 174 | 250 | 480 | 194 |
| `data/corpus_cyclic.json` | `7334afe10a4362c85b6c617e1c8c23e0766a42fe8b93121211a463fc83730bfc` | 4987 | 12 | 12 | 12 | 6 | 12 | 7 |

Structural constants, measured against the generated graph rather than trusted,
because they are the inputs the independent bound consumes:

| corpus | declared d | measured max dependents | declared L | measured max closure depth | max closure size | mean closure size | acyclic |
|---|---|---|---|---|---|---|---|
| layered | 2 | 2 | 2 | 2 | 7 | 2.833333 | True |
| cyclic | 2 | 2 | 6 | 5 | 6 | 4.500000 | False |

The committed by-construction labels agree with the labels the replay
recomputes from the committed event log: 0 of 480 disagree on the layered
corpus, 0 of 12 on the cyclic one.

**A declared constant that was wrong, and how it was caught.** The cyclic
corpus first declared a maximum impact depth of 4. The measured depth is 5 —
`c09 → c06 → c03 → c00 → c01 → c02` — and check 2 of this script failed, which
is why the declared value is now 6. The check exists because the bound in
section 5 is only valid if the declared constants are true of the graph.

## 5. The independent bound — mathematics the implementation does not use

`validate_bound_bracket.py`. "Two implementations agree" cannot catch an error
both share, and the closure comparison and the ground truth both enumerate
reachable artifacts. This bracket consumes four integers and performs no
reachability search, reads no content and computes no digest.

Upper bound, from the Moore bound on the number of nodes within distance `L` in
a graph of out-degree at most `d`, plus a union bound over the changed
artifacts:

```
stale_fraction  <=  min(1, m * min(N, k * M(d, L)) / E)
M(d, L) = 1 + d + d^2 + ... + d^L
```

`d` is the maximum number of direct dependents of any artifact, which is the
maximum out-degree of the reversed impact graph. Lower bounds are the depth-0
count (items whose own subject changed) and the depth-1 count (items whose
subject or a direct declared dependency changed); neither performs a transitive
search.

Headline, at audit revision 2000:

| quantity | value |
|---|---|
| artifacts N | 240 |
| max direct dependents d | 2 |
| declared impact depth L | 2 |
| M(d, L) | 7 |
| max items per subject m | 2 |
| evidence items E | 480 |
| changed artifacts k | 31 |
| lower bound, depth 0 | 0.110417 |
| lower bound, depth 1 | 0.268750 |
| **measured stale fraction** | **0.404167** |
| upper bound | 0.904167 |
| slack per endpoint | 8.881784e-16 |

Over a grid of ten audit revisions, with the full 480-item evidence set at each:

| revision | k | lower | measured | upper | informative | holds |
|---|---|---|---|---|---|---|
| 600 | 10 | 0.043750 | 0.077083 | 0.291667 | True | True |
| 700 | 10 | 0.054167 | 0.091667 | 0.291667 | True | True |
| 800 | 12 | 0.072917 | 0.102083 | 0.350000 | True | True |
| 900 | 12 | 0.072917 | 0.102083 | 0.350000 | True | True |
| 1000 | 16 | 0.110417 | 0.160417 | 0.466667 | True | True |
| 1200 | 18 | 0.122917 | 0.181250 | 0.525000 | True | True |
| 1400 | 20 | 0.147917 | 0.222917 | 0.583333 | True | True |
| 1600 | 26 | 0.206250 | 0.302083 | 0.758333 | True | True |
| 1800 | 28 | 0.229167 | 0.333333 | 0.816667 | True | True |
| 2000 | 31 | 0.268750 | 0.404167 | 0.904167 | True | True |

**Holds at 10 of 10, informative at 10 of 10.** Informative is reported
separately from holds, because the upper bound reaches 1 as soon as
`k * M(d, L) >= E / m`, that is for `k >= 34.29` on this corpus, and a bound
equal to 1 is still a true bound that constrains nothing. The grid was chosen
so the evidence set is complete at every point; it is not chosen to hide
revisions where the bound would be vacuous, and the condition under which it
would be is stated above.

**The slack follows the conditioning of the bound expression, not an absolute
constant.** Everything in the bracket is an exact Python integer up to one
division per endpoint, each of relative condition number 1, so the slack is
`4 * eps * max(1, |endpoint|)` — 8.881784e-16 here. An absolute constant would
be vacuous for endpoints near 1 and arbitrary for endpoints near 0.

The Moore bound is spot-checked against eight hand values:
M(2,0)=1, M(2,1)=3, M(2,2)=7, M(2,3)=15, M(3,2)=13, M(4,3)=85, M(0,5)=1,
M(1,5)=6.

### A defect in this check, found and fixed during the build

The first version took one changed set for the whole evidence set — the union
over items of the artifacts that changed in each item's own window. The union
is correct for the upper bound, because the impact set is monotone in the
changed set, and **wrong for the lower bounds**: an artifact that changed in
some other item's window need not have changed in this item's window.
Measured consequence at the corpus and grid then in use: a lower bound above
the measured fraction at 6 of 10 revisions — a check reading as a refutation of
the measurement when it was a defect in the check. The function now takes a
change set per item and forms the union itself. On the shipped corpus the
substitution over-counts the depth-1 lower bound by +0.035417 (0.304167 against
0.268750) without crossing the measurement; the literal two-item case where it
does cross is pinned as
`tests/test_regressions.py::test_regression_lower_bound_needs_a_per_item_change_set`.

## 6. Negative controls — anchored on the truth, and each asserted to fire

`validate_negative_controls.py`. **8 of 8 controls that must fire fired; 6 of 6
that must stay silent stayed silent.**

Every control applies an injection to the change history and **recomputes the
ground truth over the true dependency graph** from the modified history. No
control is anchored on a previous reading of a detector, and no control's
criterion is a confidence interval, so none can be satisfied by a lucky seed.
Injection revisions are 2001 and above and the audit revision is 2200; every
committed event lies in revisions 1..2000 and every item is produced in
1..600, and the unmodified result at revision 2200 is checked to equal the
published result at revision 2000 rather than assumed.

| # | injection | declared rate | measured | control |
|---|---|---|---|---|
| 1 | writes to leaf artifacts | 24 of 240 artifacts (0.100000) | 105 items made stale; 68 of them stale only through a dependency | fires: closure comparison recovers 105 of 105 exactly |
| 1 | the same | — | timestamp baseline newly flags 0 of those 68 | fires: it cannot see a change at depth 2 |
| 2 | touches on subjects | 30 artifacts | true stale count unchanged, 194 → 194 | silent: truth unmoved |
| 2 | the same | — | closure comparison unchanged | silent: no spurious detection |
| 2 | the same | — | timestamp false alarms 143 → 161 | fires: a touch is a false alarm |
| 3 | write then revert | 12 artifacts, 24 events | true stale count unchanged, 194 → 194 | silent |
| 3 | the same | — | 114 items an event-counting method would have flagged | fires: the value-based definition matters |
| 4 | edges hidden from the declaration | 40 of 249 | 28 items missed, 0 false alarms | fires one direction, silent the other |
| 5 | the deep-change injection re-applied | — | true fraction increase +0.218750, measured increase +0.218750, difference 0.000000e+00 against a slack of 8.881784e-16 | silent: the recovered rate equals the injected rate |

Control 1's claim was corrected during the build. It first asserted that the
timestamp baseline misses **every** newly stale item, and that failed: 3 of 105
were newly flagged, because every artifact in this corpus is also the subject
of evidence, so a written leaf is somebody's subject and its modification
revision really did move. The control now separates the 37 items whose subject
was written directly from the 68 made stale only through a dependency, and
asserts the structural claim — 0 of 68 newly flagged — rather than the
overstated one.

## 7. The learned model against the validated baselines

`validate_learned_model.py`, run after `validate_baselines.py`. Full tables are
in `MODEL_CARD.md`; the finding is here.

| split | baseline 1 AUC | baseline 1b AUC | **baseline 2 AUC** | learned AUC | leak ablation AUC |
|---|---|---|---|---|---|
| held-out artifacts, 5 grouped folds | 0.4510 | 0.5372 | **1.0000** | 0.7152 | 0.7390 |
| held-out time, 3 cutoffs | 0.4460 | 0.5460 | **1.0000** | 0.6977 | 0.7368 |

With 75 of 249 dependency edges withheld — the only circumstance in which a
learned model could add anything, because the exact method stops being exact:

| split | **baseline 2 AUC (degraded)** | learned AUC |
|---|---|---|
| held-out artifacts | **0.8495** | 0.5745 |
| held-out time | **0.8578** | 0.5211 |

**The finding: the closure comparison won, on both splits, on both views, and
is published as winning.** The pre-registered expectation in the specification
was that the closure comparison is exact and the learned model cannot beat it,
and that is what was measured. The learned model does beat the method the
specification names as the thing being refuted — baseline 1 — on both splits.
Nothing was retuned after a score was seen, and the result is not softened.

The model's confidence output is measured rather than asserted: agreement rises
0.631579 → 0.666667 → 0.721154 → 0.762195 across the four confidence bands with
at least 20 items, and **falls to 0.647059 in the sparse top band of 17 items**.
That non-monotonicity is reported, not smoothed, and is why nothing here calls
the forest's vote margin a calibrated probability.

All eight fits of one benchmark run: 1.181 s of wall clock in the final run of
this script and 1.783 s in an earlier one, against a declared 60 s budget. Only
the bound against the budget is checked; no two live wall-clock readings are
compared with each other anywhere in this repository. 0 of 8 folds were
degenerate.

## 8. `pytest.approx` hidden-default audit

`validate_approx_audit.py`. `pytest.approx(expected, rel=r)` passes if either
bound is met and the absolute default is 1e-12, so for a quantity of order 1e-5
every relative tolerance tighter than 2e-8 is silently replaced.

- **Static**, from the syntax tree of 18 test files: 41 `approx` calls, **41
  with both `rel=` and `abs=`, 0 relying on the default.**
- **Dynamic**, the whole suite re-run in a subprocess with a plugin that makes a
  bare `rel=` raise: exit 0, `213 passed in 12.13s`.
- **Control**, 2 of 2 fired: an assertion contrived to pass only through the
  hidden default passes without the plugin (exit 0) and is refused with it
  (exit 1, carrying the plugin's own message).

So **0 of 41 assertions depended on the hidden default**, measured in two
independent ways rather than read off the source.

## 9. Working directories, and tree hygiene

`probe_working_directories.py` runs every script in `validation/` and
`examples/` from four working directories — the filesystem root, the system
temporary directory, the product root and the script's own directory — inside a
**pristine copy** of the product, with `EVIDENCESTALE_FAST=1`. Every script
returned the same exit status from all four, and every status was 0. The bytes
of all committed `*_output.txt` files and all committed figures were recorded
before and after: 0 output files other than the probe's own and 0 figures
changed, so the copy really was the copy.

`validate_tree_hygiene.py` asserts what this product owns and not the third
thing: that running it ADDS no cache directory and no bytecode file (snapshot
at entry, compare at exit — 0 added), and that nothing a cold clone would carry
is a cache directory, bytecode file, build artifact or serialised model. It does
**not** assert that no such artifact exists, because release-gate check 2 runs
`ruff check` in the product root and writes `.ruff_cache` one check before the
validation scripts are re-executed; anything found that the script did not
create is reported as inherited and does not fail. No `git` command is run: the
`.gitignore` semantics the check needs are re-implemented and the covered subset
is declared in the script.

## 10. What is NOT validated

- **No real assurance case and no real repository was measured.** Every figure
  comes from a synthetic corpus with a fixed seed. See `DATASET_CARD.md` for
  the construction choices and their consequences.
- **The suite was not run at both ends of every declared dependency range.**
  The build guide asks for that; the build agent for this product is forbidden
  from running `pip` at all, for a reason recorded in the batch brief, so it
  could not be done. The declared ranges are the minor series actually
  measured, which is narrower than a floor with no ceiling and weaker than a
  measured range. Recorded in `README.md` under Limitations rather than papered
  over.
- **Dependency discovery is out of scope.** The dependency graph is declared.
  Discovering real dependencies from a build system or toolchain is the hard
  part of applying any of this, and the `partial` view is a model of imperfect
  discovery, not a measurement of any discovery tool.
- **The bias-corrected and accelerated bootstrap is not implemented**, only the
  percentile interval.
- **No timing relationship between two live wall-clock measurements is asserted
  anywhere**, in any test or script. Where a time is checked it is a direction
  (`> 0`) or a bound against a declared budget, with the measured value
  published.
