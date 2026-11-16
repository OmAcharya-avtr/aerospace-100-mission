# P100 SeedAudit — Specification

**Product:** P100 · **Package:** `seedaudit` · **Class:** compact ·
**Category:** testing · **Target validation level:** 2
**Written:** 2026-11-16, **before** the audit was run against this repository.

> This document is a **pre-registration**. The expectation in §8 was written
> and committed before the audit executed against `products/` and `scripts/`.
> A tool whose purpose is to refute reproducibility claims, and whose own
> expectation was formed after reading its own findings, would be measuring
> nothing. The commit that introduces this file precedes the commit that
> introduces `validation/`, and the ordering is verifiable in the repository
> history.

## 1. What it does

`seedaudit` audits a repository's own reproducibility claims and reports how
many of them are false. It answers three questions, and a fourth about itself.

1. **Which committed numbers move when only the random seed changes?** A
   committed number that sits in a validation output file, produced by a
   script that declares a seed, is a reproducibility claim. `seedaudit`
   re-executes that script in a pristine copy with its seed literals
   rewritten, aligns the new output against the committed one, and counts the
   numeric tokens that moved. **That count is the headline output.**
2. **Which scripts claim determinism and are not deterministic?** The same
   re-execution is also run with the seed *unchanged*. A number that moves
   there did not move because of the seed; it was never reproducible at the
   declared seed in the first place. That is a separate and worse finding, and
   it is counted separately.
3. **Which checks can create the condition they test?** The
   self-polluting-check class, detected statically and dynamically. See §4.
4. **What could the audit itself have missed?** Every verdict carries the
   closed-form detection power of the sampling that produced it, so a clean
   audit reports a quantified upper bound on the intermittency rate that could
   have escaped rather than an implied zero. See §3.

## 2. What it does that existing tools do not

Reproducibility checkers compare build outputs byte for byte. None of them
states the detection power of its own sampling. The difference between
*finding no non-determinism* and *there being none* is exactly that number,
and without it a clean report is an unearned claim of zero.

This matters because the failure it misses is the common one. This mission
shipped a 1-in-44 intermittent failure. Any small number of agreeing serial
verifications was blind to it: at `p = 1/44`, two runs detect it with
probability **0.044938016528925595**, so a clean two-run report is the
expected outcome and carries almost no information. Reaching 95 per cent
detection probability at that rate needs **131** independent repeats. No
reproducibility checker in the table below reports either figure.

Honest alternatives, named because they exist and are mature. Every version
below was read from the PyPI JSON API on 2026-11-16.

| Tool | Version | What it does better | When to use it instead |
|---|---|---|---|
| `reprotest` | 0.7.33 | varies build environment along many declared axes (path, locale, timezone, filesystem order, user) and rebuilds | you are checking a *build* for reproducibility across environment variation, not a repository's committed numbers |
| `diffoscope` | 339 | recursive content-aware diff of archives, binaries, PDFs, filesystem images | you have two artifacts and need to know *where* inside them they differ |
| `pytest-randomly` | 5.1.0 | randomises test order and reseeds `random`, `numpy.random` and `faker` per test | you want order-dependence and seed-dependence surfaced inside a pytest suite |
| `pytest-rerunfailures` | 16.7 | re-runs failing tests to let flaky ones pass | you want CI to survive a flaky test; note this *suppresses* the signal this product exists to find |
| `flaky` | 3.8.1 | same, per-test decorator, with pass/run thresholds | the same, where the flakiness is known and accepted |
| `pytest-repeat` | 0.9.4 | runs each test N times | you want a repeat count and nothing else |
| `pytest-flakefinder` | 1.1.0 | repeats tests to expose flakiness | the same, with a time budget |

None of these states a detection power or an escaped-rate bound, and none
detects the self-polluting-check class, because none of them runs a check in a
pristine copy and in a reused tree and compares the two verdicts against each
other. That pair of runs is this product's detection mechanism.

`pytest-rerunfailures` and `flaky` deserve a sharper note: they are the
*opposite* instrument. They exist to make an intermittent failure stop
blocking a pipeline. This product exists to find it and count it.

## 3. The closed-form detection power — the core of the product

### 3.1 The two designs, which are not the same formula

Let an intermittency manifest independently on each execution with rate
`p ∈ (0,1)`, sampled with `n` executions.

- **Reference-comparison design.** Each run is compared against a *committed*
  reference value. Detection requires at least one anomalous run:

  `P_ref(p, n) = 1 - (1-p)^n`

- **Pairwise-difference design.** The `n` runs are compared only against each
  other, with no reference. Detection requires at least one anomalous run
  **and** at least one normal run, because `n` identically anomalous runs
  agree with each other:

  `P_pair(p, n) = 1 - p^n - (1-p)^n`

`P_pair ≤ P_ref` always, with equality only in the limit `p^n → 0`. At
`p = 1/44, n = 2`: `P_ref = 0.044938016528925595`,
`P_pair = 0.044421487603305776`. Run-twice-and-diff is the pairwise design.
Conflating the two overstates its power by `p^n`, which is small here and is
not small at high rates: at `p = 0.5, n = 2`, `P_ref = 0.75` and
`P_pair = 0.5`. Both forms are implemented and the audit names which one each
reported verdict used.

### 3.2 Inversion: the smallest detectable rate

For a repeat budget `n` and a required detection probability `1-α`, the
smallest rate detectable with that probability is obtained by inverting
`P_ref`:

`p_min(n, α) = 1 - α^(1/n)`

### 3.3 Inversion: the escaped-rate upper bound, which is the number that matters

If `n` repeats agree — a clean audit — the one-sided upper confidence bound on
`p` at confidence `1-α` is the Clopper–Pearson bound for zero successes in `n`
Bernoulli trials, which has the closed form

`p_max(n, α) = 1 - α^(1/n)`

This is **the same expression** as `p_min`, and the coincidence is not an
accident: the smallest rate the design would have caught with probability
`1-α` is exactly the largest rate consistent with having caught nothing. For
large `n` it is the rule of three, `p_max ≈ 3/n` at `α = 0.05`.

A clean audit therefore reports `p_max`, not zero. At `n = 2` and
`α = 0.05`, `p_max = 0.7763932022500211`: two agreeing runs are consistent
with an intermittency that manifests on more than three quarters of
executions. That is the quantity a byte-for-byte reproducibility checker
leaves unstated.

### 3.4 Assumptions, stated because the bound is worthless without them

1. **Independence across repeats is the strong assumption.** Repeats within
   one process share RNG state, import caches, a fixed `PYTHONHASHSEED` and
   a warm filesystem cache, so they are positively correlated and the true
   detection power is **lower** than the formula. The bound is therefore
   **optimistic** in the direction that matters, and the only honest
   mitigation is to make the repeats as independent as the platform allows:
   each repeat is a **fresh subprocess in a fresh pristine copy with a fresh
   hash seed**. The residual correlation is not zero and is not estimated.
2. **Stationarity.** `p` is assumed constant over the repeats. A rate that
   depends on machine load is not covered.
3. **The verdict is binary and the anomaly is detectable when it manifests.**
   An anomaly below the printed precision of the output is invisible to a text
   comparison however many repeats are run; the corpus in §6 contains one
   deliberately, and the numeric comparator exists for it.
4. **Seed draws are not repeat draws, and the two axes are kept separate.**
   See §3.5.

### 3.5 Why the seed axis is a different axis

A quantity that is a deterministic function of the seed is not a Bernoulli
process over repeats. Running it twice at the same seed yields the same value
with probability 1 and gives **zero** information about seed-dependence; the
repeat-count formulae above say nothing about it. Conversely, drawing a second
distinct seed reveals seed-dependence with probability near 1 for a continuous
output, so the seed axis has far higher power per draw and needs far fewer
draws.

The two axes answer different questions and are reported separately:

- **Repeat axis, seed fixed.** Is this script intermittently non-deterministic
  at its declared seed? Governed by §3.1–3.3.
- **Seed axis, seed varied.** Does this committed number depend on the seed at
  all? Governed by the fraction of seed space on which the value differs from
  the committed one, which for a seed-sensitive continuous output is
  essentially 1 and for a seed-insensitive one is 0. A single perturbed seed
  is therefore close to sufficient, and `m` distinct seeds drawn without
  replacement from a finite declared set are **hypergeometric**, not binomial.

Reporting a single pooled "number of runs" across both axes would be wrong in
both directions at once: it would overstate the power on the repeat axis by
counting seed draws that carry no repeat information, and understate it on the
seed axis by applying a Bernoulli bound to a deterministic function.

### 3.6 The derivation is checked, not asserted

`validation/validate_power_model.py` compares the closed forms against a
Monte Carlo estimate over a grid of `(p, n)` and requires agreement within
the Monte Carlo standard error. A closed form whose only evidence is the
algebra that produced it is the P077 class of this mission: every test
compared an expression against expressions sharing its own defect.

## 4. The self-polluting-check class

A check whose own execution can create the condition it tests is not a check.
Real instances from this mission's history, which is where the motivation
comes from and not from a textbook:

- a validation probe writing its output into the directory it then scans for
  stray files;
- a `ruff` invocation writing its cache inside the product tree the audit then
  reports as dirty;
- a script whose first run creates the artifact whose absence it asserts.

Two detectors, because neither alone is sufficient.

**Static.** A script that writes under a path it also reads or globs.
`seedaudit.static` walks the `ast`, derives a conservative path expression for
every read, glob and write target from the literal segments it can resolve,
and flags containment of a write path inside a read or glob root. It reports
the pair of source lines. It is conservative by construction: a dynamic path
it cannot resolve becomes a wildcard segment, so it over-flags rather than
under-flags, and the false-positive rate on the deterministic controls in §6
is measured and published rather than tuned away.

**Dynamic, and this is the mechanism.** The script is run twice in two
*different* arrangements:

- **pristine pair** — run 1 in a fresh copy, run 2 in a *second* fresh copy;
- **reused tree** — run 1 and run 2 in the *same* copy, in sequence.

A verdict that changes between two pristine runs is **intrinsic
non-determinism**. A verdict that changes between run 1 and run 2 of the
reused tree while the pristine pair agrees is **self-pollution**: the first
run changed its own preconditions. The cross-tabulation is the classifier:

| pristine pair | reused tree | classification |
|---|---|---|
| differ | differ | intrinsic non-determinism |
| differ | agree | intrinsic non-determinism, masked in the reused tree |
| agree | differ | **self-polluting check** |
| agree | agree | no evidence, qualified by `p_max(n)` from §3.3 |

Both are reported. The fourth row is reported with its bound and never as a
pass.

## 5. Baselines

Implemented in this order. All classical and analytic; there is no fitting
step anywhere in the product.

- **Baseline 1 — run each script twice and diff.** What everyone does, built
  deliberately as the thing being refuted. Its detection power is measured
  against the §6 corpus and reported against the derived
  `P_pair(1/44, 2) = 0.044421487603305776`. A measurement agreeing with that
  number is the refutation: the method everyone uses finds a 1-in-44 defect
  about one time in twenty-two.
- **Baseline 2 — a seed-sweep and repeat-count design with the detection
  probability stated in closed form.** The product's own design. Repeat axis
  and seed axis sized separately per §3.5, every verdict carrying `P_ref`,
  `P_pair` or `p_max` as appropriate.
- **Baseline 3 — ground truth: an injected corpus of known rate.** §6. The
  audit's detection power is *measured* against it per rate, with bootstrap
  confidence intervals, and the false-positive rate on the deterministic
  negative controls is measured on the same corpus.

**No learned component.** `ai_enabled: false` in the registry, and the reason
is not a resource constraint. A model would add a second thing to trust in a
tool whose only value is being trustworthy, and it would have to be trusted
on exactly the question the tool exists to answer.

## 6. Baseline 3 — the injected corpus

A corpus of synthetic scripts, generated by a committed generator at a
declared seed, with non-determinism injected at **known** rates spanning more
than three orders of magnitude: `1.0, 0.5, 0.1, 1/44, 0.01, 0.001`.

It contains, by construction:

- **seed-dependent numeric output** — the value is a function of the seed;
- **a genuinely deterministic negative control**, which must NOT be flagged;
- **a self-polluting check** — it writes its output into the directory it
  scans, and its verdict therefore changes on a second run in the same tree
  while two pristine runs agree;
- **a script deterministic in printed output but non-deterministic in
  dict/set iteration order**, which a text diff of the final value misses;
- **a script whose non-determinism is in a float at a digit beyond its
  printed precision**, so a text diff cannot see it and only a numeric
  comparison at a declared tolerance can.

Detection power is measured per rate with **bootstrap confidence intervals**
over the repeat trials, and the false-positive rate is measured on the
negative controls. **If the audit flags a deterministic control, that is a
result and it is published, not tuned away.**

## 7. Headline outputs

1. **The count of committed numbers claimed reproducible that actually move
   under a seed change.** The headline.
2. The count that move with the seed **unchanged** — not reproducible at the
   declared seed at all, a separate and worse class.
3. The count of self-polluting checks, static and dynamic, reported separately
   with the cross-tabulation of §4.
4. For every clean verdict, `p_max(n, 0.05)` — the escaped-rate upper bound.
5. Baseline 1's measured detection power at `p = 1/44` against its derived
   value.
6. Measured detection power per injected rate with bootstrap intervals, and
   the false-positive rate on negative controls.
7. **Audit coverage, stated as a fraction and never implied.** How many
   candidate scripts were attempted, how many completed inside the timeout,
   how many errored, and how many were excluded and why.

## 8. Pre-registered expectation — written before the audit was run

**Primary. The headline count will be greater than zero and will be a small
fraction of the corpus: between 1 and 40 moved numbers, concentrated in fewer
than 15 scripts.**

Mechanism, stated in advance so the claim is falsifiable. This mission's build
guide has required declared seeds since its first batch and most scripts honour
it, so a large count would be surprising. The numbers that do move will come
from one of three mechanisms: a script that seeds one generator and lets a
second, default-seeded generator into the same computation; a quantity
derived from an unseeded container iteration order; or a figure that is not a
seeded quantity at all, such as a wall-clock duration, printed into a file the
repository treats as reproducible evidence.

**This expectation is refuted if the count is zero, or above 40.** Zero would
mean the mission's seed discipline is complete, which would make this
product's premise wrong for this corpus, and it would be published as such —
with `p_max` attached, because a zero from a bounded sample is not a zero.

**Secondary, on the unchanged-seed control: the count of numbers that move
with the seed held fixed will be non-zero, and will be dominated by wall-clock
and timing fields rather than by seeded quantities.**

**Secondary, on the static self-pollution detector: it will flag at least 6
scripts in the corpus**, since the mission's own history records six
occurrences of the class, **and more than half of the flags will be scanner
scripts writing their own `_output.txt` into a tree they scan.** Refuted below
6 flags.

**Secondary, on Baseline 1: its measured detection power at `p = 1/44` will
lie within the bootstrap interval of the derived 0.044421487603305776 and
below 0.06.**

**Secondary, on the negative controls: the dynamic detector's false-positive
rate will be 0, and the static detector's will be non-zero.** The static
detector is deliberately conservative — an unresolvable path becomes a
wildcard — so it is expected to flag at least one deterministic control. The
measured static false-positive rate is published at whatever value it takes,
and the detector is not weakened to bring it to zero.

## 9. Audit scope and the exclusion, declared in advance

The audit runs over `products/` and `scripts/` of this repository.

**`products/P094/` is excluded.** Another build agent is writing it
concurrently, so any number derived from it is not reproducible by a reader
and the exclusion is stated with every scan result rather than in a footnote.
`products/P100/` — this product — is also excluded from its own repository
scan, because a tool scanning its own tree while it is being written is the
very class it exists to detect; it is audited separately and deliberately, in
`validation/validate_self_audit.py`.

The dynamic runner **never writes outside `products/P100/` or a temporary
directory.** Every execution is in a copy. This is not hygiene: the audit runs
against a live repository that another agent is writing to, so an in-place run
would both corrupt that agent's tree and make the audit's own result
unreproducible.

## 10. This product's own reproducibility, which it cannot be exempt from

A tool that audits reproducibility and is itself non-reproducible is
worthless. Therefore:

- every test and every validation script is deterministic under a seed
  declared in one place;
- a test runs the audit **twice** over a fixed fixture corpus and asserts the
  two reports are equal, so the product's own determinism is measured and not
  claimed;
- the self-pollution self-check obeys the mission's narrowing: it asserts that
  running the script **adds** no cache directory or bytecode file and that git
  **tracks** none, and **not** that none exists, because what the release-gate
  harness leaves in the tree before calling the script is not this product's
  doing.

## 11. Deliverables

`src/seedaudit/` (closed-form power model, static analyser, dynamic
repeat-runner with pristine-copy isolation, seed sweeper, numeric-claim
extractor and aligner, injected-corpus harness, bootstrap, CLI), `tests/`,
`examples/` with real captured output, `validation/VALIDATION.md` with every
number traced to a re-runnable script, `screenshots/` generated from measured
data only, `README.md` per `templates/REPO_README_STANDARD.md`,
`pyproject.toml` with both-sided dependency ranges.

## 12. Honesty constraints

Research-grade; not flight-qualified, not certified, not approved for
operational aerospace use. No figure in `validation/VALIDATION.md` comes from
anywhere but a script in `validation/` that was executed in the session that
reports it and that exits 0 when re-run. The pre-registered expectations in §8
are reported as they land. Weak, null and refuting results are published, not
retuned. Any falsification or finding set is pinned as literals so the script
fails for real if the set changes. No other product is modified to make a
finding go away.

Credits: This is under reserved rights obtained by OPTIMA Organisation.
