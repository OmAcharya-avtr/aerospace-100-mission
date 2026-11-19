# Dataset card — `data/run_history.json`

A synthetic CI run history: 72 tests over 330 sessions, 23836 recorded attempts
of which 1417 failed. It is the only source of every number this product
asserts on.

**This model is not certified for operational flight use.**

| property | value |
|---|---|
| path | `data/run_history.json` |
| schema | `flakerank.history/1` |
| bytes | 402100 |
| sha256 | `aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a` |
| generator | `data/generate_history.py` |
| generator seed | 97044 |
| generator transcript | `data/generate_history_output.txt` |
| tests | 72 |
| sessions | 330, at 3 per calendar day over 110 days |
| attempts | 23836 |
| failed attempts | 1417 |
| pooled per-attempt failure rate | 0.059447893941936565 |
| total recorded attempt time | 2953413.293 s |
| wasted (failed-attempt) time | 253493.035 s |
| wasted fraction of total | 0.08583053228642727 |
| shortest attempt | 0.011 s |
| longest attempt | 1848.115 s |
| retry budget | 3 attempts per invocation |

Every figure above is printed by `validation/validate_history.py` and is in
`validation/validate_history_output.txt`.

## Why it is synthetic, and what that costs

There is no real CI run history here. The product needs a history in which the
TRUE per-test failure probability and runtime distribution are known, because
otherwise there is nothing to validate the analytic cost model against except
the accumulation it is supposed to predict, and "two estimates of the same
sample agree" is not evidence. A generator gives a ground truth that is known by
construction and lets every negative control be anchored on a true value rather
than on an output.

What it costs is the thing that matters most and is stated first: **every figure
in this repository is a property of this generator as much as of the code.**
Carrying a number from here into a claim about a real test suite is unsupported.
The one place a real system is referred to at all is the motivating case below,
and that is an anecdote about why the product exists, not a measurement.

## What is in it

A columnar JSON object. A row-per-attempt layout for 23836 attempts does not fit
the mission's 1 MB limit on committed data; the columnar integer layout does, at
402100 bytes.

- `tests` — one object per test: `name`, `suite`, `tier`, and the generator's
  ground-truth parameters `true_failure_probability`, `true_mean_ms`,
  `true_fail_duration_factor`, `true_regime_factor`.
- `sessions` — one object per CI session: `index` and `day`.
- `attempts` — five equal-length integer arrays: `session`, `test`, `attempt`
  (1-based within a session for that test), `failed` (0 or 1) and
  `duration_ms`.

**The ground-truth fields are the generator's parameters, not measurements.**
Nothing in `flakerank.baselines` or `flakerank.predictor` reads them; they exist
so that `validation/validate_baselines.py` can anchor its controls on a true
value.

**Durations are exact integer milliseconds.** The total is 2953413293 ms, far
below `2**53`, so every partial sum of these integers is exactly representable
in double precision and the input contributes no representation error to any
accumulation. That is why the bound check in `validation/validate_bounds.py`
can attribute all of its slack to the accumulation itself.

## How it was generated

`data/generate_history.py`, deterministically, from seed 97044, once. It lives
in `data/` and **not** in `validation/` deliberately: the release gate runs
`sorted(glob('validation/*.py'))`, so a generator there would rewrite the
evidence an alphabetically adjacent script pins the hash of.

Per test, drawn once:

| tier | count | per-attempt failure probability `p` |
|---|---|---|
| stable | 38 | exactly 0 |
| rare | 13 | log-uniform in [0.002, 0.030] |
| moderate | 13 | log-uniform in [0.030, 0.120] |
| frequent | 8 | log-uniform in [0.120, 0.450] |

- **Duration.** A lognormal attempt duration whose median is log-uniform in
  [0.05, 900] seconds, with a log-scale sigma uniform in [0.15, 0.50], drawn
  **independently of `p`**. That independence is the point of the dataset: it
  produces slow rare flakes and fast frequent ones, which is where the
  flake-rate ranking and the cost ranking disagree. It is not a rigged
  construction — a suite's slowest tests are its integration tests, and whether
  they are flaky is not a function of how slow they are. The measured sample
  correlation between `p` and `log(mean duration)` over the 72 tests is
  -0.08985773213530207, which is 0.762 standard errors from zero under
  independence; `validation/validate_history.py` reports it.
- **Fail-duration factor.** 14 of 72 tests have failing attempts that cost
  0.35x or 1.50x a passing one, because an assertion can fire early or a
  timeout can fire late. This deliberately violates assumption 3 of
  `flakerank.geometric` ("runtime independent of outcome") for part of the
  suite, so the analytic model's use of the measured FAILING-attempt mean can be
  checked against a case where the two means differ.
- **Regime change.** 10 of 72 tests have `p` multiplied by 4 or by 0.2 from
  session 230 onwards. This is what makes the held-out TIME split a real test
  rather than a relabelling of the held-out TESTS split.
- **Skips.** Each test is skipped in a given session with probability 0.05, so
  per-test invocation counts differ.
- **The retry process.** In each session each invoked test is attempted until it
  passes or until the retry budget of 3 attempts is exhausted. The recorded
  history obeys that policy and `flakerank.history` validates that it does: an
  attempt recorded after a passing attempt in the same group is rejected.

## The motivating case

One test is pinned: `soak/test_validation_probe_1_in_44`, with
`true_failure_probability` exactly `1/44 = 0.022727272727272728` and a
600-second median duration.

It is named for a failure this mission actually shipped: a validation probe
failed 1 time in 44, and no number of agreeing serial verifications was able to
see it. At that rate in a 600-second test the expected wasted time is
`600/43 = 13.953488372093023` seconds per invocation, while the flake rate is
0.0227 — below three quarters of the flaky tests in this suite.

In the committed history that test recorded 315 attempts and 4 failures, so its
**measured** rate is 0.012698412698412698, a little over half its true rate.
That is an ordinary sampling outcome at this rate and this number of
invocations, and it is left as generated rather than reseeded: a rare flake
whose measured rate is badly estimated is the case the product exists for. It
ranks 8th of 72 by accumulated wasted time and 27th of 72 by flake rate.

## What it is not

- **Not real.** No production CI system, no real test names, no real
  infrastructure.
- **Not a benchmark anyone else uses.** There is no standard public flaky-test
  cost dataset, and this product does not claim to have created one.
- **Not representative of retry dependence.** Attempts are drawn independently.
  Real retries are frequently not: a leaked file, an exhausted port or a
  poisoned cache makes the next attempt more likely to fail, and an
  infrastructure outage makes whole batches fail together. Positive dependence
  would make the analytic model an under-estimate. The generator has no
  parameter for it and the product does not estimate one.
- **Not representative of duration dependence between tests.** A loaded CI
  runner slows every test at once; here each attempt's duration is drawn
  independently.
- **Not personal data.** The dataset contains no personal data, no
  organisational identifiers and nothing traceable to a person. Test names are
  `suite/test_case_NNN`.

## Licence and provenance

Generated for this product, released under the same Apache-2.0 licence as the
code, © 2026 OPTIMA Organisation. It is derived from no external dataset.

## Regenerating it

```bash
cd products/P097
python3 data/generate_history.py
python3 validation/validate_history.py
```

The first writes `data/run_history.json` and `data/generate_history_output.txt`
and prints the digest; the second pins the digest and fails if the committed
file is not the one the generator produces. Both run from any working
directory.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
