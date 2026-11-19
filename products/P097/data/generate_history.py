"""Generate the committed recorded run history. Deterministic, run once.

THIS IS THE ONLY SOURCE OF EVERY NUMBER THIS PRODUCT ASSERTS ON.

The product is about intermittency. A product about intermittency whose own
checks are intermittent has refuted itself before a reader reaches the first
number, so nothing in this repository times anything live. Every duration every
test and every validation script asserts on is read from
``data/run_history.json``, which this script writes once, deterministically,
from a fixed seed, and which is committed together with its SHA-256 digest and
the transcript of the run that produced it
(``data/generate_history_output.txt``).

The mission this product was built in lost a whole product to checks that
compared two wall-clock readings on a shared two-core runner. Five intermittent
failures and a sixth caught in flight all traced to that one class. The
arrangement here is the direct consequence: no clock is read anywhere in the
package, the tests or the validation scripts.

This script lives in ``data/`` and **not** in ``validation/`` deliberately. The
release gate runs every ``validation/*.py`` in alphabetical order, so a
generator there would regenerate the very evidence a neighbouring script pins
the hash of, and the gate's own ordering would collide them.

THE SIMULATED PROCESS
---------------------

Sessions are CI pipeline runs, three per calendar day. In each session each test
is invoked unless it is skipped, and each invocation attempts the test until it
passes or the retry budget is exhausted, which is the stop-at-first-pass policy
:mod:`flakerank.history` validates the recorded history against.

Per test, drawn once at the declared seed:

* ``p``, the per-attempt failure probability, from one of four tiers: exactly
  zero for stable tests, and log-uniform bands for rare, moderate and frequent
  flakes. Exactly-zero tests exist because they are the majority in a real
  suite and because they make the flake-rate ranking heavily tied, which is
  the case tau-b's tie correction exists for.
* a lognormal attempt-duration distribution whose median spans 0.05 s to 900 s,
  drawn **independently of p**. That independence is the whole point: it
  produces slow rare flakes and fast frequent ones, which is where the
  flake-rate ranking and the cost ranking disagree. It is not a rigged
  construction -- a suite's slowest tests are its integration tests, and
  whether they are flaky is not a function of how slow they are.
* a ``fail_duration_factor``: failing attempts of some tests cost less than
  passing ones because an assertion fires early, and of others cost more
  because of a timeout. This deliberately violates assumption 3 of
  :mod:`flakerank.geometric` ("runtime independent of outcome") for part of the
  suite, so that the analytic model's use of the measured FAILING-attempt mean
  can be checked against a case where the two means differ.
* a regime change at a declared session for a handful of tests, multiplying
  ``p`` up or down. This is what makes the held-out TIME split a real test
  rather than a relabelling of the held-out TESTS split: for those tests the
  measured rate in the first window describes neither regime.

One test is pinned to ``p = 1/44`` with a 600-second median duration and is
named for the failure that motivated the product: the mission's own validation
probe failed 1 time in 44, and no number of agreeing serial verifications saw
it. At that rate in a 600-second test the expected wasted time per invocation
is ``600 / 43 = 13.95`` seconds, while the flake rate is 0.0227 -- below three
quarters of the flaky tests in this suite.

RUN IT
------

    python3 data/generate_history.py

from the product root, or from anywhere: the script anchors on ``__file__``.
It prints the digest of what it wrote. ``validation/validate_history.py`` pins
that digest and fails if the committed file is not the one this script
produces.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

#: The one declared seed for the generator.
SEED = 97044

#: Maximum attempts per invocation in the simulated retry policy.
RETRY_BUDGET = 3

#: Sessions, and sessions per calendar day.
N_SESSIONS = 330
SESSIONS_PER_DAY = 3

#: Session index at which the regime change takes effect.
REGIME_CHANGE_SESSION = 230

#: Probability that a test is skipped in a given session.
SKIP_PROBABILITY = 0.05

#: Suite names. Tests are distributed round-robin so that no suite is a proxy
#: for a flake tier, which would let a model learn the tier from the suite.
SUITES = (
    "unit",
    "integration",
    "hardware_in_loop",
    "regression",
    "end_to_end",
    "soak",
)

#: Tier sizes: (name, count, p_low, p_high). p_low == p_high == 0 is the stable
#: tier. Bands are sampled log-uniformly.
TIERS = (
    ("stable", 38, 0.0, 0.0),
    ("rare", 13, 0.002, 0.030),
    ("moderate", 13, 0.030, 0.120),
    ("frequent", 8, 0.120, 0.450),
)

#: Duration median band, seconds, sampled log-uniformly and independently of p.
DURATION_MEDIAN_LOW = 0.05
DURATION_MEDIAN_HIGH = 900.0

#: The pinned case, named for the mission's own recorded failure.
PINNED_NAME = "soak/test_validation_probe_1_in_44"
PINNED_P = 1.0 / 44.0
PINNED_MEDIAN_SECONDS = 600.0

#: Where the history is written, relative to the product root.
OUTPUT = Path("data") / "run_history.json"

#: Where the transcript is written.
TRANSCRIPT = Path("data") / "generate_history_output.txt"


def build() -> tuple[dict[str, object], list[str]]:
    """Build the history mapping and the transcript lines."""
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text)

    generator = np.random.default_rng(SEED)

    names: list[str] = []
    suites: list[str] = []
    probabilities: list[float] = []
    medians: list[float] = []
    sigmas: list[float] = []
    fail_factors: list[float] = []
    regime_factors: list[float] = []
    tiers: list[str] = []

    index = 0
    for tier_name, count, low, high in TIERS:
        for _ in range(count):
            suite = SUITES[index % len(SUITES)]
            names.append(f"{suite}/test_case_{index:03d}")
            suites.append(suite)
            if high <= 0.0:
                probabilities.append(0.0)
            else:
                probabilities.append(
                    float(np.exp(generator.uniform(np.log(low), np.log(high))))
                )
            medians.append(
                float(
                    np.exp(
                        generator.uniform(
                            np.log(DURATION_MEDIAN_LOW), np.log(DURATION_MEDIAN_HIGH)
                        )
                    )
                )
            )
            sigmas.append(float(generator.uniform(0.15, 0.50)))
            draw = generator.uniform()
            if draw < 0.10:
                fail_factors.append(0.35)
            elif draw < 0.16:
                fail_factors.append(1.50)
            else:
                fail_factors.append(1.0)
            change = generator.uniform()
            if tier_name != "stable" and change < 0.20:
                regime_factors.append(4.0)
            elif tier_name != "stable" and change < 0.26:
                regime_factors.append(0.2)
            else:
                regime_factors.append(1.0)
            tiers.append(tier_name)
            index += 1

    # The pinned case replaces the last rare-tier test, so the test count and
    # the suite round-robin are unchanged.
    pinned_index = TIERS[0][1] + TIERS[1][1] - 1
    names[pinned_index] = PINNED_NAME
    suites[pinned_index] = "soak"
    probabilities[pinned_index] = PINNED_P
    medians[pinned_index] = PINNED_MEDIAN_SECONDS
    sigmas[pinned_index] = 0.20
    fail_factors[pinned_index] = 1.0
    regime_factors[pinned_index] = 1.0
    tiers[pinned_index] = "rare"

    n_tests = len(names)
    say("flakerank recorded run history -- generator transcript")
    say("")
    say(f"seed: {SEED}")
    say(f"tests: {n_tests}")
    say(f"sessions: {N_SESSIONS} at {SESSIONS_PER_DAY} per calendar day")
    say(f"retry budget (maximum attempts per invocation): {RETRY_BUDGET}")
    say(f"skip probability per (session, test): {SKIP_PROBABILITY}")
    say(f"regime change takes effect at session: {REGIME_CHANGE_SESSION}")
    say(f"duration median band, seconds: [{DURATION_MEDIAN_LOW}, {DURATION_MEDIAN_HIGH}]")
    say(f"pinned case: {PINNED_NAME} at p = 1/44 = {PINNED_P!r}, median "
        f"{PINNED_MEDIAN_SECONDS} s")
    say("")
    say("tier sizes, and the duration band is sampled INDEPENDENTLY of p:")
    for tier_name, count, low, high in TIERS:
        say(f"  {tier_name:>9}: {count:>3} tests, p in [{low}, {high}]")
    say("")
    say(f"tests with a fail-duration factor != 1 (runtime depends on outcome): "
        f"{sum(1 for factor in fail_factors if factor != 1.0)}")
    say(f"tests with a regime change at session {REGIME_CHANGE_SESSION}: "
        f"{sum(1 for factor in regime_factors if factor != 1.0)}")

    session_column: list[int] = []
    test_column: list[int] = []
    attempt_column: list[int] = []
    failed_column: list[int] = []
    duration_column: list[int] = []

    for session in range(N_SESSIONS):
        for test in range(n_tests):
            if generator.uniform() < SKIP_PROBABILITY:
                continue
            probability = probabilities[test]
            if session >= REGIME_CHANGE_SESSION:
                probability = min(0.95, probability * regime_factors[test])
            for attempt in range(1, RETRY_BUDGET + 1):
                failed = bool(generator.uniform() < probability)
                scale = medians[test] * (fail_factors[test] if failed else 1.0)
                seconds = float(scale * np.exp(generator.normal(0.0, sigmas[test])))
                session_column.append(session)
                test_column.append(test)
                attempt_column.append(attempt)
                failed_column.append(int(failed))
                duration_column.append(max(1, int(round(seconds * 1000.0))))
                if not failed:
                    break

    attempts = len(session_column)
    failures = int(sum(failed_column))
    wasted_ms = int(
        sum(
            duration
            for duration, failed in zip(duration_column, failed_column, strict=True)
            if failed
        )
    )
    total_ms = int(sum(duration_column))
    say("")
    say(f"attempts recorded: {attempts}")
    say(f"failed attempts  : {failures}")
    say(f"pooled per-attempt failure rate: {failures / attempts!r}")
    say(f"total recorded attempt time, seconds: {total_ms / 1000.0!r}")
    say(f"wasted (failed-attempt) time, seconds: {wasted_ms / 1000.0!r}")
    say(f"wasted fraction of total: {wasted_ms / total_ms!r}")

    payload: dict[str, object] = {
        "schema": "flakerank.history/1",
        "generator": "data/generate_history.py",
        "seed": SEED,
        "retry_budget": RETRY_BUDGET,
        "duration_unit": "millisecond",
        "regime_change_session": REGIME_CHANGE_SESSION,
        "tests": [
            {
                "name": names[test],
                "suite": suites[test],
                "tier": tiers[test],
                "true_failure_probability": probabilities[test],
                "true_mean_ms": float(
                    medians[test] * np.exp(0.5 * sigmas[test] ** 2) * 1000.0
                ),
                "true_fail_duration_factor": fail_factors[test],
                "true_regime_factor": regime_factors[test],
            }
            for test in range(n_tests)
        ],
        "sessions": [
            {"index": session, "day": session // SESSIONS_PER_DAY}
            for session in range(N_SESSIONS)
        ],
        "attempts": {
            "session": session_column,
            "test": test_column,
            "attempt": attempt_column,
            "failed": failed_column,
            "duration_ms": duration_column,
        },
    }
    return payload, lines


def main() -> int:
    """Write the history, print and commit the transcript."""
    payload, lines = build()
    text = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    OUTPUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    size = OUTPUT.stat().st_size

    def say(value: str = "") -> None:
        lines.append(value)
        print(value)

    say("")
    say(f"written: {OUTPUT.as_posix()}")
    say(f"bytes: {size}")
    say(f"sha256: {digest}")
    say("")
    say("the mission forbids committing data over 1 MB. This file is under it;")
    say("the columnar integer layout is what makes that possible, and the")
    say("durations are exact integer milliseconds so that a sum of them carries")
    say("no representation error from the input.")
    say(f"within the 1 MB limit: {size < 1_000_000}")
    TRANSCRIPT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
