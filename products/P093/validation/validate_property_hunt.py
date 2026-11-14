"""A property hunt: search for counterexamples to the identities this product claims.

INPUT ORIGIN: SYNTHETIC and generated. Hypothesis draws the inputs; nothing here
touches measured data.

Each property below is an identity in exact arithmetic. The hunt runs
Hypothesis over it with the example count recorded in the output, and reports
either that no counterexample was found at that budget or the falsifying example
itself. A property that is falsified is NOT removed and NOT weakened into
vacuity: it is restated with the bound that explains the disagreement, and the
counterexample is pinned as a named literal regression test in
``tests/test_regressions.py``, because ``.hypothesis/`` is gitignored and an
example found by one run is lost on a fresh clone.

One property in this product has already been falsified this way and is recorded
here rather than quietly dropped. The claim that the trapezoidal integral is
invariant under a shift of the timestamps is FALSE in floating point:
Hypothesis found ``t = [0.0, 1e-4]``, ``shift = 2048.0``, where the relative
change is 2.0e-9, because the shifted spacing is not representable. The claim
this product makes instead is that the change is bounded by
``4 * EPS * max|t| / min(dt)``, which is the property hunted below.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import tempfile  # noqa: E402

# Hypothesis writes its example database into the WORKING DIRECTORY by default,
# and this script chdirs to the repository root, so a direct run would drop a
# .hypothesis directory into the tree. Point it outside before importing
# hypothesis, which reads this variable at import time.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY",
    tempfile.mkdtemp(prefix="powerenvelope-hypothesis-"),
)

import numpy as np  # noqa: E402
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.bounds import (  # noqa: E402
    EPS,
    extreme_value_bracket,
    riemann_bracket,
    slack_from_conditioning,
)
from powerenvelope.energy import (  # noqa: E402
    GAP_TOLERANCE_S,
    clip_series,
    merge_intervals,
    trapezoid_energy_mj,
)
from powerenvelope.schema import (  # noqa: E402
    POWER_COLUMNS,
    SchemaError,
    validate_power_header,
)

POWER = st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False)
SPACING = st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False)
COEFFICIENT = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)

HUNT = settings(
    max_examples=_bootstrap.HUNT_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def series(draw, min_size: int = 2, max_size: int = 60):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    spacings = draw(st.lists(SPACING, min_size=size - 1, max_size=size - 1))
    start = draw(st.floats(min_value=0.0, max_value=1e5, allow_nan=False))
    t = np.concatenate(([start], start + np.cumsum(spacings)))
    power = np.asarray(draw(st.lists(POWER, min_size=size, max_size=size)))
    return t, power


def _linearity() -> None:
    @HUNT
    @given(series(), COEFFICIENT, COEFFICIENT, st.data())
    def check(grid, alpha, beta, data):
        t, first = grid
        second = np.asarray(data.draw(st.lists(POWER, min_size=t.size, max_size=t.size)))
        mixed = trapezoid_energy_mj(t, alpha * first + beta * second)
        separate = alpha * trapezoid_energy_mj(t, first) + beta * trapezoid_energy_mj(
            t, second
        )
        assert abs(mixed / separate - 1.0) <= 1e-9

    check()


def _additivity() -> None:
    @HUNT
    @given(series(min_size=3), st.floats(min_value=0.05, max_value=0.95, allow_nan=False))
    def check(grid, split):
        t, power = grid
        low, high = float(t[0]), float(t[-1])
        middle = low + split * (high - low)
        assume(low < middle < high)
        left = clip_series(t, power, low, middle)
        right = clip_series(t, power, middle, high)
        assume(left[0].size >= 2 and right[0].size >= 2)
        whole = trapezoid_energy_mj(*clip_series(t, power, low, high))
        parts = trapezoid_energy_mj(*left) + trapezoid_energy_mj(*right)
        assert abs(parts / whole - 1.0) <= 1e-9

    check()


def _time_shift_bound() -> None:
    @HUNT
    @given(series(), st.floats(min_value=-1e4, max_value=1e4, allow_nan=False))
    def check(grid, shift):
        t, power = grid
        shifted = t + shift
        assume(np.all(np.diff(shifted) > 0.0))
        bound = 4.0 * EPS * float(np.max(np.abs(shifted))) / float(np.min(np.diff(t)))
        observed = abs(
            trapezoid_energy_mj(shifted, power) / trapezoid_energy_mj(t, power) - 1.0
        )
        assert observed <= max(bound, 1e-12)

    check()


def _brackets() -> None:
    @HUNT
    @given(series())
    def check(grid):
        t, power = grid
        value = trapezoid_energy_mj(t, power)
        assert riemann_bracket(t, power).contains(value)
        assert extreme_value_bracket(t, power).contains(value)

    check()


def _merge() -> None:
    @HUNT
    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            ),
            min_size=1,
            max_size=40,
        )
    )
    def check(intervals):
        starts = np.asarray([start for start, _width in intervals])
        ends = starts + np.asarray([width for _start, width in intervals])
        lo, hi = merge_intervals(starts, ends, GAP_TOLERANCE_S)
        assert np.all(hi >= lo)
        if lo.size > 1:
            assert np.all(lo[1:] - hi[:-1] > GAP_TOLERANCE_S)
        for start, end in zip(starts, ends, strict=True):
            assert np.any((lo <= start) & (hi >= end))

    check()


def _slack() -> None:
    @HUNT
    @given(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=40,
        )
    )
    def check(terms):
        array = np.asarray(terms, dtype=np.float64)
        assert slack_from_conditioning(array) >= 0.0
        assert slack_from_conditioning(array * 2.0) >= slack_from_conditioning(array) - 1e-300

    check()


def _header_rejection() -> None:
    @HUNT
    @given(st.sampled_from(POWER_COLUMNS), st.sampled_from(POWER_COLUMNS))
    def check(target, replacement):
        assume(target != replacement)
        header = tuple(replacement if name == target else name for name in POWER_COLUMNS)
        assume(header != POWER_COLUMNS)
        try:
            validate_power_header(header)
        except SchemaError:
            return
        raise AssertionError(f"header accepted: {header}")

    check()


PROPERTIES = (
    (
        "the energy integral is linear in power",
        "trapezoid(t, a*p + b*q) == a*trapezoid(t, p) + b*trapezoid(t, q), to 1e-9 "
        "relative. Both coefficients are positive and both powers are positive, "
        "because rail power is positive; a signed mix would be a test of catastrophic "
        "cancellation, which the bracket checks cover instead.",
        _linearity,
    ),
    (
        "the energy integral is additive over adjacent windows",
        "clipping at any interior point and summing the two parts gives the whole, to "
        "1e-9 relative. This is the identity the sleep-plus-active accounting rests on.",
        _additivity,
    ),
    (
        "a time shift changes the integral by no more than the timestamp resolution",
        "the naive invariance claim is FALSE and was falsified by this hunt; the claim "
        "is now |relative change| <= 4*EPS*max|t|/min(dt).",
        _time_shift_bound,
    ),
    (
        "the trapezoid lies inside both independent brackets",
        "the Riemann bracket and the mean-value bracket, neither of which uses the "
        "trapezoid rule.",
        _brackets,
    ),
    (
        "merged active windows are disjoint by more than the tolerance and cover the input",
        "every input interval lies inside exactly one merged window, and consecutive "
        "merged windows are separated by more than half a sampler period.",
        _merge,
    ),
    (
        "the conditioning slack is non-negative and grows with magnitude",
        "so that a narrow bracket cannot inherit an absolute floor.",
        _slack,
    ),
    (
        "the loader rejects every power header with a renamed column",
        "exhaustively over the column names, as a property rather than a list.",
        _header_rejection,
    ),
)


def main() -> int:
    out = _bootstrap.Tee("validate_property_hunt_output.txt")
    out.rule("PROPERTY HUNT -- searching for counterexamples to the claimed identities")
    out("INPUT ORIGIN: generated by Hypothesis. No measured data is involved.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(f"examples per property: {_bootstrap.HUNT_EXAMPLES}")
    out("")

    for name, description, runner in PROPERTIES:
        out.rule(name)
        out(f"  claim: {description}")
        try:
            runner()
        except AssertionError as error:
            first = str(error).splitlines()[0][:300]
            out.check(name, False, f"FALSIFIED: {first}")
            out(
                "  A falsified property is restated with the bound that explains the "
                "disagreement and the counterexample is pinned as a named literal "
                "regression test in tests/test_regressions.py. It is not deleted."
            )
        except Exception as error:  # noqa: BLE001 -- a hunt that cannot run is reported
            out.check(name, False, f"the hunt itself failed: {type(error).__name__}: {error}")
        else:
            out.check(
                name, True, f"no counterexample in {_bootstrap.HUNT_EXAMPLES} examples"
            )

    out.rule("properties already falsified, and what replaced them")
    out(
        "  1. 'the trapezoidal integral is invariant under a shift of the timestamps'. "
        "Falsified by this hunt at t = [0.0, 1e-4], shift = 2048.0, relative change "
        "2.0e-9. Replaced by the resolution bound above, which the hunt then could not "
        "falsify. Pinned as "
        "tests/test_regressions.py::test_time_shift_counterexample_t_zero_to_1e_minus_4_shift_2048."
    )
    out(
        "  The practical consequence for this product: the campaign's t_mono is the "
        "board's uptime, of order 1e5 to 1e6 seconds, with 0.05 s spacings, so every "
        "energy figure here carries about 1e-8 relative uncertainty from the timestamp "
        "representation alone. That is ten orders below the 40 mW current-quantisation "
        "floor and changes no reported figure. It is published rather than omitted."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
