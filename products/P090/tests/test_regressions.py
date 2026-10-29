"""Defects found while building this package, pinned with their literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample Hypothesis found, and every defect found
by measurement during the build, is pinned here with the exact inputs that
produced it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlecheck import benchmark, dataset
from tlecheck.elements import apply_checksum, format_tle, parse_tle
from tlecheck.epoch import julian_date
from tlecheck.history import LEVEL_ELEMENTS, TREND_ELEMENTS, modified_z, robust_scale
from tlecheck.history import check as history_check
from tlecheck.screens import run
from tlecheck.verdict import classify

from conftest import REFERENCE_FIELDS


def _set(**overrides):
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields))


def test_modified_z_affine_invariance_fails_on_a_subnormal_spread():
    """Hypothesis counterexample, pinned with its literal inputs.

    ``values=[0.0, 1.0, 6.663237840215803e-40]``, ``shift=1.0``. The robust
    scale of the sample is 9.879...e-40; adding 1.0 to every element collapses
    the third onto 1.0 exactly, the shifted sample is constant, and the modified
    z-score jumps from 1.0122567671316239e+39 to an infinity. The affine
    invariance of equation (8) is exact in exact arithmetic and conditional in
    floating point, and the docstring of
    :func:`tlecheck.history.modified_z` says which.
    """
    values = np.array([0.0, 1.0, 6.663237840215803e-40])
    shifted = values + 1.0
    assert robust_scale(values) == pytest.approx(9.879e-40, rel=1.0e-3, abs=0.0)
    assert robust_scale(shifted) == 0.0
    assert modified_z(1.0, values) == pytest.approx(
        1.0122567671316239e39, rel=1.0e-12, abs=0.0
    )
    assert modified_z(2.0, shifted) == math.inf


def test_julian_date_is_one_day_low_in_january_1900():
    """Measured boundary of equation (5), outside its declared validity range.

    1900 January 1 0h UT is Julian date 2415020.5. Equation (5) returns
    2415019.5. The two-digit epoch-year window starts at 1957 so no decodable
    element set reaches this, and the limit is pinned rather than trusted.
    """
    assert julian_date(1900, 1, 1.0) == pytest.approx(2415019.5, rel=0.0, abs=0.0)


def test_raan_level_test_produced_a_constant_z_score():
    """The node regresses, so a level test against a median is an offset.

    Measured by ``validation/validate_baseline_history.py``, which runs both
    tests on the same 217 uncorrupted element sets: the level test gives a
    modified z-score between 1.3787 and 1.5246 on every one, a spread of 0.1459.
    The element is now in the trend group and this pins it there.
    """
    assert "raan_deg" in TREND_ELEMENTS
    assert "raan_deg" not in LEVEL_ELEMENTS


def test_unfloored_scale_produced_z_scores_of_order_ten_million():
    """The quantisation floor, pinned by the behaviour it prevents.

    A history whose mean motion is constant to the eighth decimal the format
    carries has a detrended robust scale of order 1e-8. Measured by
    ``validation/validate_baseline_history.py`` on a constructed unjittered
    history whose candidate sits exactly one count from its trend prediction:
    the unfloored z-score is 2.531364e+06 times the floored one. On the corpus
    this repository ships the floor changes nothing, because the generator's
    jitter already keeps the scale above one count, and that is published too.
    """
    history = []
    for index in range(12):
        fields = dict(REFERENCE_FIELDS)
        fields["epoch_day"] = 300.0 + 1.5 * index
        history.append(parse_tle(*format_tle(**fields)))
    verdict = history_check(_set(mean_motion_rev_per_day=15.50103473), history)
    assert verdict.z_scores["mean_motion_rev_per_day"] < 10.0
    unfloored = modified_z(
        15.50103473,
        np.array([entry.mean_motion_rev_per_day for entry in history]),
    )
    assert unfloored == math.inf or unfloored > 1.0e5


def test_checksum_only_corruption_must_not_predict_a_consequence():
    """31 of the baseline's false positives came from this, measured.

    A corruption of the checksum column alone leaves the data intact and the
    propagated position error exactly 0.0 km. Carrying a checksum indicator in
    equation (15) put every such element set into the silently-wrong class.
    Integrity and consequence are separate outputs and this pins the
    separation.
    """
    history = []
    for index in range(12):
        fields = dict(REFERENCE_FIELDS)
        fields["epoch_day"] = 300.0 + 1.5 * index
        history.append(parse_tle(*format_tle(**fields)))
    base = _set(epoch_day=300.0 + 1.5 * 12)
    broken = parse_tle(base.line1[:68] + str((int(base.line1[68]) + 4) % 10), base.line2)
    evidence = run(broken, history=history)
    predicted, score, problems, intact = classify(evidence)
    assert intact is False
    assert predicted == "benign"
    assert score <= 1.0
    assert any(problem.name == "checksum_mismatch" for problem in problems)


def test_an_unreadable_field_must_not_predict_a_loud_failure():
    """Measured: predicting loud from a parse failure drove the baseline below chance.

    On the test split the rule baseline predicted a loud failure for 13
    field-shifted element sets, of which the propagator refused 1 and 10 were
    measured silently wrong. The operating characteristic for that corruption
    family had an area of 0.3686, below chance, because the records most likely
    to be silently wrong were given the lowest possible score. An unreadable
    field now sets equation (15) to infinity.
    """
    base = _set()
    broken = parse_tle(base.line1, base.line2[:26] + "ABCDEFG" + base.line2[33:])
    predicted, score, _problems, intact = classify(run(broken))
    assert intact is False
    assert math.isinf(score)
    assert predicted == "silently_wrong"


def test_brouwer_mean_motion_written_as_kozai_cost_one_hundred_kilometres():
    """The generator defect that made the continuity screen measure itself.

    Writing the propagator's post-propagation mean motion into the Kozai field
    without the inverse transformation of equation (18) biases the semi-major
    axis by about half a kilometre, which is a period error and therefore an
    along-track drift. Measured by
    ``validation/validate_baseline_history.py``, which reinstates the omission:
    consecutive uncorrupted element sets disagree by a median of 112.09 km for a
    low-Earth object and 603.64 km for a sun-synchronous one without the
    inverse, against 0.0564 km and 0.0231 km with it. This pins the inverse.
    """
    from sgp4.api import Satrec

    satrec = Satrec.twoline2rv(_set().line1, _set().line2)
    brouwer = dataset.unkozai_mean_motion(satrec.no_kozai, satrec.ecco, satrec.inclo)
    recovered = dataset.kozai_mean_motion(brouwer, satrec.ecco, satrec.inclo)
    assert recovered == pytest.approx(satrec.no_kozai, rel=0.0, abs=0.0)
    assert brouwer < satrec.no_kozai


def test_recall_at_fpr_hand_arithmetic_was_wrong_the_first_time():
    """A test, not the code, was the defect; the corrected arithmetic is pinned.

    The first version of the known-answer test for equation (22) reasoned that
    a target rate of 0.25 over four negatives at 0.1, 0.2, 0.3 and 0.4 would
    stop the threshold at 0.4 and give a recall of 0.75. The threshold can come
    down to 0.35, which still flags only the one negative at 0.4 and flags all
    four positives: the answer is a recall of 1.0.
    """
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.45, 0.55, 0.65])
    positive = np.array([False, False, False, False, True, True, True, True])
    point = benchmark.recall_at_fpr(scores, positive, 0.25)
    assert point.recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert point.threshold == pytest.approx(0.35, rel=0.0, abs=0.0)


def test_parse_faults_are_attributed_to_the_line_they_are_on():
    """A fault string without a line prefix was attributed to line 1.

    The eccentricity fault message read ``eccentricity: columns 27-33 ...`` and
    :func:`tlecheck.screens.format_screen` attributed it to line 1 because the
    text did not begin with ``line 2``. Every fault string now carries its line.
    """
    base = _set()
    broken = parse_tle(base.line1, base.line2[:26] + "ABCDEFG" + base.line2[33:])
    assert all(
        fault.startswith(("line 1:", "line 2:", "satellite number"))
        for fault in broken.faults
    )
    from tlecheck.screens import format_screen

    faults = [f for f in format_screen(broken) if f.name == "parse_fault"]
    assert faults
    assert all(finding.line == 2 for finding in faults)


def test_blank_lines_are_object_minus_one_here_and_object_zero_in_sgp4():
    """The two parsers disagree on blank input and the disagreement is pinned."""
    from sgp4.api import Satrec

    assert parse_tle(" " * 69, " " * 69).satnum == -1
    assert Satrec.twoline2rv(" " * 69, " " * 69).satnum == 0


def test_field_shift_of_the_mean_motion_reads_five_point_five_for_fifteen_point_five():
    """The measured silent corruption that motivates the whole package.

    Rotating the eleven columns of the mean-motion field one place left turns
    15.50103472 rev/day into 5.501034721 rev/day, and the recomputed checksum
    is valid, so no checksum screen can see it.
    """
    base = _set()
    field = base.line2[52:63]
    rotated = field[1:] + field[:1]
    line2 = apply_checksum(base.line2[:52] + rotated + base.line2[63:68])
    shifted = parse_tle(base.line1, line2)
    assert shifted.mean_motion_rev_per_day == pytest.approx(
        5.501034721, rel=1.0e-9, abs=0.0
    )
    assert shifted.checksum_line2_stated == shifted.checksum_line2_computed
