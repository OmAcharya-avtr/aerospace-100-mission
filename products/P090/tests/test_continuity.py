"""Ephemeris continuity, its self-calibration and its resolution limit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlecheck.continuity import (
    DEFAULT_SCALE_FLOOR_KM,
    MINIMUM_PAIRS,
    check,
    consistency_constant,
    pairwise_offsets_km,
)
from tlecheck.elements import format_tle, parse_tle

from conftest import REFERENCE_FIELDS, record_of


def _stranger():
    fields = dict(REFERENCE_FIELDS)
    fields["satnum"] = 11111
    return parse_tle(*format_tle(**fields))


def test_consistency_constant_is_shared_with_the_history_screen():
    assert consistency_constant() == 1.4826


def test_scale_floor_default_is_the_documented_one():
    assert DEFAULT_SCALE_FLOOR_KM == 0.050


def test_pairwise_offsets_are_small_for_a_generated_history(clean_record):
    offsets = pairwise_offsets_km(list(clean_record.history))
    assert len(offsets) >= MINIMUM_PAIRS
    assert all(math.isfinite(value) for value in offsets)
    assert all(value >= 0.0 for value in offsets)


def test_clean_candidate_does_not_exceed_its_own_history(clean_record):
    verdict = check(
        clean_record.candidate, list(clean_record.history), clean_record.use_jd
    )
    assert verdict.usable is True
    assert verdict.exceeded is False


def test_refusal_with_too_few_history_entries(clean_record):
    verdict = check(
        clean_record.candidate, list(clean_record.history)[:2], clean_record.use_jd
    )
    assert verdict.usable is False
    assert "needs at least" in verdict.reason
    assert math.isnan(verdict.offset_km)


def test_refusal_is_not_a_pass(small_records):
    record = record_of(small_records, "sign_flip")
    verdict = check(record.candidate, list(record.history)[:1], record.use_jd)
    assert verdict.usable is False
    assert verdict.exceeded is False


def test_resolution_is_reported_and_compared_to_the_threshold(clean_record):
    verdict = check(
        clean_record.candidate,
        list(clean_record.history),
        clean_record.use_jd,
        resolves_km=1.0,
    )
    assert verdict.resolution_km >= DEFAULT_SCALE_FLOOR_KM
    assert verdict.resolves_km == 1.0
    assert verdict.resolves_threshold == (verdict.resolution_km <= 1.0)


def test_resolution_never_falls_below_its_floor(clean_record):
    verdict = check(
        clean_record.candidate,
        list(clean_record.history),
        clean_record.use_jd,
        scale_floor_km=10.0,
    )
    assert verdict.resolution_km >= 10.0


def test_a_tiny_threshold_is_never_resolved(clean_record):
    verdict = check(
        clean_record.candidate,
        list(clean_record.history),
        clean_record.use_jd,
        resolves_km=1.0e-9,
    )
    assert verdict.resolves_threshold is False


def test_excess_is_one_sided(clean_record):
    verdict = check(
        clean_record.candidate, list(clean_record.history), clean_record.use_jd
    )
    assert verdict.excess_km >= 0.0


def test_z_score_is_one_sided(clean_record):
    verdict = check(
        clean_record.candidate, list(clean_record.history), clean_record.use_jd
    )
    assert verdict.z_score >= 0.0


def test_loud_candidate_makes_the_screen_unusable(small_records, clean_record):
    broken = parse_tle(
        clean_record.candidate.line1,
        clean_record.candidate.line2[:26]
        + "9999999"
        + clean_record.candidate.line2[33:],
    )
    verdict = check(broken, list(clean_record.history), clean_record.use_jd)
    assert verdict.usable is False
    assert "propagated loudly" in verdict.reason


def test_normal_offsets_are_returned_for_inspection(clean_record):
    verdict = check(
        clean_record.candidate, list(clean_record.history), clean_record.use_jd
    )
    assert len(verdict.normal_offsets_km) >= MINIMUM_PAIRS
    assert verdict.median_km == pytest.approx(
        float(np.median(verdict.normal_offsets_km)), rel=0.0, abs=1.0e-9
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_bad_threshold(clean_record, bad):
    with pytest.raises(ValueError, match="threshold"):
        check(
            clean_record.candidate,
            list(clean_record.history),
            clean_record.use_jd,
            threshold=bad,
        )


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_rejects_bad_scale_floor(clean_record, bad):
    with pytest.raises(ValueError, match="scale_floor_km"):
        check(
            clean_record.candidate,
            list(clean_record.history),
            clean_record.use_jd,
            scale_floor_km=bad,
        )


def test_rejects_history_of_another_object(clean_record):
    with pytest.raises(ValueError, match="history entry for object"):
        check(
            clean_record.candidate,
            [*clean_record.history, _stranger()],
            clean_record.use_jd,
        )


def test_sign_flip_record_is_screened_without_refusing(small_records):
    """The screen runs on a sign-flipped candidate and produces finite numbers.

    It deliberately does **not** assert that the offset rises. Measured over
    the two sign-flip records of this small corpus at a one-day horizon, the
    candidate's offset was below its own reference's on both, and over a
    seven-day horizon on one of two. A one-day drag-term flip is often smaller
    than the object's own element-to-element variation, which is a property of
    the problem and is reported with a population behind it in
    ``validation/validate_rule_screens.py``, not asserted on two records here.
    """
    record = record_of(small_records, "sign_flip")
    verdict = check(record.candidate, list(record.history), record.use_jd)
    assert verdict.usable is True
    assert math.isfinite(verdict.offset_km)
    assert math.isfinite(verdict.z_score)
    assert verdict.excess_km >= 0.0
