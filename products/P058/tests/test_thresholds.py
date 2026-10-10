"""Calibration tests: the procedure, its guards, and its honest failure mode."""

from __future__ import annotations

import numpy as np
import pytest

from telemdrift.benchmark import STANDARD, analytic_factory
from telemdrift.detectors import ADWIN, CUSUM
from telemdrift.streams import stationary
from telemdrift.thresholds import calibrate_threshold


def _stationary(length, seed):
    return stationary(length, seed)


def test_calibration_moves_cusum_toward_the_target():
    res = calibrate_threshold(
        name="CUSUM",
        factory_from_threshold=lambda th: CUSUM(h=th),
        default_threshold=4.0,
        stream_fn=_stationary,
        target_arl0=500.0,
        calibration_seeds=[81, 82, 83],
        evaluation_seeds=[91, 92, 93],
        calibration_length=20_000,
        evaluation_length=20_000,
    )
    assert not res.bracketing_failed
    # The default h = 4 gives ARL0 ~ 168; a target of 500 must raise h.
    assert res.threshold > 4.0
    # Within 35 % of target on held-out seeds. The band is wide on purpose: at
    # 3 x 20000 samples the relative standard error is about 7 % on each of the
    # two estimates, and bisecting against a noisy objective adds a selection
    # effect of the same order. A tighter band would be a claim the sample size
    # does not support.
    assert abs(res.target_error) < 0.35, res.summary()


def test_calibration_handles_the_inverse_relationship_of_adwin_delta():
    """ADWIN's ARL0 *decreases* with delta, so the bracket search must detect
    the sign rather than assume it. A procedure that assumed monotone-increasing
    would walk the wrong way and report a bracketing failure here."""
    res = calibrate_threshold(
        name="ADWIN",
        factory_from_threshold=lambda th: ADWIN(delta=th),
        default_threshold=0.002,
        stream_fn=_stationary,
        target_arl0=500.0,
        calibration_seeds=[84, 85],
        evaluation_seeds=[94, 95],
        calibration_length=20_000,
        evaluation_length=20_000,
        clip=(1e-12, 0.999_999),
    )
    assert not res.bracketing_failed
    assert res.threshold < 0.002
    assert abs(res.target_error) < 0.5, res.summary()


def test_calibration_reports_bracketing_failure_instead_of_raising():
    """An unreachable target is a finding, not an exception.

    ARL0 = 10^9 samples cannot be reached by any CUSUM threshold inside the
    bracket budget, so the procedure must say so and return its closest
    endpoint. A sweep over many targets records the failure and continues.
    """
    res = calibrate_threshold(
        name="CUSUM",
        factory_from_threshold=lambda th: CUSUM(h=th),
        default_threshold=4.0,
        stream_fn=_stationary,
        target_arl0=1e9,
        calibration_seeds=[86],
        evaluation_seeds=[96],
        calibration_length=4_000,
        evaluation_length=4_000,
        bracket_steps=3,
    )
    assert res.bracketing_failed
    assert np.isfinite(res.threshold)
    assert "BRACKETING FAILED" in res.summary()


def test_calibration_rejects_overlapping_seed_sets():
    """An achieved ARL0 measured on the seeds it was fitted to is not an
    achieved ARL0, so the overlap is refused rather than warned about."""
    with pytest.raises(ValueError, match="must be disjoint"):
        calibrate_threshold(
            name="CUSUM",
            factory_from_threshold=lambda th: CUSUM(h=th),
            default_threshold=4.0,
            stream_fn=_stationary,
            target_arl0=500.0,
            calibration_seeds=[1, 2],
            evaluation_seeds=[2, 3],
        )


@pytest.mark.parametrize("cal,ev", [([], [1]), ([1], [])])
def test_calibration_rejects_an_empty_seed_set(cal, ev):
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_threshold(
            name="CUSUM",
            factory_from_threshold=lambda th: CUSUM(h=th),
            default_threshold=4.0,
            stream_fn=_stationary,
            target_arl0=500.0,
            calibration_seeds=cal,
            evaluation_seeds=ev,
        )


def test_calibration_rejects_a_nonpositive_target():
    with pytest.raises(ValueError, match="target_arl0 must be > 0"):
        calibrate_threshold(
            name="CUSUM",
            factory_from_threshold=lambda th: CUSUM(h=th),
            default_threshold=4.0,
            stream_fn=_stationary,
            target_arl0=0.0,
            calibration_seeds=[1],
            evaluation_seeds=[2],
        )


def test_calibration_is_deterministic_given_the_seeds():
    kwargs = {
        "name": "CUSUM",
        "factory_from_threshold": lambda th: CUSUM(h=th),
        "default_threshold": 4.0,
        "stream_fn": _stationary,
        "target_arl0": 400.0,
        "calibration_seeds": [87, 88],
        "evaluation_seeds": [97, 98],
        "calibration_length": 8_000,
        "evaluation_length": 8_000,
    }
    a = calibrate_threshold(**kwargs)
    b = calibrate_threshold(**kwargs)
    assert a.threshold == b.threshold
    assert a.achieved.arl0 == b.achieved.arl0


def test_standard_config_seed_sets_are_disjoint():
    assert STANDARD.seeds_disjoint


def test_analytic_factory_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown analytic detector"):
        analytic_factory("kalman", 1.0)


def test_calibration_result_target_error_is_signed():
    res = calibrate_threshold(
        name="CUSUM",
        factory_from_threshold=lambda th: CUSUM(h=th),
        default_threshold=4.0,
        stream_fn=_stationary,
        target_arl0=200.0,
        calibration_seeds=[89],
        evaluation_seeds=[99],
        calibration_length=8_000,
        evaluation_length=8_000,
    )
    expected = (res.achieved.arl0 - 200.0) / 200.0
    assert res.target_error == pytest.approx(expected)
    assert res.achieved_arl0 == res.achieved.arl0
