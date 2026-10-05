"""Tests for the uniform window-score adapters."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.arl import ewma_sigma_z
from telemetryool.detectors import (
    CusumDetector,
    EwmaDetector,
    LimitDetector,
    NoveltyDetector,
    build_detector_suite,
    smoothed_features,
)
from telemetryool.novelty import HotellingT2Q
from telemetryool.synthetic import NominalModel, equicorrelation, generate_nominal


def test_limit_detector_score_is_the_channel_max_absolute_value() -> None:
    block = np.array([[[1.0, -3.0], [0.5, 0.25]]])
    assert LimitDetector().scores(block).tolist() == [[3.0, 0.5]]


def test_limit_detector_can_be_restricted_to_channels() -> None:
    block = np.array([[[1.0, -3.0], [0.5, 0.25]]])
    assert LimitDetector(channels=[0]).scores(block).tolist() == [[1.0, 0.5]]


def test_ewma_detector_score_hand_values() -> None:
    """lam = 0.5, one channel, u = [2, 0]: z = [1.0, 0.5] and the score divides
    by sigma_z = sqrt(1/3), giving [1.7320508, 0.8660254]."""
    block = np.array([[[2.0], [0.0]]])
    scores = EwmaDetector(lam=0.5).scores(block)[0]
    assert scores[0] == pytest.approx(1.0 / ewma_sigma_z(0.5), abs=1e-14)
    assert scores[1] == pytest.approx(0.5 / ewma_sigma_z(0.5), abs=1e-14)


def test_cusum_detector_score_hand_values() -> None:
    """k = 0.5, one channel, u = [1, 1, -3]: C+ = [0.5, 1.0, 0.0],
    C- = [0, 0, 2.5], so max(C+, C-) = [0.5, 1.0, 2.5]."""
    block = np.array([[[1.0], [1.0], [-3.0]]])
    assert CusumDetector(k=0.5).scores(block)[0].tolist() == [0.5, 1.0, 2.5]


def test_cusum_detector_takes_the_channel_maximum() -> None:
    block = np.array([[[1.0, 3.0]]])
    # C+ channel 0 = 0.5, channel 1 = 2.5 -> score 2.5
    assert CusumDetector(k=0.5).scores(block)[0].tolist() == [2.5]


def test_smoothed_features_doubles_the_channel_count() -> None:
    block = np.zeros((4, 10, 3))
    assert smoothed_features(block, None).shape == (4, 10, 3)
    assert smoothed_features(block, 0.2).shape == (4, 10, 6)


def test_smoothed_features_hand_values() -> None:
    """lam = 0.5, u = [2, 0]: the raw features are [2, 0] and the smoothed ones
    are [1.0, 0.5] divided by sigma_z = sqrt(1/3)."""
    block = np.array([[[2.0], [0.0]]])
    feats = smoothed_features(block, 0.5)
    assert feats[0, :, 0].tolist() == [2.0, 0.0]
    assert feats[0, 0, 1] == pytest.approx(1.0 / ewma_sigma_z(0.5), abs=1e-14)
    assert feats[0, 1, 1] == pytest.approx(0.5 / ewma_sigma_z(0.5), abs=1e-14)


def test_smoothed_features_validation() -> None:
    with pytest.raises(ValueError, match=r"lam must lie in \(0, 1\] or be None"):
        smoothed_features(np.zeros((2, 3, 1)), 0.0)
    with pytest.raises(ValueError, match="must be 3-D"):
        smoothed_features(np.zeros((2, 3)), 0.2)


def test_novelty_detector_round_trip() -> None:
    model = NominalModel(3, correlation=equicorrelation(3, 0.5))
    train = generate_nominal(model, 200, 50, np.random.default_rng(0))
    det = NoveltyDetector(HotellingT2Q(n_components=2), persistence=2, smooth_lam=0.2)
    assert det.requires_fit
    det.fit(train)
    test = generate_nominal(model, 10, 50, np.random.default_rng(1))
    scores = det.scores(test)
    assert scores.shape == (10, 50)
    assert np.all(np.isfinite(scores))


def test_novelty_detector_pvalues_single_window() -> None:
    model = NominalModel(2)
    train = generate_nominal(model, 200, 50, np.random.default_rng(2))
    det = NoveltyDetector(HotellingT2Q(n_components=1), smooth_lam=None).fit(train)
    one = generate_nominal(model, 1, 50, np.random.default_rng(3))
    pvals = det.pvalues(one)
    assert len(pvals) == 50
    assert all(0.0 < p.pvalue <= 1.0 for p in pvals)
    with pytest.raises(ValueError, match="takes a single window"):
        det.pvalues(generate_nominal(model, 2, 50, np.random.default_rng(4)))


def test_detector_suite_is_baseline_first() -> None:
    suite = build_detector_suite()
    assert [type(d).__name__ for d in suite] == [
        "LimitDetector",
        "EwmaDetector",
        "CusumDetector",
        "NoveltyDetector",
        "NoveltyDetector",
        "NoveltyDetector",
    ]
    assert len({d.name for d in suite}) == 6
    assert [d.requires_fit for d in suite] == [False, False, False, True, True, True]


def test_charts_do_not_require_fitting() -> None:
    block = np.zeros((2, 5, 1))
    for det in (LimitDetector(), EwmaDetector(), CusumDetector()):
        assert det.fit(block) is det


def test_detector_validation() -> None:
    with pytest.raises(ValueError, match="persistence must be an integer >= 1"):
        LimitDetector(persistence=0)
    with pytest.raises(ValueError, match=r"lam must lie in \(0, 1\]"):
        EwmaDetector(lam=1.5)
    with pytest.raises(ValueError, match="k must be > 0"):
        CusumDetector(k=-1.0)
    with pytest.raises(ValueError, match="must be 3-D"):
        LimitDetector().scores(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="but the block has"):
        LimitDetector(channels=[4]).scores(np.zeros((2, 3, 2)))
