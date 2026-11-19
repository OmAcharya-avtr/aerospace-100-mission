"""End to end: record to decomposition to three baselines to a learned score.

One test walks the whole product on a subset of the committed record, so a
break in the seams between modules fails here rather than in a validation
script the gate runs much later.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from passyield.baselines import (
    accounted_yield,
    analytic_shortfall_distribution,
    booked_shortfall_mbit,
    fit_event_model,
)
from passyield.bounds import inside, shortfall_bracket
from passyield.dataset import (
    events_of,
    feature_matrix,
    parse_intervals,
    station_by_name,
    station_split,
    target_vector,
    time_split,
    window_of,
)
from passyield.decompose import decompose_pass
from passyield.link import DEFAULT_BUDGET
from passyield.predictor import PassYieldPredictor, PredictorSettings
from passyield.report import (
    format_accounted,
    format_comparison,
    format_decomposition,
    format_distribution,
)
from passyield.scoring import crps_mean, interval_coverage, pit_calibration_error, pit_values


def _home_prefix() -> str:
    """``<sep>home<sep>``, assembled rather than written as a literal.

    An absolute home-directory path in any committed file fails release-gate
    check 8. The scanner that enforces that lives in
    ``tests/test_determinism.py`` and assembles its needle from ``os.sep`` so
    that its own source does not match; this helper does the same, so a test
    that CHECKS for an absolute path does not become one.
    """
    return os.sep + "home" + os.sep



#: A small forest, for the suite's compute budget. The published figures use
#: 300 trees; the difference is declared, not hidden.
SETTINGS = PredictorSettings(n_estimators=40, max_depth=8, random_state=99099)


def test_whole_pipeline_on_a_subset(satrec, record):
    """Record -> decomposition -> three baselines -> learned score -> report."""
    rows = record[:200]

    decompositions = [
        decompose_pass(satrec, station_by_name(row.station), window_of(row), events_of(row))
        for row in rows
    ]

    # The committed decomposition columns must reproduce from the committed
    # events. Compared to tolerance, never bit-exactly: the generator and this
    # run are the same arithmetic but the record round-trips through nine
    # significant digits, so each column carries about 1e-9 of relative
    # error. The SHORTFALL is a difference of two such columns of comparable
    # size, so its conditioning is worse by the ratio
    # ``nominal / (nominal - realised)``, which on this record is about 9.
    # Its tolerance is therefore an order of magnitude looser than the
    # columns it is built from -- the tolerance follows the conditioning of
    # the expression, not a single constant for the whole test.
    #
    # The ABSOLUTE tolerance is the one that matters for a pass whose realised
    # volume is only a few hundred Mbit, and it is derived in
    # ``validation/validate_record.py`` from three printing-precision effects
    # of the record's own format: the column's own nine significant digits,
    # ``duration_s``'s nine significant digits reaching the rate, and nine
    # decimal places of a second on every interval endpoint. Those sum to of
    # order 1e-4 Mbit; the bound is 1e-3, an order of magnitude above, because
    # the three do not simply add. The MEASURED worst over the whole record is
    # 1.825e-04 Mbit, a margin of 5.48 times, published in VALIDATION.md.
    for row, item in zip(rows, decompositions, strict=True):
        assert item.realised_bits == pytest.approx(row.realised_mbit, rel=1.0e-8, abs=1.0e-3)
        assert item.capacity_bits == pytest.approx(row.capacity_mbit, rel=1.0e-8, abs=1.0e-3)
        assert item.shortfall_bits == pytest.approx(row.shortfall_mbit, rel=1.0e-7, abs=1.0e-3)

    # Baseline 1.
    assert booked_shortfall_mbit() == 0.0

    # Baseline 3, with the pass as the resampling unit.
    accounted = accounted_yield(decompositions, seed=99099, resamples=300)
    assert accounted.resampling_unit == "pass"
    assert accounted.passes == len(rows)

    # Baseline 2 on one pass, inside the independent bracket.
    model = fit_event_model(
        acquisition_s=[row.acquisition_s for row in rows],
        outage_counts=[row.outage_count for row in rows],
        window_durations_s=[row.duration_s for row in rows],
        outage_durations_s=[
            duration for row in rows for _s, duration in parse_intervals(row.resync_intervals)
        ],
        idle_totals_s=[row.idle_total_s for row in rows],
    )
    target_row, target_item = rows[0], decompositions[0]
    analytic = analytic_shortfall_distribution(
        satrec, station_by_name(target_row.station), window_of(target_row), model,
        target_item.nominal_bits, target_item.capacity_bits,
    )
    bracket = shortfall_bracket(
        target_item.nominal_bits, target_item.window_duration_s,
        DEFAULT_BUDGET.max_rate_mbit_s,
    )
    for level in (0.05, 0.5, 0.95):
        assert inside(analytic.quantile(level), bracket)

    # The learned component on both splits, scored on the shortfall in Mbit.
    matrix = feature_matrix(record)
    target = target_vector(record)
    nominal = np.asarray([row.nominal_mbit for row in record])
    capacity = np.asarray([row.capacity_mbit for row in record])
    observed = np.asarray([row.shortfall_mbit for row in record])
    scores: dict[str, dict[str, float]] = {}
    for split in (station_split(record), time_split(record)):
        train, test = list(split.train), list(split.test)
        predictor = PassYieldPredictor(SETTINGS).fit(matrix[train], target[train])
        ensemble = predictor.predict_shortfall_ensemble(
            matrix[test], nominal[test], capacity[test]
        )
        low = np.quantile(ensemble, 0.10, axis=1)
        high = np.quantile(ensemble, 0.90, axis=1)
        scores[split.name] = {
            "crps": crps_mean(ensemble, observed[test]),
            "mae": float(np.mean(np.abs(ensemble.mean(axis=1) - observed[test]))),
            "pit_calibration_error": pit_calibration_error(
                pit_values(ensemble, observed[test])
            ),
            "coverage80": interval_coverage(low, high, observed[test]),
        }
        assert scores[split.name]["crps"] > 0.0

    # Baseline 1's CRPS is its absolute error, and the learned model must beat
    # it on both splits -- the booked figure is the thing being refuted, and if
    # it were not refuted that would itself be the finding and this assertion
    # would be the one to change.
    for split in (station_split(record), time_split(record)):
        test = list(split.test)
        booked = float(np.mean(np.abs(observed[test] - booked_shortfall_mbit())))
        assert scores[split.name]["crps"] < booked

    # Everything renders.
    for text in (
        format_decomposition(target_item),
        format_distribution(analytic),
        format_accounted(accounted),
        format_comparison(scores, "both splits"),
    ):
        assert isinstance(text, str) and text.strip()
        assert _home_prefix() not in text


def test_report_renders_a_zero_shortfall_without_dividing_by_zero():
    """A pass with no shortfall renders ``n/a`` shares, not a division error."""
    from passyield.decompose import Decomposition

    item = Decomposition(
        station="GS-01", nominal_bits=100.0, capacity_bits=100.0, realised_bits=100.0,
        shortfall_bits=0.0, geometry_bits=0.0, acquisition_bits=0.0, resync_bits=0.0,
        idle_bits=0.0, residual_bits=0.0, window_duration_s=10.0, active_duration_s=10.0,
        acquisition_truncated=False, max_mask_clamp_deg=0.0,
    )
    text = format_decomposition(item)
    assert "n/a" in text
