"""Integration tests for the audit. Configurations are deliberately tiny."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.audit import (
    METHODS,
    MODELS,
    breaking_point_sweep,
    coverage_audit,
    stratified_coverage,
)

SMALL = {"replicates": 3, "n_fit": 400, "n_calibration": 120, "n_test": 150, "seed": 57801}


@pytest.fixture(scope="module")
def small_audit():
    return coverage_audit(severities=(0.0, 2.0), **SMALL)


def test_row_count(small_audit):
    assert len(small_audit.rows) == len(MODELS) * len(METHODS) * 2


def test_rows_carry_the_configuration(small_audit):
    assert small_audit.replicates == 3
    assert small_audit.n_calibration == 120
    assert small_audit.severities == (0.0, 2.0)


def test_coverage_is_a_probability(small_audit):
    assert all(0.0 <= row.coverage <= 1.0 for row in small_audit.rows)


def test_confidence_interval_brackets_the_estimate(small_audit):
    for row in small_audit.rows:
        assert row.ci_low <= row.coverage <= row.ci_high


def test_clopper_pearson_is_narrower_than_the_replicate_interval(small_audit):
    # The point of reporting both: the pooled interval understates uncertainty
    # because test points inside a replicate share a quantile.
    narrower = sum(
        1
        for row in small_audit.rows
        if (row.cp_high - row.cp_low) < (row.ci_high - row.ci_low)
    )
    assert narrower >= len(small_audit.rows) // 2


def test_bound_columns_are_identical_on_every_row(small_audit):
    exact = {row.bound_exact for row in small_audit.rows}
    assert len(exact) == 1


def test_bound_matches_the_calibration_size(small_audit):
    # n = 120, alpha = 0.1: ceil(121 * 0.9) = 109, 109/121 = 0.900826446...
    row = small_audit.rows[0]
    assert row.bound_exact == pytest.approx(109 / 121, rel=1e-14)


def test_weighted_declared_equals_split_at_severity_zero(small_audit):
    for model in MODELS:
        split = small_audit.select(model=model, method="split")[0]
        weighted = small_audit.select(model=model, method="weighted_declared")[0]
        assert split.severity == 0.0
        assert weighted.severity == 0.0
        assert weighted.coverage == pytest.approx(split.coverage, rel=0.0)
        assert weighted.mean_width == pytest.approx(split.mean_width, rel=1e-12)


def test_unweighted_methods_report_full_effective_sample_size(small_audit):
    for method in ("parametric", "split", "mondrian"):
        for row in small_audit.select(method=method):
            assert row.ess == pytest.approx(120.0, rel=0.0)
            assert row.ess_fraction == pytest.approx(1.0, rel=0.0)


def test_weighted_effective_sample_size_falls_under_shift(small_audit):
    rows = small_audit.select(model="physics", method="weighted_declared")
    by_severity = {row.severity: row.ess_fraction for row in rows}
    assert by_severity[2.0] < by_severity[0.0]


def test_split_coverage_falls_under_shift(small_audit):
    rows = {row.severity: row.coverage for row in small_audit.select(model="physics", method="split")}
    assert rows[2.0] < rows[0.0]


def test_mondrian_width_varies_while_split_width_does_not(small_audit):
    split = {r.severity: r.mean_width for r in small_audit.select(model="physics", method="split")}
    mondrian = {
        r.severity: r.mean_width for r in small_audit.select(model="physics", method="mondrian")
    }
    assert split[0.0] == pytest.approx(split[2.0], rel=0.0)
    assert mondrian[0.0] != pytest.approx(mondrian[2.0], rel=1e-9)


def test_parametric_width_is_constant_across_severities(small_audit):
    widths = {
        r.severity: r.mean_width for r in small_audit.select(model="physics", method="parametric")
    }
    assert widths[0.0] == pytest.approx(widths[2.0], rel=0.0)


def test_infinite_fraction_is_zero_for_unweighted(small_audit):
    for method in ("parametric", "split", "mondrian"):
        assert all(row.infinite_fraction == 0.0 for row in small_audit.select(method=method))


def test_table_renders(small_audit):
    text = small_audit.table()
    assert "coverage" in text
    assert "weighted_declared" in text
    assert len(text.splitlines()) == len(small_audit.rows) + 2


def test_select_filters(small_audit):
    assert len(small_audit.select(model="physics")) == len(METHODS) * 2
    assert len(small_audit.select(method="split")) == len(MODELS) * 2
    assert len(small_audit.select(model="learned", method="split")) == 2


def test_below_nominal_and_covers_nominal_are_mutually_exclusive(small_audit):
    for row in small_audit.rows:
        assert not (row.below_nominal and row.covers_nominal)


def test_single_model_single_method_audit():
    result = coverage_audit(
        severities=(1.0,), models=("physics",), methods=("split",), **SMALL
    )
    assert len(result.rows) == 1
    assert result.rows[0].model == "physics"


def test_single_replicate_reports_undefined_replicate_error():
    result = coverage_audit(
        replicates=1,
        n_fit=300,
        n_calibration=120,
        n_test=100,
        seed=57802,
        severities=(0.0,),
        models=("physics",),
        methods=("split",),
    )
    assert np.isnan(result.rows[0].replicate_se)
    assert result.rows[0].per_replicate_sd == 0.0


@pytest.fixture(scope="module")
def small_sweep():
    return breaking_point_sweep(
        true_severity=3.0,
        fractions=(0.0, 0.5, 1.0, 1.5),
        replicates=3,
        n_fit=400,
        n_calibration=120,
        n_test=150,
        seed=57803,
        model="physics",
    )


def test_sweep_row_count(small_sweep):
    assert len(small_sweep.rows) == 4


def test_sweep_assumed_severity(small_sweep):
    assert [row.assumed_severity for row in small_sweep.rows] == [0.0, 1.5, 3.0, 4.5]


def test_sweep_coverage_increases_with_assumed_shift(small_sweep):
    coverage = [row.coverage for row in small_sweep.rows]
    assert coverage == sorted(coverage)


def test_sweep_width_increases_with_assumed_shift(small_sweep):
    widths = [row.mean_width for row in small_sweep.rows]
    assert widths == sorted(widths)


def test_sweep_effective_sample_size_falls_with_assumed_shift(small_sweep):
    ess = [row.ess for row in small_sweep.rows]
    assert ess == sorted(ess, reverse=True)


def test_sweep_zero_fraction_is_unweighted(small_sweep):
    assert small_sweep.rows[0].ess == pytest.approx(120.0, rel=1e-12)
    assert small_sweep.rows[0].ess_fraction == pytest.approx(1.0, rel=1e-12)


def test_sweep_table_renders(small_sweep):
    text = small_sweep.table()
    assert "fraction" in text
    assert len(text.splitlines()) == 6


def test_sweep_reports_a_breaking_point(small_sweep):
    # With three replicates the interval is wide, so only a gross failure is
    # detected; the assertion is on the shape of the answer, not its value.
    assert small_sweep.breaking_fraction is None or small_sweep.breaking_fraction <= 1.0


def test_sweep_breaking_point_is_below_last_holding(small_sweep):
    if small_sweep.breaking_fraction is not None and small_sweep.last_holding_fraction is not None:
        assert small_sweep.breaking_fraction < small_sweep.last_holding_fraction


def test_sweep_default_grid_has_thirty_one_points():
    result = breaking_point_sweep(
        true_severity=1.0, replicates=1, n_fit=300, n_calibration=120, n_test=60, seed=57804
    )
    assert len(result.rows) == 31
    assert result.rows[0].fraction == 0.0
    assert result.rows[-1].fraction == 1.5


def test_stratified_coverage_shape():
    tally = stratified_coverage(
        replicates=2, n_fit=400, n_calibration=120, n_test=150, seed=57805, severity=2.0
    )
    assert set(tally) == {"split", "mondrian"}
    assert set(tally["split"]) == {0, 1, 2}


def test_stratified_coverage_values_are_probabilities():
    tally = stratified_coverage(
        replicates=2, n_fit=400, n_calibration=120, n_test=150, seed=57806, severity=0.0
    )
    for per_bin in tally.values():
        for coverage, width, nominal, count in per_bin.values():
            assert 0.0 <= coverage <= 1.0
            assert width > 0.0
            assert nominal == pytest.approx(0.9, rel=1e-15)
            assert count > 0


def test_stratified_split_width_is_constant_across_strata():
    # One replicate only: within a replicate the marginal quantile is one
    # number, so every stratum gets the same width. Across replicates the
    # per-stratum means differ because the strata hold different counts.
    tally = stratified_coverage(
        replicates=1, n_fit=400, n_calibration=120, n_test=150, seed=57807, severity=0.0
    )
    widths = [tally["split"][b][1] for b in sorted(tally["split"])]
    assert max(widths) == pytest.approx(min(widths), rel=1e-12)


def test_stratified_mondrian_width_varies_across_strata():
    tally = stratified_coverage(
        replicates=2, n_fit=400, n_calibration=120, n_test=150, seed=57808, severity=0.0
    )
    widths = [tally["mondrian"][b][1] for b in sorted(tally["mondrian"])]
    assert max(widths) > min(widths)


def test_audit_is_reproducible():
    a = coverage_audit(severities=(1.0,), models=("physics",), methods=("split",), **SMALL)
    b = coverage_audit(severities=(1.0,), models=("physics",), methods=("split",), **SMALL)
    assert a.rows[0].coverage == pytest.approx(b.rows[0].coverage, rel=0.0)
    assert a.rows[0].mean_width == pytest.approx(b.rows[0].mean_width, rel=0.0)
