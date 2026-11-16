"""Pinned measured figures, exactly reproducible from the committed traces.

Every literal in this file is a number that depends on NOTHING but
``data/traces/`` and the committed board source. The host-calibrated
``ns_per_op`` is deliberately absent: it is a wall-clock reading on the
analysis machine and moves between runs, so pinning a figure that carries it
would pin the machine rather than the measurement. The figures that do carry it
are asserted as invariances in ``tests/test_divergence.py`` and are reported,
not pinned, in ``validation/VALIDATION.md``.

Each value was produced by running the package over the committed campaign on
2026-11-16 and is written here to full double precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiltimingdiff import BOOTSTRAP_SEED, RESAMPLES
from hiltimingdiff import divergence as divergence_of

#: Nanoseconds per declared operation, per stage. The host-calibration-free
#: form of the divergence: R[s] = ns_per_declared_op[s] / ns_per_op.
NS_PER_DECLARED_OP = {
    "ingest": 0.46875,
    "background": 1.220703125,
    "threshold": 21.651753743489582,
    "centroid": 522.5922000979172,
    "gate": 95.1902876159667,
    "track": 47.644444444444446,
}

#: Median per-pass hardware stage time over the 540 pass units, nanoseconds.
HW_MEDIAN_NS = {
    "ingest": 3840.0,
    "background": 20000.0,
    "threshold": 266056.75,
    "centroid": 421630.0,
    "gate": 6144.0,
    "track": 8576.0,
}

#: Median declared operation count per stage over the same units.
DECLARED_OPS_MEDIAN = {
    "ingest": 8192.0,
    "background": 16384.0,
    "threshold": 12288.0,
    "centroid": 806.805,
    "gate": 64.5444,
    "track": 180.0,
}

#: The divergence spread, SPEC §5 output 2. Invariant under ns_per_op, and
#: BIT-EXACTLY so since 2026-11-16: it is computed as the ratio of the
#: calibration-free per-op extremes, where ns_per_op never enters. Computed via
#: the R values it moved by one ulp with the host calibration.
SPREAD = 1114.8633602088898

#: The pass-unit bootstrap interval on the spread, also bit-exact.
SPREAD_INTERVAL = (1113.7633835519944, 1124.9177840094571)

#: Kendall's tau-b between the declared and measured stage orderings.
TAU_B = 0.2

#: The explicit permutation: entry i is the declared-order position of the
#: stage at position i of the measured order.
PERMUTATION = (3, 0, 1, 5, 4, 2)

#: Shares of measured frame time held by argmax(R) and argmin(R).
ARGMAX_SHARE = 0.5805602572403938
ARGMIN_SHARE = 0.005287459117717222

#: Board drift: the end-over-start bracket, and the straight-line fit.
DRIFT_BRACKET_RATIO = 1.0752408447278534
DRIFT_FITTED_RATIO = 1.0682256407994783
DRIFT_FIT_SLOPE_NS_PER_S = 13.4355525587356
DRIFT_FIT_R_SQUARED = 0.7268610650663789
DRIFT_MAX_BLOCK_STEP = 1.04969329916919
DRIFT_INCREASING_STEPS = 4

#: The confounding check.
CONFOUND_ORDER_R_SQUARED = 0.23312847578756135
CONFOUND_WIDTH_R_SQUARED = 0.9824296874606212
CONFOUND_ORDER_WIDTH_PEARSON = 0.5011614417507341
#: Spearman's rho on TIE-AVERAGED MIDRANKS. The value 0.5333333333333333 was
#: published here before 2026-11-16 and was wrong: it came from
#: ``argsort(argsort(x))``, an ordinal rank that breaks ties by position, on a
#: series with five tied values. See tests/test_drift.py.
CONFOUND_ORDER_WIDTH_SPEARMAN = 0.5809475019311126

#: Frequency co-variation: pooled and within-cell variance shares per stage.
FREQUENCY_POOLED_R_SQUARED = {
    "ingest": 0.0005050494884748513,
    "background": 0.002018092883406042,
    "threshold": 0.00428664980281479,
    "centroid": 0.005091334602689734,
    "gate": 0.005286479936121302,
    "track": 0.008537090338330722,
}
FREQUENCY_WITHIN_CELL_R_SQUARED = {
    "ingest": 0.00019269536848354196,
    "background": 0.0038628918194912876,
    "threshold": 0.004570282291641248,
    "centroid": 0.001027382118275355,
    "gate": 0.0005143720856498613,
    "track": 0.0012054823553220295,
}
FREQUENCY_MEAN_KHZ = 1247863.829787234
FREQUENCY_RELATIVE_SPREAD = 0.8000872981230903

#: Gate branch totals over the whole campaign.
GATE_TOTALS = {
    "accept": 174420,
    "no_candidate": 8460,
    "low_score": 4567860,
    "area_out": 158520,
    "border": 33840,
}
GATE_FRAMES = 810000


@pytest.fixture(scope="module")
def unit_divergence(units):
    """Divergence at ``ns_per_op = 1``, so every figure is host-free."""
    return divergence_of(units, 1.0, resamples=64)


@pytest.mark.parametrize("stage", sorted(NS_PER_DECLARED_OP))
def test_ns_per_declared_op_is_pinned(unit_divergence, stage):
    assert unit_divergence.ns_per_declared_op[stage] == pytest.approx(
        NS_PER_DECLARED_OP[stage], rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", sorted(HW_MEDIAN_NS))
def test_hardware_median_is_pinned(unit_divergence, stage):
    assert unit_divergence.hw_median_ns[stage] == pytest.approx(
        HW_MEDIAN_NS[stage], rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", sorted(DECLARED_OPS_MEDIAN))
def test_declared_ops_median_is_pinned(unit_divergence, stage):
    assert unit_divergence.declared_ops_median[stage] == pytest.approx(
        DECLARED_OPS_MEDIAN[stage], rel=1e-12, abs=0.0
    )


def test_spread_is_pinned(unit_divergence):
    # Bit-exact, not approx: the spread is computed where ns_per_op never
    # enters, so it is reproducible to the last bit from the committed traces.
    assert unit_divergence.spread == SPREAD


def test_spread_interval_is_pinned(units):
    # Interval ENDPOINTS depend on the resample count, so this is pinned at the
    # package's published RESAMPLES and BOOTSTRAP_SEED -- the configuration the
    # validation scripts report -- and not at the suite's reduced fixture count.
    result = divergence_of(units, 1.0, resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
    assert (result.spread_interval.low, result.spread_interval.high) == SPREAD_INTERVAL


@pytest.mark.parametrize("factor", [1e-3, 777.0])
def test_spread_interval_pin_does_not_move_with_the_host_calibration(units, factor):
    result = divergence_of(units, factor, resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
    assert (result.spread_interval.low, result.spread_interval.high) == SPREAD_INTERVAL
    assert result.spread == SPREAD


def test_spread_exceeds_the_preregistered_three_by_more_than_two_orders(unit_divergence):
    assert unit_divergence.spread / 3.0 > 100.0


def test_tau_b_is_pinned(unit_divergence):
    assert unit_divergence.tau_b == pytest.approx(TAU_B, rel=1e-12, abs=0.0)


def test_permutation_is_pinned(unit_divergence):
    assert unit_divergence.permutation == PERMUTATION


def test_permutation_is_not_the_identity(unit_divergence):
    assert unit_divergence.permutation != (0, 1, 2, 3, 4, 5)


def test_declared_order_is_pinned(unit_divergence):
    assert unit_divergence.declared_order == (
        "gate",
        "track",
        "centroid",
        "ingest",
        "threshold",
        "background",
    )


def test_measured_order_is_pinned(unit_divergence):
    assert unit_divergence.measured_order == (
        "ingest",
        "gate",
        "track",
        "background",
        "threshold",
        "centroid",
    )


def test_argmax_is_centroid(unit_divergence):
    assert unit_divergence.argmax == "centroid"


def test_argmin_is_ingest(unit_divergence):
    assert unit_divergence.argmin == "ingest"


def test_argmax_share_is_pinned(unit_divergence):
    assert unit_divergence.argmax_share == pytest.approx(
        ARGMAX_SHARE, rel=1e-12, abs=0.0
    )


def test_argmin_share_is_pinned(unit_divergence):
    assert unit_divergence.argmin_share == pytest.approx(
        ARGMIN_SHARE, rel=1e-12, abs=0.0
    )


def test_drift_bracket_ratio_is_pinned(drift):
    assert drift.bracket_ratio == pytest.approx(DRIFT_BRACKET_RATIO, rel=1e-15, abs=0.0)


def test_drift_fitted_ratio_is_pinned(drift):
    assert drift.fitted_ratio == pytest.approx(DRIFT_FITTED_RATIO, rel=1e-10, abs=0.0)


def test_drift_fit_slope_is_pinned(drift):
    assert drift.fit.slope == pytest.approx(
        DRIFT_FIT_SLOPE_NS_PER_S, rel=1e-9, abs=0.0
    )


def test_drift_fit_r_squared_is_pinned(drift):
    assert drift.fit.r_squared == pytest.approx(DRIFT_FIT_R_SQUARED, rel=1e-10, abs=0.0)


def test_drift_largest_block_step_is_pinned(drift):
    assert drift.max_block_step_ratio == pytest.approx(
        DRIFT_MAX_BLOCK_STEP, rel=1e-12, abs=0.0
    )


def test_drift_increasing_step_count_is_pinned(drift):
    assert drift.monotonic_blocks == DRIFT_INCREASING_STEPS


def test_confounding_order_r_squared_is_pinned(confound):
    assert confound.on_order.r_squared == pytest.approx(
        CONFOUND_ORDER_R_SQUARED, rel=1e-10, abs=0.0
    )


def test_confounding_width_r_squared_is_pinned(confound):
    assert confound.on_width.r_squared == pytest.approx(
        CONFOUND_WIDTH_R_SQUARED, rel=1e-10, abs=0.0
    )


def test_confounding_order_width_pearson_is_pinned(confound):
    assert confound.order_width_correlation == pytest.approx(
        CONFOUND_ORDER_WIDTH_PEARSON, rel=1e-12, abs=0.0
    )


def test_confounding_order_width_spearman_is_pinned(confound):
    assert confound.order_width_spearman == pytest.approx(
        CONFOUND_ORDER_WIDTH_SPEARMAN, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", sorted(FREQUENCY_POOLED_R_SQUARED))
def test_pooled_frequency_variance_share_is_pinned(covariation, stage):
    assert covariation.per_stage[stage].pooled_variance_share == pytest.approx(
        FREQUENCY_POOLED_R_SQUARED[stage], rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize("stage", sorted(FREQUENCY_WITHIN_CELL_R_SQUARED))
def test_within_cell_frequency_variance_share_is_pinned(covariation, stage):
    assert covariation.per_stage[stage].within_cell_variance_share == pytest.approx(
        FREQUENCY_WITHIN_CELL_R_SQUARED[stage], rel=1e-9, abs=0.0
    )


def test_mean_core_frequency_is_pinned(covariation):
    assert covariation.frequency_mean_khz == pytest.approx(
        FREQUENCY_MEAN_KHZ, rel=1e-12, abs=0.0
    )


def test_frequency_relative_spread_is_pinned(covariation):
    assert covariation.frequency_relative_spread == pytest.approx(
        FREQUENCY_RELATIVE_SPREAD, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("branch", sorted(GATE_TOTALS))
def test_gate_branch_total_is_pinned(census, branch):
    assert census.totals[branch] == GATE_TOTALS[branch]


def test_gate_frame_total_is_pinned(census):
    assert census.n_frames_total == GATE_FRAMES


def test_the_published_ordering_inputs_are_far_from_every_float_edge(unit_divergence, units):
    # Asserts, rather than argues, that the `ordering` rescaling defect cannot
    # touch a published figure. `divergence()` applies `ordering` DIRECTLY to
    # the per-stage medians and to the declared operation counts -- there is no
    # rescaling in the published path at all -- and both arrays are finite,
    # pairwise distinct, and roughly 300 orders of magnitude from the nearest
    # float64 edge, with neighbouring values a third apart in relative terms
    # against a one-ulp spacing of about 2e-16.
    hardware = np.median(units.hw, axis=0)
    declared = np.median(units.ops, axis=0)
    for array in (hardware, declared):
        assert bool(np.all(np.isfinite(array)))
        assert len(set(array.tolist())) == array.size
        assert float(np.abs(array).min()) > 1e-290
        assert float(np.abs(array).max()) < 1e290
        gaps = np.diff(np.sort(array)) / np.sort(array)[:-1]
        assert float(gaps.min()) > 0.3


@pytest.mark.parametrize("factor", [1e-6, 0.1, 0.5, 2.0, 10.0, 1000.0, 1e6])
def test_published_rank_outputs_do_not_move_under_any_ns_per_op(units, factor):
    # The same statement from the other direction, over the full range of
    # host calibrations: the permutation, both orders, both arg-extrema, the
    # spread and tau-b are bit-identical across six orders of magnitude of
    # ns_per_op, because `ordering` never sees a rescaled value.
    base = divergence_of(units, 1.0, resamples=16)
    scaled = divergence_of(units, factor, resamples=16)
    assert scaled.permutation == base.permutation == PERMUTATION
    assert scaled.declared_order == base.declared_order
    assert scaled.measured_order == base.measured_order
    assert (scaled.argmax, scaled.argmin) == (base.argmax, base.argmin)
    assert scaled.spread == pytest.approx(base.spread, rel=1e-12, abs=0.0)
    assert scaled.tau_b == pytest.approx(base.tau_b, rel=1e-12, abs=0.0)


def test_acceptance_rate_is_pinned(census):
    assert census.acceptance_rate == pytest.approx(
        174420.0 / 810000.0, rel=1e-15, abs=0.0
    )
