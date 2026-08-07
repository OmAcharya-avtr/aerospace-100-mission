"""Quadrature-convergence and regression tests.

The metrics are only as good as the integrals behind them, so the grid
behaviour is pinned here:

* on a smooth profile (Hufnagel-Valley) composite Simpson must converge
  towards the adaptive-quadrature answer at a fast rate;
* on a discontinuous profile (SLC) convergence is slow and not monotone, which is a property of
  the rule, not a bug - the regression bounds below record how slow, so that a
  future change cannot silently make it worse;
* the adaptive result must be independent of the (unused) node count and must
  reproduce closed forms exactly where they exist.
"""

import numpy as np
import pytest

from atmoprofile import (
    ConvergenceRecord,
    constant_profile,
    effective_turbulence_height,
    fried_parameter,
    grid_convergence,
    hv57,
    slc_day,
    slc_night,
    turbulence_moment,
    weighted_integral,
)

LAM = 500e-9


class TestExactCases:
    def test_simpson_is_exact_for_a_constant_profile(self):
        # Simpson integrates a constant exactly (any node count).
        slab = constant_profile(3e-15, 0.0, 1500.0)
        exact = 3e-15 * 1500.0
        for n in (3, 51, 1001):
            got = turbulence_moment(slab, 0.0, method="simpson", n_nodes=n)
            assert got == pytest.approx(exact, rel=1e-14)

    def test_simpson_matches_quad_on_a_power_law_weight(self):
        # int_0^H h^(5/3) dh = (3/8) H^(8/3); Simpson on a smooth integrand
        # should reach ~1e-9 relative accuracy with 2001 nodes.
        slab = constant_profile(1e-15, 0.0, 1000.0)
        exact = 1e-15 * (3 / 8) * 1000 ** (8 / 3)
        simp = turbulence_moment(slab, 5 / 3, method="simpson", n_nodes=2001)
        assert simp == pytest.approx(exact, rel=1e-8)

    def test_node_count_is_ignored_by_quad(self):
        profile = hv57()
        a = fried_parameter(profile, LAM, method="quad", n_nodes=11)
        b = fried_parameter(profile, LAM, method="quad", n_nodes=99_999)
        assert a == b


class TestConvergenceSmoothProfile:
    def test_hv57_r0_converges(self):
        profile = hv57()
        reference = fried_parameter(profile, LAM)  # adaptive quadrature
        records = grid_convergence(
            lambda n: fried_parameter(profile, LAM, method="simpson", n_nodes=n),
            (201, 1001, 4001, 16001),
            reference,
        )
        assert all(isinstance(r, ConvergenceRecord) for r in records)
        errors = [r.rel_error_vs_reference for r in records]
        # Monotone improvement, and 4001 nodes are already good to 1e-7.
        assert errors == sorted(errors, reverse=True)
        assert errors[0] < 1e-2  # 201 nodes: measured 2.26e-3
        assert errors[2] < 1e-7  # 4001 nodes: measured 1.58e-8
        assert errors[3] < 1e-9  # 16001 nodes: measured 6.19e-11

    def test_reported_relative_change_shrinks(self):
        profile = hv57()
        reference = turbulence_moment(profile, 5 / 3)
        records = grid_convergence(
            lambda n: turbulence_moment(profile, 5 / 3, method="simpson", n_nodes=n),
            (501, 2001, 8001),
            reference,
        )
        changes = [r.rel_change for r in records[1:]]
        assert np.isnan(records[0].rel_change)
        assert changes[1] < changes[0]
        assert changes[-1] < 1e-6  # measured 1.28e-7 between 2001 and 8001 nodes


class TestConvergenceDiscontinuousProfile:
    @pytest.mark.parametrize("factory", [slc_day, slc_night])
    def test_slc_simpson_converges_slowly(self, factory):
        profile = factory()
        reference = fried_parameter(profile, LAM)
        records = grid_convergence(
            lambda n: fried_parameter(profile, LAM, method="simpson", n_nodes=n),
            (1001, 4001, 16001, 64001),
            reference,
        )
        errors = [r.rel_error_vs_reference for r in records]
        # Regression bounds from the build run (see validation/VALIDATION.md):
        # SLC-Day  1001 -> 2.8e-3, 64001 -> 3.1e-5
        # SLC-Night 1001 -> 1.3e-2, 64001 -> 7.3e-5
        assert errors[0] < 2e-2
        assert errors[-1] < 2e-4
        assert errors[-1] < errors[0] / 10.0

    def test_quad_uses_the_declared_breakpoints(self):
        # With the discontinuities declared, the adaptive rule integrates each
        # smooth piece exactly; compare against a hand-assembled piecewise sum
        # of the SLC-Day analytic segments.
        profile = slc_day()
        # int_0^18.5 1.7e-14 dh
        seg1 = 1.7e-14 * 18.5
        # int_18.5^110 3.13e-13 h^-1.05 dh = 3.13e-13 [h^-0.05 / -0.05]
        seg2 = 3.13e-13 * (110.0**-0.05 - 18.5**-0.05) / -0.05
        # int_110^1500 1.3e-15 dh
        seg3 = 1.3e-15 * (1500.0 - 110.0)
        # int_1500^7200 8.87e-7 h^-3 dh = 8.87e-7 * (1/2)(1500^-2 - 7200^-2)
        seg4 = 8.87e-7 * 0.5 * (1500.0**-2 - 7200.0**-2)
        # int_7200^20000 2.0e-16 h^-0.5 dh = 2.0e-16 * 2 (20000^0.5 - 7200^0.5)
        seg5 = 2.0e-16 * 2.0 * (20000.0**0.5 - 7200.0**0.5)
        exact = seg1 + seg2 + seg3 + seg4 + seg5
        assert turbulence_moment(profile, 0.0) == pytest.approx(exact, rel=1e-9)


class TestQuadratureRegressionPins:
    """Pinned integral values so a numerical regression is caught immediately."""

    def test_hv57_moments(self):
        profile = hv57()
        assert turbulence_moment(profile, 0.0) == pytest.approx(2.2339844e-12, rel=1e-6)
        assert turbulence_moment(profile, 5 / 3) == pytest.approx(8.4619740e-07, rel=1e-6)
        assert turbulence_moment(profile, 5 / 6) == pytest.approx(5.3957326e-10, rel=1e-6)

    def test_effective_heights(self):
        assert effective_turbulence_height(hv57()) == pytest.approx(2223.49, rel=1e-4)
        assert effective_turbulence_height(slc_day()) == pytest.approx(1258.67, rel=1e-4)
        assert effective_turbulence_height(slc_night()) == pytest.approx(1902.44, rel=1e-4)

    def test_weighted_integral_with_unit_weight_matches_moment_zero(self):
        profile = hv57()
        assert weighted_integral(profile, lambda h: 1.0) == pytest.approx(
            turbulence_moment(profile, 0.0), rel=1e-12
        )


class TestGridConvergenceHelper:
    def test_zero_reference_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            grid_convergence(lambda n: 1.0, (3, 5), 0.0)

    def test_records_carry_node_counts(self):
        records = grid_convergence(lambda n: 1.0 + 1.0 / n, (11, 101), 1.0)
        assert [r.n_nodes for r in records] == [11, 101]
        assert records[-1].rel_error_vs_reference == pytest.approx(1 / 101, rel=1e-12)
