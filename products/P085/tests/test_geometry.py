"""The exact rotation model, and the bound that checks the design matrix.

This is the file that would catch a wrong sign or a wrong trigonometric factor
in :mod:`pointmodel.terms`, because nothing here shares an expression with it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pointmodel import geometry, terms


class TestSkyVectorAndTangents:
    """Equations (7) and (17)."""

    def test_sky_vector_at_the_cardinal_directions(self) -> None:
        # Hand calculation: north at the horizon is [1, 0, 0]; east at the
        # horizon is [0, 1, 0]; the zenith is [0, 0, 1].
        assert geometry.sky_vector(0.0, 0.0) == pytest.approx(
            np.array([1.0, 0.0, 0.0]), rel=0.0, abs=1e-15
        )
        assert geometry.sky_vector(math.pi / 2.0, 0.0) == pytest.approx(
            np.array([0.0, 1.0, 0.0]), rel=0.0, abs=1e-15
        )
        assert geometry.sky_vector(0.0, math.pi / 2.0) == pytest.approx(
            np.array([0.0, 0.0, 1.0]), rel=0.0, abs=1e-15
        )

    def test_tangents_are_orthonormal_and_perpendicular_to_the_vector(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(50):
            azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
            elevation = float(rng.uniform(math.radians(3.0), math.radians(87.0)))
            vector = geometry.sky_vector(azimuth, elevation)
            cross, elev = geometry.tangent_basis(azimuth, elevation)
            assert float(np.linalg.norm(cross)) == pytest.approx(1.0, rel=1e-14, abs=0.0)
            assert float(np.linalg.norm(elev)) == pytest.approx(1.0, rel=1e-14, abs=0.0)
            assert float(cross @ elev) == pytest.approx(0.0, rel=0.0, abs=1e-15)
            assert float(cross @ vector) == pytest.approx(0.0, rel=0.0, abs=1e-15)
            assert float(elev @ vector) == pytest.approx(0.0, rel=0.0, abs=1e-15)


class TestBeamDirection:
    """Equation (12) as an exact composition."""

    def test_zero_coefficients_give_the_nominal_direction(self) -> None:
        azimuth, elevation = 1.3, 0.7
        exact = geometry.beam_direction(azimuth, elevation, np.zeros(terms.N_TERMS))
        assert exact == pytest.approx(
            geometry.sky_vector(azimuth, elevation), rel=0.0, abs=1e-15
        )

    def test_result_is_a_unit_vector(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(30):
            coefficients = rng.uniform(-500.0, 500.0, terms.N_TERMS)
            vector = geometry.beam_direction(
                float(rng.uniform(0.0, 2.0 * math.pi)),
                float(rng.uniform(0.1, 1.5)),
                coefficients,
            )
            assert float(np.linalg.norm(vector)) == pytest.approx(1.0, rel=1e-14, abs=0.0)

    def test_azimuth_index_rotates_the_beam_in_azimuth_exactly(self) -> None:
        # IA is a rotation about the (untilted) vertical by exactly IA, so the
        # exact beam for IA alone is the nominal beam at azimuth A + IA. This
        # is an exact identity in exact arithmetic and holds to rounding here.
        azimuth, elevation = 0.9, 0.6
        index_arcsec = 137.0
        coefficients = terms.coefficients_from_mapping({"IA": index_arcsec})
        exact = geometry.beam_direction(azimuth, elevation, coefficients)
        shifted = geometry.sky_vector(
            azimuth + index_arcsec * geometry.RAD_PER_ARCSEC, elevation
        )
        assert exact == pytest.approx(shifted, rel=0.0, abs=1e-15)

    def test_rejects_wrong_shape_and_non_finite(self) -> None:
        with pytest.raises(ValueError, match="length-9"):
            geometry.beam_direction(0.0, 0.5, np.zeros(3))
        with pytest.raises(ValueError, match="non-finite"):
            geometry.beam_direction(0.0, 0.5, np.full(terms.N_TERMS, np.nan))


class TestLinearisationBound:
    """Equation (15) and equation (16)."""

    def test_bound_is_zero_at_zero_and_increasing(self) -> None:
        assert geometry.linearisation_bound(0.0) == pytest.approx(0.0, rel=0.0, abs=0.0)
        values = [geometry.linearisation_bound(s) for s in (1e-6, 1e-5, 1e-4, 1e-3)]
        assert all(b > a for a, b in zip(values, values[1:], strict=False))

    def test_bound_is_quadratic_for_small_angles(self) -> None:
        # B(s) = exp(s + s^2 e^s / 2) - 1 - s is s^2 + O(s^3), so a factor of
        # ten in s must give a factor of one hundred in B to within O(s).
        small = geometry.linearisation_bound(1e-6)
        larger = geometry.linearisation_bound(1e-5)
        assert larger / small == pytest.approx(100.0, rel=1e-4, abs=0.0)
        assert small == pytest.approx(1e-12, rel=1e-5, abs=0.0)

    def test_rejects_a_negative_angle(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            geometry.linearisation_bound(-1.0)

    @pytest.mark.parametrize("amplitude", [1.0, 10.0, 100.0, 1000.0])
    def test_design_matrix_satisfies_the_bound(self, amplitude: float) -> None:
        # The check the design matrix exists to pass. The bound comes from the
        # operator-norm Taylor remainder of the matrix exponential, which no
        # part of pointmodel.terms uses, and it carries no fitted constant.
        rng = np.random.default_rng(int(amplitude) + 1)
        azimuth = rng.uniform(0.0, 2.0 * math.pi, 120)
        elevation = np.radians(rng.uniform(5.0, 88.0, 120))
        coefficients = rng.uniform(-amplitude, amplitude, terms.N_TERMS)
        error, bound = geometry.linearisation_error(azimuth, elevation, coefficients)
        assert np.all(error <= bound), (
            f"largest error/bound ratio {float(np.max(error / bound)):.6f} at "
            f"amplitude {amplitude}"
        )
        # The bound is tight: it is not satisfied by being vacuous.
        assert float(np.max(error / bound)) > 0.05

    def test_a_corrupted_column_violates_the_bound(self) -> None:
        # Negative control, anchored on the TRUE geometry rather than on any
        # output. Flipping the sign of one design-matrix column must put the
        # first-order reconstruction outside the bound, and it is asserted
        # that the control FIRES.
        rng = np.random.default_rng(77)
        azimuth = rng.uniform(0.0, 2.0 * math.pi, 60)
        elevation = np.radians(rng.uniform(10.0, 80.0, 60))
        coefficients = terms.coefficients_from_mapping({"NPAE": 50.0})
        cross, elev = terms.design_matrix(azimuth, elevation)
        corrupted_cross = cross.copy()
        corrupted_cross[:, terms.TERM_INDEX["NPAE"]] *= -1.0
        violations = 0
        for position in range(azimuth.size):
            exact = geometry.beam_direction(
                float(azimuth[position]), float(elevation[position]), coefficients
            )
            nominal = geometry.sky_vector(
                float(azimuth[position]), float(elevation[position])
            )
            tangent_cross, tangent_elev = geometry.tangent_basis(
                float(azimuth[position]), float(elevation[position])
            )
            linear = (
                nominal
                + float(corrupted_cross[position] @ coefficients)
                * geometry.RAD_PER_ARCSEC
                * tangent_cross
                + float(elev[position] @ coefficients)
                * geometry.RAD_PER_ARCSEC
                * tangent_elev
            )
            bound = geometry.linearisation_bound(
                geometry.total_small_angle(coefficients, float(elevation[position]))
            )
            if float(np.linalg.norm(exact - linear)) > bound:
                violations += 1
        assert violations == azimuth.size, (
            f"the sign-flip control fired on only {violations} of {azimuth.size} "
            "observations; it must fire on all of them"
        )


class TestExactResiduals:
    """The residuals a perfect instrument with these misalignments would show."""

    def test_zero_coefficients_give_zero_residuals(self) -> None:
        azimuth = np.linspace(0.0, 2.0 * math.pi, 12)[:-1]
        elevation = np.full(11, 0.8)
        cross, elev = geometry.exact_residuals(azimuth, elevation, np.zeros(terms.N_TERMS))
        assert cross == pytest.approx(np.zeros(11), rel=0.0, abs=1e-12)
        assert elev == pytest.approx(np.zeros(11), rel=0.0, abs=1e-12)

    def test_agrees_with_the_design_matrix_to_second_order(self) -> None:
        # At 20 arcsec total misalignment the first-order truncation is below
        # 1e-3 arcsec on the sky; at 200 arcsec it is a hundred times larger.
        # The ratio of the two is the quadratic scaling and is the statement
        # being made, not an absolute tolerance.
        rng = np.random.default_rng(3)
        azimuth = rng.uniform(0.0, 2.0 * math.pi, 80)
        elevation = np.radians(rng.uniform(10.0, 80.0, 80))
        cross_block, elev_block = terms.design_matrix(azimuth, elevation)
        worst = {}
        for amplitude in (20.0, 200.0):
            coefficients = np.full(terms.N_TERMS, amplitude / terms.N_TERMS)
            cross, elev = geometry.exact_residuals(azimuth, elevation, coefficients)
            worst[amplitude] = max(
                float(np.max(np.abs(cross - cross_block @ coefficients))),
                float(np.max(np.abs(elev - elev_block @ coefficients))),
            )
        assert worst[20.0] < 1e-3
        assert worst[200.0] / worst[20.0] == pytest.approx(100.0, rel=0.2, abs=0.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="entries"):
            geometry.exact_residuals(
                np.array([0.0, 1.0]), np.array([0.5]), np.zeros(terms.N_TERMS)
            )


class TestTotalSmallAngle:
    """Equation (14)."""

    def test_sum_of_the_five_rotation_magnitudes(self) -> None:
        # Hand calculation at E = 45 degrees with IA = 10, CA = 20, NPAE = 30,
        # AN = 3, AW = 4 and no elevation-side terms: the tilt magnitude is
        # hypot(3, 4) = 5 exactly, so s = (10 + 0 + 30 + 20 + 5) arcsec =
        # 65 arcsec = 65 / 206264.80624709636 radians = 3.1512889e-4 rad.
        coefficients = terms.coefficients_from_mapping(
            {"IA": 10.0, "CA": 20.0, "NPAE": 30.0, "AN": 3.0, "AW": 4.0}
        )
        got = geometry.total_small_angle(coefficients, math.radians(45.0))
        assert got == pytest.approx(65.0 * geometry.RAD_PER_ARCSEC, rel=1e-14, abs=0.0)
        assert got == pytest.approx(3.1512889e-4, rel=1e-6, abs=0.0)

    def test_elevation_side_terms_combine_before_taking_the_modulus(self) -> None:
        # IE, TF, TS and RF enter the composition as ONE rotation, so a
        # cancellation between them reduces s. At E = 45 degrees with
        # IE = +100 and TF = -100/cos(45 deg) = -141.4213562373095 the two
        # cancel exactly and s is zero.
        coefficients = terms.coefficients_from_mapping(
            {"IE": 100.0, "TF": -100.0 / math.cos(math.radians(45.0))}
        )
        got = geometry.total_small_angle(coefficients, math.radians(45.0))
        assert got == pytest.approx(0.0, rel=0.0, abs=1e-18)
