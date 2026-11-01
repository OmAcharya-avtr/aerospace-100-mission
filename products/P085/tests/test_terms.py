"""The design matrix: hand-calculated columns, and every input rejection.

Every ``pytest.approx`` call passes an explicit ``abs=``; see the note at the
top of ``tests/test_refraction.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pointmodel import terms
from pointmodel.refraction import BENNETT_MAX_ALTITUDE_DEG, BENNETT_MIN_ALTITUDE_DEG


class TestDeclaredTermSet:
    """Structure of the term set itself."""

    def test_nine_terms_in_a_fixed_order(self) -> None:
        assert terms.N_TERMS == 9
        assert terms.TERM_NAMES == (
            "IA",
            "IE",
            "CA",
            "AN",
            "AW",
            "NPAE",
            "TF",
            "TS",
            "RF",
        )
        assert len(terms.TERMS) == terms.N_TERMS
        assert {term.unit for term in terms.TERMS} == {"arcsec"}

    def test_index_map_matches_the_order(self) -> None:
        for position, name in enumerate(terms.TERM_NAMES):
            assert terms.TERM_INDEX[name] == position

    def test_table_has_one_row_per_term(self) -> None:
        table = terms.term_table()
        assert len(table) == terms.N_TERMS
        assert all(len(row) == 5 for row in table)


class TestDesignMatrixKnownAnswers:
    """Hand calculations of every column at chosen directions."""

    def test_north_at_thirty_degrees(self) -> None:
        # Hand calculation at A = 0 (due north), E = 30 degrees.
        #   cos E = cos 30 = sqrt(3)/2 = 0.8660254037844387
        #   sin E = 0.5 exactly
        #   sin A = 0, cos A = 1
        # Cross-elevation row, in TERM_NAMES order, from the table in
        # pointmodel.terms:
        #   IA   -> cos E                = 0.8660254037844387
        #   IE   -> 0
        #   CA   -> 1
        #   AN   -> sin A sin E          = 0 * 0.5 = 0
        #   AW   -> -cos A sin E         = -1 * 0.5 = -0.5
        #   NPAE -> sin E                = 0.5
        #   TF, TS, RF -> 0
        # Elevation row:
        #   IA   -> 0
        #   IE   -> 1
        #   CA   -> 0
        #   AN   -> cos A                = 1
        #   AW   -> sin A                = 0
        #   NPAE -> 0
        #   TF   -> cos E                = 0.8660254037844387
        #   TS   -> sin E                = 0.5
        #   RF   -> rho_hat(30 deg)      = 1.726203589934048, the ratio of
        #           tangents hand-calculated in tests/test_refraction.py
        azimuth = np.array([0.0])
        elevation = np.array([math.radians(30.0)])
        cross, elev = terms.design_matrix(azimuth, elevation)
        root_three_over_two = math.sqrt(3.0) / 2.0
        assert root_three_over_two == pytest.approx(0.8660254037844387, rel=1e-15, abs=0.0)
        expected_cross = np.array(
            [root_three_over_two, 0.0, 1.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
        )
        expected_elev = np.array(
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, root_three_over_two, 0.5, 1.726203589934048]
        )
        assert cross[0] == pytest.approx(expected_cross, rel=1e-13, abs=1e-15)
        assert elev[0] == pytest.approx(expected_elev, rel=1e-12, abs=1e-15)

    def test_east_at_sixty_degrees(self) -> None:
        # Hand calculation at A = 90 degrees (due east), E = 60 degrees.
        #   cos E = 0.5 exactly, sin E = sqrt(3)/2 = 0.8660254037844386
        #   sin A = 1, cos A = 0 (to within the rounding of radians(90))
        # Cross-elevation: IA -> 0.5, CA -> 1, AN -> 1 * sqrt(3)/2, AW -> 0,
        # NPAE -> sqrt(3)/2. Elevation: IE -> 1, AN -> 0, AW -> 1, TF -> 0.5,
        # TS -> sqrt(3)/2.
        azimuth = np.array([math.radians(90.0)])
        elevation = np.array([math.radians(60.0)])
        cross, elev = terms.design_matrix(azimuth, elevation)
        half_root_three = math.sqrt(3.0) / 2.0
        assert cross[0, terms.TERM_INDEX["IA"]] == pytest.approx(0.5, rel=1e-13, abs=0.0)
        assert cross[0, terms.TERM_INDEX["CA"]] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert cross[0, terms.TERM_INDEX["AN"]] == pytest.approx(
            half_root_three, rel=1e-13, abs=0.0
        )
        assert cross[0, terms.TERM_INDEX["AW"]] == pytest.approx(0.0, rel=0.0, abs=1e-15)
        assert cross[0, terms.TERM_INDEX["NPAE"]] == pytest.approx(
            half_root_three, rel=1e-13, abs=0.0
        )
        assert elev[0, terms.TERM_INDEX["IE"]] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert elev[0, terms.TERM_INDEX["AN"]] == pytest.approx(0.0, rel=0.0, abs=1e-15)
        assert elev[0, terms.TERM_INDEX["AW"]] == pytest.approx(1.0, rel=1e-15, abs=0.0)
        assert elev[0, terms.TERM_INDEX["TF"]] == pytest.approx(0.5, rel=1e-13, abs=0.0)
        assert elev[0, terms.TERM_INDEX["TS"]] == pytest.approx(
            half_root_three, rel=1e-13, abs=0.0
        )

    def test_tilt_pair_is_one_physical_rotation(self) -> None:
        # The AN/AW pair comes from one tilt of magnitude theta toward azimuth
        # phi, so equations (9) and (10) must reproduce
        #   dE = theta cos(A - phi),  dX = theta sin(A - phi) sin(E).
        # Hand check at theta = 10 arcsec, phi = 40 degrees, A = 115 degrees,
        # E = 50 degrees:  A - phi = 75 degrees, so dE = 10 cos 75 =
        # 2.5881904510252074 arcsec and dX = 10 sin 75 sin 50 =
        # 9.659258262890683 * 0.766044443118978 = 7.3994211169384805 arcsec.
        theta, phi_deg = 10.0, 40.0
        coefficients = terms.coefficients_from_mapping(
            {
                "AN": theta * math.cos(math.radians(phi_deg)),
                "AW": theta * math.sin(math.radians(phi_deg)),
            }
        )
        azimuth = np.array([math.radians(115.0)])
        elevation = np.array([math.radians(50.0)])
        cross, elev = terms.design_matrix(azimuth, elevation)
        expected_elevation = theta * math.cos(math.radians(115.0 - phi_deg))
        expected_cross = (
            theta * math.sin(math.radians(115.0 - phi_deg)) * math.sin(math.radians(50.0))
        )
        assert expected_elevation == pytest.approx(2.5881904510252074, rel=1e-13, abs=0.0)
        assert expected_cross == pytest.approx(7.3994211169384805, rel=1e-13, abs=0.0)
        assert float(elev[0] @ coefficients) == pytest.approx(
            expected_elevation, rel=1e-13, abs=0.0
        )
        assert float(cross[0] @ coefficients) == pytest.approx(
            expected_cross, rel=1e-13, abs=0.0
        )

    def test_shape_and_block_order(self) -> None:
        azimuth = np.linspace(0.0, 2.0 * np.pi, 11)[:-1]
        elevation = np.full(10, math.radians(45.0))
        cross, elev = terms.design_matrix(azimuth, elevation)
        assert cross.shape == (10, terms.N_TERMS)
        assert elev.shape == (10, terms.N_TERMS)
        stacked = terms.stack_residuals(cross, elev)
        assert stacked.shape == (20, terms.N_TERMS)
        assert np.array_equal(stacked[:10], cross)
        assert np.array_equal(stacked[10:], elev)


class TestDesignMatrixEdgeCases:
    """Behaviour at the declared boundaries."""

    def test_at_the_lowest_declared_elevation(self) -> None:
        elevation = np.array([math.radians(BENNETT_MIN_ALTITUDE_DEG)])
        _cross, elev = terms.design_matrix(np.array([0.0]), elevation)
        # At 2 degrees the refraction shape is large: Bennett gives 1058 arcsec
        # there against 59.69 at 45 degrees, a ratio near 17.7.
        assert float(elev[0, terms.TERM_INDEX["RF"]]) > 17.0

    def test_at_the_highest_declared_elevation(self) -> None:
        elevation = np.array([math.radians(BENNETT_MAX_ALTITUDE_DEG)])
        _cross, elev = terms.design_matrix(np.array([0.0]), elevation)
        shape = float(elev[0, terms.TERM_INDEX["RF"]])
        assert 0.0 < shape < 0.01

    def test_single_observation_is_allowed(self) -> None:
        cross, elev = terms.design_matrix(np.array([1.0]), np.array([1.0]))
        assert cross.shape == (1, terms.N_TERMS)
        assert elev.shape == (1, terms.N_TERMS)

    def test_columns_are_finite_across_the_whole_declared_range(self) -> None:
        elevation = np.radians(
            np.linspace(BENNETT_MIN_ALTITUDE_DEG, BENNETT_MAX_ALTITUDE_DEG, 500)
        )
        azimuth = np.linspace(0.0, 2.0 * np.pi, 500)
        cross, elev = terms.design_matrix(azimuth, elevation)
        assert np.all(np.isfinite(cross))
        assert np.all(np.isfinite(elev))


class TestInputValidation:
    """Every rejection path, with the message that names the problem."""

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="entries"):
            terms.design_matrix(np.array([0.0, 1.0]), np.array([1.0]))

    def test_rejects_two_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            terms.design_matrix(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            terms.design_matrix(np.array([]), np.array([]))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            terms.design_matrix(np.array([np.inf]), np.array([1.0]))

    def test_rejects_elevation_below_the_declared_range(self) -> None:
        with pytest.raises(ValueError, match="declared validity range"):
            terms.design_matrix(np.array([0.0]), np.array([math.radians(1.0)]))

    def test_rejects_elevation_above_the_declared_range(self) -> None:
        with pytest.raises(ValueError, match="declared validity range"):
            terms.design_matrix(np.array([0.0]), np.array([math.radians(89.95)]))

    def test_rejects_mismatched_blocks_in_stack(self) -> None:
        with pytest.raises(ValueError, match="different row counts"):
            terms.stack_residuals(np.zeros((3, 9)), np.zeros((4, 9)))

    def test_rejects_unknown_term_name(self) -> None:
        with pytest.raises(ValueError, match="unknown pointing-model term"):
            terms.coefficients_from_mapping({"IA": 1.0, "NOPE": 2.0})

    def test_rejects_non_finite_coefficient(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            terms.coefficients_from_mapping({"IA": float("nan")})

    def test_rejects_wrong_length_coefficient_vector(self) -> None:
        with pytest.raises(ValueError, match="length-9"):
            terms.mapping_from_coefficients(np.zeros(5))


class TestMappingRoundTrip:
    """Name-to-value conversion in both directions."""

    def test_round_trip(self) -> None:
        mapping = {name: float(index + 1) for index, name in enumerate(terms.TERM_NAMES)}
        vector = terms.coefficients_from_mapping(mapping)
        assert terms.mapping_from_coefficients(vector) == mapping

    def test_missing_terms_are_zero(self) -> None:
        vector = terms.coefficients_from_mapping({"RF": 58.0})
        assert float(vector[terms.TERM_INDEX["RF"]]) == pytest.approx(
            58.0, rel=0.0, abs=0.0
        )
        assert float(np.sum(np.abs(vector))) == pytest.approx(58.0, rel=0.0, abs=0.0)
