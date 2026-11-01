"""Property-based tests for the algebraic identities this package rests on.

Hypothesis is used only where there is a real identity to state. Where a
counterexample was found it is pinned in ``tests/test_regressions.py`` with
literal inputs, because ``.hypothesis/`` is gitignored and a falsifying example
found by one run is lost on a fresh clone.

Deadlines are disabled: these tests build design matrices and take singular
value decompositions, and the first example of a run pays the import cost of
scipy, which makes a per-example deadline meaningless rather than informative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pointmodel import geometry, identifiability, refraction, terms, wls

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

ELEVATION_DEG = st.floats(min_value=3.0, max_value=88.0, allow_nan=False, allow_infinity=False)
AZIMUTH_DEG = st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False)
COEFFICIENT = st.floats(
    min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False
)


@st.composite
def coefficient_vectors(draw: st.DrawFn) -> np.ndarray:
    """A length-9 coefficient vector in arcseconds."""
    return np.array([draw(COEFFICIENT) for _ in range(terms.N_TERMS)])


class TestDesignMatrixIdentities:
    """Identities the design matrix satisfies exactly in exact arithmetic."""

    @SETTINGS
    @given(azimuth=AZIMUTH_DEG, elevation=ELEVATION_DEG)
    def test_azimuth_is_periodic(self, azimuth: float, elevation: float) -> None:
        # Adding a full turn to the azimuth cannot change any column. In exact
        # arithmetic this is an identity; in floating point radians(a) and
        # radians(a + 360) differ in the last places, so the comparison is to
        # a tolerance scaled by the magnitude of the angle.
        first = terms.design_matrix(
            np.array([math.radians(azimuth)]), np.array([math.radians(elevation)])
        )
        second = terms.design_matrix(
            np.array([math.radians(azimuth + 360.0)]), np.array([math.radians(elevation)])
        )
        for block_a, block_b in zip(first, second, strict=True):
            assert block_a == pytest.approx(block_b, rel=0.0, abs=1e-12)

    @SETTINGS
    @given(
        azimuth=AZIMUTH_DEG,
        elevation=ELEVATION_DEG,
        scale=st.floats(min_value=-10.0, max_value=10.0),
        coefficients=coefficient_vectors(),
    )
    def test_the_model_is_linear_in_the_coefficients(
        self, azimuth: float, elevation: float, scale: float, coefficients: np.ndarray
    ) -> None:
        cross, elev = terms.design_matrix(
            np.array([math.radians(azimuth)]), np.array([math.radians(elevation)])
        )
        for block in (cross, elev):
            scaled = float(block[0] @ (scale * coefficients))
            direct = scale * float(block[0] @ coefficients)
            assert scaled == pytest.approx(direct, rel=1e-12, abs=1e-12)

    @SETTINGS
    @given(azimuth=AZIMUTH_DEG, elevation=ELEVATION_DEG, tilt=COEFFICIENT)
    def test_a_pure_tilt_is_a_single_rotation(
        self, azimuth: float, elevation: float, tilt: float
    ) -> None:
        # The AN/AW pair comes from one physical rotation of magnitude theta,
        # so the quantity AN^2 + AW^2 is the squared tilt whatever the azimuth
        # of the lean, and equations (9) and (10) must reproduce
        # theta cos(A - phi) and theta sin(A - phi) sin(E) for every phi.
        phi = math.radians(azimuth * 0.37)
        coefficients = terms.coefficients_from_mapping(
            {"AN": tilt * math.cos(phi), "AW": tilt * math.sin(phi)}
        )
        cross, elev = terms.design_matrix(
            np.array([math.radians(azimuth)]), np.array([math.radians(elevation)])
        )
        delta = math.radians(azimuth) - phi
        expected_elevation = tilt * math.cos(delta)
        expected_cross = tilt * math.sin(delta) * math.sin(math.radians(elevation))
        assert float(elev[0] @ coefficients) == pytest.approx(
            expected_elevation, rel=1e-11, abs=1e-11
        )
        assert float(cross[0] @ coefficients) == pytest.approx(
            expected_cross, rel=1e-11, abs=1e-11
        )


class TestLinearisationBoundHolds:
    """Equation (16) must hold for every coefficient vector and direction."""

    @SETTINGS
    @given(
        azimuth=AZIMUTH_DEG,
        elevation=st.floats(min_value=3.0, max_value=88.0),
        coefficients=coefficient_vectors(),
    )
    def test_bound_is_never_violated(
        self, azimuth: float, elevation: float, coefficients: np.ndarray
    ) -> None:
        error, bound = geometry.linearisation_error(
            np.array([math.radians(azimuth)]),
            np.array([math.radians(elevation)]),
            coefficients,
        )
        assert float(error[0]) <= float(bound[0])


class TestFitIdentities:
    """Identities of the weighted least squares solution."""

    @SETTINGS
    @given(
        coefficients=coefficient_vectors(),
        offset=st.floats(min_value=-50.0, max_value=50.0),
    )
    def test_fit_is_equivariant_under_adding_a_model_signal(
        self, coefficients: np.ndarray, offset: float
    ) -> None:
        # Adding A*delta to the observations must add delta to the estimate
        # exactly in exact arithmetic. The comparison is to a tolerance that
        # follows the conditioning: the error is bounded by the condition
        # number times the rounding of the inputs.
        azimuth, elevation = np.linspace(0.0, 2.0 * np.pi, 41)[:-1], np.radians(
            np.linspace(15.0, 80.0, 40)
        )
        cross, elev = terms.design_matrix(azimuth, elevation)
        design = terms.stack_residuals(cross, elev)
        observed = design @ coefficients
        delta = np.full(terms.N_TERMS, offset)
        base = wls.fit(design, observed, 1.0, error_model="declared")
        shifted = wls.fit(design, observed + design @ delta, 1.0, error_model="declared")
        scale = max(1.0, float(np.max(np.abs(coefficients))) + abs(offset))
        tolerance = 1e-12 * base.condition_number * scale
        assert shifted.coefficients == pytest.approx(
            base.coefficients + delta, rel=0.0, abs=tolerance
        )

    @SETTINGS
    @given(factor=st.floats(min_value=0.05, max_value=20.0))
    def test_uniform_reweighting_does_not_move_the_estimate(self, factor: float) -> None:
        azimuth, elevation = np.linspace(0.0, 2.0 * np.pi, 41)[:-1], np.radians(
            np.linspace(15.0, 80.0, 40)
        )
        cross, elev = terms.design_matrix(azimuth, elevation)
        design = terms.stack_residuals(cross, elev)
        rng = np.random.default_rng(3)
        observed = design @ np.full(terms.N_TERMS, 10.0) + rng.normal(0.0, 2.0, 80)
        one = wls.fit(design, observed, 1.0, error_model="declared")
        other = wls.fit(design, observed, factor, error_model="declared")
        assert other.coefficients == pytest.approx(one.coefficients, rel=1e-8, abs=1e-8)
        assert other.standard_errors == pytest.approx(
            factor * one.standard_errors, rel=1e-10, abs=0.0
        )


class TestIdentifiabilityIdentities:
    """Identities of the rank analysis."""

    @SETTINGS
    @given(
        low=st.floats(min_value=10.0, max_value=50.0),
        span=st.floats(min_value=1.0, max_value=30.0),
    )
    def test_column_scaling_does_not_change_the_singular_value_ratios(
        self, low: float, span: float
    ) -> None:
        # The analysis normalises the columns, so rescaling any column must
        # leave every ratio s_k / s_1 unchanged. This is the property that
        # makes the condition number a statement about geometry and not units.
        high = min(low + span, 85.0)
        azimuth = np.linspace(0.0, 2.0 * np.pi, 61)[:-1]
        elevation = np.radians(np.linspace(low, high, 60))
        cross, elev = terms.design_matrix(azimuth, elevation)
        design = terms.stack_residuals(cross, elev)
        base = identifiability.analyse(design, np.zeros(terms.N_TERMS))
        rescaled_design = design.copy()
        rescaled_design[:, terms.TERM_INDEX["RF"]] *= 137.0
        rescaled = identifiability.analyse(rescaled_design, np.zeros(terms.N_TERMS))
        assert rescaled.ratios == pytest.approx(base.ratios, rel=1e-6, abs=1e-12)

    @SETTINGS
    @given(
        low=st.floats(min_value=12.0, max_value=40.0),
        span=st.floats(min_value=2.0, max_value=40.0),
    )
    def test_widening_the_elevation_range_never_worsens_the_condition_number(
        self, low: float, span: float
    ) -> None:
        # Not an exact monotonicity -- the grid changes too -- so the claim is
        # a factor of two, which is far inside the four to twelve orders of
        # magnitude the restricted geometries differ by.
        high = min(low + span, 85.0)
        azimuth = np.linspace(0.0, 2.0 * np.pi, 81)[:-1]
        narrow = np.radians(np.linspace(low, min(low + 0.25 * (high - low), 85.0), 80))
        wide = np.radians(np.linspace(low, high, 80))
        narrow_design = terms.stack_residuals(*terms.design_matrix(azimuth, narrow))
        wide_design = terms.stack_residuals(*terms.design_matrix(azimuth, wide))
        narrow_report = identifiability.analyse(narrow_design, np.zeros(terms.N_TERMS))
        wide_report = identifiability.analyse(wide_design, np.zeros(terms.N_TERMS))
        assert wide_report.condition_number <= 2.0 * narrow_report.condition_number


class TestRefractionIdentities:
    """Identities of the refraction model."""

    @SETTINGS
    @given(
        elevation=st.floats(min_value=2.0, max_value=89.8),
        pressure=st.floats(min_value=300.0, max_value=1100.0),
        temperature=st.floats(min_value=-50.0, max_value=50.0),
    )
    def test_conditions_factor_is_exactly_multiplicative(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        standard = float(refraction.bennett_refraction(elevation, 1010.0, 10.0))
        scaled = float(refraction.bennett_refraction(elevation, pressure, temperature))
        factor = (pressure / 1010.0) * (283.0 / (273.0 + temperature))
        assert scaled == pytest.approx(factor * standard, rel=1e-12, abs=0.0)

    @SETTINGS
    @given(elevation=st.floats(min_value=2.0, max_value=89.8))
    def test_the_shape_is_the_refraction_divided_by_its_value_at_forty_five(
        self, elevation: float
    ) -> None:
        shape = float(
            refraction.refraction_shape(np.array([math.radians(elevation)]))[0]
        )
        direct = float(refraction.bennett_refraction(elevation)) / float(
            refraction.bennett_refraction(45.0)
        )
        assert shape == pytest.approx(direct, rel=1e-13, abs=0.0)
