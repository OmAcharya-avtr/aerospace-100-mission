"""Pinned regressions and the numerical benchmark.

Two kinds of test live here.

**Pinned counterexamples.** ``.hypothesis/`` is gitignored, so a falsifying
example found by one run is lost on a fresh clone. Every counterexample
Hypothesis found during this build is pinned below with its literal inputs.

**Benchmark and regression.** Numbers this package produces on fixed inputs,
pinned so that a change in behaviour is caught by the suite rather than by a
reader of ``validation/``. The tolerances are stated and the reason for each
is given; none of them is bit-exact, because none of these quantities is
exactly reproducible across a change of BLAS or of numpy's SVD driver.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

import pointmodel
from pointmodel import geometry, identifiability, refraction, synthetic, terms, wls


class TestPinnedCounterexamples:
    """Falsifying examples found by Hypothesis during this build."""

    def test_equivariance_tolerance_must_follow_the_conditioning(self) -> None:
        """An absolute 1e-12 tolerance on the equivariance identity is wrong.

        Found by Hypothesis on 2026-11-01 at 2000 examples, on the identity
        "adding ``A delta`` to the observations adds ``delta`` to the
        estimate". The falsifying example is

            coefficients = [0, 0, 0, 0, 0, 0, 0, 0, 60.0]   (RF alone)
            offset       = 19.0

        for which the largest coefficient discrepancy is 1.2363443602225743e-12
        and an absolute bound of 1e-12 fails. The identity is exact in exact
        arithmetic; the discrepancy is the conditioning of the solve acting on
        the rounding of the inputs, and the honest bound is therefore
        ``1e-12 * condition_number * scale``, which is the form used in
        ``tests/test_properties.py``. This is the defect class recorded in the
        build guide: a slack floored at an absolute constant does not follow
        the conditioning of the expression it bounds.
        """
        coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0])
        offset = 19.0
        azimuth = np.linspace(0.0, 2.0 * np.pi, 41)[:-1]
        elevation = np.radians(np.linspace(15.0, 80.0, 40))
        design = terms.stack_residuals(*terms.design_matrix(azimuth, elevation))
        observed = design @ coefficients
        delta = np.full(terms.N_TERMS, offset)
        base = wls.fit(design, observed, 1.0, error_model="declared")
        shifted = wls.fit(design, observed + design @ delta, 1.0, error_model="declared")
        error = float(np.max(np.abs(shifted.coefficients - (base.coefficients + delta))))
        # The naive absolute bound fails on this input. Asserting that it
        # fails is what keeps the lesson in the suite.
        assert error > 1e-12
        # The conditioning-following bound holds, with margin.
        scale = max(1.0, float(np.max(np.abs(coefficients))) + abs(offset))
        assert error <= 1e-12 * base.condition_number * scale


class TestPinnedBuildDefects:
    """Defects found during this build, with the behaviour that fixes them."""

    def test_variance_inflation_uses_the_full_spectrum(
        self, single_azimuth: synthetic.ResidualSet
    ) -> None:
        """A VIF computed from the truncated pseudo-inverse hides the degeneracy.

        The first implementation computed the variance inflation factors from
        the same truncated inverse the estimator uses, which sets the
        directions below the rank cut to zero. On the ``single_az`` geometry
        that reported a largest VIF of 1304, which reads as a mildly awkward
        design, while the geometry is exactly rank deficient by two and the
        design supports no estimate of those directions at all. The factors
        now use the full spectrum and the same case reports above 1e20. The
        standard errors still come from the truncated inverse, and the two
        disagreeing is the signal.
        """
        result = wls.fit(single_azimuth.design, single_azimuth.observed, 2.0)
        assert float(np.max(result.vif)) > 1e20
        assert float(np.max(result.standard_errors)) < 1e3
        assert result.rank == 7

    def test_bennett_is_refused_before_its_sign_change(self) -> None:
        """Equation (1) returns a negative refraction above 89.9225 degrees.

        Caught while writing the design matrix: the refraction column would
        have changed sign near the zenith and a fit would have attributed a
        positive refraction to a negative coefficient. The formula is refused
        above 89.9 degrees, and the design matrix refuses the same elevation.
        """
        with pytest.raises(ValueError, match="refused above"):
            refraction.bennett_refraction(89.93)
        with pytest.raises(ValueError, match="declared validity range"):
            terms.design_matrix(np.array([0.0]), np.array([math.radians(89.93)]))
        # And just below the limit it is still positive and small.
        assert float(refraction.bennett_refraction(89.89)) > 0.0
        assert float(refraction.bennett_refraction(89.89)) < 0.2


class TestPinnedVerdicts:
    """The rank verdicts, pinned at a fixed geometry size and seed."""

    SIZE = 300
    SEED = 20260101
    SIGMA = 2.0

    def _report(self, name: str) -> identifiability.IdentifiabilityReport:
        residuals = synthetic.generate(
            name, self.SIZE, sigma_arcsec=self.SIGMA, seed=self.SEED
        )
        result = wls.fit(residuals.design, residuals.observed, self.SIGMA)
        return identifiability.analyse_fit(residuals.design, result)

    @pytest.mark.parametrize(
        ("name", "expected_rank", "condition_low", "condition_high"),
        [
            # Condition numbers are pinned to a factor of two either side.
            # They depend on the random draw of the geometry at the pinned
            # seed and on the SVD driver, neither of which is exactly
            # reproducible across platforms, and a factor of two is far
            # inside the four orders of magnitude that separate these cases.
            ("full_sky", 9, 64.0, 258.0),
            ("two_az", 9, 64.0, 258.0),
            ("low_el", 8, 1586.0, 6344.0),
            ("narrow_el", 6, 3.2e5, 1.3e6),
            ("single_az", 7, 1.0e14, 1.0e17),
        ],
    )
    def test_rank_and_condition_number(
        self, name: str, expected_rank: int, condition_low: float, condition_high: float
    ) -> None:
        report = self._report(name)
        assert report.numerical_rank == expected_rank
        assert condition_low <= report.condition_number <= condition_high

    @pytest.mark.parametrize(
        ("name", "refused"),
        [
            ("full_sky", set()),
            ("two_az", set()),
            ("low_el", {"IE", "TF", "TS"}),
            ("narrow_el", {"IA", "CA", "NPAE", "IE", "TF", "TS", "RF"}),
            ("single_az", {"AN", "AW", "NPAE", "IE"}),
        ],
    )
    def test_refused_terms(self, name: str, refused: set[str]) -> None:
        report = self._report(name)
        assert set(report.unidentifiable_terms) == refused


class TestPinnedRefractionNumbers:
    """The cross-check figures, pinned."""

    GRID = np.linspace(10.0, 85.0, 16)

    def test_radio_dry_comparison(self) -> None:
        result = refraction.compare_refraction(
            self.GRID, refractivity_constant=refraction.K_RADIO_DRY
        )
        # Pinned to three significant figures. The quadrature's own reported
        # absolute error is below 1e-8 arcsec, so the only instability here is
        # the adaptive subdivision of scipy.integrate.quad, which is why the
        # bound is 1e-3 relative and not tighter.
        assert result.max_absolute_arcsec == pytest.approx(8.474, rel=1e-3, abs=0.0)
        assert result.mean_ratio == pytest.approx(1.04352, rel=1e-3, abs=0.0)
        assert result.max_shape_relative == pytest.approx(1.96409e-2, rel=1e-3, abs=0.0)

    def test_optical_comparison_is_within_the_published_accuracy(self) -> None:
        result = refraction.compare_refraction(
            self.GRID, refractivity_constant=refraction.K_OPTICAL_550NM
        )
        # 4.127 arcsec against the 4.2 arcsec that Meeus quotes for equation
        # (1). It passes with 1.7 per cent of margin, which is thin, and the
        # margin is stated rather than left for a reader to work out.
        assert result.max_absolute_arcsec == pytest.approx(4.12697, rel=1e-3, abs=0.0)
        assert result.max_absolute_arcsec < 0.07 * 60.0
        assert result.mean_ratio == pytest.approx(1.02553, rel=1e-3, abs=0.0)

    def test_bennett_at_the_grid_ends(self) -> None:
        # Hand-checkable: 60 / tan(radians(10 + 7.31/14.4)) and
        # 60 / tan(radians(85 + 7.31/89.4)).
        low = 60.0 / math.tan(math.radians(10.0 + 7.31 / 14.4))
        high = 60.0 / math.tan(math.radians(85.0 + 7.31 / 89.4))
        assert low == pytest.approx(323.4903, rel=1e-6, abs=0.0)
        assert high == pytest.approx(5.1630, rel=1e-4, abs=0.0)
        assert float(refraction.bennett_refraction(10.0)) == pytest.approx(
            low, rel=1e-13, abs=0.0
        )
        assert float(refraction.bennett_refraction(85.0)) == pytest.approx(
            high, rel=1e-13, abs=0.0
        )


class TestPinnedLinearisation:
    """The measured tightness of equation (16), pinned."""

    @pytest.mark.parametrize("amplitude", [1.0, 10.0, 100.0, 1000.0])
    def test_worst_ratio_is_stable_across_four_decades(self, amplitude: float) -> None:
        # The bound is derived, not fitted, so the ratio of the observed error
        # to it is a property of the model and not of a tolerance. It sits
        # between 0.1 and 1.0 across four decades of misalignment amplitude,
        # which is what says the bound is neither violated nor vacuous.
        rng = np.random.default_rng(2026)
        azimuth = rng.uniform(0.0, 2.0 * np.pi, 200)
        elevation = np.radians(rng.uniform(5.0, 88.0, 200))
        coefficients = rng.uniform(-amplitude, amplitude, terms.N_TERMS)
        error, bound = geometry.linearisation_error(azimuth, elevation, coefficients)
        ratio = float(np.max(error / bound))
        assert 0.1 < ratio < 1.0


class TestBenchmark:
    """Wall-clock budget. Timings are unstable and the bound is generous."""

    def test_full_decomposition_of_three_hundred_observations(self) -> None:
        # The compute budget for this product is two cores and nothing here is
        # a Monte Carlo, so the whole pipeline on a 300-observation run --
        # design matrix, SVD fit, rank analysis, Huber IRLS and 200 RANSAC
        # trials -- is expected in well under a second. The bound asserted is
        # 30 seconds, which is a regression guard against an accidental
        # quadratic, not a performance claim.
        residuals = synthetic.generate("full_sky", 300, sigma_arcsec=2.0, seed=1)
        started = time.perf_counter()
        budget = pointmodel.decompose(residuals.design, residuals.observed, sigma_arcsec=2.0)
        elapsed = time.perf_counter() - started
        assert budget.least_squares.rank == terms.N_TERMS
        assert elapsed < 30.0

    def test_residual_set_generation_scales_linearly(self) -> None:
        # The exact rotation model loops in Python, one composition per
        # observation, so generation is linear in the observation count. A
        # four-fold increase must not cost more than twelve times as long;
        # the factor of three of headroom covers the fixed import and setup
        # cost, which dominates at the small end.
        def timed(count: int) -> float:
            started = time.perf_counter()
            synthetic.generate("full_sky", count, sigma_arcsec=1.0, seed=2)
            return time.perf_counter() - started

        timed(50)  # warm the import path, discard
        small = timed(100)
        large = timed(400)
        assert large < 12.0 * small
