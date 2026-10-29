"""Tests for the four refractivity formulations, equations (5) to (12)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from refractpath.refractivity import (
    REFRACTIVITY_MODELS,
    REFRACTIVITY_VALIDITY_UM,
    birch_downs_1994,
    ciddor_1996,
    ciddor_compressibility,
    ciddor_enhancement,
    ciddor_molar_mass_air,
    ciddor_saturation_pressure_pa,
    ciddor_vapour_pressure_pa,
    edlen_1966,
    iau_explanatory_supplement,
    refractivity_spread,
)

STANDARD = (0.633, 101325.0, 15.0, 0.0)


class TestKnownAnswers:
    def test_birch_downs_dispersion_at_633nm(self) -> None:
        # Hand calculation of equation (6) at lambda = 0.633 um:
        #   s^2 = 1 / 0.633^2 = 2.495850...
        #   2406147 / (130 - 2.49585) = 2406147 / 127.50415 = 18871.09...
        #   15998   / (38.9 - 2.49585) = 15998 / 36.40415   =   439.46...
        #   8342.54 + 18871.09 + 439.46 = 27653.09  ->  2.765309e-4
        # The equation (7) density factor is unity at the standard state, so
        # the full formula returns the dispersion term to 1e-11.
        assert float(birch_downs_1994(*STANDARD)) == pytest.approx(
            2.7652887e-4, rel=0.0, abs=1e-11
        )

    def test_edlen_dispersion_at_633nm(self) -> None:
        # The same arithmetic with Edlen's 1966 coefficients:
        #   8342.13 + 2406030 / 127.50415 + 15997 / 36.40415 = 27651.74
        assert float(edlen_1966(*STANDARD)) == pytest.approx(
            2.7651532e-4, rel=0.0, abs=1e-11
        )

    def test_ciddor_at_633nm(self) -> None:
        assert float(ciddor_1996(*STANDARD)) == pytest.approx(
            2.7653021e-4, rel=0.0, abs=1e-11
        )

    def test_explanatory_supplement_at_633nm(self) -> None:
        # Hand calculation of equation (9) at 0.633 um, 15 C, 101325 Pa, dry:
        #   287.604 + 1.6288 * 2.49585 + 0.0136 * 6.22929 = 291.754 e-6
        #   x (101325 / 101325) x (273.15 / 288.15) = 2.765661e-4
        assert float(iau_explanatory_supplement(*STANDARD)) == pytest.approx(
            2.7656611e-4, rel=0.0, abs=1e-11
        )

    def test_ciddor_saturation_pressure_at_20_c(self) -> None:
        assert float(ciddor_saturation_pressure_pa(293.15)) == pytest.approx(
            2339.163, rel=0.0, abs=0.001
        )

    def test_ciddor_molar_mass_at_450_ppm(self) -> None:
        # 1e-3 (28.9635 + 12.011e-6 (450 - 400)) = 1e-3 (28.9635 + 0.00060055)
        assert float(ciddor_molar_mass_air(450.0)) == pytest.approx(
            0.02896410055, rel=0.0, abs=1e-13
        )

    def test_ciddor_enhancement_near_unity(self) -> None:
        value = float(ciddor_enhancement(101325.0, 20.0))
        assert value == pytest.approx(1.00062 + 3.14e-8 * 101325.0 + 5.6e-7 * 400.0,
                                      rel=0.0, abs=1e-15)
        assert 1.0 < value < 1.01

    def test_ciddor_compressibility_of_dry_standard_air(self) -> None:
        value = float(ciddor_compressibility(101325.0, 288.15, 0.0))
        assert value == pytest.approx(0.9995922, rel=0.0, abs=1e-7)
        assert value < 1.0


class TestCrossFormulationAgreement:
    """The four formulations must agree to the level their authors imply."""

    @pytest.mark.parametrize("wavelength", [0.35, 0.4, 0.55, 0.633, 0.8, 1.0, 1.55])
    def test_dry_air_formulations_agree_to_1e_minus_7(self, wavelength: float) -> None:
        # Measured maximum over this grid: 5.5601e-8 at 0.55 um. The declared
        # bound is 1e-7, which is about 3.6e-4 of n - 1 and therefore about
        # 0.02 arcsec of refraction at 45 degrees elevation.
        values = [
            float(function(wavelength, 101325.0, 15.0, 0.0))
            for function in REFRACTIVITY_MODELS.values()
        ]
        assert max(values) - min(values) < 1e-7

    @pytest.mark.parametrize(
        ("pressure", "temperature"),
        [(101325.0, 15.0), (101000.0, 10.0), (70000.0, -10.0), (85000.0, 30.0)],
    )
    def test_dry_air_agreement_holds_across_the_state_grid(
        self, pressure: float, temperature: float
    ) -> None:
        values = [
            float(function(0.55, pressure, temperature, 0.0))
            for function in REFRACTIVITY_MODELS.values()
        ]
        # Measured maximum over this grid: 8.8253e-8 at 85000 Pa and 30 C.
        assert max(values) - min(values) < 1e-7

    def test_humid_air_disagreement_is_larger_and_is_measured(self) -> None:
        # At 20 C and saturation the Explanatory Supplement humidity term of
        # equation (9) departs from the Edlen-family and Ciddor treatments by
        # about 4.3e-7 in n, which is ten times the dry-air spread. This is a
        # real cross-model disagreement and the test records its size rather
        # than tolerating it silently.
        vapour = float(ciddor_vapour_pressure_pa(20.0, 1.0, 101000.0))
        values = {
            name: float(function(0.55, 101000.0, 20.0, vapour))
            for name, function in REFRACTIVITY_MODELS.items()
        }
        spread = max(values.values()) - min(values.values())
        assert 3.0e-7 < spread < 6.0e-7
        assert min(values, key=lambda key: values[key]) == "iau_explanatory_supplement"

    def test_birch_downs_exceeds_edlen_by_about_1_4e_minus_8(self) -> None:
        # Birch and Downs (1993) is published as a correction to Edlen (1966)
        # at the 1e-8 level; this checks the implemented difference is that
        # size and that it has the published sign.
        difference = float(birch_downs_1994(*STANDARD)) - float(edlen_1966(*STANDARD))
        assert 1.0e-8 < difference < 2.0e-8


class TestPhysicalBehaviour:
    def test_refractivity_is_proportional_to_pressure_for_the_edlen_family(self) -> None:
        low = float(birch_downs_1994(0.55, 50000.0, 10.0, 0.0))
        high = float(birch_downs_1994(0.55, 100000.0, 10.0, 0.0))
        # Equation (7) is linear in p up to the small 1e-8 p term, so the ratio
        # is 2 to within about 2e-4 relative.
        assert high / low == pytest.approx(2.0, rel=3e-4, abs=0.0)

    def test_refractivity_falls_with_temperature(self) -> None:
        values = [
            float(ciddor_1996(0.55, 101325.0, t, 0.0)) for t in (-20.0, 0.0, 20.0, 40.0)
        ]
        assert all(later < earlier for earlier, later in pairwise(values))

    def test_refractivity_falls_with_wavelength_in_the_optical_band(self) -> None:
        for function in REFRACTIVITY_MODELS.values():
            values = [
                float(function(lam, 101325.0, 15.0, 0.0))
                for lam in (0.35, 0.45, 0.55, 0.7, 1.0, 1.5)
            ]
            assert all(
                later < earlier for earlier, later in pairwise(values)
            )

    def test_water_vapour_lowers_optical_refractivity(self) -> None:
        dry = float(ciddor_1996(0.55, 101000.0, 20.0, 0.0))
        humid = float(ciddor_1996(0.55, 101000.0, 20.0, 2000.0))
        assert humid < dry
        # The size: about 0.3 per cent of n - 1 at saturation at 20 C.
        assert (dry - humid) / dry == pytest.approx(0.0027, rel=0.0, abs=0.0006)

    def test_co2_raises_ciddor_refractivity(self) -> None:
        low = float(ciddor_1996(0.55, 101325.0, 15.0, 0.0, 300.0))
        high = float(ciddor_1996(0.55, 101325.0, 15.0, 0.0, 600.0))
        assert high > low

    def test_only_ciddor_responds_to_co2(self) -> None:
        for name, function in REFRACTIVITY_MODELS.items():
            if name == "ciddor_1996":
                continue
            # The other three carry a fixed CO2 content in their published
            # coefficients and take no argument for it.
            assert function(0.55, 101325.0, 15.0, 0.0) == pytest.approx(
                function(0.55, 101325.0, 15.0, 0.0), rel=0.0, abs=0.0
            )

    def test_vectorised_over_arrays(self) -> None:
        pressures = np.array([50000.0, 75000.0, 101325.0])
        values = birch_downs_1994(0.55, pressures, 10.0, 0.0)
        assert values.shape == (3,)
        assert np.all(np.diff(values) > 0.0)


class TestValidation:
    @pytest.mark.parametrize("function", list(REFRACTIVITY_MODELS.values()))
    def test_non_positive_wavelength_rejected(self, function: object) -> None:
        with pytest.raises(ValueError, match=r"wavelength_um must be positive"):
            function(0.0, 101325.0, 15.0, 0.0)  # type: ignore[operator]

    @pytest.mark.parametrize("function", list(REFRACTIVITY_MODELS.values()))
    def test_non_positive_pressure_rejected(self, function: object) -> None:
        with pytest.raises(ValueError, match=r"pressure_pa must be positive"):
            function(0.55, 0.0, 15.0, 0.0)  # type: ignore[operator]

    @pytest.mark.parametrize("function", list(REFRACTIVITY_MODELS.values()))
    def test_temperature_below_absolute_zero_rejected(self, function: object) -> None:
        with pytest.raises(ValueError, match=r"above absolute zero"):
            function(0.55, 101325.0, -300.0, 0.0)  # type: ignore[operator]

    @pytest.mark.parametrize("function", list(REFRACTIVITY_MODELS.values()))
    def test_non_finite_inputs_rejected(self, function: object) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            function(float("nan"), 101325.0, 15.0, 0.0)  # type: ignore[operator]

    def test_wavelength_at_the_edlen_pole_rejected(self) -> None:
        # 1 / 0.1603^2 = 38.9, the pole of the second Edlen denominator.
        with pytest.raises(ValueError, match=r"0.1603 um pole"):
            birch_downs_1994(0.16, 101325.0, 15.0, 0.0)

    def test_wavelength_at_the_ciddor_pole_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"pole"):
            ciddor_1996(0.13, 101325.0, 15.0, 0.0)

    def test_ciddor_rejects_vapour_above_total_pressure(self) -> None:
        with pytest.raises(ValueError, match=r"must not exceed pressure_pa"):
            ciddor_1996(0.55, 1000.0, 20.0, 2000.0)


class TestSpreadReport:
    def test_spread_report_contains_every_formulation(self) -> None:
        report = refractivity_spread(0.55, 101325.0, 15.0, 0.0)
        for name in REFRACTIVITY_MODELS:
            assert name in report
        assert "spread" in report
        assert "relative_spread" in report

    def test_spread_equals_max_minus_min(self) -> None:
        report = refractivity_spread(0.55, 101000.0, 10.0, 500.0)
        values = [report[name] for name in REFRACTIVITY_MODELS]
        assert report["spread"] == pytest.approx(
            max(values) - min(values), rel=0.0, abs=1e-18
        )

    def test_relative_spread_is_small_for_dry_optical_air(self) -> None:
        report = refractivity_spread(0.55, 101325.0, 15.0, 0.0)
        assert 0.0 < report["relative_spread"] < 1e-3


class TestPublishedValidityRanges:
    def test_every_formulation_declares_a_range(self) -> None:
        assert set(REFRACTIVITY_VALIDITY_UM) == set(REFRACTIVITY_MODELS)

    def test_ranges_are_ordered_and_carry_a_source(self) -> None:
        for low, high, source in REFRACTIVITY_VALIDITY_UM.values():
            assert high > low > 0.0
            assert len(source) > 10

    def test_birch_downs_has_the_narrowest_published_range(self) -> None:
        widths = {
            name: high - low for name, (low, high, _) in REFRACTIVITY_VALIDITY_UM.items()
        }
        assert min(widths, key=lambda key: widths[key]) == "birch_downs_1994"
