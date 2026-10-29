"""Tests for the external published references and the checks against them.

One of these checks is expected to FAIL and the test asserts that it fails.
Bennett's published maximum error of 0.07 arcminutes was stated against
Garfinkel (1967), which this package does not implement; measured against this
package's own ray trace it is exceeded below about 19 degrees of apparent
elevation. The honest handling is to measure it, publish the miss, and refuse
to widen the tolerance, so the test pins the failure as the expected result
and would itself fail if someone made the check pass by loosening it.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from refractpath.reference_checks import (
    BENNETT_STATE,
    REFERENCE_CHECKS,
    STANDARD_STATE,
    bennett_departure_arcsec,
    check_bennett_accuracy_claim,
    check_constant_of_refraction,
    check_density_standard_air,
    check_horizon_refraction,
    check_refractivity_633nm,
    check_saturation_vapour_pressure,
    constant_of_refraction_arcsec,
    run_reference_checks,
)
from refractpath.references import (
    BENNETT_ACCURACY_CLAIM,
    PUBLISHED_BY_KEY,
    PUBLISHED_REFERENCES,
    PublishedReference,
    compare,
)


class TestReferenceCatalogue:
    def test_six_references_in_three_kinds(self) -> None:
        assert len(PUBLISHED_REFERENCES) == 6
        kinds = {reference.kind for reference in PUBLISHED_REFERENCES}
        assert kinds == {"published_value", "published_claim", "convention"}

    def test_every_reference_declares_its_provenance(self) -> None:
        for reference in PUBLISHED_REFERENCES:
            assert len(reference.citation) > 20
            assert len(reference.conditions) > 10
            assert len(reference.tolerance_source) > 20
            assert len(reference.note) > 40
            assert reference.tolerance >= 0.0

    def test_lookup_by_key_is_complete(self) -> None:
        assert set(PUBLISHED_BY_KEY) == {
            reference.key for reference in PUBLISHED_REFERENCES
        }

    def test_only_the_claim_carries_a_zero_tolerance(self) -> None:
        zero = [r.key for r in PUBLISHED_REFERENCES if r.tolerance == 0.0]
        assert zero == ["bennett_accuracy_claim"]

    def test_the_convention_is_labelled_as_one(self) -> None:
        assert PUBLISHED_BY_KEY["horizon_refraction_convention"].kind == "convention"
        assert "rounded convention" in PUBLISHED_BY_KEY["horizon_refraction_convention"].note


class TestCompare:
    def test_two_sided_comparison_inside_tolerance(self) -> None:
        reference = PUBLISHED_BY_KEY["saturation_vapour_pressure_20c"]
        assert compare(reference, 2338.0).within_tolerance is True
        assert compare(reference, 2340.0).within_tolerance is True

    def test_two_sided_comparison_outside_tolerance(self) -> None:
        reference = PUBLISHED_BY_KEY["saturation_vapour_pressure_20c"]
        assert compare(reference, 2330.0).within_tolerance is False

    def test_claim_comparison_is_one_sided(self) -> None:
        # A measured maximum error below the published bound passes; above it
        # fails. There is no symmetric tolerance here.
        assert compare(BENNETT_ACCURACY_CLAIM, 1.0).within_tolerance is True
        assert compare(BENNETT_ACCURACY_CLAIM, 4.2).within_tolerance is True
        assert compare(BENNETT_ACCURACY_CLAIM, 4.3).within_tolerance is False

    def test_non_finite_measurement_never_passes(self) -> None:
        reference = PUBLISHED_BY_KEY["saturation_vapour_pressure_20c"]
        assert compare(reference, float("nan")).within_tolerance is False

    def test_difference_is_measured_minus_published(self) -> None:
        reference = PUBLISHED_BY_KEY["saturation_vapour_pressure_20c"]
        assert compare(reference, 2341.0).difference == pytest.approx(
            2.0, rel=0.0, abs=1e-12
        )

    def test_detail_is_carried_through(self) -> None:
        reference = PUBLISHED_BY_KEY["saturation_vapour_pressure_20c"]
        assert compare(reference, 2339.0, "note").detail == "note"

    def test_reference_is_frozen(self) -> None:
        reference = PublishedReference(
            key="k",
            quantity="q",
            value=1.0,
            tolerance=0.1,
            tolerance_source="s",
            conditions="c",
            citation="cit",
            kind="published_value",
            note="n",
        )
        with pytest.raises(Exception, match=r"cannot assign|frozen"):
            reference.value = 2.0  # type: ignore[misc]


class TestThePassingChecks:
    def test_refractivity_at_633nm_passes(self) -> None:
        comparison = check_refractivity_633nm()
        assert comparison.within_tolerance is True
        assert abs(comparison.difference) < 5e-9
        assert "dispersion term" in comparison.detail

    def test_density_of_standard_air_passes_and_explains_the_departure(self) -> None:
        comparison = check_density_standard_air()
        assert comparison.within_tolerance is True
        assert comparison.measured > comparison.reference.value
        assert "compressibility correction" in comparison.detail

    def test_saturation_vapour_pressure_passes_for_both_formulations(self) -> None:
        comparison = check_saturation_vapour_pressure()
        assert comparison.within_tolerance is True
        assert "both are inside the tolerance of this check" in comparison.detail
        # The enhancement-factor value is reported but deliberately NOT compared
        # against a reference published for pure water.
        assert "NOT" in comparison.detail

    def test_the_enhancement_factor_value_is_outside_the_pure_water_tolerance(self) -> None:
        # Measured: 2348.5798 Pa, which is +9.4165 Pa above the bare IAPWS
        # saturation value of 2339.1632 Pa.
        # This is a real and correct difference -- the enhancement factor
        # accounts for the presence of air -- and comparing it against a
        # pure-water figure would be comparing two different quantities.
        from refractpath.refractivity import ciddor_vapour_pressure_pa

        in_air = float(ciddor_vapour_pressure_pa(20.0, 1.0, 101325.0))
        assert in_air == pytest.approx(2348.58, rel=0.0, abs=0.01)
        assert abs(in_air - 2339.0) > 2.0

    def test_constant_of_refraction_passes_and_reports_its_wavelength_sensitivity(
        self,
    ) -> None:
        comparison = check_constant_of_refraction()
        assert comparison.within_tolerance is True
        assert abs(comparison.difference) < 0.01
        assert "times the tolerance" in comparison.detail

    def test_horizon_convention_passes_but_says_so_weakly(self) -> None:
        comparison = check_horizon_refraction()
        assert comparison.within_tolerance is True
        assert "says very little" in comparison.detail


class TestTheFailingCheck:
    def test_bennett_claim_fails_against_this_packages_ray_trace(self) -> None:
        comparison = check_bennett_accuracy_claim()
        assert comparison.within_tolerance is False
        assert comparison.measured > BENNETT_ACCURACY_CLAIM.value

    def test_the_measured_maximum_departure_is_about_34_arcsec(self) -> None:
        comparison = check_bennett_accuracy_claim()
        assert comparison.measured == pytest.approx(34.07, rel=0.0, abs=0.2)

    def test_the_worst_departure_is_at_the_horizon(self) -> None:
        grid, departure = bennett_departure_arcsec()
        assert grid[int(np.argmax(departure))] == pytest.approx(0.0, rel=0.0, abs=1e-9)

    def test_the_claim_holds_above_about_nineteen_degrees(self) -> None:
        grid, departure = bennett_departure_arcsec()
        inside = departure <= BENNETT_ACCURACY_CLAIM.value
        last_violation = int(np.flatnonzero(~inside)[-1])
        assert grid[last_violation + 1] == pytest.approx(19.1, rel=0.0, abs=0.5)

    def test_the_claim_holds_everywhere_above_thirty_degrees(self) -> None:
        grid, departure = bennett_departure_arcsec()
        high = grid >= 30.0
        assert np.all(departure[high] <= BENNETT_ACCURACY_CLAIM.value)

    def test_the_note_states_what_the_miss_does_not_prove(self) -> None:
        assert "does not show Bennett wrong" in BENNETT_ACCURACY_CLAIM.note
        assert "Garfinkel" in BENNETT_ACCURACY_CLAIM.note

    def test_the_wavelength_choice_moves_the_measured_departure(self) -> None:
        comparison = check_bennett_accuracy_claim()
        assert "633 nm" in comparison.detail


class TestConstantOfRefraction:
    def test_known_answer_at_the_published_conditions(self) -> None:
        # kappa (1 - beta) at 10 C, 1013.25 hPa, dry, 550 nm, with
        # kappa = 2.8275930e-4 and beta = 1.299516e-3.
        assert constant_of_refraction_arcsec(STANDARD_STATE) == pytest.approx(
            58.2475, rel=0.0, abs=0.001
        )

    def test_it_is_below_the_bare_refractivity(self) -> None:
        from refractpath.models import observer_refractivity

        kappa_arcsec = observer_refractivity(STANDARD_STATE) * 180.0 * 3600.0 / math.pi
        assert constant_of_refraction_arcsec(STANDARD_STATE) < kappa_arcsec

    def test_it_falls_with_wavelength(self) -> None:
        from refractpath.atmosphere import AtmosphericState

        values = [
            constant_of_refraction_arcsec(
                AtmosphericState(pressure_hpa=1013.25, temperature_c=10.0, wavelength_um=lam)
            )
            for lam in (0.40, 0.55, 0.70, 1.00)
        ]
        assert all(later < earlier for earlier, later in pairwise(values))


class TestRunAll:
    def test_one_comparison_per_reference_in_order(self) -> None:
        comparisons = run_reference_checks()
        assert len(comparisons) == len(PUBLISHED_REFERENCES) == len(REFERENCE_CHECKS)
        assert [c.reference.key for c in comparisons] == [
            r.key for r in PUBLISHED_REFERENCES
        ]

    def test_exactly_one_failure_and_it_is_the_bennett_claim(self) -> None:
        failures = [c for c in run_reference_checks() if not c.within_tolerance]
        assert len(failures) == 1
        assert failures[0].reference.key == "bennett_accuracy_claim"

    def test_nothing_raises_on_the_failure(self) -> None:
        # The miss is a result, not an exception. If this ever starts raising,
        # the failing reference will stop being reported.
        assert run_reference_checks() is not None

    def test_the_published_states_are_the_documented_ones(self) -> None:
        assert BENNETT_STATE.pressure_hpa == pytest.approx(1010.0, rel=0.0, abs=0.0)
        assert BENNETT_STATE.temperature_c == pytest.approx(10.0, rel=0.0, abs=0.0)
        assert STANDARD_STATE.pressure_hpa == pytest.approx(1013.25, rel=0.0, abs=0.0)
        assert STANDARD_STATE.wavelength_um == pytest.approx(0.55, rel=0.0, abs=0.0)
