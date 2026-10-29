"""Tests for the validity verdict: binding rule 17, a computed and returned output."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from refractpath import DEFAULT_TOLERANCE_ARCSEC, RefractionResult, refraction
from refractpath.atmosphere import AtmosphericState, Observer
from refractpath.disagreement import default_ensemble
from refractpath.verdict import (
    PROFILE_ENVELOPE_LAPSE_RATES,
    UNDETERMINABLE,
    ValidityVerdict,
    compute_verdict,
    profile_envelope_arcsec,
)

PUBLISHED = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.55)


class TestProfileEnvelope:
    def test_three_declared_lapse_rates(self) -> None:
        assert len(PROFILE_ENVELOPE_LAPSE_RATES) == 3
        labels = [label for label, _ in PROFILE_ENVELOPE_LAPSE_RATES]
        assert labels == ["dry_adiabatic", "us_standard_1976", "isothermal"]
        rates = [rate for _, rate in PROFILE_ENVELOPE_LAPSE_RATES]
        assert rates == [-0.0098, -0.0065, 0.0]

    def test_envelope_is_negligible_at_high_elevation(self) -> None:
        width, members = profile_envelope_arcsec(45.0, PUBLISHED)
        assert len(members) == 3
        assert width < 0.01

    def test_envelope_dominates_at_the_horizon(self) -> None:
        # Measured: the three declared surface lapse rates move the horizon
        # refraction by about 300 arcsec, which is five arcminutes and more
        # than two orders of magnitude larger than the cross-model spread at
        # the same elevation.
        width, _ = profile_envelope_arcsec(0.0, PUBLISHED)
        assert 200.0 < width < 400.0

    def test_isothermal_gives_the_largest_horizon_refraction(self) -> None:
        _, members = profile_envelope_arcsec(0.0, PUBLISHED)
        assert members["isothermal"] > members["us_standard_1976"]
        assert members["us_standard_1976"] > members["dry_adiabatic"]

    def test_envelope_grows_monotonically_toward_the_horizon(self) -> None:
        widths = [
            profile_envelope_arcsec(h, PUBLISHED)[0] for h in (45.0, 20.0, 10.0, 5.0, 1.0, 0.0)
        ]
        assert all(later > earlier for earlier, later in pairwise(widths))

    def test_the_callers_lapse_rate_is_overridden_not_honoured(self) -> None:
        odd = AtmosphericState(
            pressure_hpa=1010.0, temperature_c=10.0, surface_lapse_rate_k_per_m=0.005
        )
        assert profile_envelope_arcsec(10.0, odd)[0] == pytest.approx(
            profile_envelope_arcsec(10.0, PUBLISHED)[0], rel=1e-9, abs=0.0
        )


class TestVerdictLabels:
    def test_high_elevation_is_usable(self) -> None:
        verdict = compute_verdict(80.0, PUBLISHED, 10.0)
        assert verdict.verdict == "usable"
        assert verdict.usable is True
        assert verdict.refused is False
        assert verdict.reasons == ()

    def test_moderate_elevation_is_degraded_by_a_model_out_of_range(self) -> None:
        # At 25 degrees the ensemble agrees to 5.7 arcsec, inside a 10 arcsec
        # tolerance, but plane_parallel is below its declared 30 degree floor.
        verdict = compute_verdict(25.0, PUBLISHED, 10.0)
        assert verdict.verdict == "degraded"
        assert any("plane_parallel" in reason for reason in verdict.reasons)

    def test_low_elevation_is_refused(self) -> None:
        verdict = compute_verdict(5.0, PUBLISHED, 10.0)
        assert verdict.verdict == "refuse"
        assert verdict.refused is True
        assert verdict.above_validity_floor is False

    def test_the_horizon_is_refused_even_at_a_wide_tolerance(self) -> None:
        verdict = compute_verdict(0.0, PUBLISHED, 60.0)
        assert verdict.verdict == "refuse"
        assert any("unmeasured temperature profile" in reason for reason in verdict.reasons)

    def test_refusal_names_the_widest_pair(self) -> None:
        verdict = compute_verdict(5.0, PUBLISHED, 10.0)
        joined = " ".join(verdict.reasons)
        assert "oriani_series" in joined or "plane_parallel" in joined

    def test_wavelength_outside_the_declared_domain_degrades_but_does_not_refuse(self) -> None:
        state = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=1.9)
        verdict = compute_verdict(80.0, state, 10.0)
        assert verdict.verdict == "degraded"
        assert verdict.state_inside_declared_domains["wavelength_um"] is False
        assert any("declared domain" in reason for reason in verdict.reasons)

    def test_wavelength_outside_a_formulation_range_is_named(self) -> None:
        state = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=1.2)
        verdict = compute_verdict(80.0, state, 10.0)
        assert verdict.refractivity_inside_published_wavelength_range[
            "iau_explanatory_supplement"
        ] is False
        assert verdict.refractivity_inside_published_wavelength_range["ciddor_1996"] is True
        assert any("outside the published range" in reason for reason in verdict.reasons)

    def test_pressure_outside_the_declared_domain_is_named(self) -> None:
        state = AtmosphericState(pressure_hpa=250.0, temperature_c=-50.0, wavelength_um=0.55)
        verdict = compute_verdict(80.0, state, 10.0)
        assert verdict.state_inside_declared_domains["pressure_hpa"] is False

    def test_undefined_methods_are_reported_as_a_reason(self) -> None:
        verdict = compute_verdict(0.0, PUBLISHED, 1.0e9)
        assert set(verdict.models_undefined) == {"plane_parallel", "oriani_series"}
        assert any("undefined at this elevation" in reason for reason in verdict.reasons)

    def test_half_tolerance_envelope_degrades(self) -> None:
        # Isolate the half-tolerance envelope gate by running a single-member
        # ensemble, so the cross-model spread is zero by construction and the
        # envelope is the only thing that can fire. The tolerance is set to
        # 1.5 times the measured envelope, which puts the envelope between
        # half the tolerance and the whole of it.
        single = tuple(
            member for member in default_ensemble() if member.name == "raytrace"
        )
        width, _ = profile_envelope_arcsec(10.0, PUBLISHED)
        verdict = compute_verdict(10.0, PUBLISHED, width * 1.5, None, single)
        assert verdict.disagreement_arcsec == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert verdict.verdict == "degraded"
        assert any("half the declared tolerance" in reason for reason in verdict.reasons)

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan")])
    def test_bad_tolerance_rejected(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match=r"tolerance_arcsec must be positive and finite"):
            compute_verdict(45.0, PUBLISHED, tolerance)


class TestVerdictContent:
    def test_every_field_is_populated(self) -> None:
        verdict = compute_verdict(45.0, PUBLISHED, 10.0)
        assert isinstance(verdict, ValidityVerdict)
        assert verdict.apparent_elevation_deg == pytest.approx(45.0, rel=0.0, abs=0.0)
        assert verdict.tolerance_arcsec == pytest.approx(10.0, rel=0.0, abs=0.0)
        assert verdict.disagreement_arcsec > 0.0
        assert math.isfinite(verdict.profile_envelope_arcsec)
        assert math.isfinite(verdict.validity_floor_deg)
        assert len(verdict.models_inside_published_validity) == 5
        assert verdict.plane_parallel_analytic_floor_deg > 0.0
        assert verdict.oriani_breakdown_elevation_deg > 0.0
        assert 0.0 < verdict.refractivity_relative_spread < 1e-2

    def test_both_computed_floors_are_closed_form_not_declared(self) -> None:
        verdict = compute_verdict(45.0, PUBLISHED, 10.0)
        assert verdict.plane_parallel_analytic_floor_deg == pytest.approx(
            1.3601824, rel=0.0, abs=1e-6
        )
        assert verdict.oriani_breakdown_elevation_deg == pytest.approx(
            1.9507071, rel=0.0, abs=1e-6
        )

    def test_undeterminable_list_is_structural_and_non_empty(self) -> None:
        verdict = compute_verdict(45.0, PUBLISHED, 10.0)
        assert verdict.undeterminable == UNDETERMINABLE
        assert len(UNDETERMINABLE) == 5
        for item in UNDETERMINABLE:
            assert len(item) > 40

    def test_undeterminable_names_the_profile_first(self) -> None:
        assert "temperature profile" in UNDETERMINABLE[0]

    def test_verdict_is_frozen(self) -> None:
        verdict = compute_verdict(45.0, PUBLISHED, 10.0)
        with pytest.raises(Exception, match=r"cannot assign|frozen"):
            verdict.verdict = "usable"  # type: ignore[misc]


class TestRefractionEntryPoint:
    def test_the_verdict_is_the_first_field(self) -> None:
        fields = list(RefractionResult.__dataclass_fields__)
        assert fields[0] == "verdict"

    def test_result_carries_the_ensemble_and_the_reference(self) -> None:
        result = refraction(45.0, PUBLISHED, tolerance_arcsec=10.0)
        assert result.reference_method == "raytrace"
        assert set(result.methods_arcsec) == {member.name for member in default_ensemble()}
        assert result.reference_refraction_arcsec == pytest.approx(
            result.methods_arcsec["raytrace"], rel=0.0, abs=1e-12
        )

    def test_true_elevation_is_the_apparent_minus_the_reference(self) -> None:
        result = refraction(20.0, PUBLISHED, tolerance_arcsec=10.0)
        assert result.true_elevation_deg == pytest.approx(
            20.0 - result.reference_refraction_arcsec / 3600.0, rel=0.0, abs=1e-12
        )
        assert result.true_elevation_deg < 20.0

    def test_default_state_and_observer(self) -> None:
        result = refraction(45.0)
        assert result.state == AtmosphericState()
        assert result.observer == Observer()
        assert result.verdict.tolerance_arcsec == pytest.approx(
            DEFAULT_TOLERANCE_ARCSEC, rel=0.0, abs=0.0
        )

    def test_widest_pair_matches_the_spread(self) -> None:
        result = refraction(20.0, PUBLISHED, tolerance_arcsec=10.0)
        low, high = result.widest_pair
        assert result.disagreement_arcsec == pytest.approx(
            result.methods_arcsec[high] - result.methods_arcsec[low], rel=0.0, abs=1e-12
        )

    def test_refractivity_report_is_attached(self) -> None:
        result = refraction(45.0, PUBLISHED, tolerance_arcsec=10.0)
        assert "ciddor_1996" in result.refractivity_at_observer
        assert "spread" in result.refractivity_at_observer

    def test_usable_shorthand_matches_the_verdict(self) -> None:
        assert refraction(80.0, PUBLISHED, tolerance_arcsec=10.0).usable is True
        assert refraction(2.0, PUBLISHED, tolerance_arcsec=10.0).usable is False

    def test_unknown_refractivity_model_rejected(self) -> None:
        with pytest.raises(KeyError, match=r"choose from"):
            refraction(45.0, PUBLISHED, refractivity_model="nope")

    def test_elevation_outside_the_range_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[-5, 90\]"):
            refraction(95.0, PUBLISHED)

    def test_a_humid_state_changes_the_reference_but_not_the_two_fits(self) -> None:
        dry = refraction(45.0, PUBLISHED, tolerance_arcsec=10.0)
        humid_state = AtmosphericState(
            pressure_hpa=1010.0,
            temperature_c=10.0,
            relative_humidity=1.0,
            wavelength_um=0.55,
        )
        humid = refraction(45.0, humid_state, tolerance_arcsec=10.0)
        assert humid.reference_refraction_arcsec < dry.reference_refraction_arcsec
        assert humid.methods_arcsec["bennett_1982"] == pytest.approx(
            dry.methods_arcsec["bennett_1982"], rel=0.0, abs=0.0
        )


class TestControlsThatMustFire:
    """Controls anchored on the truth, each asserted to fire."""

    def test_control_horizon_at_one_arcsec_refuses(self) -> None:
        assert compute_verdict(0.0, PUBLISHED, 1.0).verdict == "refuse"

    def test_control_two_degrees_at_one_arcsec_refuses(self) -> None:
        assert compute_verdict(2.0, PUBLISHED, 1.0).verdict == "refuse"

    def test_control_one_degree_at_ten_arcsec_refuses(self) -> None:
        assert compute_verdict(1.0, PUBLISHED, 10.0).verdict == "refuse"

    def test_control_profile_envelope_alone_can_refuse(self) -> None:
        # At 3 degrees the envelope is large enough to refuse on its own even
        # when the ensemble spread is inside the tolerance. The control checks
        # that the envelope gate is load-bearing and not decorative.
        width, _ = profile_envelope_arcsec(3.0, PUBLISHED)
        tolerance = width * 0.5
        verdict = compute_verdict(3.0, PUBLISHED, tolerance)
        assert verdict.verdict == "refuse"
        assert any("unmeasured temperature profile" in reason for reason in verdict.reasons)

    def test_control_plane_parallel_is_undefined_below_its_analytic_floor(self) -> None:
        verdict = compute_verdict(1.0, PUBLISHED, 10.0)
        assert "plane_parallel" in verdict.models_undefined

    def test_control_oriani_is_undefined_below_its_computed_breakdown(self) -> None:
        verdict = compute_verdict(1.0, PUBLISHED, 10.0)
        assert "oriani_series" in verdict.models_undefined


class TestControlsThatMustNotFire:
    """Controls that must stay quiet, asserted in the other direction."""

    def test_control_eighty_degrees_at_ten_arcsec_is_usable(self) -> None:
        assert compute_verdict(80.0, PUBLISHED, 10.0).verdict == "usable"

    def test_control_forty_five_degrees_at_ten_arcsec_is_usable(self) -> None:
        verdict = compute_verdict(45.0, PUBLISHED, 10.0)
        assert verdict.verdict == "usable"
        assert verdict.reasons == ()

    def test_control_the_zenith_at_one_arcsec_is_usable(self) -> None:
        assert compute_verdict(90.0, PUBLISHED, 1.0).verdict == "usable"

    def test_control_a_physical_profile_variation_does_not_refuse_at_high_elevation(
        self,
    ) -> None:
        inverted = AtmosphericState(
            pressure_hpa=1010.0,
            temperature_c=10.0,
            wavelength_um=0.55,
            surface_lapse_rate_k_per_m=0.02,
            surface_layer_top_m=200.0,
        )
        assert compute_verdict(80.0, inverted, 10.0).verdict == "usable"
