"""Profile-model, wind-model, summary and CLI tests."""

import io
import json
import math

import numpy as np
import pytest

from atmoprofile import (
    STANDARD_PROFILES,
    Cn2Profile,
    bufton_wind,
    coherence_length_to_seeing,
    constant_profile,
    fried_parameter,
    hufnagel_valley,
    hv57,
    rms_upper_wind,
    slc_day,
    slc_night,
    standard_profile,
    summarize,
    tabulated_profile,
)
from atmoprofile.__main__ import main

LAM = 500e-9


class TestHufnagelValley:
    def test_hv57_parameters(self):
        profile = hv57()
        assert profile.meta["v_rms_ms"] == 21.0
        assert profile.meta["ground_a"] == 1.7e-14

    def test_ground_value_is_dominated_by_the_A_term(self):
        # At h = 0 the HV model reduces to 0 + 2.7e-16 + A = 2.7e-16 + 1.7e-14
        profile = hv57()
        assert float(profile(0.0)) == pytest.approx(1.7e-14 + 2.7e-16, rel=1e-12)

    def test_tropopause_bump_present(self):
        # The (1e-5 h)^10 exp(-h/1000) term peaks near h = 10 km; Cn^2 there must
        # exceed the value one decade lower in altitude.
        profile = hv57()
        assert float(profile(10_000.0)) > float(profile(5_000.0))

    def test_stronger_wind_increases_high_altitude_turbulence(self):
        weak = hufnagel_valley(10.0)
        strong = hufnagel_valley(40.0)
        assert float(strong(10_000.0)) > float(weak(10_000.0))
        # ... and r0 must shrink
        assert fried_parameter(strong, LAM) < fried_parameter(weak, LAM)

    def test_ground_term_dominates_r0(self):
        # Removing the surface term (A = 0) must increase r0 substantially.
        no_ground = hufnagel_valley(21.0, 0.0)
        assert fried_parameter(no_ground, LAM) > fried_parameter(hv57(), LAM)


class TestSLCModels:
    @pytest.mark.parametrize("factory", [slc_day, slc_night])
    def test_positive_everywhere(self, factory):
        profile = factory()
        h = np.linspace(0.0, 20_000.0, 5001)
        values = np.asarray(profile(h))
        assert np.all(values > 0.0)
        assert np.all(np.isfinite(values))

    def test_slc_day_branch_values(self):
        # Hand evaluation of each analytic branch.
        profile = slc_day()
        assert float(profile(10.0)) == pytest.approx(1.7e-14, rel=1e-12)
        assert float(profile(50.0)) == pytest.approx(3.13e-13 / 50.0**1.05, rel=1e-12)
        assert float(profile(1000.0)) == pytest.approx(1.3e-15, rel=1e-12)
        assert float(profile(3000.0)) == pytest.approx(8.87e-7 / 3000.0**3, rel=1e-12)
        assert float(profile(10_000.0)) == pytest.approx(2.0e-16 / 10_000.0**0.5, rel=1e-12)

    def test_slc_night_branch_values(self):
        profile = slc_night()
        assert float(profile(10.0)) == pytest.approx(8.4e-15, rel=1e-12)
        assert float(profile(50.0)) == pytest.approx(2.87e-12 / 50.0**2, rel=1e-12)
        assert float(profile(500.0)) == pytest.approx(2.5e-16, rel=1e-12)
        assert float(profile(3000.0)) == pytest.approx(8.87e-7 / 3000.0**3, rel=1e-12)
        assert float(profile(10_000.0)) == pytest.approx(2.0e-16 / 10_000.0**0.5, rel=1e-12)

    def test_breakpoints_declared(self):
        assert slc_day().breakpoints == (18.5, 110.0, 1500.0, 7200.0)
        assert slc_night().breakpoints == (18.5, 110.0, 850.0, 7000.0)

    def test_night_is_weaker_than_day_near_the_ground(self):
        assert float(slc_night()(5.0)) < float(slc_day()(5.0))


class TestTabulatedProfile:
    def test_interpolation_is_log_linear(self):
        profile = tabulated_profile([0.0, 1000.0], [1e-14, 1e-16])
        # geometric mean at the midpoint: sqrt(1e-14 * 1e-16) = 1e-15
        assert float(profile(500.0)) == pytest.approx(1e-15, rel=1e-12)

    def test_round_trip_against_an_analytic_profile(self):
        # Sample HV5/7 densely, rebuild it as a table, and check r0 agrees.
        source = hv57()
        h = np.unique(np.concatenate([np.linspace(0.0, 2000.0, 400),
                                      np.geomspace(2100.0, 20000.0, 400)]))
        table = tabulated_profile(h, np.asarray(source(h)), name="HV5/7 sampled")
        # 800 log-linear knots reproduce the analytic profile's r0 to 4.5e-6
        assert fried_parameter(table, LAM) == pytest.approx(fried_parameter(source, LAM), rel=1e-4)

    def test_metadata_preserved(self):
        profile = tabulated_profile(
            [0.0, 100.0], [1e-14, 1e-15], name="site A", reference="campaign 2026"
        )
        assert isinstance(profile, Cn2Profile)
        assert "site A" in profile.describe()
        assert "campaign 2026" in profile.describe()


class TestWindModels:
    def test_bufton_ground_and_peak(self):
        wind = bufton_wind(5.0)
        # v(0) = 5 + 30 exp(-(9400/4800)^2) = 5 + 30 exp(-3.8351) = 5 + 0.6472
        assert float(wind(0.0)) == pytest.approx(5.0 + 30.0 * math.exp(-((9400 / 4800) ** 2)),
                                                 rel=1e-12)
        # peak at 9400 m is exactly v_g + 30
        assert float(wind(9400.0)) == pytest.approx(35.0, rel=1e-12)

    def test_rms_upper_wind_value(self):
        # Computed in this build: 22.9637 m/s for v_g = 5 m/s over 5-20 km.
        # NOTE: the literature associates the Bufton wind with the HV pseudowind
        # 21 m/s; the 9.3 % discrepancy is documented in validation/VALIDATION.md
        # and is deliberately NOT tuned away.
        assert rms_upper_wind(bufton_wind(5.0)) == pytest.approx(22.9637, rel=1e-4)

    def test_rms_upper_wind_of_uniform_wind_is_that_wind(self):
        from atmoprofile import constant_wind

        assert rms_upper_wind(constant_wind(17.0)) == pytest.approx(17.0, rel=1e-9)

    def test_rms_band_must_lie_inside_support(self):
        with pytest.raises(ValueError, match="outside the support"):
            rms_upper_wind(bufton_wind(5.0), 5_000.0, 50_000.0)


class TestSeeing:
    def test_seeing_from_r0(self):
        # epsilon = 0.98 * 500e-9 / 0.10 m = 4.9e-6 rad = 1.0106 arcsec
        eps = coherence_length_to_seeing(0.10, LAM)
        assert eps == pytest.approx(4.9e-6, rel=1e-12)
        assert math.degrees(eps) * 3600.0 == pytest.approx(1.01069, rel=1e-4)

    def test_negative_r0_rejected(self):
        with pytest.raises(ValueError, match="r0_m must be > 0"):
            coherence_length_to_seeing(-1.0, LAM)


class TestSummary:
    def test_summary_fields(self):
        summary = summarize(hv57(), LAM, wind=bufton_wind(5.0))
        assert summary.profile == "HV5/7"
        assert summary.r0_m == pytest.approx(0.0496245, rel=1e-4)
        assert summary.theta0_urad == pytest.approx(7.010862, rel=1e-4)
        assert summary.f_greenwood_hz == pytest.approx(71.8705, rel=1e-3)
        assert summary.weak_fluctuation_valid is True
        assert set(summary.as_dict()) == set(summary.__dataclass_fields__)

    def test_summary_without_wind_has_no_greenwood_frequency(self):
        summary = summarize(hv57(), LAM)
        assert summary.f_greenwood_hz is None

    def test_summary_is_quiet_in_the_strong_regime(self):
        # summarize() suppresses the strong-fluctuation warning but still reports
        # the regime through weak_fluctuation_valid.
        import warnings

        strong = constant_profile(1e-13, 0.0, 20_000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = summarize(strong, LAM)
        assert summary.weak_fluctuation_valid is False


class TestRegistry:
    def test_all_standard_profiles_build(self):
        for key in STANDARD_PROFILES:
            profile = standard_profile(key)
            assert isinstance(profile, Cn2Profile)
            assert profile.reference
            assert profile.validity

    def test_lookup_is_case_insensitive(self):
        assert standard_profile("HV57").name == "HV5/7"


class TestCLI:
    def test_summary_table(self):
        out = io.StringIO()
        code = main(["summary", "--profile", "hv57", "--zenith-deg", "0", "45"], stream=out)
        text = out.getvalue()
        assert code == 0
        assert "HV5/7" in text
        assert "r0[cm]" in text
        assert text.count("\n") > 5

    def test_summary_json(self):
        out = io.StringIO()
        code = main(["summary", "--profile", "slc_night", "--json"], stream=out)
        payload = json.loads(out.getvalue())
        assert code == 0
        assert payload[0]["profile"] == "SLC-Night"
        assert payload[0]["r0_m"] == pytest.approx(0.0760229, rel=1e-3)

    def test_profile_dump(self):
        out = io.StringIO()
        code = main(["profile", "--profile", "slc_day", "--n", "5"], stream=out)
        text = out.getvalue()
        assert code == 0
        assert "SLC-Day" in text
        assert "Cn^2 [m^-2/3]" in text

    def test_unknown_profile_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            main(["summary", "--profile", "not-a-model"], stream=io.StringIO())
