"""Case-level known answers, with the hand arithmetic written out.

Every expected value below was computed by hand from the YAML before the
code was run on it. Comparisons carry a two-sided tolerance.
"""

from __future__ import annotations

import math

import pytest

from budgetroll import (
    linear_rollup_uncertainty,
    rollup,
    rss_rollup_uncertainty,
    run_checks,
    severity_counts,
)

# --- clean_mass.yaml ----------------------------------------------------


def test_clean_mass_rollup(clean_mass):
    # nominal    2.00 + 1.50 + 0.80 + 1.20                      = 5.500 kg
    # allowance  5 % of each                                     = 0.275 kg
    # predicted                                                  = 5.775 kg
    # 1 sigma    sqrt(0.0100 + 0.0064 + 0.0025 + 0.0036)
    #            = sqrt(0.0225)                                  = 0.150 kg
    # margin     10.000 - 5.775                                  = 4.225 kg
    # margin@1s  10.000 - 5.925                                  = 4.075 kg
    root = rollup(clean_mass).root
    assert root.nominal == pytest.approx(5.500, rel=0, abs=1e-12)
    assert root.allowance == pytest.approx(0.275, rel=0, abs=1e-12)
    assert root.predicted == pytest.approx(5.775, rel=0, abs=1e-12)
    assert root.uncertainty == pytest.approx(0.150, rel=0, abs=1e-12)
    assert root.margin_absolute == pytest.approx(4.225, rel=0, abs=1e-12)
    assert root.margin_at_k == pytest.approx(4.075, rel=0, abs=1e-12)
    assert root.margin_fraction == pytest.approx(0.4225, rel=0, abs=1e-12)


def test_clean_mass_subtotals(clean_mass):
    roll = rollup(clean_mass)
    # structure sqrt(0.0100 + 0.0064) = sqrt(0.0164) = 0.1280624847...
    assert roll.by_id("structure").uncertainty == pytest.approx(math.sqrt(0.0164), rel=1e-12)
    # avionics  sqrt(0.0025 + 0.0036) = sqrt(0.0061) = 0.0781024967...
    assert roll.by_id("avionics").uncertainty == pytest.approx(math.sqrt(0.0061), rel=1e-12)
    # and sqrt(0.0164 + 0.0061) = sqrt(0.0225) = 0.15 exactly
    assert math.isclose(
        math.sqrt(0.0164 + 0.0061), 0.15, rel_tol=1e-12, abs_tol=1e-12
    )


def test_clean_mass_raises_nothing(clean_mass):
    assert run_checks(clean_mass) == []


def test_clean_mass_combination_comparison(clean_mass):
    # linear 0.10 + 0.08 + 0.05 + 0.06 = 0.29; rss 0.15
    assert linear_rollup_uncertainty(clean_mass) == pytest.approx(0.29, rel=0, abs=1e-12)
    assert rss_rollup_uncertainty(clean_mass) == pytest.approx(0.15, rel=0, abs=1e-12)


# --- planted_findings.yaml ---------------------------------------------


EXPECTED_PLANTED_COUNTS = {
    "BR001": 1,
    "BR002": 3,
    "BR003": 1,
    "BR004": 1,
    "BR005": 1,
    "BR006": 1,
    "BR007": 1,
    "BR008": 1,
    "BR009": 1,
    "BR010": 1,
}


def test_planted_finding_counts_per_code(planted):
    counts: dict[str, int] = {}
    for finding in run_checks(planted):
        counts[finding.code] = counts.get(finding.code, 0) + 1
    assert counts == EXPECTED_PLANTED_COUNTS


def test_planted_severity_counts(planted):
    # BR001, two BR002 errors, BR003, BR008, BR009 are errors: 6.
    # BR004, BR005, BR006, BR010 are warnings: 4.
    # One BR002 info and BR007: 2.
    assert severity_counts(run_checks(planted)) == {"error": 6, "warning": 4, "info": 2}
    assert sum(EXPECTED_PLANTED_COUNTS.values()) == 12


def test_planted_structure_arithmetic(planted):
    # structure nominal 6.0 + 3.0 + 2.0                          = 11.00 kg
    # allowance 6.0*0.05 + 3.0*0.10 + 2.0*0.20
    #           = 0.30 + 0.30 + 0.40                             =  1.00 kg
    # predicted                                                   = 12.00 kg
    # allocation 11.50, so the overrun is                            0.50 kg
    # declared rss sqrt(4 + 1 + 9) = sqrt(14)                     = 3.741657
    # exact: var = 14 + 2*0.9*2*1 + 2*0.9*2*3
    #            = 14 + 3.6 + 10.8 = 28.4, sqrt = 5.329165
    node = rollup(planted).by_id("structure")
    assert node.nominal == pytest.approx(11.0, rel=0, abs=1e-12)
    assert node.allowance == pytest.approx(1.0, rel=0, abs=1e-12)
    assert node.predicted == pytest.approx(12.0, rel=0, abs=1e-12)
    assert node.margin_absolute == pytest.approx(-0.5, rel=0, abs=1e-12)
    assert node.uncertainty == pytest.approx(math.sqrt(14.0), rel=1e-12)
    assert node.exact_uncertainty == pytest.approx(math.sqrt(28.4), rel=1e-12)


def test_planted_root_arithmetic(planted):
    # root nominal 11.0 + 3.5 + 1.8 + 1.8                        = 18.10 kg
    # allowance 1.00 + 0.60 + 0.18 + 0.13                        =  1.91 kg
    # predicted                                                   = 20.01 kg
    # root rule is linear: 3.741657 + 0.5 + 0.4 + 0.5            =  5.141657
    # exact over all eight leaves:
    #   28.4 + 0.25 + 0 + 0.16 + 0.04 + 0.09 = 28.94, sqrt       =  5.379591
    root = rollup(planted).root
    assert root.nominal == pytest.approx(18.10, rel=0, abs=1e-12)
    assert root.allowance == pytest.approx(1.91, rel=0, abs=1e-12)
    assert root.predicted == pytest.approx(20.01, rel=0, abs=1e-12)
    assert root.uncertainty == pytest.approx(math.sqrt(14.0) + 1.4, rel=1e-12)
    assert root.exact_uncertainty == pytest.approx(math.sqrt(28.94), rel=1e-12)
    # margin 22.00 - 20.01 = 1.99; at k = 1, 22.00 - 25.151657 = -3.151657
    assert root.margin_absolute == pytest.approx(1.99, rel=0, abs=1e-12)
    assert root.margin_at_k == pytest.approx(22.0 - 20.01 - (math.sqrt(14.0) + 1.4), rel=1e-12)


def test_planted_is_blocking(planted):
    from budgetroll import has_blocking

    assert has_blocking(run_checks(planted)) is True


# --- cubesat_power.yaml -------------------------------------------------


def test_cubesat_power_rollup(cubesat_power):
    # nominal 1.20 + 0.90 + 1.10 + 0.35 + 0.55 + 0.40 + 1.60 + 0.45 + 0.60
    #         = 7.1500 W
    # allowance 0.2400 + 0.0900 + 0.0550 + 0.0175 + 0.0275 + 0.0200
    #           + 0.1600 + 0.0225 + 0.1200 = 0.7525 W
    # predicted 7.9025 W; allocation 8.4000 W; margin 0.4975 W
    root = rollup(cubesat_power).root
    assert root.nominal == pytest.approx(7.1500, rel=0, abs=1e-12)
    assert root.allowance == pytest.approx(0.7525, rel=0, abs=1e-12)
    assert root.predicted == pytest.approx(7.9025, rel=0, abs=1e-12)
    assert root.margin_absolute == pytest.approx(0.4975, rel=0, abs=1e-12)
    # margin fraction 0.4975 / 8.4 = 0.059226190...
    assert root.margin_fraction == pytest.approx(0.4975 / 8.4, rel=1e-12)


def test_cubesat_power_uncertainty_three_ways(cubesat_power):
    # leaves 0.18, 0.09, 0.06, 0.03, 0.04, 0.02, 0.20, 0.05, 0.12
    # linear sum                                                = 0.790000 W
    # flat rss: 0.0324 + 0.0081 + 0.0036 + 0.0009 + 0.0016
    #           + 0.0004 + 0.0400 + 0.0025 + 0.0144 = 0.1039
    #           sqrt(0.1039)                                    = 0.322335 W
    # declared hierarchy:
    #   payload sqrt(0.0405) = 0.201246
    #   adcs    linear 0.06 + 0.03 = 0.09
    #   comms   sqrt(0.0404) = 0.200998
    #   bus     sqrt(0.0081 + 0.0016 + 0.0404 + 0.0025) = sqrt(0.0526) = 0.229347
    #   thermal 0.12
    #   root    sqrt(0.0405 + 0.0526 + 0.0144) = sqrt(0.1075)   = 0.327872 W
    # exact with rho = 0.6 between the two payload items:
    #   0.1039 + 2 * 0.6 * 0.18 * 0.09 = 0.1039 + 0.01944 = 0.12334
    #   sqrt(0.12334)                                           = 0.351198 W
    roll = rollup(cubesat_power)
    assert linear_rollup_uncertainty(cubesat_power) == pytest.approx(0.79, rel=0, abs=1e-12)
    assert rss_rollup_uncertainty(cubesat_power) == pytest.approx(
        math.sqrt(0.1039), rel=1e-12
    )
    assert roll.by_id("payload").uncertainty == pytest.approx(math.sqrt(0.0405), rel=1e-12)
    assert roll.by_id("adcs").uncertainty == pytest.approx(0.09, rel=0, abs=1e-12)
    assert roll.by_id("comms").uncertainty == pytest.approx(math.sqrt(0.0404), rel=1e-12)
    assert roll.by_id("bus").uncertainty == pytest.approx(math.sqrt(0.0526), rel=1e-12)
    assert roll.root.uncertainty == pytest.approx(math.sqrt(0.1075), rel=1e-12)
    assert roll.root.exact_uncertainty == pytest.approx(math.sqrt(0.12334), rel=1e-12)


def test_cubesat_power_margin_at_two_sigma(cubesat_power):
    # 8.4000 - (7.9025 + 2 * 0.327872) = -0.158244 W
    root = rollup(cubesat_power).root
    assert cubesat_power.k_sigma == 2.0
    assert root.margin_at_k == pytest.approx(8.4 - 7.9025 - 2 * math.sqrt(0.1075), rel=1e-12)
    assert root.margin_at_k < 0.0


def test_cubesat_power_findings(cubesat_power):
    counts: dict[str, int] = {}
    for finding in run_checks(cubesat_power):
        counts[finding.code] = counts.get(finding.code, 0) + 1
    assert counts == {"BR001": 2, "BR002": 4, "BR004": 1, "BR007": 1, "BR010": 1}
    assert severity_counts(run_checks(cubesat_power)) == {
        "error": 4,
        "warning": 2,
        "info": 3,
    }


# --- cubesat_link.yaml --------------------------------------------------


def test_cubesat_link_rollup(cubesat_link):
    # nominal 1.50 + 0.50 + 1.00 + 0.30 + 0.50 + 0.40 + 0.80 + 0.70 + 0.60
    #         = 6.3000 dB
    # allowance 0.0750 + 0.0500 + 0.2000 + 0.0150 + 0.0250
    #           + 0.0200 + 0.0800 + 0.0700 + 0.0300 = 0.5650 dB
    # predicted 6.8650 dB; allocation 17.6655 dB; margin 10.8005 dB
    root = rollup(cubesat_link).root
    assert root.nominal == pytest.approx(6.3000, rel=0, abs=1e-12)
    assert root.allowance == pytest.approx(0.5650, rel=0, abs=1e-12)
    assert root.predicted == pytest.approx(6.8650, rel=0, abs=1e-12)
    assert root.margin_absolute == pytest.approx(10.8005, rel=0, abs=1e-12)


def test_cubesat_link_uncertainty(cubesat_link):
    # declared hierarchy:
    #   implementation sqrt(0.09 + 0.04)          = sqrt(0.1300)
    #   geometry       sqrt(0.16 + 0.01 + 0.0225) = sqrt(0.1925)
    #   propagation    sqrt(0.01 + 0.25)          = sqrt(0.2600)
    #   ageing         linear 0.20 + 0.15 = 0.35  -> 0.1225 in variance
    #   root           sqrt(0.1300 + 0.1925 + 0.2600 + 0.1225) = sqrt(0.7050)
    # flat rss over the nine leaves: sqrt(0.6450)
    roll = rollup(cubesat_link)
    assert roll.root.uncertainty == pytest.approx(math.sqrt(0.7050), rel=1e-12)
    assert rss_rollup_uncertainty(cubesat_link) == pytest.approx(math.sqrt(0.6450), rel=1e-12)
    assert roll.root.exact_uncertainty == pytest.approx(math.sqrt(0.6450), rel=1e-12)
    # margin at 1 sigma: 17.6655 - (6.8650 + 0.839643) = 9.960857 dB
    assert roll.root.margin_at_k == pytest.approx(
        17.6655 - 6.8650 - math.sqrt(0.7050), rel=1e-12
    )


def test_cubesat_link_findings(cubesat_link):
    findings = run_checks(cubesat_link)
    assert severity_counts(findings) == {"error": 0, "warning": 0, "info": 2}
    assert {f.code for f in findings} == {"BR002"}
