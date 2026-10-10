"""The Brier decomposition identities, to machine precision, over many inputs.

Two claims are checked:

1. ``BS = REL - RES + UNC`` holds exactly for the distinct-value (exact)
   decomposition. This is Murphy (1973).
2. ``BS = REL - RES + UNC + WBV - 2 WBC`` holds exactly for *any* binning of
   *any* forecast. This is the five-term identity derived in
   ``src/calibaudit/decomposition.py``.

The third number reported is the size of the three-term residual on binned
continuous forecasts, which is what a library reporting only three terms
silently drops. It is not an error to be driven to zero; it is a quantity to
be measured.
"""

from __future__ import annotations

import numpy as np
from _harness import Recorder

from calibaudit.decomposition import binned_decomposition, murphy_decomposition
from calibaudit.synthetic import (
    SPEC_NAMES,
    analytic_truth,
    get_spec,
    sample_forecast,
)

SEED = 56
BIN_COUNTS = (1, 2, 3, 5, 10, 15, 20, 50, 100, 500)
SAMPLE_SIZES = (10, 100, 1000, 8000)
ROUNDINGS = (1, 2, 3)


def main() -> int:
    rec = Recorder("validate_decomposition_identity")
    rec.header("Brier decomposition identities - calibaudit 0.1.0")
    rng = np.random.default_rng(SEED)

    # --- 1. exact decomposition, distinct forecast values -------------------
    worst_exact = 0.0
    worst_exact_where = ""
    n_exact = 0
    for name in SPEC_NAMES:
        spec = get_spec(name)
        for n in SAMPLE_SIZES:
            s = sample_forecast(spec, n, seed=SEED * 1000 + n)
            for digits in ROUNDINGS:
                d = murphy_decomposition(np.round(s.forecasts, digits), s.outcomes)
                n_exact += 1
                if abs(d.three_term_residual) > worst_exact:
                    worst_exact = abs(d.three_term_residual)
                    worst_exact_where = f"{name}, n = {n}, {digits} digits"
    rec.say(f"exact decomposition cases : {n_exact}")
    rec.say(f"worst three-term residual : {worst_exact:.6e}  ({worst_exact_where})")
    rec.say()
    rec.check(
        "exact Murphy identity BS = REL - RES + UNC",
        reference="Murphy 1973, J. Appl. Meteorol. 12(4) 595-600",
        measured=f"worst |residual| = {worst_exact:.6e} over {n_exact} cases",
        expectation="<= 1e-14 (machine precision at this magnitude)",
        passed=worst_exact <= 1e-14,
    )

    # --- 2. five-term identity, arbitrary binning ---------------------------
    worst_five = 0.0
    worst_five_where = ""
    n_five = 0
    for name in SPEC_NAMES:
        spec = get_spec(name)
        for n in SAMPLE_SIZES:
            s = sample_forecast(spec, n, seed=SEED * 2000 + n)
            for n_bins in BIN_COUNTS:
                for strategy in ("equal_width", "equal_mass"):
                    d = binned_decomposition(
                        s.forecasts, s.outcomes, n_bins=n_bins, strategy=strategy
                    )
                    n_five += 1
                    if abs(d.identity_residual) > worst_five:
                        worst_five = abs(d.identity_residual)
                        worst_five_where = f"{name}, n = {n}, B = {n_bins}, {strategy}"
    rec.say(f"binned decomposition cases : {n_five}")
    rec.say(f"worst identity residual    : {worst_five:.6e}  ({worst_five_where})")
    rec.say()
    rec.check(
        "five-term identity BS = REL - RES + UNC + WBV - 2 WBC",
        reference="elementary algebra, derived in src/calibaudit/decomposition.py",
        measured=f"worst |residual| = {worst_five:.6e} over {n_five} cases",
        expectation="<= 1e-14",
        passed=worst_five <= 1e-14,
    )

    # --- 3. random adversarial inputs, not just the shipped specs -----------
    worst_random = 0.0
    n_random = 0
    for _ in range(600):
        n = int(rng.integers(1, 400))
        mode = rng.integers(0, 4)
        if mode == 0:
            f = rng.random(n)
        elif mode == 1:
            f = rng.choice([0.0, 1.0], size=n).astype(float)
        elif mode == 2:
            f = np.round(rng.random(n), 1)
        else:
            f = np.clip(rng.normal(0.5, 0.05, size=n), 0.0, 1.0)
        o = (rng.random(n) < 0.5).astype(float)
        n_bins = int(rng.integers(1, 60))
        strategy = "equal_width" if rng.random() < 0.5 else "equal_mass"
        d = binned_decomposition(f, o, n_bins=n_bins, strategy=strategy)
        n_random += 1
        worst_random = max(worst_random, abs(d.identity_residual))
    rec.say(f"random adversarial cases   : {n_random}")
    rec.say(f"worst identity residual    : {worst_random:.6e}")
    rec.say()
    rec.check(
        "five-term identity on random inputs including all-0/1 and tied forecasts",
        reference="same algebra, inputs drawn from four adversarial families",
        measured=f"worst |residual| = {worst_random:.6e} over {n_random} cases",
        expectation="<= 1e-14",
        passed=worst_random <= 1e-14,
    )

    # --- 4. what a three-term report drops ---------------------------------
    rec.say("How much a three-term binned report drops, spec 'overconfident', n = 8000")
    rec.say()
    head = (
        f"{'B':>5} {'strategy':>12} {'BS':>12} {'REL-RES+UNC':>13} "
        f"{'WBV':>11} {'WBC':>12} {'3-term resid':>13} {'% of BS':>9}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    s = sample_forecast(get_spec("overconfident"), 8000, seed=SEED)
    max_frac = 0.0
    for strategy in ("equal_width", "equal_mass"):
        for n_bins in (2, 5, 10, 20, 50, 200):
            d = binned_decomposition(
                s.forecasts, s.outcomes, n_bins=n_bins, strategy=strategy
            )
            frac = 100.0 * abs(d.three_term_residual) / d.brier
            max_frac = max(max_frac, frac)
            rec.say(
                f"{n_bins:>5d} {strategy:>12} {d.brier:>12.8f} {d.three_term_sum:>13.8f} "
                f"{d.within_bin_variance:>11.8f} {d.within_bin_covariance:>12.8f} "
                f"{d.three_term_residual:>+13.8f} {frac:>9.4f}"
            )
    rec.say()
    rec.check(
        "the dropped three-term residual is measurable, not negligible",
        reference="same sample, same code path, two bins vs 200 bins",
        measured=f"largest |three-term residual| = {max_frac:.4f} % of the Brier score",
        expectation="> 0.01 % at some bin count, i.e. the drop is real",
        passed=max_frac > 0.01,
    )

    # --- 5. the terms against their population values ----------------------
    rec.say("Binned terms against population values, n = 200000, 50 equal-mass bins")
    rec.say()
    head = (
        f"{'spec':>19} {'UNC_pop':>10} {'UNC_meas':>10} {'RES_pop':>10} "
        f"{'RES_meas':>10} {'REL_pop':>11} {'REL_meas':>11} {'BS_pop':>10} {'BS_meas':>10}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    worst_unc = worst_res = 0.0
    for name in SPEC_NAMES:
        spec = get_spec(name)
        t = analytic_truth(spec)
        big = sample_forecast(spec, 200_000, seed=SEED * 3000)
        d = binned_decomposition(
            big.forecasts, big.outcomes, n_bins=50, strategy="equal_mass"
        )
        worst_unc = max(worst_unc, abs(d.uncertainty - t.uncertainty))
        worst_res = max(worst_res, abs(d.resolution - t.resolution))
        rec.say(
            f"{name:>19} {t.uncertainty:>10.6f} {d.uncertainty:>10.6f} "
            f"{t.resolution:>10.6f} {d.resolution:>10.6f} {t.reliability:>11.3e} "
            f"{d.reliability:>11.3e} {t.brier:>10.6f} {d.brier:>10.6f}"
        )
    rec.say()
    rec.check(
        "measured uncertainty term converges to the Beta closed form",
        reference="UNC = obar(1-obar) with obar = a/(a+b), exact",
        measured=f"worst |difference| = {worst_unc:.6e} at n = 200000",
        expectation="<= 5e-3 (five standard errors of a base-rate estimate at this n)",
        passed=worst_unc <= 5e-3,
    )
    rec.check(
        "measured resolution term converges to Var(p) of the Beta latent",
        reference="RES = ab / ((a+b)^2 (a+b+1)), exact",
        measured=f"worst |difference| = {worst_res:.6e} at n = 200000, 50 bins",
        expectation="<= 1e-2; binning truncates resolution from below, so a "
        "one-sided shortfall is expected",
        passed=worst_res <= 1e-2,
    )

    # --- 6. the population values themselves, with their quadrature error ---
    rec.say()
    rec.say(
        "Population values of every shipped spec, with the absolute-error estimate\n"
        "scipy.integrate.quad returns for each integral it had to evaluate. These\n"
        "are the reference values every other script measures against, and the\n"
        "abserr column is the tolerance to quote when doing so. The three calibrated\n"
        "specs need no quadrature for REL or ECE: both are exactly 0 and are returned\n"
        "as such, so their abserr is 0."
    )
    rec.say()
    head = (
        f"{'spec':>19} {'obar':>8} {'UNC':>9} {'RES':>9} {'REL':>11} {'REL err':>10} "
        f"{'ECE':>9} {'ECE err':>10} {'LS':>18} {'LS err':>10}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    for name in SPEC_NAMES:
        t = analytic_truth(get_spec(name))
        rec.say(
            f"{name:>19} {t.base_rate:>8.4f} {t.uncertainty:>9.6f} {t.resolution:>9.6f} "
            f"{t.reliability:>11.3e} {t.reliability_abserr:>10.2e} {t.ece:>9.6f} "
            f"{t.ece_abserr:>10.2e} {t.log_score!r:>18} {t.log_score_abserr:>10.2e}"
        )
    rec.say()

    seven_twelfths = 7.0 / 12.0
    cal = analytic_truth(get_spec("calibrated"))
    err = abs(cal.log_score - seven_twelfths)
    rec.say(
        "Hand integral, Beta(2,2) identity forecaster: density 6p(1-p), and by the\n"
        "p <-> 1-p symmetry of both density and integrand,\n"
        "  LS = -2 * int 6p(1-p) p ln p dp = -12 int (p^2 - p^3) ln p dp,\n"
        "and with int_0^1 p^n ln p dp = -1/(n+1)^2,\n"
        "  int (p^2 - p^3) ln p dp = -1/9 + 1/16 = -7/144,  so LS = 7/12."
    )
    rec.say(f"  7/12               = {seven_twelfths!r}")
    rec.say(f"  quadrature         = {cal.log_score!r}")
    rec.say(f"  |difference|       = {err:.3e}")
    rec.say(f"  quad's own abserr  = {cal.log_score_abserr:.3e}")
    rec.say()
    rec.check(
        "quadrature log score of the Beta(2,2) identity spec matches the hand integral "
        "7/12, and quad's own abserr is conservative",
        reference="7/12 = 0.5833333333333333, derived above with no numerics",
        measured=f"quadrature {cal.log_score!r}, |difference| {err:.3e}, against a "
        f"reported abserr of {cal.log_score_abserr:.3e}",
        expectation="|difference| <= 1e-10, and the reported abserr larger than the "
        "actual error, i.e. conservative",
        passed=err <= 1e-10 and cal.log_score_abserr > err,
    )

    uni = analytic_truth(get_spec("calibrated_uniform"))
    err_uni = abs(uni.log_score - 0.5)
    rec.say(
        "Hand integral, uniform identity forecaster: density 1, so\n"
        "  LS = -2 int_0^1 p ln p dp = -2 * (-1/4) = 1/2."
    )
    rec.say(f"  1/2                = {0.5!r}")
    rec.say(f"  quadrature         = {uni.log_score!r}")
    rec.say(f"  |difference|       = {err_uni:.3e}")
    rec.say()
    rec.check(
        "quadrature log score of the uniform identity spec matches the hand integral 1/2",
        reference="1/2 = 0.5, derived above with no numerics",
        measured=f"quadrature {uni.log_score!r}, |difference| {err_uni:.3e}",
        expectation="|difference| <= 1e-10",
        passed=err_uni <= 1e-10,
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
