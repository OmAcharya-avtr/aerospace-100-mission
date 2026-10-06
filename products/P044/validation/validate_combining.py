"""Validation: MRC, EGC and SC, outage, diversity order, correlation gap.

Checks
------
1. Pointwise ordering. ``gamma_MRC >= gamma_EGC`` and
   ``gamma_MRC >= gamma_SC`` for every realisation, by Cauchy-Schwarz.
   ``gamma_EGC >= gamma_SC`` is **not** claimed: the fraction of
   realisations on which selection beats equal gain is measured and
   reported, because it is a real and widely mis-stated property.
2. Known-answer checks of the three gains on hand-written inputs, shown in
   the printed table.
3. MRC gain statistics against closed forms: ``E[sum_k I_k] = L`` exactly,
   and for the lognormal case ``var(sum_k I_k) = si * sum_jk corr(I_j,I_k)``
   using the exact irradiance correlation of
   ``aperturediv.correlation.log_to_irradiance_correlation``.
4. Outage probability for ``L = 1`` against the lognormal closed form
   ``F(gamma_th / gamma_bar)``.
5. Diversity order for independent gamma-gamma branches against the
   asymptotic prediction ``L * min(alpha, beta)``: the small-irradiance
   density of the gamma-gamma model behaves as ``I^(min(a,b)-1)``, so the
   sum of ``L`` independent branches has ``P(sum < x) ~ x^(L min(a,b))`` and
   the asymptotic log-log slope is ``L min(a,b)``. The **measured**
   finite-window slope is reported next to it; it is lower, and the size of
   the shortfall is the point of the check, not an error in it.
6. Diversity order with correlated branches, same geometry, same sample
   count, same window: the measured shortfall against the independent case
   is the gap the README quotes.
7. The correlation gap in dB: mean SNR needed to hit a target outage,
   independent minus correlated, per ``L``.
8. The lognormal case has **no finite asymptotic diversity order** at all --
   its outage decays faster than any power of the mean SNR -- so the
   measured slope grows without bound as the window moves down. That is
   demonstrated by measuring the slope in three successive windows.

Sizing: 2000000 realisations per configuration, which resolves outage to
about 1e-5 (20 counts), and a 0.25 dB SNR grid so that a two-decade outage
window holds enough points for a least-squares slope. Runtime about 120 s on
one core; this is the heaviest script in the repository.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from aperturediv.channel import (  # noqa: E402
    gamma_gamma_params_from_rytov,
    lognormal_cdf,
)
from aperturediv.combining import (  # noqa: E402
    COMBINERS,
    combined_gain,
    diversity_order,
    egc_gain,
    mrc_gain,
    outage_probability,
    sc_gain,
    weighted_snr,
)
from aperturediv.correlation import (  # noqa: E402
    correlation_matrix,
    equispaced_positions,
    log_to_irradiance_correlation,
    sample_correlated_gamma_gamma,
    sample_correlated_lognormal,
)

N = 2_000_000
SEED = 44044
L_MAX = 4
SI = 0.9
RYTOV = 1.0
SPACING_M = 0.05
RHO_C_M = 0.10
THRESHOLD_DB = 5.0
SNR_GRID = np.arange(0.0, 60.0, 0.25)
WINDOW = (1e-5, 1e-3)
TARGET_OUTAGE = 1e-3


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _snr_for_outage(gain: np.ndarray, target: float, threshold_db: float) -> float:
    """Branch mean SNR in dB at which the outage equals ``target``."""
    quant = float(np.quantile(gain, target))
    return float(threshold_db - 10.0 * np.log10(quant))


def main() -> int:
    failures: list[str] = []
    rng = np.random.default_rng(SEED)
    pos = equispaced_positions(L_MAX, SPACING_M)
    r_corr = correlation_matrix(pos, RHO_C_M, "gaussian")
    r_indep = np.eye(L_MAX)

    _rule("1. Pointwise ordering of the three combiners")
    irr = sample_correlated_lognormal(N, SI, r_corr, rng)
    print(f"  si {SI}, L {L_MAX}, spacing {SPACING_M} m, rho_c {RHO_C_M} m, n {N}")
    print("   L   MRC>=EGC   MRC>=SC   P(SC > EGC)   mean MRC   mean EGC   mean SC")
    for n_ap in range(1, L_MAX + 1):
        sub = irr[:, :n_ap]
        g_mrc, g_egc, g_sc = mrc_gain(sub), egc_gain(sub), sc_gain(sub)
        ok_me = bool(np.all(g_mrc >= g_egc - 1e-12))
        ok_ms = bool(np.all(g_mrc >= g_sc - 1e-12))
        # A relative tolerance, because at L = 1 the three gains are the same
        # quantity computed three ways: EGC forms (sqrt I)^2, which differs
        # from I in the last bit. Without the tolerance that rounding is
        # reported as selection winning a quarter of the time, which would be
        # an artefact and not a finding.
        p_sc_wins = float(np.mean(g_sc > g_egc * (1.0 + 1e-12)))
        print(f"{n_ap:4d} {str(ok_me):>10s} {str(ok_ms):>9s} {p_sc_wins:13.6f} "
              f"{g_mrc.mean():10.6f} {g_egc.mean():10.6f} {g_sc.mean():9.6f}")
        if not ok_me:
            failures.append(f"MRC < EGC somewhere at L={n_ap}")
        if not ok_ms:
            failures.append(f"MRC < SC somewhere at L={n_ap}")
    print()
    print("  For L >= 2, P(SC > EGC) is not zero, so the textbook ordering EGC >= SC")
    print("  is false pointwise; it holds only in the mean. Selection wins on exactly")
    print("  the realisations where one branch is in a deep fade. At L = 1 all three")
    print("  schemes are the same quantity and the probability is identically zero.")
    print()

    _rule("2. Known-answer checks, hand-written inputs")
    cases = (
        ("single branch, I=0.25", np.array([[0.25]]), 0.25, 0.25, 0.25),
        ("two equal, I=(1,1)", np.array([[1.0, 1.0]]), 2.0, 2.0, 1.0),
        ("two unequal, I=(1,0)", np.array([[1.0, 0.0]]), 1.0, 0.5, 1.0),
        ("two unequal, I=(4,1)", np.array([[4.0, 1.0]]), 5.0, 4.5, 4.0),
        ("four equal, I=(0.5 x4)", np.full((1, 4), 0.5), 2.0, 2.0, 0.5),
    )
    print("  case                       MRC exp   MRC got   EGC exp   EGC got"
          "   SC exp    SC got")
    for name, arr, m_e, e_e, s_e in cases:
        m_g = float(mrc_gain(arr)[0])
        e_g = float(egc_gain(arr)[0])
        s_g = float(sc_gain(arr)[0])
        print(f"  {name:25s} {m_e:8.4f} {m_g:9.4f} {e_e:9.4f} {e_g:9.4f} {s_e:8.4f} {s_g:9.4f}")
        for label, exp, got in (("MRC", m_e, m_g), ("EGC", e_e, e_g), ("SC", s_e, s_g)):
            if abs(exp - got) > 1e-12:
                failures.append(f"known answer {name} {label}: expected {exp}, got {got}")
    print("  hand check, two unequal (1,0): EGC = (sqrt1 + sqrt0)^2 / 2 = 0.5 < SC = 1")
    print("  hand check, two unequal (4,1): EGC = (2+1)^2 / 2 = 4.5, MRC = 5, SC = 4")
    w = np.array([[1.0, 1.0]])
    h = np.array([[2.0, 1.0]])
    got = float(weighted_snr(w, h, 1.0)[0])
    print(f"  weighted_snr(w=(1,1), h=(2,1), gamma_bar=1) = {got:.6f}, "
          "expected (2+1)^2/2 = 4.5")
    if abs(got - 4.5) > 1e-12:
        failures.append(f"weighted_snr known answer: {got}")
    print()

    _rule("3. MRC gain statistics against closed forms")
    print("   L   E[sum I] sampled   exact   var sampled    var closed    rel diff")
    for n_ap in range(1, L_MAX + 1):
        sub = irr[:, :n_ap]
        g = mrc_gain(sub)
        corr_i = log_to_irradiance_correlation(r_corr[:n_ap, :n_ap], SI)
        np.fill_diagonal(corr_i, 1.0)
        var_closed = SI * float(corr_i.sum())
        rel = abs(g.var() - var_closed) / var_closed
        print(f"{n_ap:4d} {g.mean():17.6f} {float(n_ap):7.1f} {g.var():13.6f} "
              f"{var_closed:13.6f} {rel:11.3e}")
        if abs(g.mean() - n_ap) / n_ap > 0.01:
            failures.append(f"E[sum I] at L={n_ap}: {g.mean()} vs {n_ap}")
        if rel > 0.02:
            failures.append(f"var(sum I) at L={n_ap}: {g.var()} vs {var_closed}")
    print("  var(sum_k I_k) = si * sum_jk corr(I_j, I_k) with the exact lognormal")
    print("  irradiance correlation; tolerance 2 % at this sample size.")
    print()

    _rule("4. L = 1 outage against the lognormal closed form")
    g1 = mrc_gain(irr[:, :1])
    print("  branch mean SNR dB   P_out empirical   P_out closed form   MC se       z")
    worst_z = 0.0
    for snr_db in (5.0, 10.0, 15.0, 20.0, 25.0):
        p_e = float(outage_probability(g1, snr_db, THRESHOLD_DB))
        p_c = float(lognormal_cdf(10.0 ** ((THRESHOLD_DB - snr_db) / 10.0), SI))
        se = float(np.sqrt(max(p_c * (1 - p_c), 1e-15) / N))
        z = (p_e - p_c) / se
        worst_z = max(worst_z, abs(z))
        print(f"{snr_db:20.2f} {p_e:17.9f} {p_c:19.9f} {se:10.2e} {z:+7.3f}")
    print(f"  worst |z| {worst_z:.3f}, tolerance 4.0")
    if worst_z > 4.0:
        failures.append(f"L=1 outage vs closed form worst |z| {worst_z}")
    print()

    _rule("5-6. Diversity order, gamma-gamma, independent and correlated")
    alpha, beta = gamma_gamma_params_from_rytov(RYTOV)
    tail = min(alpha, beta)
    print(f"  Rytov variance {RYTOV}  alpha {alpha:.6f}  beta {beta:.6f}")
    print(f"  tail index min(alpha,beta) = {tail:.6f}")
    print(f"  n {N}  threshold {THRESHOLD_DB} dB  SNR grid step 0.25 dB  window {WINDOW}")
    print("  (asymptotic prediction applies to the independent case only)")
    print()
    gg_gains: dict[str, dict[int, np.ndarray]] = {}
    measured: dict[str, dict[int, float]] = {}
    for label, matrix in (("independent", r_indep), ("correlated", r_corr)):
        gg = sample_correlated_gamma_gamma(N, alpha, beta, matrix, rng)
        gg_gains[label] = {}
        measured[label] = {}
        print(f"-- {label}")
        print("   L   measured order   asymptotic L*min(a,b)   shortfall   points"
              "    SNR window dB   fit rms")
        for n_ap in range(1, L_MAX + 1):
            gain = mrc_gain(gg[:, :n_ap])
            gg_gains[label][n_ap] = gain
            p_curve = outage_probability(gain, SNR_GRID, THRESHOLD_DB)
            try:
                res = diversity_order(SNR_GRID, p_curve, window=WINDOW)
            except ValueError as exc:
                failures.append(f"diversity order not measurable, {label} L={n_ap}: {exc}")
                print(f"{n_ap:4d}   NOT MEASURABLE: {exc}")
                continue
            measured[label][n_ap] = res.order
            pred = n_ap * tail
            print(
                f"{n_ap:4d} {res.order:16.3f} {pred:23.3f} {pred - res.order:11.3f} "
                f"{res.n_points:8d}  {res.snr_db_range[0]:6.2f}-{res.snr_db_range[1]:6.2f}"
                f" {res.residual_rms:9.4f}"
            )
        print()
    print("  The measured slope is below the asymptotic value in every case. That is")
    print("  not an error: the asymptote is reached only as P_out -> 0, and the window")
    print("  used here (1e-5 to 1e-3) is the range a link designer actually operates")
    print("  in. The honest statement is the measured slope with its window attached.")
    print()
    print("   L   order independent   order correlated   loss from correlation")
    for n_ap in range(1, L_MAX + 1):
        if n_ap in measured["independent"] and n_ap in measured["correlated"]:
            a_i, a_c = measured["independent"][n_ap], measured["correlated"][n_ap]
            print(f"{n_ap:4d} {a_i:18.3f} {a_c:18.3f} {a_i - a_c:22.3f}")
            if n_ap > 1 and a_c > a_i + 0.05:
                failures.append(
                    f"correlated order {a_c} exceeds independent {a_i} at L={n_ap}"
                )
    print(f"  adjacent-aperture log-domain correlation: {r_corr[0, 1]:.6f}")
    print()

    _rule("7. The correlation gap in dB at a target outage")
    print(f"  target outage {TARGET_OUTAGE:.0e}, threshold {THRESHOLD_DB} dB")
    print("   L  scheme   SNR indep dB   SNR corr dB   gap dB   gain over L=1 indep dB")
    print("  the two arrays below are drawn from the same seeded normal stream, so the")
    print("  independent and correlated cases are a matched comparison, not two")
    print("  unrelated Monte Carlo runs.")
    print()
    gg_i_arr = sample_correlated_gamma_gamma(
        N, alpha, beta, r_indep, np.random.default_rng(SEED + 101)
    )
    gg_c_arr = sample_correlated_gamma_gamma(
        N, alpha, beta, r_corr, np.random.default_rng(SEED + 101)
    )
    for scheme in COMBINERS:
        ref = None
        for n_ap in range(1, L_MAX + 1):
            s_i = _snr_for_outage(
                combined_gain(gg_i_arr[:, :n_ap], scheme), TARGET_OUTAGE, THRESHOLD_DB
            )
            s_c = _snr_for_outage(
                combined_gain(gg_c_arr[:, :n_ap], scheme), TARGET_OUTAGE, THRESHOLD_DB
            )
            if n_ap == 1:
                ref = s_i
            print(f"{n_ap:4d}  {scheme:6s} {s_i:14.3f} {s_c:13.3f} {s_c - s_i:8.3f} "
                  f"{(ref or 0.0) - s_i:22.3f}")
            if n_ap > 1 and s_c < s_i - 0.05:
                failures.append(
                    f"correlated needs less SNR than independent, {scheme} L={n_ap}"
                )
        print()
    print("  'gap dB' is the extra branch mean SNR the correlated array needs to reach")
    print("  the same outage as the independent idealisation. It is the price of the")
    print("  independence assumption, per aperture count and per scheme.")
    print()

    _rule("8. The lognormal case has no finite asymptotic diversity order")
    print("  Measured slope for independent lognormal branches in three successive")
    print("  windows. A finite diversity order would give the same slope in all three.")
    ln_i = sample_correlated_lognormal(N, SI, r_indep, np.random.default_rng(SEED + 7))
    windows = ((1e-2, 1e-1), (1e-3, 1e-2), (1e-5, 1e-3))
    print("   L" + "".join(f"   window {w[0]:.0e}-{w[1]:.0e}" for w in windows))
    rising = True
    for n_ap in range(1, L_MAX + 1):
        gain = mrc_gain(ln_i[:, :n_ap])
        row = []
        for win in windows:
            p_curve = outage_probability(gain, SNR_GRID, THRESHOLD_DB)
            try:
                row.append(diversity_order(SNR_GRID, p_curve, window=win).order)
            except ValueError:
                row.append(float("nan"))
        print(f"{n_ap:4d}" + "".join(f"{v:22.3f}" for v in row))
        clean = [v for v in row if np.isfinite(v)]
        rising = rising and all(b > a for a, b in zip(clean, clean[1:], strict=False))
    print(f"  slope increases monotonically as the window moves down, every L: {rising}")
    print("  So 'diversity order L' is not a property of a lognormal channel. For")
    print("  gamma-gamma it is, because the gamma-gamma density has a power-law tail")
    print("  at small irradiance and the lognormal does not.")
    if not rising:
        failures.append("lognormal slope did not increase monotonically across windows")
    print()

    _rule("Result")
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
