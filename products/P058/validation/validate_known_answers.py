"""Known-answer checks: measured ARL against published values, and detector internals.

Section 1 is the one that matters most. Every ARL figure in this repository comes
out of the same estimator, so a published ARL for the one detector that has one
tells the reader whether the estimator is measuring what it claims.
"""

from __future__ import annotations

import math

import numpy as np
from _harness import run
from scipy.stats import ks_2samp


def body(report) -> None:
    from telemdrift.detectors import (
        ADWIN,
        CUSUM,
        EWMA,
        PageHinkley,
        WindowedKS,
        ks_two_sample_statistic,
    )
    from telemdrift.reference import (
        NIST_CUSUM_ARL,
        ks_asymptotic_tail_probability,
        siegmund_one_sided_arl,
    )
    from telemdrift.scoring import measure_arl0
    from telemdrift.streams import stationary

    def sf(length, seed):
        return stationary(length, seed)

    report.section("1. CUSUM ARL0 against the NIST handbook and Siegmund's formula")
    print("  Source 1: NIST/SEMATECH e-Handbook of Statistical Methods, section")
    print("            6.3.2.3.1, 'Cusum Average Run Length',")
    print("            https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3231.htm")
    print("            read 2026-10-10. For k = 0.5 it tabulates the ONE-SIDED")
    print("            in-control ARL as 336 at h = 4 and 930 at h = 5. The page")
    print("            states: 'If one has to control both positive and negative")
    print("            deviations, as is usually the case, two one-sided charts are")
    print("            used'. The handbook credits no external author for the table.")
    print("  Source 2: Siegmund's closed form, ARL = (exp(-2 D b) + 2 D b - 1)/(2 D^2)")
    print("            with b = h + 1.166 and D = delta - k.")
    print()
    print("  This package's CUSUM runs BOTH one-sided charts against the same h, so")
    print("  it false-alarms at twice the rate and the reference is the handbook")
    print("  value HALVED. Quoting 336 for a two-sided chart would be wrong by a")
    print("  factor of two, which is the error this section exists to prevent.")
    print()
    for h in (4, 5):
        one = NIST_CUSUM_ARL[h]["arl0"]
        sieg = siegmund_one_sided_arl(h, 0.5, 0.0)
        rel = abs(sieg - one) / one
        print(f"  h = {h}: handbook one-sided {one:7.1f}   Siegmund {sieg:7.2f}   "
              f"relative difference {100 * rel:.2f} %")
        report.check(f"the two references agree within 1 % at h = {h}", rel < 0.01,
                     f"{100 * rel:.2f} %")
    print()
    print("  Measured, 12 seeds x 60000 stationary samples = 720000 samples per h.")
    print()
    print("  h    reference   measured      SEM   rel.diff   runs   censored tail")
    for h in (4, 5):
        ref = 0.5 * NIST_CUSUM_ARL[h]["arl0"]
        res = measure_arl0(lambda hh=h: CUSUM(h=hh, k=0.5), sf, range(59_001, 59_013),
                           60_000)
        rel = (res.arl0 - ref) / ref
        print(f"  {h}    {ref:9.1f} {res.arl0:10.1f} {res.sem:8.1f} "
              f"{100 * rel:+8.1f} % {res.n_runs:6d} "
              f"{100 * res.censored_fraction:9.2f} %")
        report.check(
            f"measured two-sided CUSUM ARL0 within 10 % of the halved handbook value"
            f" at h = {h}",
            abs(rel) < 0.10,
            f"measured {res.arl0:.1f} +/- {res.sem:.1f} vs reference {ref:.0f}, "
            f"{100 * rel:+.1f} %",
        )
        report.check(
            f"measured ARL0 is below the reference at h = {h}, as the censored-tail "
            f"exclusion predicts",
            res.arl0 < ref,
            f"{100 * rel:+.1f} %",
        )
    print()
    print("  The tolerance is 10 % and was set from the Monte Carlo standard error")
    print("  before the measurement, not after it: at these budgets the relative")
    print("  SEM is 1.5-2.6 %, so 10 % is four to seven standard errors. The halving")
    print("  is itself an approximation - the two arms are driven by the same")
    print("  observations and are not independent - which is why the band is not")
    print("  tighter.")

    report.section("2. CUSUM ARL1 against the handbook, one-sigma shift")
    print("  The handbook gives one-sided ARL1 = 8.38 at h = 4 and 10.4 at h = 5 for")
    print("  a one-sigma shift. For a two-sided chart detecting a +1 sigma shift the")
    print("  upper arm does essentially all the work, so the one-sided value is the")
    print("  reference and no halving applies. Measured with the zero-state")
    print("  convention (chart started at the change) to match the handbook, which")
    print("  is NOT the steady-state convention used everywhere else in this")
    print("  repository; the two are different quantities and the difference is")
    print("  measured in validate_arl_calibration.py section 4.")
    print()
    print("  h    reference   measured      SEM   rel.diff   replicates")
    for h in (4, 5):
        ref = NIST_CUSUM_ARL[h]["arl1_shift1"]
        delays = []
        rng_seeds = range(60_001, 60_601)
        for s in rng_seeds:
            det = CUSUM(h=h, k=0.5)
            det.reset()
            x = stationary(400, s) + 1.0
            fired = -1
            for i, v in enumerate(x):
                if det.update(v):
                    fired = i
                    break
            delays.append(400 if fired < 0 else fired + 1)
        arr = np.asarray(delays, dtype=float)
        sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
        rel = (arr.mean() - ref) / ref
        print(f"  {h}    {ref:9.2f} {arr.mean():10.2f} {sem:8.2f} "
              f"{100 * rel:+8.1f} % {arr.size:12d}")
        report.check(
            f"zero-state CUSUM ARL1 within 15 % of the handbook value at h = {h}",
            abs(rel) < 0.15,
            f"measured {arr.mean():.2f} +/- {sem:.2f} vs {ref}",
        )

    report.section("3. Windowed KS statistic against scipy.stats.ks_2samp")
    print("  The detector implements the two-sample KS statistic directly because")
    print("  the benchmark evaluates it of order 10^5 times. SciPy is the reference.")
    worst = 0.0
    rng = np.random.default_rng(58_700)
    for _ in range(500):
        n, m = int(rng.integers(5, 120)), int(rng.integers(5, 120))
        a = rng.standard_normal(n)
        b = rng.standard_normal(m) * rng.uniform(0.5, 2.0) + rng.uniform(-1.5, 1.5)
        worst = max(worst, abs(ks_two_sample_statistic(np.sort(a), b)
                               - ks_2samp(a, b).statistic))
    print(f"  500 random pairs, sizes 5-119, worst absolute difference {worst:.3e}")
    report.check("KS statistic matches SciPy to machine precision", worst < 1e-12,
                 f"worst {worst:.3e}")

    report.section("4. Windowed KS default threshold against its own asymptotic formula")
    c = WindowedKS.default_threshold()
    p = ks_asymptotic_tail_probability(c, WindowedKS.N_REF, WindowedKS.N_DET)
    print(f"  n_ref = {WindowedKS.N_REF}, n_det = {WindowedKS.N_DET}, "
          f"alpha = {WindowedKS.ALPHA_DEFAULT}")
    print(f"  c_alpha  = {c:.9f}")
    print(f"  P(D > c) = {p:.9f}   (asymptotic, should equal alpha)")
    report.check("the shipped default c inverts to its declared alpha",
                 abs(p - WindowedKS.ALPHA_DEFAULT) < 1e-9, f"{p:.9f}")

    report.section("5. ADWIN cut rule, hand-computed")
    print("  Transcribed from Bifet and Gavalda's technical report 'Adaptive")
    print("  Parameter-free Learning from Evolving Data Streams' section 4.1.1")
    print("  (upcommons.upc.edu), the author-hosted source this session could read:")
    print()
    print("      m       = 2 / (1/|W0| + 1/|W1|)")
    print("      eps_cut = sqrt( (1 / (2 m)) * ln(4 |W| / delta) )")
    print()
    print("  With |W0| = |W1| = 10, |W| = 20, delta = 0.05:")
    print("      m       = 2 / (1/10 + 1/10) = 10")
    print("      eps_cut = sqrt( 0.05 * ln(1600) ) = sqrt(0.05 * 7.3777589)")
    print("              = sqrt(0.36888794) = 0.60736146")
    hand = math.sqrt(0.05 * math.log(1600.0))
    print(f"  recomputed here: {hand:.8f}")
    report.check("hand arithmetic reproduces the formula", abs(hand - 0.60736146) < 1e-7,
                 f"{hand:.8f}")
    det = ADWIN(delta=0.05, min_sub=5, check_every=20, max_buckets=40)
    for _ in range(10):
        det.update(0.0)
    for _ in range(9):
        det.update(10.0)
    fired = det.update(10.0)
    print(f"  detector on 10 zeros then 10 tens: cut declared = {fired}, "
          f"cut evidence = {det.cut_evidence():.4f} (ratio to eps_cut)")
    report.check("the implementation cuts on a 10-sigma split", fired)
    print()
    print("  NOTE, recorded rather than resolved: the SDM 2007 paper is widely")
    print("  quoted with m = 1/(1/n0 + 1/n1), a factor of two smaller, which makes")
    print("  eps_cut larger by sqrt(2). This package implements the harmonic-mean")
    print("  form above because that is what the source it could verify states, and")
    print("  calibrates delta to a measured ARL0, which absorbs a constant factor on")
    print("  eps_cut exactly. No number in this repository depends on which form is")
    print("  meant, because none of them uses the nominal delta as a false-alarm")
    print("  rate.")
    report.finding(
        "ADWIN cut rule: the verified author-hosted source states m = 2/(1/n0+1/n1); "
        "the SDM 2007 paper is widely quoted with m = 1/(1/n0+1/n1). Implemented as "
        "verified; delta is calibrated so the factor is absorbed."
    )

    report.section("6. Detector internals, hand-computed")
    print("  CUSUM, k = 0.5, h = 2, four samples of 1.0 then a fifth:")
    print("    S+ = 0.5, 1.0, 1.5, 2.0 (not > 2), 2.5 -> alarm on the fifth")
    cs = CUSUM(h=2.0, k=0.5)
    fired = [cs.update(1.0) for _ in range(5)]
    print(f"    fired sequence {fired}, final S+ = {cs.s_plus}, S- = {cs.s_minus}")
    report.check("CUSUM alarms on exactly the fifth sample",
                 fired == [False, False, False, False, True])
    report.check("CUSUM lower arm stays at zero", cs.s_minus == 0.0)

    print()
    print("  EWMA, r = 0.5, L = 3, two samples of 1.0:")
    print("    z1 = 0.5, var1 = (1/3)(1 - 0.25) = 0.25, limit = 1.5")
    print("    z2 = 0.75, var2 = (1/3)(1 - 0.0625) = 0.3125, limit = 1.677051")
    ew = EWMA(L=3.0, r=0.5)
    ew.update(1.0)
    z1, l1 = ew.z, ew._limit()
    ew.update(1.0)
    z2, l2 = ew.z, ew._limit()
    print(f"    measured z1 = {z1:.6f} limit1 = {l1:.6f}")
    print(f"    measured z2 = {z2:.6f} limit2 = {l2:.6f}")
    report.check("EWMA z follows the hand recursion",
                 abs(z1 - 0.5) < 1e-12 and abs(z2 - 0.75) < 1e-12)
    report.check("EWMA exact limit matches the hand variance",
                 abs(l1 - 1.5) < 1e-12 and abs(l2 - 3.0 * math.sqrt(0.3125)) < 1e-12)

    print()
    print("  Page-Hinkley, delta = 0, samples 0.0, 2.0, 4.0:")
    print("    running means 0.0, 1.0, 2.0; m = 0.0, 1.0, 3.0; m_min = 0, m_max = 3")
    ph = PageHinkley(lambda_=10.0, delta=0.0)
    for v in (0.0, 2.0, 4.0):
        ph.update(v)
    print(f"    measured mean = {ph.mean:.6f}, m = {ph.m:.6f}, "
          f"m_min = {ph.m_min:.6f}, m_max = {ph.m_max:.6f}")
    report.check("Page-Hinkley follows the hand recursion",
                 abs(ph.mean - 2.0) < 1e-12 and abs(ph.m - 3.0) < 1e-12)

    print()
    print("  Page-Hinkley on a constant stream cannot fire, at any threshold:")
    print("    the running mean equals the value, so every increment is -delta.")
    ph2 = PageHinkley(lambda_=1.0, delta=0.0)
    never = not any(ph2.update(1000.0) for _ in range(1000))
    print(f"    1000 samples of 1000.0 at lambda = 1.0: fired = {not never}")
    report.check("Page-Hinkley is a detector of changes in level, not of level",
                 never)


if __name__ == "__main__":
    raise SystemExit(run("validate_known_answers",
                         "telemdrift 0.1.0 - known-answer validation",
                         body))
