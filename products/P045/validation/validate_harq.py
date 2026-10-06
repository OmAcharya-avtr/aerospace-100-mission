"""Validation: hybrid-ARQ goodput, three computation paths, and the crossover.

Checks
------
1. ``block_fer`` (equation 7) against an exact rational-free binomial sum built
   from :func:`math.comb`, which shares no code with the scipy implementation
   used in the package.
2. Type-I chase combining: the analytic path against Monte Carlo, as z scores
   against the binomial standard error of the simulation.
3. Type-II incremental redundancy, the three paths compared:
   the independent-round *approximation*, the exact nested dynamic program, and
   Monte Carlo.  The approximation is reported with its error, because a
   package that ships an approximation without measuring it is shipping an
   unknown.
4. The crossover: the feedback latency D at which paying for redundancy up
   front overtakes retransmitting, and its sensitivity to every assumption that
   fixes its location -- code-family efficiency alpha, per-symbol Es/N0, the
   increment size, and the maximum number of rounds.
5. The optimal first-transmission rate as a function of D, and the value of D
   on each link preset, which is what says whether a real space link is on the
   retransmit side of the crossover or the redundancy side.

Runtime: about 120 s on one core.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul.harq import (  # noqa: E402
    HarqConfig,
    block_fer,
    crossover_rtt,
    harq_goodput,
    harq_goodput_exact,
    harq_simulate,
    optimal_first_rate,
)
from arqlonghaul.link import PRESETS  # noqa: E402

K = 200
ALPHA = 0.5
ESN0_DB = 1.0
DELTA = 40
MAX_ROUNDS = 4
MC_FRAMES = 60_000


def exact_block_fer(n: int, t: int, p: float) -> float:
    """Equation (7) by direct summation with :func:`math.comb`."""
    acc = 0.0
    for i in range(t + 1):
        acc += math.comb(n, i) * p**i * (1.0 - p) ** (n - i)
    return 1.0 - acc


def main() -> None:
    print("=" * 94)
    print("1. block_fer (equation 7) against a direct math.comb summation")
    print("=" * 94)
    print(f"{'n':>6}{'t':>5}{'p':>9}{'package':>16}{'math.comb':>16}{'abs diff':>12}")
    worst = 0.0
    for n, t in ((64, 3), (250, 12), (300, 25), (400, 50), (500, 10)):
        for p in (0.001, 0.01, 0.0563, 0.1, 0.3):
            a = block_fer(n, t, p)
            b = exact_block_fer(n, t, p)
            worst = max(worst, abs(a - b))
            print(f"{n:>6}{t:>5}{p:>9g}{a:>16.10e}{b:>16.10e}{abs(a - b):>12.2e}")
    print(f"largest absolute difference: {worst:.2e}")
    print("Tolerance adopted: 1e-12 absolute.")
    print(f"PASS: {worst < 1e-12}")

    print()
    print("=" * 94)
    print("2. Type-I chase combining: analytic against Monte Carlo")
    print(f"   {MC_FRAMES} frames per Monte Carlo point")
    print("=" * 94)
    rng = np.random.default_rng(45_045_4)
    print(f"{'n1':>5}{'Es/N0':>7}{'M':>3}{'analytic gp':>14}{'MC gp':>13}"
          f"{'rel %':>9}{'analytic res':>14}{'MC res':>11}{'z':>8}")
    worst_z = 0.0
    for n1 in (250, 300):
        for snr in (-1.0, 1.0, 3.0):
            cfg = HarqConfig(
                k=K,
                n1=n1,
                max_rounds=MAX_ROUNDS,
                esn0_db=snr,
                rtt_symbols=200.0,
                alpha=ALPHA,
                scheme="type_i",
            )
            a = harq_goodput(cfg)
            m = harq_simulate(cfg, MC_FRAMES, rng)
            se = math.sqrt(max(a.residual_fer, 1.0 / MC_FRAMES) / MC_FRAMES)
            z = (m.residual_fer - a.residual_fer) / se if se > 0 else 0.0
            worst_z = max(worst_z, abs(z))
            print(
                f"{n1:>5}{snr:>7g}{MAX_ROUNDS:>3}{a.goodput:>14.6f}{m.goodput:>13.6f}"
                f"{100 * (m.goodput / a.goodput - 1):>9.3f}{a.residual_fer:>14.4e}"
                f"{m.residual_fer:>11.4e}{z:>8.2f}"
            )
    print(f"worst |z| on the residual frame error rate: {worst_z:.2f}")
    print("Tolerance adopted: |z| <= 4.")
    print(f"PASS: {worst_z <= 4.0}")

    print()
    print("=" * 94)
    print("3. Type-II incremental redundancy: approximation, exact DP, Monte Carlo")
    print("=" * 94)
    print(f"{'n1':>5}{'Es/N0':>7}{'approx res':>13}{'exact res':>13}"
          f"{'MC res':>11}{'approx/exact-1 %':>18}{'MC z vs exact':>15}")
    worst_approx = 0.0
    worst_mc_z = 0.0
    for n1 in (240, 260, 300, 340):
        for snr in (0.0, 1.0, 2.0):
            cfg = HarqConfig(
                k=K,
                n1=n1,
                delta=DELTA,
                max_rounds=MAX_ROUNDS,
                esn0_db=snr,
                rtt_symbols=200.0,
                alpha=ALPHA,
                scheme="type_ii",
            )
            a = harq_goodput(cfg)
            e = harq_goodput_exact(cfg)
            m = harq_simulate(cfg, MC_FRAMES, rng)
            rel = a.residual_fer / e.residual_fer - 1.0 if e.residual_fer > 0 else 0.0
            worst_approx = max(worst_approx, abs(rel))
            se = math.sqrt(max(e.residual_fer, 1.0 / MC_FRAMES) / MC_FRAMES)
            z = (m.residual_fer - e.residual_fer) / se if se > 0 else 0.0
            worst_mc_z = max(worst_mc_z, abs(z))
            print(
                f"{n1:>5}{snr:>7g}{a.residual_fer:>13.4e}{e.residual_fer:>13.4e}"
                f"{m.residual_fer:>11.4e}{100 * rel:>18.4f}{z:>15.2f}"
            )
    print()
    print("worst error of the independent-round approximation on the residual")
    print(f"frame error rate, against the exact nested DP: {100 * worst_approx:.4f} %")
    print(f"worst |z| of the Monte Carlo against the exact DP:  {worst_mc_z:.2f}")
    print("Tolerance adopted: |z| <= 4 for the Monte Carlo. The approximation")
    print("error is reported, not bounded: it is a property of the")
    print("approximation and the exact DP is what the package publishes.")
    print(f"PASS (Monte Carlo): {worst_mc_z <= 4.0}")

    print()
    print("=" * 94)
    print("4. The crossover: when is up-front redundancy cheaper than")
    print("   retransmitting? R=0.90 with IR versus R=0.50 with IR.")
    print("=" * 94)
    base = crossover_rtt(
        k=K,
        esn0_db=ESN0_DB,
        delta=DELTA,
        retransmit_rate=0.90,
        upfront_rate=0.50,
        max_rounds=MAX_ROUNDS,
        alpha=ALPHA,
    )
    print(f"baseline: k={K}, Es/N0={ESN0_DB} dB, alpha={ALPHA}, delta={DELTA}, "
          f"M={MAX_ROUNDS}")
    print(f"crossover D = {base['crossover_d']:.2f} symbol times")
    print()
    print("sensitivity of the crossover to each assumption, one at a time:")
    print(f"{'varied':<14}{'value':>9}{'crossover D':>14}{'change %':>11}")
    ref = base["crossover_d"]
    for alpha in (0.4, 0.5, 0.6, 0.8):
        c = crossover_rtt(
            k=K, esn0_db=ESN0_DB, delta=DELTA, retransmit_rate=0.90,
            upfront_rate=0.50, max_rounds=MAX_ROUNDS, alpha=alpha,
        )["crossover_d"]
        print(f"{'alpha':<14}{alpha:>9g}{c:>14.2f}{100 * (c / ref - 1):>11.1f}")
    for snr in (-1.0, 0.0, 1.0, 2.0, 3.0):
        c = crossover_rtt(
            k=K, esn0_db=snr, delta=DELTA, retransmit_rate=0.90,
            upfront_rate=0.50, max_rounds=MAX_ROUNDS, alpha=ALPHA,
        )["crossover_d"]
        print(f"{'Es/N0 dB':<14}{snr:>9g}{c:>14.2f}{100 * (c / ref - 1):>11.1f}")
    for delta in (20, 40, 80, 160):
        c = crossover_rtt(
            k=K, esn0_db=ESN0_DB, delta=delta, retransmit_rate=0.90,
            upfront_rate=0.50, max_rounds=MAX_ROUNDS, alpha=ALPHA,
        )["crossover_d"]
        print(f"{'delta':<14}{delta:>9}{c:>14.2f}{100 * (c / ref - 1):>11.1f}")
    for m_max in (2, 3, 4, 6, 8):
        c = crossover_rtt(
            k=K, esn0_db=ESN0_DB, delta=DELTA, retransmit_rate=0.90,
            upfront_rate=0.50, max_rounds=m_max, alpha=ALPHA,
        )["crossover_d"]
        print(f"{'max_rounds':<14}{m_max:>9}{c:>14.2f}{100 * (c / ref - 1):>11.1f}")
    print()
    print("A 'nan' crossover means no sign change inside the search grid, which")
    print("is reported rather than extrapolated.")

    print()
    print("=" * 94)
    print("5. Optimal first-transmission rate against D, and D on real links")
    print("=" * 94)
    print(f"{'D symbols':>12}{'optimal R1':>12}{'n1':>6}{'goodput':>11}"
          f"{'mean rounds':>13}{'residual':>12}")
    for d in (1.0, 10.0, 50.0, 100.0, 300.0, 1e3, 1e4, 1e5, 1e6):
        rate, n1, res = optimal_first_rate(
            K, d, ESN0_DB, DELTA, MAX_ROUNDS, ALPHA
        )
        print(
            f"{d:>12.0f}{rate:>12.4f}{n1:>6}{res.goodput:>11.6f}"
            f"{res.mean_rounds:>13.4f}{res.residual_fer:>12.3e}"
        )
    print()
    print("D on each link preset, taking BPSK so that one channel symbol carries")
    print("one bit and D is the bandwidth-delay product in bits:")
    print(f"{'preset':<11}{'D symbols':>14}{'D / crossover':>16}{'side':>26}")
    for name, link in PRESETS.items():
        d = link.bdp_bits
        ratio = d / ref if ref and math.isfinite(ref) else float("nan")
        side = "pay redundancy up front" if ratio > 1 else "retransmit"
        print(f"{name:<11}{d:>14.4g}{ratio:>16.4g}{side:>26}")
    print()
    print("Reading: every link preset here is between four and six orders of")
    print("magnitude past the crossover, so for a single HARQ process with no")
    print("pipelining the answer on a space link is always to pay for redundancy")
    print("before knowing whether it is needed. The assumption that produces that")
    print("conclusion is the no-pipelining one: a sender that runs enough")
    print("parallel HARQ processes to fill the round trip hides D entirely, and")
    print("then the crossover moves back to D near zero and retransmission wins")
    print("again. The crossover is a statement about one process, not about the")
    print("link.")


if __name__ == "__main__":
    main()
