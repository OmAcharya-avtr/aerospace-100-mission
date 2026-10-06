"""Validation: the Gilbert-Elliott channel reproduces its own analytic statistics.

Before the correlated channel is used to contradict the independent-error
closed forms, it has to be shown to be the channel it claims to be.  Four
statistics of a realisation are compared with the analytic expressions in
``arqlonghaul.channel``:

1. Marginal frame error rate against equation (3),
   ``p_bar = (1 - pi_b) eps_g + pi_b eps_b``.
2. Mean Bad-state sojourn against equation (4), ``1/p_bg``, measured from the
   state trace by run-length counting.
3. Lag-k autocorrelation of the error indicator against equation (5),
   ``rho_k = pi_b (1-pi_b) (eps_b - eps_g)^2 lambda^k / (p_bar (1-p_bar))``.
4. The inverse construction :meth:`GilbertElliottChannel.from_mean_and_burst`:
   asking for a marginal rate and a burst length and getting them back.

Every comparison is reported as a z score against the appropriate sampling
standard error, which for the marginal rate accounts for the correlation: the
variance of the sample mean of a correlated binary series is inflated by the
factor ``1 + 2 sum_k rho_k``, evaluated in closed form for the geometric
autocorrelation of this chain as ``(1 + lambda_eff) / (1 - lambda_eff)`` with
``lambda_eff = lambda``.  Using the independent-sample standard error here
would overstate the precision by a factor of about six at a burst length of 25
and would make a correct channel look broken.

Runtime: about 40 s on one core.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul.channel import GilbertElliottChannel, IndependentFrameChannel  # noqa: E402

SLOTS = 2_000_000
CASES = (
    (0.02, 2.0),
    (0.05, 5.0),
    (0.05, 25.0),
    (0.10, 25.0),
    (0.10, 100.0),
    (0.30, 50.0),
)


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True in ``flags``."""
    padded = np.concatenate(([False], flags, [False]))
    change = np.diff(padded.astype(np.int8))
    starts = np.nonzero(change == 1)[0]
    ends = np.nonzero(change == -1)[0]
    return ends - starts


def sample_acf(x: np.ndarray, lag: int) -> float:
    """Sample lag-``lag`` autocorrelation of a 1-D series."""
    c = x - x.mean()
    denom = float(np.dot(c, c))
    if denom == 0.0:
        return 0.0
    return float(np.dot(c[:-lag], c[lag:]) / denom)


def main() -> None:
    rng = np.random.default_rng(45_045_1)
    print(f"slots per case: {SLOTS}")
    print()
    print("=" * 84)
    print("1. Marginal frame error rate, equation (3)")
    print("   standard error inflated by the correlation factor (1+lam)/(1-lam)")
    print("=" * 84)
    print(f"{'target fer':>11}{'burst':>8}{'analytic':>11}{'sampled':>11}"
          f"{'iid se':>10}{'corr se':>10}{'z':>8}")
    worst_z = 0.0
    for fer, burst in CASES:
        ch = GilbertElliottChannel.from_mean_and_burst(fer, burst)
        errors = ch.errors(SLOTS, rng)
        sampled = float(errors.mean())
        p = ch.mean_fer
        iid_se = (p * (1 - p) / SLOTS) ** 0.5
        lam = ch.lam
        inflation = ((1.0 + lam) / (1.0 - lam)) ** 0.5 if lam < 1.0 else float("inf")
        corr_se = iid_se * inflation
        z = (sampled - p) / corr_se
        worst_z = max(worst_z, abs(z))
        print(
            f"{fer:>11g}{burst:>8g}{p:>11.6f}{sampled:>11.6f}"
            f"{iid_se:>10.2e}{corr_se:>10.2e}{z:>8.2f}"
        )
    print(f"worst |z| on the marginal rate: {worst_z:.2f}")

    print()
    print("=" * 84)
    print("2. Mean Bad-state sojourn, equation (4), from run-length counting")
    print("=" * 84)
    print(f"{'target burst':>13}{'analytic':>11}{'sampled':>11}{'runs':>9}"
          f"{'se':>9}{'z':>8}")
    worst_z2 = 0.0
    for fer, burst in CASES:
        ch = GilbertElliottChannel.from_mean_and_burst(fer, burst)
        states = ch.state_trace(SLOTS, rng)
        lens = run_lengths(states)
        # Geometric with mean L has variance L(L-1); use the sample sd.
        mean = float(lens.mean())
        se = float(lens.std(ddof=1) / np.sqrt(lens.size))
        z = (mean - ch.mean_burst_slots) / se
        worst_z2 = max(worst_z2, abs(z))
        print(
            f"{burst:>13g}{ch.mean_burst_slots:>11.4f}{mean:>11.4f}"
            f"{lens.size:>9}{se:>9.4f}{z:>8.2f}"
        )
    print(f"worst |z| on the mean sojourn: {worst_z2:.2f}")

    print()
    print("=" * 84)
    print("3. Autocorrelation of the error indicator, equation (5)")
    print("=" * 84)
    print(f"{'fer':>7}{'burst':>7}{'lag':>6}{'analytic':>11}{'sampled':>11}"
          f"{'abs diff':>11}")
    worst_acf = 0.0
    for fer, burst in ((0.05, 25.0), (0.10, 100.0), (0.30, 50.0)):
        ch = GilbertElliottChannel.from_mean_and_burst(fer, burst)
        errors = ch.errors(SLOTS, rng).astype(float)
        for lag in (1, 5, 25, 100):
            a = ch.autocorrelation(lag)
            s = sample_acf(errors, lag)
            worst_acf = max(worst_acf, abs(a - s))
            print(f"{fer:>7g}{burst:>7g}{lag:>6}{a:>11.6f}{s:>11.6f}{abs(a - s):>11.2e}")
    print(f"worst absolute autocorrelation error: {worst_acf:.2e}")
    print("Tolerance adopted: 5e-3 absolute, which is about the sampling")
    print(f"standard error of an autocorrelation at n = {SLOTS}.")
    print(f"PASS: {worst_acf < 5e-3}")

    print()
    print("=" * 84)
    print("4. The inverse construction returns what it was asked for")
    print("=" * 84)
    print(f"{'asked fer':>11}{'got fer':>12}{'asked burst':>12}{'got burst':>11}"
          f"{'pi_bad':>9}{'lambda':>9}")
    for fer, burst in CASES:
        ch = GilbertElliottChannel.from_mean_and_burst(fer, burst)
        print(
            f"{fer:>11g}{ch.mean_fer:>12.8f}{burst:>12g}"
            f"{ch.mean_burst_slots:>11.6f}{ch.pi_bad:>9.5f}{ch.lam:>9.5f}"
        )

    print()
    print("=" * 84)
    print("5. For comparison: the independent channel's own run lengths")
    print("   An independent channel still has error 'bursts' of mean")
    print("   1/(1-p); the Gilbert-Elliott channel is only interesting")
    print("   because it has longer ones at the same marginal rate.")
    print("=" * 84)
    print(f"{'fer':>7}{'analytic 1/(1-p)':>18}{'sampled':>11}")
    for fer in (0.05, 0.10, 0.30):
        ch = IndependentFrameChannel(fer)
        lens = run_lengths(ch.errors(SLOTS, rng))
        print(f"{fer:>7g}{ch.mean_burst_slots:>18.6f}{lens.mean():>11.6f}")


if __name__ == "__main__":
    main()
