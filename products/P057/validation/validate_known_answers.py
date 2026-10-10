"""Known answers: the exact statements, confirmed by Monte Carlo and by hand.

Every expected value below is derived in a comment and asserted against the
measured one. Exits 0.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from conformalband.bounds import (
    clopper_pearson,
    effective_sample_size,
    split_conformal_coverage_bound,
)
from conformalband.conformal import (
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    conformal_rank,
    weighted_quantile,
)
from conformalband.data import make_dataset
from conformalband.physics import leg_energy, level_flight_power
from conformalband.shift import (
    CALIBRATION_HEADWIND_MEAN,
    CALIBRATION_HEADWIND_SD,
    CALIBRATION_MASS_MEAN,
    CALIBRATION_MASS_SD,
    CovariateShift,
)

REPLICATES = 400_000
SEED = 57101

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def empirical_split_coverage(n: int, alpha: float, seed: int, law: str) -> float:
    """Pooled coverage of split conformal on exchangeable scores."""
    rng = np.random.default_rng(seed)
    if law == "uniform":
        scores = rng.random((REPLICATES, n + 1))
    elif law == "lognormal":
        scores = rng.lognormal(0.0, 1.0, (REPLICATES, n + 1))
    else:
        scores = rng.standard_cauchy((REPLICATES, n + 1))
        scores = np.abs(scores)
    rank = conformal_rank(n, alpha)
    quantile = np.partition(scores[:, :n], rank - 1, axis=1)[:, rank - 1]
    return float(np.mean(scores[:, n] <= quantile))


def section_bound() -> None:
    print("1. finite-sample coverage bound, k = ceil((n+1)(1-alpha)), exact = k/(n+1)")
    header = f"{'n':>7s} {'alpha':>6s} {'k':>5s} {'exact':>15s} {'hand':>15s} {'gap':>14s}"
    print(header)
    print("  " + "-" * len(header))
    # Hand arithmetic:
    #   n=19,  a=0.10: 20*0.90 = 18      -> k=18, 18/20  = 0.900000000000
    #   n=18,  a=0.10: 19*0.90 = 17.1    -> k=18, 18/19  = 0.947368421053
    #   n=9,   a=0.10: 10*0.90 = 9       -> k=9,   9/10  = 0.900000000000
    #   n=100, a=0.10: 101*0.90 = 90.9   -> k=91, 91/101 = 0.900990099010
    #   n=500, a=0.10: 501*0.90 = 450.9  -> k=451, 451/501 = 0.900199600798
    #   n=99,  a=0.05: 100*0.95 = 95     -> k=95, 95/100 = 0.950000000000
    hand = [
        (19, 0.10, 18, 18 / 20),
        (18, 0.10, 18, 18 / 19),
        (9, 0.10, 9, 9 / 10),
        (100, 0.10, 91, 91 / 101),
        (500, 0.10, 451, 451 / 501),
        (99, 0.05, 95, 95 / 100),
    ]
    ok = True
    for n, alpha, rank, exact in hand:
        bound = split_conformal_coverage_bound(n, alpha)
        print(
            f"  {n:7d} {alpha:6.2f} {bound.rank:5d} {bound.exact:15.12f} {exact:15.12f} "
            f"{bound.conservatism:14.12f}"
        )
        ok = ok and bound.rank == rank and abs(bound.exact - exact) < 1e-15
    record("every bound equals the hand arithmetic to 1e-15", ok)
    record(
        "n=8 with alpha=0.1 is refused because k=9 > 8",
        _raises(lambda: split_conformal_coverage_bound(8, 0.1)),
    )
    print()


def _raises(call) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def section_monte_carlo() -> None:
    print("2. Monte-Carlo confirmation of the exact coverage, exchangeable scores")
    print(f"   {REPLICATES} replicates per row, standard error about "
          f"{math.sqrt(0.9 * 0.1 / REPLICATES):.6f}")
    header = f"{'n':>5s} {'alpha':>6s} {'law':>10s} {'exact':>15s} {'measured':>15s} {'error':>12s}"
    print(header)
    print("  " + "-" * len(header))
    tolerance = 4e-3
    ok = True
    for n, alpha in ((19, 0.1), (18, 0.1), (100, 0.1), (99, 0.05)):
        exact = split_conformal_coverage_bound(n, alpha).exact
        for law in ("uniform", "lognormal", "halfcauchy"):
            measured = empirical_split_coverage(n, alpha, SEED + n, law)
            print(
                f"  {n:5d} {alpha:6.2f} {law:>10s} {exact:15.12f} {measured:15.12f} "
                f"{measured - exact:+12.9f}"
            )
            ok = ok and abs(measured - exact) < tolerance
    record(f"every measured coverage is within {tolerance} of k/(n+1), any score law", ok)
    separation = empirical_split_coverage(18, 0.1, SEED + 18, "uniform") - empirical_split_coverage(
        19, 0.1, SEED + 19, "uniform"
    )
    print(f"  coverage at n=18 minus coverage at n=19 : {separation:+.9f}")
    print(f"  hand value 18/19 - 18/20                : {18 / 19 - 18 / 20:+.9f}")
    record(
        "dropping one calibration point raises coverage by 18/19 - 18/20 = 0.047368421",
        abs(separation - (18 / 19 - 18 / 20)) < 2 * tolerance,
    )
    print()


def section_weighted_quantile() -> None:
    print("3. the weighted quantile reduces to the order statistic")
    values = np.arange(1.0, 25.0)
    got = weighted_quantile(values, np.ones(24), 1.0 - 0.44, tail_weight=1.0)
    print(f"  n=24, alpha=0.44: 25*(1-0.44) = {25 * (1 - 0.44):.17f}")
    print(f"  unguarded ceil                 : {math.ceil(25 * (1 - 0.44))}")
    print(f"  conformal_rank (guarded)       : {conformal_rank(24, 0.44)}")
    print(f"  weighted_quantile              : {got}")
    record("the floating-point ceiling guard selects rank 14, not 15", got == 14.0)

    rng = np.random.default_rng(SEED + 1)
    y = rng.normal(3.0, 0.3, 500)
    p = rng.normal(3.0, 0.3, 500)
    split = SplitConformal(0.1).calibrate(y, p)
    weighted = WeightedConformal(0.1).calibrate(y, p, np.ones(500))
    mondrian = MondrianConformal(0.1).calibrate(y, p, np.zeros(500, dtype=int))
    print(f"  split quantile                 : {split.quantile:.15f}")
    print(f"  weighted with unit weights     : {weighted.quantiles(np.ones(1))[0]:.15f}")
    print(f"  mondrian with one bin          : {mondrian.quantiles[0]:.15f}")
    record(
        "weighted with unit weights equals split exactly",
        weighted.quantiles(np.ones(1))[0] == split.quantile,
    )
    record("mondrian with one bin equals split exactly", mondrian.quantiles[0] == split.quantile)

    zero_shift = CovariateShift(severity=0.0)
    mass = rng.normal(CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD, 500)
    wind = rng.normal(CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD, 500)
    unshifted = WeightedConformal(0.1).calibrate(y, p, zero_shift.likelihood_ratio(mass, wind))
    record(
        "weighted conformal at severity 0 equals split exactly",
        unshifted.quantiles(np.ones(1))[0] == split.quantile,
    )
    print()


def section_likelihood_ratio() -> None:
    print("4. the declared likelihood ratio against the Gaussian densities")
    header = (
        f"{'severity':>9s} {'mahalanobis':>12s} {'max rel err':>13s} {'E[w]':>10s} "
        f"{'E[w^2]':>11s} {'exp(M^2)':>11s}"
    )
    print(header)
    print("  " + "-" * len(header))
    rng = np.random.default_rng(SEED + 2)
    mass = rng.normal(CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD, 400_000)
    wind = rng.normal(CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD, 400_000)
    ok_ratio = True
    ok_mean = True
    for severity in (0.0, 1.0, 2.0, 3.0):
        shift = CovariateShift(severity=severity)
        numerator = stats.norm.pdf(mass, shift.mass_mean, CALIBRATION_MASS_SD) * stats.norm.pdf(
            wind, shift.headwind_mean, CALIBRATION_HEADWIND_SD
        )
        denominator = stats.norm.pdf(
            mass, CALIBRATION_MASS_MEAN, CALIBRATION_MASS_SD
        ) * stats.norm.pdf(wind, CALIBRATION_HEADWIND_MEAN, CALIBRATION_HEADWIND_SD)
        weights = shift.likelihood_ratio(mass, wind)
        relative = float(np.max(np.abs(weights / (numerator / denominator) - 1.0)))
        first = float(np.mean(weights))
        second = float(np.mean(weights**2))
        print(
            f"  {severity:9.1f} {shift.mahalanobis:12.6f} {relative:13.3e} {first:10.6f} "
            f"{second:11.6f} {math.exp(shift.mahalanobis**2):11.6f}"
        )
        ok_ratio = ok_ratio and relative < 1e-12
        ok_mean = ok_mean and abs(first - 1.0) < 0.01
    record("closed-form log ratio matches the density ratio to 1e-12 relative", ok_ratio)
    record("E_cal[w(X)] = 1 to within 0.01 at every severity", ok_mean)
    print()


def section_physics() -> None:
    print("5. the energy model, by hand")
    # V = 20 m/s, m = 6 kg, rho = 1.18 kg/m^3, constant efficiency.
    #   P_parasite = 0.5 * 1.18 * 8000 * 0.30 * 0.035          =  49.560000000000 W
    #   W = 6 * 9.80665 = 58.839900000000 N, W^2 = 3462.133832010000 N^2
    #   denominator = 1.18 * 20 * pi * 1.44 * 0.85             =  90.749302028656
    #   P_induced  = 2 * 3462.133832010000 / 90.749302028656   =  76.301056969380 W
    #   P_total    = (49.56 + 76.301056969380) / 0.62 + 12     = 215.001704789323 W
    #   t = 1000 / 20 = 50 s, E = 215.001704789323 * 50 / 3600 =   2.986134788741 Wh
    power = float(level_flight_power(20.0, 6.0, 1.18, constant_efficiency=True))
    energy = float(leg_energy(20.0, 6.0, 1.18, 0.0, 1000.0, constant_efficiency=True))
    print(f"  P_total  measured : {power:.12f} W   hand : 215.001704789323 W")
    print(f"  E_leg    measured : {energy:.12f} Wh  hand :   2.986134788741 Wh")
    record(
        "level flight power matches the hand value to 1e-12 relative",
        abs(power / 215.001704789323 - 1.0) < 1e-12,
    )
    record(
        "leg energy matches the hand value to 1e-12 relative",
        abs(energy / 2.986134788741 - 1.0) < 1e-12,
    )
    doubled = float(leg_energy(20.0, 6.0, 1.18, 0.0, 2000.0, constant_efficiency=True))
    print(f"  doubling the distance : {doubled:.12f} Wh = {doubled / energy:.15f} x")
    record("energy is exactly linear in distance", abs(doubled / energy - 2.0) < 1e-14)
    print()


def section_diagnostics() -> None:
    print("6. diagnostics, by hand")
    print(f"  ESS of 37 equal weights          : {effective_sample_size(np.full(37, 2.5)):.12f}")
    pair = effective_sample_size(np.array([1.0, 3.0]))
    print(f"  ESS of weights [1, 3]            : {pair:.12f}")
    one_hot = np.zeros(50)
    one_hot[7] = 3.0
    print(f"  ESS of a single non-zero weight  : {effective_sample_size(one_hot):.12f}")
    record(
        "ESS of n equal weights is n",
        abs(effective_sample_size(np.full(37, 2.5)) - 37.0) < 1e-12,
    )
    record(
        "ESS of [1, 3] is (1+3)^2/(1+9) = 1.6",
        abs(pair - 1.6) < 1e-15,
    )
    record("ESS of one non-zero weight is 1", abs(effective_sample_size(one_hot) - 1.0) < 1e-12)
    low, high = clopper_pearson(0, 20)
    print(f"  Clopper-Pearson(0, 20)           : [{low:.12f}, {high:.12f}]")
    print(f"  hand upper limit 1 - 0.025^(1/20): {1.0 - 0.025 ** (1 / 20):.12f}")
    record(
        "Clopper-Pearson(0, 20) upper limit matches 1 - 0.025^(1/20)",
        abs(high - (1.0 - 0.025 ** (1 / 20))) < 1e-10,
    )
    print()


def section_dataset() -> None:
    print("7. dataset statistics quoted in DATASET_CARD.md")
    # The observation noise is multiplicative: e = truth * 0.06 * z with
    # z ~ N(0, 1) independent of truth, so
    #   sd(e) = 0.06 * sqrt(E[truth^2]) = 0.06 * sqrt(mean^2 + sd^2).
    # That identity is the known answer; the three statistics are measured.
    data = make_dataset(200_000, seed=1)
    mean = float(np.mean(data.truth))
    sd = float(np.std(data.truth))
    noise_sd = float(np.std(data.energy - data.truth))
    predicted = 0.06 * math.sqrt(mean**2 + sd**2)
    print("  make_dataset(200000, seed=1), calibration distribution")
    print(f"  mean noiseless energy        : {mean:.6f} Wh")
    print(f"  sd across covariate draws    : {sd:.6f} Wh")
    print(f"  observation-noise sd         : {noise_sd:.6f} Wh")
    print(f"  0.06 * sqrt(mean^2 + sd^2)   : {predicted:.6f} Wh")
    record(
        "the multiplicative-noise identity sd(e) = 0.06 sqrt(E[truth^2]) holds to 1 per cent",
        abs(noise_sd / predicted - 1.0) < 0.01,
    )
    record("mean noiseless energy is between 3.0 and 3.2 Wh", 3.0 < mean < 3.2)
    print()


def main() -> None:
    print("known-answer validation for conformalband 0.1.0")
    print(f"seed {SEED}, {REPLICATES} Monte-Carlo replicates where used")
    print()
    section_bound()
    section_monte_carlo()
    section_weighted_quantile()
    section_likelihood_ratio()
    section_physics()
    section_diagnostics()
    section_dataset()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (these are findings, recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("ALL KNOWN-ANSWER CHECKS PASSED")


if __name__ == "__main__":
    main()
