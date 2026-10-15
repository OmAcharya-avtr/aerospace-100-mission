"""The learned predictor against the analytic model, and interval coverage.

Level 3 check 4. Four measurements:

1. **Accuracy against the analytic baseline** on held-out activities, for both
   targets, over several dataset seeds, with the paired difference reported.
2. **The irreducible noise floor**, so the win is readable against what is
   achievable rather than against zero. The floor is computed two ways: in
   closed form for the duration, where the generator's noise is
   ``Normal(0, 0.08 * d_a)`` and the mean absolute deviation of a centred normal
   is ``sigma * sqrt(2/pi)``; and by Monte Carlo over repeated realisations for
   both targets. The two must agree for the duration.
3. **Marginal coverage** of the split-conformal interval, both the absolute and
   the normalised variant, at three levels, over seeds, with the spread shown.
4. **Conditional coverage by duration tercile**, which is where the absolute
   variant is expected to fail: the generator's noise is heteroscedastic, so a
   constant-width interval must undercover long activities and overcover short
   ones. The normalised variant exists to fix exactly that, and the measured
   improvement -- or its absence -- is the result.

The honest framing, repeated from activityplan.features: the learned model beats
the analytic model because the analytic model omits two terms the generator
contains, and it is handed the analytic prediction as a feature. What is
measured here without that caveat is the interval's coverage, which is a
property of the conformal construction and of exchangeability.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

add_src_to_path()

from activityplan.features import (  # noqa: E402
    REFERENCE_TEMPERATURE_C,
    analytic_duration_s,
    feature_matrix,
    realise,
    sample_requests,
)
from activityplan.predictor import (  # noqa: E402
    DurationEnergyPredictor,
    analytic_predictions,
    coverage,
    mean_interval_width,
)

OUTPUT = Path(__file__).with_name("validate_predictor_output.txt")

SEEDS = (61, 62, 63, 64, 65)
N_TOTAL = 1400
N_TEST = 400
N_ESTIMATORS = 160
ALPHAS = (0.05, 0.10, 0.20)
DURATION_NOISE_FRACTION = 0.08


def _dataset(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    requests = sample_requests(N_TOTAL, rng)
    realised = [realise(request, rng) for request in requests]
    features = feature_matrix(list(requests))
    durations = np.array([item.duration_s for item in realised], dtype=float)
    energies = np.array([item.energy_wh for item in realised], dtype=float)
    return requests, features, durations, energies


def _mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(truth - prediction)))


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_predictor.py")
    say("learned duration and energy predictor against the analytic model, and coverage")
    for line in environment_lines():
        say(line)
    say(f"dataset          : {N_TOTAL} activities per seed, {N_TEST} held out")
    say(f"seeds            : {SEEDS}")
    say(f"forest           : {N_ESTIMATORS} trees, min_samples_leaf 2, n_jobs 1")

    say.rule("check 1: accuracy against the analytic model on held-out activities")
    say("the learned model is given the analytic prediction as a feature, so it fits the")
    say("correction, and it wins because the analytic model omits the generator's")
    say("cold-settling, gain and heater terms. This is misspecification, measured.")
    say("")
    say(
        f"{'seed':>5s} {'target':12s} {'analytic MAE':>13s} {'learned MAE':>12s} "
        f"{'MAE ratio':>10s} {'analytic RMSE':>14s} {'learned RMSE':>13s} {'fit s':>7s}"
    )
    duration_ratios: list[float] = []
    energy_ratios: list[float] = []
    learned_duration_mae: list[float] = []
    analytic_duration_mae: list[float] = []
    learned_energy_mae: list[float] = []
    analytic_energy_mae: list[float] = []
    for seed in SEEDS:
        requests, features, durations, energies = _dataset(seed)
        split = N_TOTAL - N_TEST
        t0 = time.perf_counter()
        predictor = DurationEnergyPredictor(
            alpha=0.10, mode="normalised", n_estimators=N_ESTIMATORS, random_state=seed
        ).fit(features[:split], durations[:split], energies[:split])
        fit_s = time.perf_counter() - t0
        learned_duration = predictor.duration.predict_point(features[split:])
        learned_energy = predictor.energy.predict_point(features[split:])
        analytic_duration, analytic_energy = analytic_predictions(requests[split:])
        for label, truth, analytic, learned in (
            ("duration [s]", durations[split:], analytic_duration, learned_duration),
            ("energy [W*h]", energies[split:], analytic_energy, learned_energy),
        ):
            a_mae = _mae(truth, analytic)
            l_mae = _mae(truth, learned)
            say(
                f"{seed:5d} {label:12s} {a_mae:13.5f} {l_mae:12.5f} {l_mae / a_mae:10.4f} "
                f"{_rmse(truth, analytic):14.5f} {_rmse(truth, learned):13.5f} {fit_s:7.2f}"
            )
            if label.startswith("duration"):
                duration_ratios.append(l_mae / a_mae)
                learned_duration_mae.append(l_mae)
                analytic_duration_mae.append(a_mae)
            else:
                energy_ratios.append(l_mae / a_mae)
                learned_energy_mae.append(l_mae)
                analytic_energy_mae.append(a_mae)
    say("")
    say(
        f"duration MAE ratio learned/analytic : mean {np.mean(duration_ratios):.4f}, "
        f"range {np.min(duration_ratios):.4f} to {np.max(duration_ratios):.4f}"
    )
    say(
        f"energy   MAE ratio learned/analytic : mean {np.mean(energy_ratios):.4f}, "
        f"range {np.min(energy_ratios):.4f} to {np.max(energy_ratios):.4f}"
    )
    say("criterion                           : ratio < 1 on every seed for both targets")
    beat = all(ratio < 1.0 for ratio in duration_ratios + energy_ratios)
    say(f"verdict                             : {'PASS' if beat else 'FAIL'}")

    say.rule("check 2: the irreducible noise floor, so the win is readable")
    say("duration closed form: the generator adds Normal(0, 0.08 * analytic_duration), and")
    say("the mean absolute deviation of a centred normal is sigma * sqrt(2/pi), so the")
    say("best achievable MAE for any predictor of the duration is the mean of that over")
    say("the test set. The Monte Carlo estimate repeats the generator 200 times per")
    say("request and scores against the per-request sample mean, which also covers the")
    say("floor applied at 15 s and the energy target, where the noise enters as a product.")
    say("")
    requests, features, durations, energies = _dataset(SEEDS[0])
    split = N_TOTAL - N_TEST
    test_requests = requests[split:]
    closed_form = float(
        np.mean(
            [
                DURATION_NOISE_FRACTION * analytic_duration_s(r) * math.sqrt(2.0 / math.pi)
                for r in test_requests
            ]
        )
    )
    subset = test_requests[:150]
    rng = np.random.default_rng(777)
    duration_floor = []
    energy_floor = []
    for request in subset:
        draws = [realise(request, rng) for _ in range(200)]
        d = np.array([item.duration_s for item in draws], dtype=float)
        e = np.array([item.energy_wh for item in draws], dtype=float)
        duration_floor.append(float(np.mean(np.abs(d - d.mean()))))
        energy_floor.append(float(np.mean(np.abs(e - e.mean()))))
    mc_duration = float(np.mean(duration_floor))
    mc_energy = float(np.mean(energy_floor))
    say(f"duration floor, closed form     : {closed_form:.5f} s")
    say(f"duration floor, Monte Carlo     : {mc_duration:.5f} s over {len(subset)} requests")
    say(f"  relative difference           : {abs(mc_duration - closed_form) / closed_form:.4f}")
    say(f"energy   floor, Monte Carlo     : {mc_energy:.5f} W*h")
    say("")
    mean_learned_duration = float(np.mean(learned_duration_mae))
    mean_analytic_duration = float(np.mean(analytic_duration_mae))
    mean_learned_energy = float(np.mean(learned_energy_mae))
    mean_analytic_energy = float(np.mean(analytic_energy_mae))
    say(
        f"duration : analytic {mean_analytic_duration:.5f} s, learned "
        f"{mean_learned_duration:.5f} s, floor {closed_form:.5f} s"
    )
    recovered_duration = (mean_analytic_duration - mean_learned_duration) / max(
        mean_analytic_duration - closed_form, 1e-12
    )
    say(
        f"  the learned model closes {100.0 * recovered_duration:.1f} % of the gap between "
        "the analytic model and the floor"
    )
    say(
        f"energy   : analytic {mean_analytic_energy:.5f} W*h, learned "
        f"{mean_learned_energy:.5f} W*h, floor {mc_energy:.5f} W*h"
    )
    recovered_energy = (mean_analytic_energy - mean_learned_energy) / max(
        mean_analytic_energy - mc_energy, 1e-12
    )
    say(
        f"  the learned model closes {100.0 * recovered_energy:.1f} % of the gap between "
        "the analytic model and the floor"
    )

    say.rule("check 3: marginal coverage of the split-conformal interval")
    say("the guarantee is marginal, distribution-free and finite-sample under")
    say("exchangeability (Lei et al. 2018). Measured on the held-out set, per seed.")
    say("")
    say(
        f"{'alpha':>6s} {'mode':11s} {'target':12s} {'nominal':>8s} {'mean cov':>9s} "
        f"{'min cov':>8s} {'max cov':>8s} {'mean width':>11s}"
    )
    coverage_failures = 0
    coverage_rows: dict[tuple[float, str, str], tuple[float, float]] = {}
    for alpha in ALPHAS:
        for mode in ("absolute", "normalised"):
            per_target: dict[str, list[tuple[float, float]]] = {
                "duration [s]": [],
                "energy [W*h]": [],
            }
            for seed in SEEDS:
                requests, features, durations, energies = _dataset(seed)
                predictor = DurationEnergyPredictor(
                    alpha=alpha, mode=mode, n_estimators=N_ESTIMATORS, random_state=seed
                ).fit(features[:split], durations[:split], energies[:split])
                duration_interval = predictor.duration.predict(features[split:])
                energy_interval = predictor.energy.predict(features[split:])
                per_target["duration [s]"].append(
                    (
                        coverage(durations[split:], duration_interval),
                        mean_interval_width(duration_interval),
                    )
                )
                per_target["energy [W*h]"].append(
                    (
                        coverage(energies[split:], energy_interval),
                        mean_interval_width(energy_interval),
                    )
                )
            for target, pairs in per_target.items():
                covers = np.array([p[0] for p in pairs], dtype=float)
                widths = np.array([p[1] for p in pairs], dtype=float)
                say(
                    f"{alpha:6.2f} {mode:11s} {target:12s} {1.0 - alpha:8.4f} "
                    f"{float(np.mean(covers)):9.4f} {float(np.min(covers)):8.4f} "
                    f"{float(np.max(covers)):8.4f} {float(np.mean(widths)):11.4f}"
                )
                coverage_rows[(alpha, mode, target)] = (
                    float(np.mean(covers)),
                    float(np.mean(widths)),
                )
                # Binomial standard error at n = N_TEST, three of them as the band.
                band = 3.0 * math.sqrt(alpha * (1.0 - alpha) / N_TEST)
                if float(np.mean(covers)) < (1.0 - alpha) - band:
                    coverage_failures += 1
    say("")
    say(
        "criterion : mean coverage not below nominal by more than three binomial standard"
    )
    say(f"            errors at n = {N_TEST}, which is 3*sqrt(a(1-a)/n)")
    say(f"rows outside that band : {coverage_failures} of {len(coverage_rows)}")
    say(f"verdict                : {'FAIL' if coverage_failures else 'PASS'}")

    say.rule("check 4: conditional coverage by duration tercile")
    say("the generator's duration noise scales with the analytic duration, so a")
    say("constant-width interval must undercover long activities. This is the")
    say("measurement the normalised variant exists for.")
    say("")
    say(
        f"{'mode':11s} {'tercile':10s} {'mean cov':>9s} {'min cov':>8s} {'max cov':>8s} "
        f"{'mean width':>11s} {'dev from nominal':>17s}"
    )
    alpha = 0.10
    worst_absolute_deviation = 0.0
    worst_normalised_deviation = 0.0
    for mode in ("absolute", "normalised"):
        per_tercile: dict[str, list[tuple[float, float]]] = {
            "short": [],
            "middle": [],
            "long": [],
        }
        for seed in SEEDS:
            requests, features, durations, energies = _dataset(seed)
            predictor = DurationEnergyPredictor(
                alpha=alpha, mode=mode, n_estimators=N_ESTIMATORS, random_state=seed
            ).fit(features[:split], durations[:split], energies[:split])
            interval = predictor.duration.predict(features[split:])
            analytic = np.array(
                [analytic_duration_s(r) for r in requests[split:]], dtype=float
            )
            lower_cut, upper_cut = np.percentile(analytic, [100.0 / 3.0, 200.0 / 3.0])
            masks = {
                "short": analytic <= lower_cut,
                "middle": (analytic > lower_cut) & (analytic <= upper_cut),
                "long": analytic > upper_cut,
            }
            for label, mask in masks.items():
                from activityplan.predictor import Interval

                sliced = Interval(
                    point=interval.point[mask],
                    lower=interval.lower[mask],
                    upper=interval.upper[mask],
                )
                per_tercile[label].append(
                    (
                        coverage(durations[split:][mask], sliced),
                        mean_interval_width(sliced),
                    )
                )
        for label, pairs in per_tercile.items():
            covers = np.array([p[0] for p in pairs], dtype=float)
            widths = np.array([p[1] for p in pairs], dtype=float)
            deviation = float(np.mean(covers)) - (1.0 - alpha)
            if mode == "absolute":
                worst_absolute_deviation = min(worst_absolute_deviation, deviation)
            else:
                worst_normalised_deviation = min(worst_normalised_deviation, deviation)
            say(
                f"{mode:11s} {label:10s} {float(np.mean(covers)):9.4f} "
                f"{float(np.min(covers)):8.4f} {float(np.max(covers)):8.4f} "
                f"{float(np.mean(widths)):11.4f} {deviation:+17.4f}"
            )
    say("")
    say(f"worst tercile shortfall, absolute mode   : {worst_absolute_deviation:+.4f}")
    say(f"worst tercile shortfall, normalised mode : {worst_normalised_deviation:+.4f}")
    say("this check has no pass criterion: both numbers are reported because conditional")
    say("coverage is not guaranteed by either construction, and the comparison between")
    say("them is the result.")

    failed = bool(coverage_failures) or not beat
    say.rule("summary")
    say(f"duration MAE ratio learned/analytic   mean {np.mean(duration_ratios):.4f}")
    say(f"energy   MAE ratio learned/analytic   mean {np.mean(energy_ratios):.4f}")
    say(f"duration irreducible floor            {closed_form:.5f} s (closed form)")
    say(
        "marginal coverage at alpha 0.10       "
        f"absolute {coverage_rows[(0.10, 'absolute', 'duration [s]')][0]:.4f}, "
        f"normalised {coverage_rows[(0.10, 'normalised', 'duration [s]')][0]:.4f}"
    )
    say(
        "mean interval width at alpha 0.10     "
        f"absolute {coverage_rows[(0.10, 'absolute', 'duration [s]')][1]:.4f} s, "
        f"normalised {coverage_rows[(0.10, 'normalised', 'duration [s]')][1]:.4f} s"
    )
    say(f"worst tercile shortfall absolute      {worst_absolute_deviation:+.4f}")
    say(f"worst tercile shortfall normalised    {worst_normalised_deviation:+.4f}")
    say(f"wall clock                            {time.perf_counter() - started:.1f} s")
    say(f"overall                               {'FAIL' if failed else 'PASS'}")
    say(f"raw output written to                 {OUTPUT.name}")
    say(f"reference temperature used by the generator: {REFERENCE_TEMPERATURE_C:.1f} C")
    say.save()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
