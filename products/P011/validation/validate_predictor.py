"""Validation 6 — learned predictive controller against the classical baselines.

Everything here is measured on phase-screen realisations the model never saw.
The comparison is deliberately unkind to the learned model:

* the classical integrator's **gain is tuned** on a separate tuning screen at
  every latency and noise level, so it is presented at its best configuration;
* the pure-delay baseline uses the identical pseudo-open-loop control path as
  the learned controller, so the comparison isolates *prediction* from the
  control formulation;
* the final section deliberately breaks the model's assumptions (wind speed and
  photon flux different from training) and reports where it loses.

Run from products/P011:
    PYTHONPATH=src python validation/validate_predictor.py
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from waveforge.control import stability_limit_gain
from waveforge.datasets import make_slope_dataset
from waveforge.loop import AOConfig, AOSystem
from waveforge.predictor import (
    LinearSlopePredictor,
    PureDelayPredictor,
    build_lagged_dataset,
)

BASE = AOConfig(seed=0)
TRAIN_SEEDS = (101, 102, 103)
TEST_SEEDS = (901,)
TUNING_SEED = 555
EVAL_SEEDS = (901, 902)
TRAIN_FRAMES = 400
RUN_FRAMES = 500
WARMUP = 150
N_HISTORY = 4
GAIN_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def tune_gain(system: AOSystem, delay: int, predictor=None) -> tuple[float, float]:
    """Best gain on the tuning screen, for whichever controller is passed."""
    limit = stability_limit_gain(delay)
    best_gain, best_value = None, np.inf
    for gain in GAIN_GRID:
        if gain >= limit:
            continue
        result = system.run(
            300,
            warmup_frames=100,
            rng=4242,
            gain=gain,
            delay_frames=delay,
            predictor=predictor,
        )
        if result.mean_residual_variance < best_value:
            best_gain, best_value = gain, result.mean_residual_variance
    return float(best_gain), float(best_value)


def main() -> None:
    start = time.perf_counter()
    print("=" * 78)
    print("WaveForge validation 6 — learned predictive control vs the baselines")
    print("=" * 78)
    print(f"aperture / r0        : {BASE.diameter_m} m / {BASE.r0_m} m "
          f"(D/r0 = {BASE.d_over_r0:.1f})")
    print(f"wind / frame rate    : {BASE.wind_speed_m_s} m/s / {BASE.frame_rate_hz:.0f} Hz")
    print(f"sensor / mirror      : {BASE.n_sub}x{BASE.n_sub} SH, {BASE.n_act}x{BASE.n_act} DM")
    print(f"training screens     : seeds {list(TRAIN_SEEDS)}, {TRAIN_FRAMES} frames each")
    print(f"held-out screens     : seeds {list(TEST_SEEDS)} (open loop), "
          f"{list(EVAL_SEEDS)} (closed loop)")
    print(f"gain-tuning screen   : seed {TUNING_SEED} (used for no reported number)")
    print("Train and test screens are different atmospheric realisations, never")
    print("different slices of one sequence.")
    print()

    data = make_slope_dataset(
        BASE, n_frames=TRAIN_FRAMES, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS
    )
    print(f"dataset              : {data.n_train_frames} training frames, "
          f"{data.n_test_frames} test frames, {data.n_slopes} slopes")
    print()

    print("--- 1. Open-loop forecast accuracy on held-out screens ---")
    print("Root-mean-square slope error [rad/m] of a forecast h frames ahead.")
    print(f"{'h':>3} {'ridge alpha':>12} {'ML RMSE':>10} {'delay RMSE':>12} "
          f"{'ratio':>8} {'winner':>10}")
    models: dict[int, LinearSlopePredictor] = {}
    for horizon in (1, 2, 3, 4):
        model = LinearSlopePredictor(
            n_history=N_HISTORY, horizon=horizon, n_members=8, random_state=0
        ).fit(data.train)
        models[horizon] = model
        x, y = build_lagged_dataset(data.test, N_HISTORY, horizon)
        mean, _ = model.predict_batch(x)
        delay = x[:, -data.n_slopes :]
        ml_rmse = float(np.sqrt(np.mean((mean - y) ** 2)))
        delay_rmse = float(np.sqrt(np.mean((delay - y) ** 2)))
        winner = "learned" if ml_rmse < delay_rmse else "delay"
        print(
            f"{horizon:>3} {model.chosen_alpha:>12g} {ml_rmse:>10.4f} "
            f"{delay_rmse:>12.4f} {ml_rmse / delay_rmse:>8.3f} {winner:>10}"
        )
    print()

    print("--- 2. Uncertainty calibration on held-out screens ---")
    print("The predictor reports sigma = sqrt(ensemble variance + out-of-bag")
    print("residual variance). Nominal coverage is the Gaussian value; measured")
    print("coverage is the fraction of held-out slopes inside k sigma.")
    print(f"{'h':>3} {'1 sigma':>9} {'(68.3%)':>9} {'2 sigma':>9} {'(95.4%)':>9} "
          f"{'3 sigma':>9} {'(99.7%)':>9} {'mean sigma':>11} {'RMSE':>8}")
    for horizon in (1, 2, 3, 4):
        model = models[horizon]
        x, y = build_lagged_dataset(data.test, N_HISTORY, horizon)
        mean, sigma = model.predict_batch(x)
        z = np.abs(mean - y) / sigma
        print(
            f"{horizon:>3} {float(np.mean(z < 1)) * 100:>8.1f}% {'68.3%':>9} "
            f"{float(np.mean(z < 2)) * 100:>8.1f}% {'95.4%':>9} "
            f"{float(np.mean(z < 3)) * 100:>8.1f}% {'99.7%':>9} "
            f"{float(sigma.mean()):>11.4f} {float(np.sqrt(np.mean((mean - y) ** 2))):>8.4f}"
        )
    print()
    print("Reading: the intervals are NOT calibrated Gaussian sigmas, and they")
    print("miscalibrate in both directions. At h = 1 and h = 2 they are")
    print("conservative (measured coverage above nominal, the safe direction);")
    print("at h = 3 and especially h = 4 they become optimistic - 62% inside one")
    print("sigma at h = 4 against a nominal 68%, with the mean sigma well below")
    print("the actual RMSE. The out-of-bag residual term is estimated on the")
    print("training screens and does not grow enough to cover the extra error")
    print("the model makes further ahead on unseen turbulence. Treat the output")
    print("as a relative confidence signal only, and never as a probability.")
    print()

    print("--- 3. Closed loop: classical integrator vs pure delay vs learned ---")
    print("Gains are tuned separately for each controller on the tuning screen,")
    print("then frozen; the numbers below are from the evaluation screens.")
    tuning_system = AOSystem(replace(BASE, seed=TUNING_SEED))
    eval_systems = [AOSystem(replace(BASE, seed=seed)) for seed in EVAL_SEEDS]
    print(
        f"{'d':>3} {'g_int':>6} {'g_del':>6} {'g_ml':>6} {'integrator':>11} "
        f"{'pure delay':>11} {'learned':>10} {'S_int':>7} {'S_del':>7} {'S_ml':>7} {'winner':>9}"
    )
    for delay in (1, 2, 3, 4):
        model = models[delay]
        g_int, _ = tune_gain(tuning_system, delay)
        g_del, _ = tune_gain(tuning_system, delay, PureDelayPredictor())
        g_ml, _ = tune_gain(tuning_system, delay, model)
        variances, strehls = [], []
        for controller, gain in (
            (None, g_int),
            (PureDelayPredictor(), g_del),
            (model, g_ml),
        ):
            runs = [
                system.run(
                    RUN_FRAMES,
                    warmup_frames=WARMUP,
                    rng=31337,
                    gain=gain,
                    delay_frames=delay,
                    predictor=controller,
                )
                for system in eval_systems
            ]
            variances.append(float(np.mean([r.mean_residual_variance for r in runs])))
            strehls.append(float(np.mean([r.mean_strehl for r in runs])))
        names = ["integrator", "pure delay", "learned"]
        winner = names[int(np.argmin(variances))]
        print(
            f"{delay:>3} {g_int:>6.2f} {g_del:>6.2f} {g_ml:>6.2f} "
            f"{variances[0]:>11.4f} {variances[1]:>11.4f} {variances[2]:>10.4f} "
            f"{strehls[0]:>7.4f} {strehls[1]:>7.4f} {strehls[2]:>7.4f} {winner:>9}"
        )
    print()

    print("--- 4. Failure mode: wind speed different from training ---")
    print(f"The model was trained at {BASE.wind_speed_m_s} m/s only.")
    print(f"{'v [m/s]':>8} {'integrator':>11} {'learned':>10} {'ratio':>8} {'winner':>11}")
    model = models[2]
    for wind in (5.0, 10.0, 15.0, 20.0):
        system = AOSystem(replace(BASE, seed=EVAL_SEEDS[0], wind_speed_m_s=wind))
        frames = min(300, int(system.atmosphere.max_frames) - 1)
        classical = system.run(frames, warmup_frames=100, rng=31337, gain=0.4, delay_frames=2)
        learned = system.run(
            frames, warmup_frames=100, rng=31337, gain=0.4, delay_frames=2, predictor=model
        )
        ratio = learned.mean_residual_variance / classical.mean_residual_variance
        print(
            f"{wind:>8.1f} {classical.mean_residual_variance:>11.4f} "
            f"{learned.mean_residual_variance:>10.4f} {ratio:>8.3f} "
            f"{'learned' if ratio < 1 else 'INTEGRATOR':>11}"
        )
    print()

    print("--- 5. Failure mode: measurement noise, trained clean vs noise-matched ---")
    print("A predictor trained on noise-free slopes and then deployed on a noisy")
    print("sensor amplifies the noise. Retraining at the deployed noise level")
    print("fixes it, and is the only supported way to use the model under noise.")
    print(
        f"{'flux [e-]':>10} {'sigma':>8} {'integrator':>11} {'ML clean':>10} "
        f"{'ML matched':>11} {'winner':>11}"
    )
    for flux in (float("inf"), 1000.0, 300.0, 100.0):
        config = replace(BASE, seed=EVAL_SEEDS[0], photon_flux=flux, read_noise_e=1.0)
        system = AOSystem(config)
        sigma = system.sensor.slope_noise_sigma()
        clean_model = models[2]
        if sigma > 0.0:
            noisy_data = make_slope_dataset(
                BASE,
                n_frames=TRAIN_FRAMES,
                train_seeds=TRAIN_SEEDS,
                test_seeds=TEST_SEEDS,
                noise_sigma=sigma,
                noise_seed=8888,
            )
            matched = LinearSlopePredictor(
                n_history=N_HISTORY, horizon=2, n_members=8, random_state=0
            ).fit(noisy_data.train)
        else:
            matched = clean_model
        g_int, _ = tune_gain(AOSystem(replace(config, seed=TUNING_SEED)), 2)
        results = {}
        for label, controller, gain in (
            ("int", None, g_int),
            ("clean", clean_model, 0.4),
            ("matched", matched, 0.4),
        ):
            results[label] = system.run(
                RUN_FRAMES,
                warmup_frames=WARMUP,
                rng=31337,
                gain=gain,
                delay_frames=2,
                predictor=controller,
            ).mean_residual_variance
        best = min(results, key=lambda k: results[k])
        label = {"int": "INTEGRATOR", "clean": "ML clean", "matched": "ML matched"}[best]
        print(
            f"{flux:>10.0f} {sigma:>8.2f} {results['int']:>11.4f} "
            f"{results['clean']:>10.4f} {results['matched']:>11.4f} {label:>11}"
        )
    print()
    print(f"elapsed: {time.perf_counter() - start:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
