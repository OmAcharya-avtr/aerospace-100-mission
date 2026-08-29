"""Validation 5: learned predictive controller vs the classical baselines.

Two independent comparisons, both on held-out screens the models never saw:

1. **Open-loop forecast accuracy.** RMS error of the forecast slope vector
   against the *noise-free* slopes of the same screens (so measurement noise on
   the target does not inflate every method equally).
2. **Closed-loop performance.** Residual phase variance and exact Strehl ratio
   with (a) the classical integrator at its best gain, (b) pseudo-open-loop
   control with a pure-delay forecast, (c) the same controller driven by the
   linear auto-regressive predictor, and (d) by the learned MLP ensemble.

Also reported: the calibration of the ensemble's uncertainty output, and the
wall-clock training cost.

Run from ``products/P011/``:

    PYTHONPATH=src python validation/validate_predictor.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from waveforge.dataset import generate_dataset
from waveforge.loop import run_closed_loop
from waveforge.predictor import (
    EnsemblePredictor,
    LinearPredictor,
    PersistencePredictor,
    build_windows,
)
from waveforge.presets import ReferenceConfig, build_flow, build_system

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- experiment
N_HISTORY = 3
N_COMPONENTS = 32
TRAIN_SEEDS = tuple(range(1, 9))
TEST_SEEDS = (50, 51, 52, 53)
N_FRAMES = 600
N_PHOTONS = 200.0
READ_NOISE = 1.0
LATENCIES = (1, 2, 3)
BOILING = 0.2
INTEGRATOR_GAINS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8)
POLC_GAINS = (0.3, 0.5, 0.7, 0.9, 1.0)


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main() -> None:
    cfg = ReferenceConfig(boiling=BOILING)
    system = build_system(cfg)
    print("=" * 78)
    print("WaveForge validation 5 -- predictive control vs classical baselines")
    print("=" * 78)
    print(f"config: D={cfg.diameter} m, lambda={cfg.wavelength * 1e9:.0f} nm, "
          f"r0={cfg.r0} m, D/r0={cfg.d_over_r0:.1f}")
    print(f"        v={cfg.wind_speed} m/s, f_s={cfg.frame_rate:.0f} Hz, "
          f"boiling={BOILING}, tau0={cfg.coherence_time * 1e3:.2f} ms")
    print(f"        {system.sensor.n_valid} subapertures -> {system.sensor.n_slopes} slopes, "
          f"{system.dm.n_actuators} actuators, {system.n_modes_kept} modes kept")
    print(f"        photons={N_PHOTONS:.0f} e-/subap/frame, read noise={READ_NOISE} e-/px, "
          f"slope noise sigma="
          f"{np.sqrt(system.sensor.noise_variance(N_PHOTONS, READ_NOISE)):.3f} rad/m")
    print(f"        train seeds {TRAIN_SEEDS}, test seeds {TEST_SEEDS}, "
          f"{N_FRAMES} frames each")

    t0 = time.time()
    train = generate_dataset(
        system.sensor, cfg, TRAIN_SEEDS, N_FRAMES, n_photons=N_PHOTONS, read_noise=READ_NOISE
    )
    test = generate_dataset(
        system.sensor, cfg, TEST_SEEDS, N_FRAMES, n_photons=N_PHOTONS, read_noise=READ_NOISE
    )
    truth = generate_dataset(system.sensor, cfg, TEST_SEEDS, N_FRAMES, n_photons=None)
    print(f"        data generated in {time.time() - t0:.1f} s "
          f"({len(train)} train x {N_FRAMES} frames)")

    print()
    print("-" * 78)
    print("5.1  Open-loop forecast RMS error [rad/m], scored against noise-free truth")
    print("-" * 78)
    print(f"{'L':>2} {'persistence':>12} {'linear AR':>11} {'MLP(pers)':>11} "
          f"{'MLP(linear)':>12} {'train [s]':>10}")

    models: dict[int, dict[str, object]] = {}
    forecast_rows = []
    for lat in LATENCIES:
        x_te, _ = build_windows(test, N_HISTORY, lat)
        _, y_true = build_windows(truth, N_HISTORY, lat)

        pers = PersistencePredictor(horizon=lat).fit(train)
        m_p = x_te[:, -1, :]

        lin = LinearPredictor(
            n_history=N_HISTORY, horizon=lat, n_components=N_COMPONENTS
        ).fit(train)
        m_l = lin.predict_many(x_te)[0]

        t0 = time.time()
        ens_p = EnsemblePredictor(
            n_history=N_HISTORY, horizon=lat, n_components=N_COMPONENTS,
            base="persistence", random_state=11,
        ).fit(train)
        ens_l = EnsemblePredictor(
            n_history=N_HISTORY, horizon=lat, n_components=N_COMPONENTS,
            base="linear", random_state=11,
        ).fit(train)
        t_train = time.time() - t0
        m_ep = ens_p.predict_many(x_te)[0]
        m_el, s_el = ens_l.predict_many(x_te)

        row = (rms(y_true - m_p), rms(y_true - m_l), rms(y_true - m_ep), rms(y_true - m_el))
        forecast_rows.append((lat, *row, t_train))
        print(f"{lat:>2} {row[0]:12.4f} {row[1]:11.4f} {row[2]:11.4f} {row[3]:12.4f} "
              f"{t_train:10.1f}")
        models[lat] = {
            "persistence": pers, "linear": lin, "mlp_pers": ens_p, "mlp_lin": ens_l,
            "err": np.abs(y_true - m_el), "std": s_el,
        }

    print()
    print("  gain of each method over the pure-delay baseline (>1 means better):")
    for lat, p, ln, ep, el, _ in forecast_rows:
        print(f"    L={lat}: linear {p / ln:6.3f}x   MLP(pers) {p / ep:6.3f}x   "
              f"MLP(linear) {p / el:6.3f}x   MLP(linear) vs linear {ln / el:6.3f}x")

    print()
    print("-" * 78)
    print("5.2  Uncertainty calibration of the learned ensemble, MLP(linear)")
    print("-" * 78)
    print(f"{'L':>2} {'mean|err|':>10} {'mean sigma':>11} {'ratio':>8} "
          f"{'cov 1sig':>9} {'cov 2sig':>9}")
    for lat in LATENCIES:
        err = np.asarray(models[lat]["err"])
        sd = np.asarray(models[lat]["std"])
        cov1 = float(np.mean(err <= sd))
        cov2 = float(np.mean(err <= 2.0 * sd))
        print(f"{lat:>2} {err.mean():10.4f} {sd.mean():11.4f} {sd.mean() / err.mean():8.3f} "
              f"{cov1:9.3f} {cov2:9.3f}")
    print("  reference for a correctly calibrated Gaussian: 0.683 and 0.954")

    print()
    print("-" * 78)
    print("5.3  Closed loop on the held-out screens: residual variance / Strehl")
    print("-" * 78)

    def loop_stats(controller, gain, lat, predictor=None):
        variances, strehls = [], []
        for seed in TEST_SEEDS:
            flow = build_flow(cfg, seed=seed)
            res = run_closed_loop(
                system, flow, N_FRAMES, gain=gain, latency=lat,
                n_photons=N_PHOTONS, read_noise=READ_NOISE,
                controller=controller, predictor=predictor, rng=seed + 4242,
            )
            variances.append(res.mean_residual_variance)
            strehls.append(res.mean_strehl)
        return float(np.mean(variances)), float(np.mean(strehls))

    print("  tuning the classical integrator gain on the same held-out screens")
    print("  (the baseline is therefore presented at its best, not handicapped)")
    best_integrator = {}
    for lat in LATENCIES:
        rows = []
        for g in INTEGRATOR_GAINS:
            v, s = loop_stats("integrator", g, lat)
            rows.append((g, v, s))
        g, v, s = min(rows, key=lambda r: r[1])
        best_integrator[lat] = (g, v, s)
        sweep = "  ".join(f"{r[0]:.1f}:{r[1]:.3f}" for r in rows)
        print(f"    L={lat} sweep {sweep}")
        print(f"    L={lat} best gain {g:.2f} -> var {v:.4f} rad^2, Strehl {s:.4f}")

    print()
    print("  tuning the pseudo-open-loop gain for each predictor")
    results: dict[int, dict[str, tuple[float, float, float]]] = {}
    for lat in LATENCIES:
        results[lat] = {}
        for name, pred in (
            ("polc_delay", None),
            ("linear", models[lat]["linear"]),
            ("mlp_pers", models[lat]["mlp_pers"]),
            ("mlp_linear", models[lat]["mlp_lin"]),
        ):
            controller = "polc_delay" if pred is None else "polc_predict"
            rows = []
            for g in POLC_GAINS:
                v, s = loop_stats(controller, g, lat, predictor=pred)
                rows.append((g, v, s))
            g, v, s = min(rows, key=lambda r: r[1])
            results[lat][name] = (g, v, s)

    print()
    print(f"{'L':>2} {'controller':>12} {'gain':>6} {'residual var':>13} {'Strehl':>8} "
          f"{'vs integrator':>14}")
    for lat in LATENCIES:
        gi, vi, si = best_integrator[lat]
        print(f"{lat:>2} {'integrator':>12} {gi:6.2f} {vi:13.4f} {si:8.4f} {1.0:14.3f}")
        for name in ("polc_delay", "linear", "mlp_pers", "mlp_linear"):
            g, v, s = results[lat][name]
            print(f"{lat:>2} {name:>12} {g:6.2f} {v:13.4f} {s:8.4f} {vi / v:14.3f}")

    print()
    print("-" * 78)
    print("5.4  Verdict")
    print("-" * 78)
    for lat in LATENCIES:
        _, vi, _ = best_integrator[lat]
        _, vd, _ = results[lat]["polc_delay"]
        _, vl, _ = results[lat]["linear"]
        _, vm, _ = results[lat]["mlp_linear"]
        winner = min(
            [("integrator", vi), ("pure delay", vd), ("linear AR", vl), ("MLP ensemble", vm)],
            key=lambda r: r[1],
        )
        print(f"  L={lat}: best = {winner[0]} ({winner[1]:.4f} rad^2). "
              f"MLP/integrator = {vm / vi:.3f}, MLP/linear = {vm / vl:.3f} "
              f"(<1 means the MLP wins)")
    print()
    print("done.")


if __name__ == "__main__":
    main()
