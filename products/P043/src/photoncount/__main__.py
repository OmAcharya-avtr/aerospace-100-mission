"""Command-line interface: ``python -m photoncount <subcommand>``.

Subcommands mirror the library's layers. Every one prints numbers and nothing
else, so the output is diffable: no progress bars, no banners.

``ppm``         slot statistics, exact symbol error probability and capacity
``deadtime``    forward and inverse dead-time relations with the branch warning
``webb``        Webb moments and its Gaussian/Poisson limit residuals
``capacity``    rates and photons per bit for a PPM configuration
``acquire``     a simulated or dry-run acquisition through the HAL, with preflight
``correct``     closed-form and learned rate corrections for an observed rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .afterpulse import effective_afterpulse_probability, fano_factor_prediction
from .capacity import (
    erasure_channel_capacity,
    hard_decision_capacity,
    minimum_photons_per_bit,
    photons_per_bit,
    soft_decision_achievable_rate,
)
from .correction import (
    composed_baseline,
    matched_baseline,
    nonparalyzable_baseline,
    paralyzable_baseline,
)
from .deadtime import (
    dead_time_loss_fraction,
    is_observable,
    live_time_fraction,
    observed_rate,
    paralyzable_maximum,
    true_rate,
)
from .hal import AcquisitionRequest, BackendMode, DeviceBackend, SimulatedBackend
from .ops import RunJournal, RunRecord, run_preflight
from .ppm import (
    PPMConfig,
    bits_per_symbol,
    erasure_probability,
    symbol_error_probability,
)
from .simulate import DetectorSpec
from .webb import WebbParameters, binned_pmf, moments, pdf


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))


def _cmd_ppm(args: argparse.Namespace) -> int:
    cfg = PPMConfig(args.order, args.signal, args.background, args.dark)
    exact = symbol_error_probability(cfg)
    payload: dict[str, object] = {
        "order": cfg.order,
        "bits_per_symbol": bits_per_symbol(cfg.order),
        "signal_counts": cfg.signal_counts,
        "noise_counts_per_slot": cfg.noise_counts,
        "symbol_error_probability": exact["error"],
        "truncation": exact["truncation"],
        "tail_mass": exact["tail_mass"],
        "all_slots_empty_probability": erasure_probability(cfg),
    }
    if cfg.is_background_free:
        payload["erasure_capacity_bits_per_symbol"] = erasure_channel_capacity(cfg)[
            "bits_per_symbol"
        ]
    else:
        payload["hard_decision_capacity_bits_per_symbol"] = hard_decision_capacity(cfg)[
            "bits_per_symbol"
        ]
    _emit(payload)
    return 0


def _cmd_deadtime(args: argparse.Namespace) -> int:
    tau = args.dead_time
    n_max, m_max = paralyzable_maximum(tau)
    payload: dict[str, object] = {
        "dead_time_s": tau,
        "model": args.model,
        "paralyzable_peak_true_rate_hz": n_max,
        "paralyzable_max_observed_rate_hz": m_max,
    }
    if args.true_rate is not None:
        obs = float(observed_rate(args.true_rate, tau, args.model)[0])
        payload["true_rate_hz"] = args.true_rate
        payload["observed_rate_hz"] = obs
        payload["loss_fraction"] = float(
            dead_time_loss_fraction(args.true_rate, tau, args.model)[0]
        )
        payload["live_time_fraction"] = float(
            live_time_fraction(args.true_rate, tau, args.model)[0]
        )
    if args.observed_rate is not None:
        ok = bool(is_observable(args.observed_rate, tau, args.model)[0])
        payload["observed_rate_hz"] = args.observed_rate
        payload["observable"] = ok
        if ok:
            payload["true_rate_lower_branch_hz"] = float(
                true_rate(args.observed_rate, tau, args.model, "lower")[0]
            )
            if args.model == "paralyzable":
                payload["true_rate_upper_branch_hz"] = float(
                    true_rate(args.observed_rate, tau, args.model, "upper")[0]
                )
                payload["branch_warning"] = (
                    "the paralyzable inverse is two-valued; both roots reproduce this "
                    "observed rate and only knowledge of the illumination picks one"
                )
        else:
            payload["branch_warning"] = (
                "no true rate reproduces this observed rate for this model and dead time"
            )
    _emit(payload)
    return 0


def _cmd_webb(args: argparse.Namespace) -> int:
    params = WebbParameters(args.mean, args.excess_noise, args.gain)
    mom = moments(params)
    k_max = int(args.mean + 10.0 * np.sqrt(args.mean * args.excess_noise) + 10)
    probs = binned_pmf(params, k_max)
    from scipy import stats

    poisson = stats.poisson.pmf(np.arange(k_max + 1), args.mean)
    gauss_ref = WebbParameters(args.mean, 1.0, args.gain)
    grid = np.linspace(
        args.mean - 6.0 * np.sqrt(args.mean), args.mean + 6.0 * np.sqrt(args.mean), 2001
    )
    _emit(
        {
            "mean_primary": params.mean_primary,
            "excess_noise_factor": params.excess_noise_factor,
            "gain": params.gain,
            "moments": mom,
            "total_variation_to_poisson": float(0.5 * np.abs(probs - poisson).sum()),
            "sup_norm_to_gaussian_limit": float(
                np.max(np.abs(pdf(grid, params) - pdf(grid, gauss_ref)))
            ),
            "binned_pmf_sum": float(probs.sum()),
        }
    )
    return 0


def _cmd_capacity(args: argparse.Namespace) -> int:
    cfg = PPMConfig(args.order, args.signal, args.background, args.dark)
    payload: dict[str, object] = {
        "order": cfg.order,
        "minimum_photons_per_bit_limit": minimum_photons_per_bit(cfg.order),
    }
    if cfg.is_background_free:
        cap = erasure_channel_capacity(cfg)
        payload["erasure_capacity"] = cap
        payload["photons_per_bit"] = photons_per_bit(cfg, cap["bits_per_symbol"])
    else:
        hard = hard_decision_capacity(cfg)
        soft = soft_decision_achievable_rate(
            cfg, args.monte_carlo, np.random.default_rng(args.seed)
        )
        payload["hard_decision"] = hard
        payload["soft_decision"] = soft
        payload["photons_per_bit_hard"] = photons_per_bit(cfg, hard["bits_per_symbol"])
        payload["photons_per_bit_soft"] = photons_per_bit(cfg, soft["bits_per_symbol"])
    _emit(payload)
    return 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    spec = DetectorSpec(
        dead_time_s=args.dead_time,
        model=args.model,
        afterpulse_probability=args.afterpulse,
        afterpulse_mean_delay_s=args.afterpulse_delay,
    )
    request = AcquisitionRequest(args.window, args.windows, args.label)
    mode = BackendMode.DRY_RUN if args.dry_run else BackendMode.SIMULATION
    if args.backend == "device":
        backend = DeviceBackend(
            dead_time_s=args.dead_time,
            dead_time_model=args.model,
            afterpulse_probability=args.afterpulse,
            afterpulse_mean_delay_s=args.afterpulse_delay,
        )
    else:
        backend = SimulatedBackend(spec, args.rate, np.random.default_rng(args.seed))
    report = run_preflight(backend, request, args.rate, directory=args.directory)
    payload: dict[str, object] = {
        "backend": backend.name,
        "mode": mode.value,
        "preflight_passed": report.passed,
        "preflight": report.as_dicts(),
    }
    if not report.passed:
        payload["aborted"] = "preflight failed; no acquisition attempted"
        _emit(payload)
        return 1
    journal = RunJournal(request.label, args.directory)
    try:
        journal.begin({"window_s": request.window_s, "n_windows": int(request.n_windows)})
        with backend:
            acq = backend.acquire(request, mode)
        record = RunRecord.from_acquisition(acq, request, backend, report, seed=args.seed)
        payload["record_file"] = journal.commit(record)
        payload["observed_rate_hz"] = acq.observed_rate_hz
        payload["fano_factor"] = acq.fano_factor
        payload["is_measurement"] = acq.is_measurement
        payload["wall_seconds"] = acq.wall_seconds
        payload["effective_afterpulse_probability"] = effective_afterpulse_probability(
            args.afterpulse, args.dead_time, args.afterpulse_delay
        )
        payload["afterpulse_fano_prediction"] = fano_factor_prediction(args.afterpulse)
    except NotImplementedError as exc:
        payload["not_implemented"] = str(exc)
        _emit(payload)
        return 2
    _emit(payload)
    return 0


def _cmd_correct(args: argparse.Namespace) -> int:
    m = np.atleast_1d(float(args.observed_rate))
    tau = np.atleast_1d(float(args.dead_time))
    par = np.atleast_1d(1.0 if args.model == "paralyzable" else 0.0)
    p_ap = np.atleast_1d(float(args.afterpulse))
    payload: dict[str, object] = {
        "observed_rate_hz": float(m[0]),
        "dead_time_s": float(tau[0]),
        "model": args.model,
        "afterpulse_probability": float(p_ap[0]),
        "observed_x": float(m[0] * tau[0]),
        "nonparalyzable_closed_form_hz": float(nonparalyzable_baseline(m, tau)[0]),
        "paralyzable_closed_form_hz": float(paralyzable_baseline(m, tau)[0]),
        "matched_closed_form_hz": float(matched_baseline(m, tau, par)[0]),
        "composed_closed_form_hz": float(composed_baseline(m, tau, par, p_ap)[0]),
    }
    if args.model_file is not None:
        from .correction import RateCorrector
        from .dataset import make_features

        path = Path(args.model_file)
        if not path.exists():
            payload["learned"] = f"model file {path.name!r} not found; run the training script"
        else:
            corrector = RateCorrector.load(path)
            feats = make_features(
                m, tau, np.atleast_1d(args.fano), p_ap,
                np.atleast_1d(args.afterpulse_delay / args.dead_time), par,
            )
            out = corrector.predict_rate(feats, tau)
            payload["learned_hz"] = float(out["median"][0])
            payload["learned_interval_hz"] = [
                float(out["lower"][0]),
                float(out["upper"][0]),
            ]
    _emit(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser. Exposed so tests can exercise it without a subprocess."""
    parser = argparse.ArgumentParser(
        prog="photoncount",
        description=(
            "Photon-counting optical receiver chain: Poisson and Webb detection "
            "statistics, PPM slot statistics, dead time and afterpulsing, PPM "
            "Poisson-channel rates, and a hardware abstraction layer. "
            "Research-grade; not flight-qualified, not certified, not approved "
            "for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"photoncount {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ppm = sub.add_parser("ppm", help="PPM slot statistics and symbol error probability")
    p_ppm.add_argument("--order", type=int, default=16, help="M, slots per symbol")
    p_ppm.add_argument("--signal", type=float, default=2.0, help="n_s, counts per pulsed slot")
    p_ppm.add_argument("--background", type=float, default=0.0, help="n_b, counts per slot")
    p_ppm.add_argument("--dark", type=float, default=0.0, help="n_d, counts per slot")
    p_ppm.set_defaults(func=_cmd_ppm)

    p_dt = sub.add_parser("deadtime", help="dead-time forward and inverse relations")
    p_dt.add_argument("--dead-time", type=float, required=True, help="tau, seconds")
    p_dt.add_argument(
        "--model", choices=("paralyzable", "nonparalyzable"), default="paralyzable"
    )
    p_dt.add_argument("--true-rate", type=float, default=None, help="n, counts/s")
    p_dt.add_argument("--observed-rate", type=float, default=None, help="m, counts/s")
    p_dt.set_defaults(func=_cmd_deadtime)

    p_webb = sub.add_parser("webb", help="Webb moments and limit residuals")
    p_webb.add_argument("--mean", type=float, default=100.0, help="m, primary photoelectrons")
    p_webb.add_argument("--excess-noise", type=float, default=2.0, help="F, >= 1")
    p_webb.add_argument("--gain", type=float, default=1.0, help="G, mean avalanche gain")
    p_webb.set_defaults(func=_cmd_webb)

    p_cap = sub.add_parser("capacity", help="PPM Poisson-channel rates and photons per bit")
    p_cap.add_argument("--order", type=int, default=16)
    p_cap.add_argument("--signal", type=float, default=2.0)
    p_cap.add_argument("--background", type=float, default=0.0)
    p_cap.add_argument("--dark", type=float, default=0.0)
    p_cap.add_argument("--monte-carlo", type=int, default=20000)
    p_cap.add_argument("--seed", type=int, default=0)
    p_cap.set_defaults(func=_cmd_capacity)

    p_acq = sub.add_parser("acquire", help="run an acquisition through the HAL")
    p_acq.add_argument("--backend", choices=("simulated", "device"), default="simulated")
    p_acq.add_argument("--rate", type=float, default=1e5, help="true rate, counts/s")
    p_acq.add_argument("--dead-time", type=float, default=1e-7)
    p_acq.add_argument(
        "--model", choices=("paralyzable", "nonparalyzable"), default="paralyzable"
    )
    p_acq.add_argument("--afterpulse", type=float, default=0.03)
    p_acq.add_argument("--afterpulse-delay", type=float, default=3e-7)
    p_acq.add_argument("--window", type=float, default=1e-3, help="window length, s")
    p_acq.add_argument("--windows", type=int, default=50)
    p_acq.add_argument("--label", default="cli_run")
    p_acq.add_argument("--directory", default="runs", help="capture directory, relative")
    p_acq.add_argument("--dry-run", action="store_true", help="real path, discarded counts")
    p_acq.add_argument("--seed", type=int, default=0)
    p_acq.set_defaults(func=_cmd_acquire)

    p_cor = sub.add_parser("correct", help="closed-form and learned rate correction")
    p_cor.add_argument("--observed-rate", type=float, required=True, help="m, counts/s")
    p_cor.add_argument("--dead-time", type=float, required=True, help="tau, s")
    p_cor.add_argument(
        "--model", choices=("paralyzable", "nonparalyzable"), default="paralyzable"
    )
    p_cor.add_argument("--afterpulse", type=float, default=0.0)
    p_cor.add_argument("--afterpulse-delay", type=float, default=3e-7)
    p_cor.add_argument("--fano", type=float, default=1.0, help="measured Fano factor")
    p_cor.add_argument(
        "--model-file", default=None, help="path to a .joblib corrector from the training script"
    )
    p_cor.set_defaults(func=_cmd_correct)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess in tests
    sys.exit(main())
