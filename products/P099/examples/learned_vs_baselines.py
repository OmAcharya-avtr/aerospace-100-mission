"""Plot the learned predictor against the baselines on both held-out splits.

The left panel is the CRPS of every predictor on both splits, in Mbit, lower
better. The right panel is the calibration: the empirical probability
integral transform of each predictor against the uniform it should be, on the
harder split. A forecast on the diagonal is calibrated; a forecast bowed away
from it is over-confident, and a scheduler cannot book a quantile of it.

Writes ``screenshots/learned_vs_baselines.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from passyield.baselines import (  # noqa: E402
    analytic_shortfall_distribution,
    backlog_dry_time_s,
    booked_shortfall_mbit,
    fit_event_model,
)
from passyield.dataset import (  # noqa: E402
    feature_matrix,
    load_passes,
    parse_intervals,
    station_by_name,
    station_split,
    target_vector,
    time_split,
    window_of,
)
from passyield.geometry import satellite  # noqa: E402
from passyield.predictor import (  # noqa: E402
    DECLARED_UNCERTAINTY_MODE,
    PassYieldPredictor,
    PredictorSettings,
)
from passyield.report import format_comparison  # noqa: E402
from passyield.scoring import (  # noqa: E402
    crps_mean,
    interval_coverage,
    mean_absolute_error,
    pit_calibration_error,
    pit_values,
)

#: Ensemble members in every predictive distribution.
MEMBERS = 256

#: Colours, one per predictor.
COLOURS = {
    "baseline 1 booked": "#d73027",
    "baseline 2 convolution": "#4575b4",
    "baseline 2b backlog-aware": "#91bfdb",
    "null constant fraction": "#878787",
    "learned tree-spread": "#fdae61",
    f"learned {DECLARED_UNCERTAINTY_MODE}": "#1a9850",
}


def _ensembles(satrec, record, matrix, target, nominal, capacity, split):
    """Every predictor's ensemble on one split's held-out side."""
    train, test = list(split.train), list(split.test)
    rows = [record[index] for index in train]
    laws = fit_event_model(
        acquisition_s=[row.acquisition_s for row in rows],
        outage_counts=[row.outage_count for row in rows],
        window_durations_s=[row.duration_s for row in rows],
        outage_durations_s=[
            duration for row in rows
            for _start, duration in parse_intervals(row.resync_intervals)
        ],
        idle_totals_s=[row.idle_total_s for row in rows],
    )
    settings = (
        PredictorSettings(n_estimators=20) if _bootstrap.FAST else PredictorSettings()
    )
    predictor = PassYieldPredictor(settings).fit(matrix[train], target[train])
    levels = (np.arange(MEMBERS) + 0.5) / MEMBERS
    out: dict[str, np.ndarray] = {
        "baseline 1 booked": np.full((len(test), 1), booked_shortfall_mbit()),
        "null constant fraction": (
            nominal[test][:, None]
            - np.quantile(target[train], levels)[None, :] * capacity[test][:, None]
        ),
    }
    for label, aware in (
        ("baseline 2 convolution", False),
        ("baseline 2b backlog-aware", True),
    ):
        members = []
        for index in test:
            row = record[index]
            station = station_by_name(row.station)
            window = window_of(row)
            dry = (
                backlog_dry_time_s(satrec, station, window, row.backlog_mbit)
                if aware
                else None
            )
            law = analytic_shortfall_distribution(
                satrec, station, window, laws, nominal[index], capacity[index],
                deterministic_idle_s=dry,
            )
            members.append(law.samples_for_scoring(MEMBERS))
        width = min(item.size for item in members)
        out[label] = np.asarray([item[:width] for item in members])
    out["learned tree-spread"] = predictor.predict_shortfall_ensemble(
        matrix[test], nominal[test], capacity[test], "tree-spread"
    )
    out[f"learned {DECLARED_UNCERTAINTY_MODE}"] = predictor.predict_shortfall_ensemble(
        matrix[test], nominal[test], capacity[test], DECLARED_UNCERTAINTY_MODE
    )
    return out, test


def main() -> int:
    """Score every predictor on both splits and draw the two panels."""
    print(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    record = load_passes()[: _bootstrap.record_limit()]
    satrec = satellite()
    matrix = feature_matrix(record)
    target = target_vector(record)
    nominal = np.asarray([row.nominal_mbit for row in record])
    capacity = np.asarray([row.capacity_mbit for row in record])
    observed = np.asarray([row.shortfall_mbit for row in record])

    results: dict[str, dict[str, dict[str, float]]] = {}
    pit_by_predictor: dict[str, np.ndarray] = {}
    for split in (station_split(record), time_split(record)):
        ensembles, test = _ensembles(
            satrec, record, matrix, target, nominal, capacity, split
        )
        scores: dict[str, dict[str, float]] = {}
        for name, ensemble in ensembles.items():
            pit = pit_values(ensemble, observed[test])
            scores[name] = {
                "crps": crps_mean(ensemble, observed[test]),
                "mae": mean_absolute_error(ensemble.mean(axis=1), observed[test]),
                "pit_calibration_error": pit_calibration_error(pit),
                "coverage80": interval_coverage(
                    np.quantile(ensemble, 0.10, axis=1),
                    np.quantile(ensemble, 0.90, axis=1),
                    observed[test],
                ),
            }
            if split.name == "held-out stations":
                pit_by_predictor[name] = pit
        results[split.name] = scores
        print()
        print(format_comparison(scores, f"scored on the shortfall in Mbit -- {split.name}"))

    names = list(results["held-out stations"])
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

    left = axes[0]
    width = 0.38
    positions = np.arange(len(names))
    for offset, split_name in ((-width / 2, "held-out stations"), (width / 2, "held-out time")):
        values = [results[split_name][name]["crps"] for name in names]
        left.bar(
            positions + offset, values, width=width,
            color=["#2166ac" if offset < 0 else "#92c5de"] * len(names),
            label=split_name,
        )
    left.set_xticks(positions)
    left.set_xticklabels([name.replace(" ", "\n", 1) for name in names], fontsize=7)
    left.set_ylabel("CRPS, Mbit (lower is better)")
    left.set_title("error: the booked figure is refuted by an order of magnitude")
    left.legend(fontsize=8)

    right = axes[1]
    grid = np.linspace(0.0, 1.0, 101)
    right.plot(grid, grid, ls="--", lw=1.0, color="0.4", label="calibrated")
    for name in names:
        pit = np.sort(pit_by_predictor[name])
        empirical = np.arange(1, pit.size + 1) / pit.size
        right.plot(pit, empirical, lw=1.5, color=COLOURS[name], label=name)
    right.set_xlabel("probability integral transform")
    right.set_ylabel("empirical cumulative probability")
    right.set_title("calibration on the harder split (held-out stations)")
    right.legend(fontsize=7, loc="lower right")

    best = min(names, key=lambda name: results["held-out stations"][name]["crps"])
    calibrated = min(
        names,
        key=lambda name: results["held-out stations"][name]["pit_calibration_error"],
    )
    figure.suptitle(
        f"held-out stations: lowest CRPS {best}; best calibrated {calibrated}",
        fontsize=9,
    )
    figure.tight_layout()
    target_path = ROOT / _bootstrap.SCREENSHOTS / "learned_vs_baselines.png"
    figure.savefig(target_path)
    plt.close(figure)
    print()
    print(f"wrote {_bootstrap.relative(target_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
