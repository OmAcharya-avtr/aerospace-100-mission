"""A benchmark protocol that tunes every baseline before comparing it to the model.

Why this module exists
----------------------
A learned model compared against an untuned baseline tells you nothing. The
fixed-margin policy has one free parameter, the hysteresis policy has two, and
the confidence gate of :class:`acmpilot.predictor.PredictivePolicy` has one. If
the gate is tuned and the margins are not, the comparison is rigged, and
``validation/validate_policies.py`` shows that the shipped 3 dB default margin is
measurably *not* the best margin on this channel.

So every policy here gets the same treatment:

* hyperparameters are selected on a **tuning** set of seeds,
* the selected policy is then scored on a **disjoint test** set of seeds,
* the learned predictor is fitted on a **third disjoint** set of seeds.

All three sets index into the same seeded channel generator, so no policy ever
sees another's data, and within a set every policy sees identical sample paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channel import ChannelConfig, snr_db_path
from .modcod import ModcodTable
from .policy import ClairvoyantUpperBound, FixedMargin, Policy, ThresholdHysteresis
from .predictor import (
    GaussMarkovPredictor,
    PredictivePolicy,
    QuantilePredictor,
    make_lag_features,
)
from .simulate import run_policy

#: Fixed-margin values searched during tuning, dB.
MARGIN_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)

#: ``(up_margin_db, down_margin_db)`` pairs searched during tuning.
HYSTERESIS_GRID: tuple[tuple[float, float], ...] = (
    (0.5, 0.5), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5), (4.0, 0.5), (5.0, 0.5),
    (6.0, 0.5), (3.0, 1.5), (4.0, 2.0), (5.0, 3.0),
)

#: Confidence-gate strengths searched during tuning, dimensionless.
GATE_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


@dataclass(frozen=True)
class BenchmarkSplit:
    """Disjoint seed sets. Overlap is rejected at construction.

    Attributes
    ----------
    train_seeds
        Used only to fit the predictors.
    tune_seeds
        Used only to select hyperparameters.
    test_seeds
        Used only to report. Every published number comes from these.
    """

    train_seeds: tuple[int, ...] = (101, 102, 103)
    tune_seeds: tuple[int, ...] = (201, 202, 203, 204, 205)
    test_seeds: tuple[int, ...] = (301, 302, 303, 304, 305, 306, 307, 308, 309, 310)

    def __post_init__(self) -> None:
        sets = [set(self.train_seeds), set(self.tune_seeds), set(self.test_seeds)]
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = sets[i] & sets[j]
                if overlap:
                    raise ValueError(f"seed sets must be disjoint; shared: {sorted(overlap)}")
        if not all(sets):
            raise ValueError("every seed set must be non-empty")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Sizes for the benchmark. Defaults fit the 2-core, 3-minute budget.

    Attributes
    ----------
    n_slots
        Slots per scored episode.
    train_slots
        Slots per training episode (three episodes by default).
    n_lags
        Lagged reports per feature row. The default is 4 rather than 8 because
        more lags were measured to buy nothing: at ``n_lags = 8`` and 100 trees
        the held-out median-absolute error was 2.068 dB against 2.063 dB at
        ``n_lags = 4`` and 60 trees, for 3.6x the fit time
        (``validation/validate_predictor.txt``). That is itself a result --- the
        driver process is first-order Markov, so there is no information in the
        older reports to find.
    n_estimators
        Trees per quantile model.
    split
        Seed sets.
    """

    n_slots: int = 20_000
    train_slots: int = 5_200
    n_lags: int = 4
    n_estimators: int = 60
    split: BenchmarkSplit = field(default_factory=BenchmarkSplit)


def score_policy(
    table: ModcodTable,
    config: ChannelConfig,
    policy: Policy,
    *,
    tau_s: float,
    seeds: tuple[int, ...],
    n_slots: int,
) -> dict[str, float]:
    """Mean accounting of ``policy`` over ``seeds``, plus the goodput standard error.

    Returns
    -------
    dict
        Every :class:`acmpilot.accounting.Accounting` field averaged, plus
        ``goodput_sem`` and ``n_seeds``.
    """
    d = config.delay_slots(tau_s)
    records = []
    for seed in seeds:
        snr = snr_db_path(config, n_slots, seed)
        records.append(run_policy(table, snr, policy, delay_slots=d).accounting.as_dict())
    out = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
    goodputs = np.array([r["goodput_bit_per_symbol"] for r in records])
    out["goodput_sem"] = float(
        goodputs.std(ddof=1) / np.sqrt(goodputs.size) if goodputs.size > 1 else 0.0
    )
    out["n_seeds"] = float(len(seeds))
    return out


def tune_fixed_margin(
    table: ModcodTable,
    config: ChannelConfig,
    *,
    tau_s: float,
    seeds: tuple[int, ...],
    n_slots: int,
    grid: tuple[float, ...] = MARGIN_GRID,
) -> tuple[FixedMargin, list[tuple[float, float]]]:
    """Select the margin maximising mean goodput on ``seeds``.

    Returns
    -------
    (policy, trace)
        ``trace`` is ``[(margin_db, goodput), ...]`` over the whole grid, so the
        tuning is reportable rather than hidden.
    """
    trace = [
        (
            float(margin),
            score_policy(
                table, config, FixedMargin(margin_db=margin),
                tau_s=tau_s, seeds=seeds, n_slots=n_slots,
            )["goodput_bit_per_symbol"],
        )
        for margin in grid
    ]
    best = max(trace, key=lambda row: row[1])[0]
    return FixedMargin(margin_db=best), trace


def tune_hysteresis(
    table: ModcodTable,
    config: ChannelConfig,
    *,
    tau_s: float,
    seeds: tuple[int, ...],
    n_slots: int,
    grid: tuple[tuple[float, float], ...] = HYSTERESIS_GRID,
) -> tuple[ThresholdHysteresis, list[tuple[float, float, float]]]:
    """Select ``(up, down)`` maximising mean goodput on ``seeds``.

    Returns
    -------
    (policy, trace)
        ``trace`` is ``[(up_db, down_db, goodput), ...]``.
    """
    trace = [
        (
            float(up),
            float(down),
            score_policy(
                table, config,
                ThresholdHysteresis(up_margin_db=up, down_margin_db=down),
                tau_s=tau_s, seeds=seeds, n_slots=n_slots,
            )["goodput_bit_per_symbol"],
        )
        for up, down in grid
    ]
    best = max(trace, key=lambda row: row[2])
    return (
        ThresholdHysteresis(up_margin_db=best[0], down_margin_db=best[1]),
        trace,
    )


def tune_gate(
    table: ModcodTable,
    config: ChannelConfig,
    predictor: GaussMarkovPredictor | QuantilePredictor,
    *,
    tau_s: float,
    seeds: tuple[int, ...],
    n_slots: int,
    n_lags: int,
    grid: tuple[float, ...] = GATE_GRID,
) -> tuple[PredictivePolicy, list[tuple[float, float]]]:
    """Select the confidence-gate strength maximising mean goodput on ``seeds``.

    Returns
    -------
    (policy, trace)
        ``trace`` is ``[(gate_k, goodput), ...]``. ``gate_k = 0`` is in the grid
        on purpose: it is the ablation that shows whether the confidence output
        is doing any work at all.
    """
    d = config.delay_slots(tau_s)
    trace = [
        (
            float(gate),
            score_policy(
                table, config,
                PredictivePolicy(predictor, delay_slots=d, n_lags=n_lags, gate_k=gate),
                tau_s=tau_s, seeds=seeds, n_slots=n_slots,
            )["goodput_bit_per_symbol"],
        )
        for gate in grid
    ]
    best = max(trace, key=lambda row: row[1])[0]
    return (
        PredictivePolicy(predictor, delay_slots=d, n_lags=n_lags, gate_k=best),
        trace,
    )


def fit_predictors(
    config: ChannelConfig,
    *,
    tau_s: float,
    bench: BenchmarkConfig,
) -> tuple[GaussMarkovPredictor, QuantilePredictor, np.ndarray, np.ndarray]:
    """Fit the analytic and learned predictors on the training seeds only.

    Returns
    -------
    (analytic, learned, features, target)
        The training design matrix is returned so a caller can report its size.
    """
    d = config.delay_slots(tau_s)
    train = np.concatenate(
        [snr_db_path(config, bench.train_slots, s) for s in bench.split.train_seeds]
    )
    features, target, _ = make_lag_features(train, delay_slots=d, n_lags=bench.n_lags)
    analytic = GaussMarkovPredictor(delay_slots=d).fit(train)
    learned = QuantilePredictor(
        delay_slots=d, n_estimators=bench.n_estimators
    ).fit(features, target)
    return analytic, learned, features, target


def benchmark_at_delay(
    table: ModcodTable,
    config: ChannelConfig,
    *,
    tau_s: float,
    bench: BenchmarkConfig | None = None,
) -> dict[str, object]:
    """Tune every policy, then score every policy, at one feedback delay.

    Returns
    -------
    dict
        ``tau_s``, ``delay_slots``, ``n_train_rows``, ``tuned`` (the selected
        hyperparameters), ``traces`` (every tuning grid), and ``scores``: a list
        of rows, each with ``policy``, ``variant`` (``"default"``, ``"tuned"``,
        ``"predictive"`` or ``"bound"``), ``causal``, ``learned`` and every
        scored field, measured on the **test** seeds only. When tuning happens to
        select the shipped default, the default and tuned rows are identical;
        ``variant`` is what distinguishes them.
    """
    cfg = BenchmarkConfig() if bench is None else bench
    d = config.delay_slots(tau_s)
    analytic, learned, features, _ = fit_predictors(config, tau_s=tau_s, bench=cfg)

    fixed, margin_trace = tune_fixed_margin(
        table, config, tau_s=tau_s, seeds=cfg.split.tune_seeds, n_slots=cfg.n_slots
    )
    hyst, hyst_trace = tune_hysteresis(
        table, config, tau_s=tau_s, seeds=cfg.split.tune_seeds, n_slots=cfg.n_slots
    )
    analytic_policy, analytic_trace = tune_gate(
        table, config, analytic, tau_s=tau_s, seeds=cfg.split.tune_seeds,
        n_slots=cfg.n_slots, n_lags=cfg.n_lags,
    )
    learned_policy, learned_trace = tune_gate(
        table, config, learned, tau_s=tau_s, seeds=cfg.split.tune_seeds,
        n_slots=cfg.n_slots, n_lags=cfg.n_lags,
    )

    entries: list[tuple[Policy, bool, str]] = [
        (FixedMargin(margin_db=3.0), False, "default"),
        (fixed, False, "tuned"),
        (ThresholdHysteresis(), False, "default"),
        (hyst, False, "tuned"),
        (analytic_policy, False, "predictive"),
        (learned_policy, True, "predictive"),
        (ClairvoyantUpperBound(), False, "bound"),
    ]
    scores = []
    for policy, is_learned, variant in entries:
        row: dict[str, object] = {
            "policy": policy.name,
            "variant": variant,
            "causal": bool(policy.causal),
            "learned": is_learned,
        }
        row.update(
            score_policy(
                table, config, policy, tau_s=tau_s,
                seeds=cfg.split.test_seeds, n_slots=cfg.n_slots,
            )
        )
        scores.append(row)

    return {
        "tau_s": float(tau_s),
        "delay_slots": d,
        "n_train_rows": int(features.shape[0]),
        "tuned": {
            "fixed_margin_db": fixed.margin_db,
            "hysteresis_up_db": hyst.up_margin_db,
            "hysteresis_down_db": hyst.down_margin_db,
            "analytic_gate_k": analytic_policy.gate_k,
            "learned_gate_k": learned_policy.gate_k,
        },
        "traces": {
            "fixed_margin": margin_trace,
            "hysteresis": hyst_trace,
            "analytic_gate": analytic_trace,
            "learned_gate": learned_trace,
        },
        "predictors": {"analytic": analytic, "learned": learned},
        "scores": scores,
    }
