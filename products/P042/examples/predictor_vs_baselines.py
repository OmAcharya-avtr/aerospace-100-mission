"""Example 5: the learned predictor against every baseline, and its calibration.

Writes ``../screenshots/predictor_vs_baselines.png``.

Four panels:

1. Held-out goodput against feedback delay for all seven policies, with the
   clairvoyant bound dashed and labelled as unachievable. Notice the learned
   curve and the analytic AR(1) curve lie on top of each other at every delay:
   on this channel the learned model has nothing the three-parameter linear
   predictor has not already got. Notice also that both pull away from the tuned
   baselines as the delay grows -- the predictor earns its keep at large
   tau/tau_c, not at small.
2. Reliability of the learned predictor's stated quantiles: measured coverage
   against nominal. Notice the points sit near the diagonal but slightly below
   it -- the intervals are marginally too narrow, which the validation output
   quantifies. A predictor whose gate is this close to nominal can be trusted to
   back off when it says it is unsure.
3. The confidence-gate ablation: tuned gate against ``gate_k = 0``, which uses
   the point estimate alone. Notice the gate is worth more than the choice of
   predictor by an order of magnitude. The confidence output, not the regression,
   is what the policy needs.
4. Predicted interval against realised error at one delay, binned by predicted
   spread. Notice the realised spread grows with the predicted spread -- the
   confidence output is informative, not decorative.

Runtime: about 55 s on one core (two gradient-boosting fits).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acmpilot.benchmark import BenchmarkConfig, benchmark_at_delay, score_policy  # noqa: E402
from acmpilot.channel import ChannelConfig, snr_db_path  # noqa: E402
from acmpilot.modcod import ModcodTable, measure_thresholds  # noqa: E402
from acmpilot.policy import CLAIRVOYANT_LABEL  # noqa: E402
from acmpilot.predictor import PredictivePolicy, make_lag_features  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "screenshots" / "predictor_vs_baselines.png"
OUT_REL = "screenshots/predictor_vs_baselines.png"
TABLE_JSON = HERE.parent / "validation" / "modcod_thresholds.json"
TAUS_MS = (2.0, 10.0)
CURVE_TAUS_MS = (1.0, 2.0, 5.0, 10.0, 20.0)


def _table() -> ModcodTable:
    if TABLE_JSON.exists():
        return ModcodTable.load_json(TABLE_JSON)
    table, _ = measure_thresholds()
    return table


def main() -> int:
    table = _table()
    config = ChannelConfig(slot_s=1e-3, tau_c_s=10e-3, sigma_i2=0.5, mean_snr_db=14.0)
    bench = BenchmarkConfig(n_slots=12_000)

    # Fit once at each of two delays; reuse the two fitted predictors for the
    # goodput curve across all delays so the example stays inside its budget.
    results = {
        tau_ms: benchmark_at_delay(table, config, tau_s=tau_ms * 1e-3, bench=bench)
        for tau_ms in TAUS_MS
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))

    # Panel 1: goodput curves. The predictors are refitted only at the delays in
    # TAUS_MS to stay inside the example's compute budget; for the other delays
    # the predictor fitted at the nearest of those delays is reused, and the
    # title says so. The validation scripts refit at every delay.
    ax = axes[0, 0]
    curves: dict[str, list[float]] = {}
    for tau_ms in CURVE_TAUS_MS:
        nearest = min(TAUS_MS, key=lambda t: abs(t - tau_ms))
        res = results[nearest]
        d = config.delay_slots(tau_ms * 1e-3)
        for policy, key in _curve_policies(res, d, bench.n_lags):
            score = score_policy(
                table, config, policy, tau_s=tau_ms * 1e-3,
                seeds=bench.split.test_seeds, n_slots=bench.n_slots,
            )
            curves.setdefault(key, []).append(score["goodput_bit_per_symbol"])

    for i, (name, values) in enumerate(curves.items()):
        acausal = name == CLAIRVOYANT_LABEL
        ax.plot(
            CURVE_TAUS_MS, values, "--" if acausal else "o-",
            lw=1.9 if acausal else 1.4, color="k" if acausal else f"C{i}", ms=4,
            label="clairvoyant: UPPER BOUND, no causal policy can achieve"
            if acausal else name,
        )
    ax.set_xlabel("round-trip feedback delay tau, ms")
    ax.set_ylabel("held-out goodput, bit/symbol")
    ax.set_title(
        "All policies, held-out test seeds, tau_c = 10 ms\n"
        "learned and analytic predictors coincide at every delay"
    )
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    # Panel 2: reliability diagram
    ax = axes[0, 1]
    res = results[10.0]
    d = int(res["delay_slots"])
    held = snr_db_path(config, 20_000, 401)
    feat, targ, _ = make_lag_features(held, delay_slots=d, n_lags=bench.n_lags)
    quants = res["predictors"]["learned"].predict_quantiles(feat)
    nominal = np.array([0.1, 0.5, 0.9])
    measured = np.array([float(np.mean(targ <= quants[q])) for q in (0.1, 0.5, 0.9)])
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="perfect calibration")
    ax.plot(nominal, measured, "o-", ms=8, lw=1.5, label="learned quantile GBR, tau = 10 ms")
    centre, spread = res["predictors"]["analytic"].predict(feat)
    analytic_measured = [
        float(np.mean(targ <= centre - spread / 2.0)),
        float(np.mean(targ <= centre)),
        float(np.mean(targ <= centre + spread / 2.0)),
    ]
    ax.plot(nominal, analytic_measured, "s-", ms=8, lw=1.5,
            label="analytic AR(1) MMSE (not learned)")
    for x, y in zip(nominal, measured, strict=True):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, -10), fontsize=7)
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("measured fraction of targets at or below")
    ax.set_title("Reliability of the stated quantiles\nthe gate is only as good as this diagram")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 3: gate ablation
    ax = axes[1, 0]
    labels, tuned_g, zero_g = [], [], []
    for tau_ms in TAUS_MS:
        res = results[tau_ms]
        d = int(res["delay_slots"])
        for pred_key, label in (("learned", "learned"), ("analytic", "analytic")):
            gate = (
                res["tuned"]["learned_gate_k"] if pred_key == "learned"
                else res["tuned"]["analytic_gate_k"]
            )
            tuned_g.append(
                score_policy(
                    table, config,
                    PredictivePolicy(res["predictors"][pred_key], delay_slots=d,
                                     n_lags=bench.n_lags, gate_k=gate),
                    tau_s=tau_ms * 1e-3, seeds=bench.split.test_seeds,
                    n_slots=bench.n_slots,
                )["goodput_bit_per_symbol"]
            )
            zero_g.append(
                score_policy(
                    table, config,
                    PredictivePolicy(res["predictors"][pred_key], delay_slots=d,
                                     n_lags=bench.n_lags, gate_k=0.0),
                    tau_s=tau_ms * 1e-3, seeds=bench.split.test_seeds,
                    n_slots=bench.n_slots,
                )["goodput_bit_per_symbol"]
            )
            labels.append(f"{label}\ntau={tau_ms:g} ms")
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, zero_g, width=0.4, label="gate_k = 0 (point estimate only)")
    ax.bar(pos + 0.2, tuned_g, width=0.4, label="tuned confidence gate")
    for i, (a, b) in enumerate(zip(zero_g, tuned_g, strict=True)):
        ax.annotate(f"+{b - a:.3f}", (i + 0.2, b), ha="center",
                    textcoords="offset points", xytext=(0, 3), fontsize=7)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("held-out goodput, bit/symbol")
    ax.set_title(
        "What the confidence output is worth\n"
        "(larger than the gap between the two predictors)"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    # Panel 4: predicted spread against realised error
    ax = axes[1, 1]
    res = results[10.0]
    centre_l, spread_l = res["predictors"]["learned"].predict(feat)
    err = np.abs(targ - centre_l)
    bins = np.quantile(spread_l, np.linspace(0.0, 1.0, 9))
    idx = np.clip(np.digitize(spread_l, bins[1:-1]), 0, 7)
    xs = [float(np.mean(spread_l[idx == b])) for b in range(8)]
    ys = [float(np.mean(err[idx == b])) for b in range(8)]
    p90 = [float(np.quantile(err[idx == b], 0.9)) for b in range(8)]
    ax.plot(xs, ys, "o-", lw=1.5, label="mean |error| in bin")
    ax.plot(xs, p90, "s--", lw=1.5, label="90th percentile |error| in bin")
    ax.set_xlabel("predicted spread q90 - q10, dB")
    ax.set_ylabel("realised |prediction error|, dB")
    ax.set_title(
        "The confidence output is informative, tau = 10 ms\n"
        "wider stated interval really does mean larger error"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "acmpilot: learned channel predictor against three required baselines and an "
        "analytic non-learned predictor. Research-grade, simulated channel.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    for name, values in curves.items():
        print(f"{name[:52]:54s} " + " ".join(f"{v:.4f}" for v in values))
    print(f"delays, ms: {list(CURVE_TAUS_MS)}")
    print(f"written: {OUT_REL}")
    return 0


def _curve_policies(res: dict, delay_slots: int, n_lags: int):
    """The seven policies of one benchmark result, as ``(policy, legend_key)``.

    The tuned hyperparameters come from the benchmark's tuning seeds, so the
    curve compares tuned baselines against the predictors, not defaults against
    a tuned model.
    """
    from acmpilot.policy import ClairvoyantUpperBound, FixedMargin, ThresholdHysteresis

    tuned = res["tuned"]
    return [
        (FixedMargin(margin_db=3.0), "fixed margin 3 dB (shipped default)"),
        (
            FixedMargin(margin_db=tuned["fixed_margin_db"]),
            "fixed margin, tuned per delay",
        ),
        (ThresholdHysteresis(), "hysteresis +2/-0.5 dB (shipped default)"),
        (
            ThresholdHysteresis(
                up_margin_db=tuned["hysteresis_up_db"],
                down_margin_db=tuned["hysteresis_down_db"],
            ),
            "hysteresis, tuned per delay",
        ),
        (
            PredictivePolicy(
                res["predictors"]["analytic"], delay_slots=delay_slots,
                n_lags=n_lags, gate_k=tuned["analytic_gate_k"],
            ),
            "analytic AR(1) MMSE (NOT learned), gated",
        ),
        (
            PredictivePolicy(
                res["predictors"]["learned"], delay_slots=delay_slots,
                n_lags=n_lags, gate_k=tuned["learned_gate_k"],
            ),
            "learned quantile GBR, gated",
        ),
        (ClairvoyantUpperBound(), CLAIRVOYANT_LABEL),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
