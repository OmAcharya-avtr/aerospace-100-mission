"""Validation V4: learned predictor against all three baselines, lognormal channel.

Protocol (``src/acmpilot/benchmark.py``)
----------------------------------------
Three disjoint seed sets: the predictors are fitted on the training seeds, EVERY
policy's free parameters -- the fixed margin, the hysteresis dead band, and the
confidence gate -- are tuned on the tuning seeds, and all numbers reported are
measured on the test seeds. Comparing a tuned model against an untuned baseline
would be worthless, and ``validate_policies.py`` V3.6 already shows the shipped
3 dB default margin is not the best margin on this channel. Both the default and
the tuned baseline are reported.

Four non-learned references are reported, three of them required:

  1. fixed margin                 (required baseline)
  2. threshold with hysteresis    (required baseline)
  3. clairvoyant upper bound      (required; ACAUSAL, not achievable)
  4. analytic AR(1) MMSE          (additional; also NOT learned)

Reference 4 is in the table because the lognormal channel of
``acmpilot.channel`` is an AR(1) process in log-amplitude by construction, so the
minimum-mean-square-error predictor at any horizon is exactly linear in the last
report. If a learned model cannot beat three estimated parameters on this
channel, the honest conclusion is that there is nothing on this channel for it to
learn, and that conclusion is reported rather than buried.

Checks:

V4.1  Calibration of both predictors' stated intervals against nominal 80%
      coverage and nominal 10% below-q10 exceedance, at every delay. Mean error
      is reported alongside, not instead.
V4.2  Goodput, efficiency, outage and conservatism of all seven policies at
      every delay, on held-out seeds, with the standard error of the mean.
V4.3  The confidence-gate ablation: ``gate_k = 0`` uses only the point estimate.
      The difference between the tuned gate and ``gate_k = 0`` is what the
      confidence output is worth.
V4.4  Which policy wins at which delay, stated plainly, including when a
      baseline wins and when two policies are not separated by their standard
      errors.

Runtime: about 150 s on one core (five gradient-boosting fits).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from acmpilot.benchmark import (  # noqa: E402
    BenchmarkConfig,
    benchmark_at_delay,
    score_policy,
)
from acmpilot.channel import ChannelConfig, snr_db_path  # noqa: E402
from acmpilot.modcod import ModcodTable, measure_thresholds  # noqa: E402
from acmpilot.predictor import (  # noqa: E402
    PredictivePolicy,
    analytic_calibration_report,
    calibration_report,
    make_lag_features,
)

TABLE_JSON = HERE / "modcod_thresholds.json"
TAUS_MS = (1.0, 2.0, 5.0, 10.0, 20.0)
MARGINAL = "lognormal"


def load_table() -> ModcodTable:
    """MODCOD table from ``validation/modcod_thresholds.json`` if present."""
    if TABLE_JSON.exists():
        print("  MODCOD table loaded from validation/modcod_thresholds.json")
        return ModcodTable.load_json(TABLE_JSON)
    print("  MODCOD table measured in-process")
    table, _ = measure_thresholds()
    return table


def run(marginal: str, taus_ms: tuple[float, ...], tag: str) -> int:
    """Run the whole V4 protocol on one channel marginal."""
    print(f"V4 PREDICTOR VALIDATION ({tag}) -- acmpilot")
    print("script: validation/validate_predictor.py" if marginal == "lognormal"
          else "script: validation/validate_predictor_gammagamma.py")
    table = load_table()
    bench = BenchmarkConfig()
    config = ChannelConfig(
        slot_s=1e-3, tau_c_s=10e-3, sigma_i2=0.5, mean_snr_db=14.0, marginal=marginal
    )
    print(
        f"  channel: {marginal}, slot = {config.slot_s * 1e3:g} ms, "
        f"tau_c = {config.tau_c_s * 1e3:g} ms, sigma_I^2 = {config.sigma_i2:g}, "
        f"SNR at mean irradiance = {config.mean_snr_db:g} dB"
    )
    print(
        f"  seeds: train {bench.split.train_seeds}, tune {bench.split.tune_seeds}, "
        f"test {bench.split.test_seeds}"
    )
    print(f"  {bench.n_slots} slots per scored episode; n_lags = {bench.n_lags}; "
          f"{bench.n_estimators} trees per quantile model")
    print()

    results = {}
    for tau_ms in taus_ms:
        results[tau_ms] = benchmark_at_delay(
            table, config, tau_s=tau_ms * 1e-3, bench=bench
        )

    # ---- V4.1 calibration ----------------------------------------------------
    print("V4.1 calibration of the stated intervals, measured on a held-out path")
    print("  nominal coverage_80 = 0.80, below_q10 = 0.10, above_q90 = 0.10")
    print(
        f"{'tau_ms':>7} {'predictor':>34} {'cover80':>8} {'below_q10':>10} "
        f"{'above_q90':>10} {'mae_dB':>7} {'rmse_dB':>8} {'spread_dB':>10} {'cross':>7}"
    )
    for tau_ms in taus_ms:
        res = results[tau_ms]
        d = int(res["delay_slots"])
        held = snr_db_path(config, bench.n_slots, 401)
        feat, targ, _ = make_lag_features(held, delay_slots=d, n_lags=bench.n_lags)
        preds = res["predictors"]
        rep_l = calibration_report(preds["learned"], feat, targ)
        rep_a = analytic_calibration_report(preds["analytic"], feat, targ)
        for label, rep in (
            ("learned quantile GBR", rep_l),
            ("analytic AR(1) MMSE (not learned)", rep_a),
        ):
            print(
                f"{tau_ms:7.1f} {label:>34} {rep['coverage_80']:8.4f} "
                f"{rep['below_q10']:10.4f} {rep['above_q90']:10.4f} "
                f"{rep['mae_q50']:7.4f} {rep['rmse_q50']:8.4f} "
                f"{rep['mean_spread_db']:10.4f} {rep['crossing_fraction']:7.4f}"
            )
    print()

    # ---- tuning record -------------------------------------------------------
    print("V4.2a hyperparameters selected on the tuning seeds")
    print(
        f"{'tau_ms':>7} {'fixed_margin_dB':>16} {'hyst_up_dB':>11} "
        f"{'hyst_down_dB':>13} {'analytic_gate':>14} {'learned_gate':>13} "
        f"{'train_rows':>11}"
    )
    for tau_ms in taus_ms:
        t = results[tau_ms]["tuned"]
        print(
            f"{tau_ms:7.1f} {t['fixed_margin_db']:16.2f} {t['hysteresis_up_db']:11.2f} "
            f"{t['hysteresis_down_db']:13.2f} {t['analytic_gate_k']:14.2f} "
            f"{t['learned_gate_k']:13.2f} {results[tau_ms]['n_train_rows']:11d}"
        )
    print()

    # ---- V4.2 scores ---------------------------------------------------------
    print("V4.2b held-out goodput, bit/symbol, 10 test seeds, paired sample paths")
    print(
        f"{'tau_ms':>7} {'variant':>11} {'policy':>54} {'goodput':>8} {'sem':>7} "
        f"{'eff':>6} {'outage':>8} {'conserv':>8} {'wasted':>7} {'lost':>7} "
        f"{'switch':>7} {'L?':>3}"
    )
    for tau_ms in taus_ms:
        for row in results[tau_ms]["scores"]:
            print(
                f"{tau_ms:7.1f} {str(row['variant']):>11} {str(row['policy'])[:54]:>54} "
                f"{row['goodput_bit_per_symbol']:8.4f} {row['goodput_sem']:7.4f} "
                f"{row['efficiency']:6.4f} {row['outage_fraction']:8.5f} "
                f"{row['conservative_fraction']:8.4f} "
                f"{row['wasted_bit_per_symbol']:7.4f} {row['lost_bit_per_symbol']:7.4f} "
                f"{row['switch_rate_per_slot']:7.4f} "
                f"{'yes' if row['learned'] else 'no':>3}"
            )
    print()

    # ---- V4.3 gate ablation --------------------------------------------------
    print("V4.3 confidence-gate ablation on the test seeds: tuned gate vs gate_k = 0")
    print("  gate_k = 0 selects on the point estimate alone, so the delta is exactly")
    print("  what the confidence output buys.")
    print(
        f"{'tau_ms':>7} {'predictor':>34} {'gate_k':>7} {'G_tuned':>8} "
        f"{'G_gate0':>8} {'delta':>8} {'out_tuned':>10} {'out_gate0':>10}"
    )
    for tau_ms in taus_ms:
        res = results[tau_ms]
        d = int(res["delay_slots"])
        for key, label in (
            ("learned", "learned quantile GBR"),
            ("analytic", "analytic AR(1) MMSE (not learned)"),
        ):
            gate = (
                res["tuned"]["learned_gate_k"] if key == "learned"
                else res["tuned"]["analytic_gate_k"]
            )
            tuned_row = score_policy(
                table, config,
                PredictivePolicy(
                    res["predictors"][key], delay_slots=d, n_lags=bench.n_lags,
                    gate_k=gate,
                ),
                tau_s=tau_ms * 1e-3, seeds=bench.split.test_seeds, n_slots=bench.n_slots,
            )
            zero_row = score_policy(
                table, config,
                PredictivePolicy(
                    res["predictors"][key], delay_slots=d, n_lags=bench.n_lags,
                    gate_k=0.0,
                ),
                tau_s=tau_ms * 1e-3, seeds=bench.split.test_seeds, n_slots=bench.n_slots,
            )
            print(
                f"{tau_ms:7.1f} {label:>34} {gate:7.2f} "
                f"{tuned_row['goodput_bit_per_symbol']:8.4f} "
                f"{zero_row['goodput_bit_per_symbol']:8.4f} "
                f"{tuned_row['goodput_bit_per_symbol'] - zero_row['goodput_bit_per_symbol']:8.4f} "
                f"{tuned_row['outage_fraction']:10.5f} {zero_row['outage_fraction']:10.5f}"
            )
    print()

    # ---- V4.4 verdict --------------------------------------------------------
    def _sep(a: dict, b: dict) -> tuple[float, float, str]:
        gap = float(a["goodput_bit_per_symbol"]) - float(b["goodput_bit_per_symbol"])
        comb = float(np.hypot(float(a["goodput_sem"]), float(b["goodput_sem"])))
        verdict = "yes" if comb > 0 and abs(gap) > 2 * comb else "NO"
        return gap, gap / comb if comb > 0 else float("nan"), verdict

    print("V4.4a learned vs the analytic (non-learned) predictor, per delay")
    print(
        f"{'tau_ms':>7} {'G_learned':>10} {'G_analytic':>11} {'gap':>8} "
        f"{'gap/sem':>8} {'separated':>10} {'winner':>26}"
    )
    for tau_ms in taus_ms:
        rows_ = {r["policy"]: r for r in results[tau_ms]["scores"]}
        learned_row = next(r for r in rows_.values() if r["learned"])
        analytic_row = next(
            r for r in rows_.values()
            if r["variant"] == "predictive" and not r["learned"]
        )
        gap, ratio, verdict = _sep(learned_row, analytic_row)
        winner = (
            "indistinguishable" if verdict == "NO"
            else ("learned" if gap > 0 else "analytic (not learned)")
        )
        print(
            f"{tau_ms:7.1f} {learned_row['goodput_bit_per_symbol']:10.4f} "
            f"{analytic_row['goodput_bit_per_symbol']:11.4f} {gap:8.4f} "
            f"{ratio:8.2f} {verdict:>10} {winner:>26}"
        )
    print()
    print("V4.4b best predictive policy vs the best TUNED non-predictive baseline")
    print(
        f"{'tau_ms':>7} {'best_predictive':>34} {'G':>8} {'best_tuned_baseline':>26} "
        f"{'G2':>8} {'gap':>8} {'gap_pct':>8} {'gap/sem':>8} {'separated':>10}"
    )
    for tau_ms in taus_ms:
        scores = results[tau_ms]["scores"]
        pred = max(
            (r for r in scores if r["variant"] == "predictive"),
            key=lambda r: float(r["goodput_bit_per_symbol"]),
        )
        base = max(
            (r for r in scores if r["variant"] == "tuned"),
            key=lambda r: float(r["goodput_bit_per_symbol"]),
        )
        gap, ratio, verdict = _sep(pred, base)
        print(
            f"{tau_ms:7.1f} {str(pred['policy'])[:34]:>34} "
            f"{pred['goodput_bit_per_symbol']:8.4f} "
            f"{str(base['policy'])[:26]:>26} {base['goodput_bit_per_symbol']:8.4f} "
            f"{gap:8.4f} "
            f"{100.0 * gap / float(base['goodput_bit_per_symbol']):7.2f}% "
            f"{ratio:8.2f} {verdict:>10}"
        )
    print("  'separated' means |gap| exceeds twice the combined standard error of the")
    print("  two means over 10 test seeds. A 'NO' means this trial count cannot tell")
    print("  the two apart, and the result must not be read as a win either way.")
    print()
    print(f"V4 COMPLETE ({tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(MARGINAL, TAUS_MS, "lognormal"))
