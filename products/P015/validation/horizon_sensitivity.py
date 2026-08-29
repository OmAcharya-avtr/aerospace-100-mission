"""V3 — Sensitivity of the learned policy to the prediction horizon.

Trains and evaluates the learned predictive policy at several prediction
horizons H (in steps), holding everything else fixed, and reports how
throughput / outage fraction / switch count vary with H. A short horizon
gives the model little lead time to react; a long horizon labels far more
steps as "imminent outage" (see ``features.label_imminent_outage`` -- the
label event is monotonically more likely as H grows), which is expected to
increase both switch count and false-positive-driven throughput loss.

Run: ``python3 validation/horizon_sensitivity.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkswitch.learn import train_outage_predictor  # noqa: E402
from linkswitch.metrics import compare_policies  # noqa: E402
from linkswitch.optical import OpticalParams  # noqa: E402
from linkswitch.policies import LearnedPolicy  # noqa: E402
from linkswitch.rf import RFParams  # noqa: E402
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry  # noqa: E402


def main() -> None:
    t_start = time.perf_counter()

    opt = OpticalParams(sigma_i2=0.4, coherence_steps=4.0, margin_db=4.0, rate_mbps=1000.0)
    rf = RFParams(rate_mbps=150.0)
    cfg = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=1))
    tau = opt.tau_phys
    window = 6
    confidence = 0.5
    n_steps, n_reps = 1500, 120

    horizons = [1, 2, 3, 5, 8, 12, 20]

    print(f"Scenario: sigma_i2={opt.sigma_i2}, margin_db={opt.margin_db}, "
          f"coherence={opt.coherence_steps}, downtime_steps=1, window={window}, "
          f"confidence_threshold={confidence}")
    print(f"n_steps={n_steps}, n_reps={n_reps} per horizon (paired Monte Carlo, 95% CI)")
    print(f"{'horizon':>8}{'throughput Mb/s':>28}{'outage frac':>26}{'switches':>14}")

    rows = []
    for h in horizons:
        train_tels = [generate_telemetry(cfg, 500, seed=70_000 + i) for i in range(15)]
        model = train_outage_predictor(train_tels, tau_phys=tau, horizon=h, window=window,
                                        random_state=0)
        factories = {
            "learned": lambda m=model: LearnedPolicy(m, tau_phys=tau,
                                                      confidence_threshold=confidence,
                                                      window=window),
        }
        results = compare_policies(cfg, factories, n_steps=n_steps, n_reps=n_reps, seed0=1000)
        agg = results["learned"]
        t, o, s = agg["throughput_mbps"], agg["outage_fraction"], agg["switch_count"]
        print(f"{h:>8}{t.mean:>10.3f} [{t.ci_low:.3f}, {t.ci_high:.3f}]"
              f"{o.mean:>12.4f} [{o.ci_low:.4f}, {o.ci_high:.4f}]{s.mean:>8.2f}")
        rows.append((h, t.mean, o.mean, s.mean))

    best_throughput_h = max(rows, key=lambda r: r[1])[0]
    best_outage_h = min(rows, key=lambda r: r[2])[0]
    fewest_switch_h = min(rows, key=lambda r: r[3])[0]
    print(f"\nHorizon maximising throughput : H={best_throughput_h}")
    print(f"Horizon minimising outage      : H={best_outage_h}")
    print(f"Horizon minimising switches    : H={fewest_switch_h}")

    switches_by_h = {h: s for h, _, _, s in rows}
    monotone_switches_nondecreasing = all(
        switches_by_h[horizons[i]] <= switches_by_h[horizons[i + 1]] + 1e-9
        for i in range(len(horizons) - 1)
    )
    print(f"\nSwitch count monotonically non-decreasing in H across the full sweep: "
          f"{monotone_switches_nondecreasing}")
    print("(expected per the module docstring: a longer horizon labels more steps as "
          "'imminent outage', so the trained model triggers preemptive switches more "
          "often; whether this is monotone in practice, given RandomForest training "
          "noise, is reported as measured above, not assumed.)")

    elapsed = time.perf_counter() - t_start
    print(f"\nWall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
