"""V2 — Policy comparison with confidence intervals over seeded Monte Carlo.

Compares the two classical baselines (fixed-threshold, hysteresis) against
the learned predictive policy on the same paired, seeded Monte Carlo
episodes, reporting 95% confidence intervals on delivered throughput,
outage fraction and switch count (not just point means).

**Whichever policy wins on a metric is reported as measured, without
retuning to change the outcome** (per the mission's engineering-honesty
requirement for AI products).

Run: ``python3 validation/policy_comparison_ci.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkswitch.learn import train_outage_predictor  # noqa: E402
from linkswitch.metrics import compare_policies  # noqa: E402
from linkswitch.optical import OpticalParams  # noqa: E402
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy  # noqa: E402
from linkswitch.rf import RFParams  # noqa: E402
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry  # noqa: E402


def run_scenario(name: str, opt: OpticalParams, rf: RFParams, downtime_steps: int,
                  n_steps: int, n_reps: int, horizon: int, window: int,
                  confidence: float) -> dict:
    cfg = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=downtime_steps))
    tau = opt.tau_phys

    train_tels = [generate_telemetry(cfg, 500, seed=40_000 + i) for i in range(15)]
    model = train_outage_predictor(train_tels, tau_phys=tau, horizon=horizon, window=window,
                                    random_state=0)

    factories = {
        "fixed_threshold": lambda: FixedThresholdPolicy(tau=tau),
        "hysteresis": lambda: HysteresisPolicy(tau_low=tau * 0.85, tau_high=tau * 1.15),
        "learned": lambda: LearnedPolicy(model, tau_phys=tau, confidence_threshold=confidence,
                                          window=window),
    }
    results = compare_policies(cfg, factories, n_steps=n_steps, n_reps=n_reps, seed0=0)

    print(f"\n--- Scenario: {name} "
          f"(sigma_i2={opt.sigma_i2}, margin_db={opt.margin_db}, coherence={opt.coherence_steps}, "
          f"downtime_steps={downtime_steps}, horizon={horizon}) ---")
    print(f"n_steps={n_steps}, n_reps={n_reps} (paired Monte Carlo, 95% CI)")
    header = f"{'policy':<18}{'throughput Mb/s':>28}{'outage frac':>26}{'switches':>14}"
    print(header)
    for pname, agg in results.items():
        t = agg["throughput_mbps"]
        o = agg["outage_fraction"]
        s = agg["switch_count"]
        print(
            f"{pname:<18}"
            f"{t.mean:>10.3f} [{t.ci_low:.3f}, {t.ci_high:.3f}]"
            f"{o.mean:>12.4f} [{o.ci_low:.4f}, {o.ci_high:.4f}]"
            f"{s.mean:>8.2f}"
        )
    best_throughput = max(results, key=lambda n: results[n]["throughput_mbps"].mean)
    best_outage = min(results, key=lambda n: results[n]["outage_fraction"].mean)
    best_switches = min(results, key=lambda n: results[n]["switch_count"].mean)
    print(f"Highest mean throughput : {best_throughput}")
    print(f"Lowest mean outage      : {best_outage}")
    print(f"Fewest mean switches    : {best_switches}")
    if best_throughput != "learned":
        print(f"HONEST RESULT: the learned policy does NOT win on throughput in this "
              f"scenario ('{best_throughput}' does).")
    return results


def main() -> None:
    t_start = time.perf_counter()

    # Scenario A: the package default (mild turbulence, generous margin) --
    # outages are rare, so all policies cluster near the ceiling.
    run_scenario(
        "mild (package defaults)",
        OpticalParams(sigma_i2=0.25, coherence_steps=5.0, margin_db=6.0, rate_mbps=1000.0),
        RFParams(rate_mbps=150.0),
        downtime_steps=1, n_steps=2000, n_reps=200, horizon=5, window=8, confidence=0.5,
    )

    # Scenario B: moderate turbulence, tighter margin -- outages and
    # switching decisions matter much more; this is the scenario used in
    # examples/policy_comparison.py and MODEL_CARD.md.
    run_scenario(
        "moderate (tighter margin, more turbulence)",
        OpticalParams(sigma_i2=0.4, coherence_steps=4.0, margin_db=4.0, rate_mbps=1000.0),
        RFParams(rate_mbps=150.0),
        downtime_steps=1, n_steps=2000, n_reps=200, horizon=5, window=6, confidence=0.5,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nWall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
