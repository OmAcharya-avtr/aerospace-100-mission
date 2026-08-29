"""Example 2: Monte Carlo policy comparison with confidence intervals.

Simulated data only (see package docstring / README). Saves
``../screenshots/policy_comparison.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkswitch.learn import train_outage_predictor  # noqa: E402
from linkswitch.metrics import compare_policies  # noqa: E402
from linkswitch.optical import OpticalParams  # noqa: E402
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy  # noqa: E402
from linkswitch.rf import RFParams  # noqa: E402
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry  # noqa: E402


def main() -> None:
    opt = OpticalParams(sigma_i2=0.4, coherence_steps=4.0, margin_db=4.0, rate_mbps=1000.0)
    rf = RFParams(rate_mbps=150.0)
    cfg = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=1))
    tau = opt.tau_phys

    train_tels = [generate_telemetry(cfg, 500, seed=300 + i) for i in range(15)]
    model = train_outage_predictor(train_tels, tau_phys=tau, horizon=5, window=6, random_state=0)

    factories = {
        "fixed_threshold": lambda: FixedThresholdPolicy(tau=tau),
        "hysteresis": lambda: HysteresisPolicy(tau_low=tau * 0.85, tau_high=tau * 1.15),
        "learned": lambda: LearnedPolicy(model, tau_phys=tau, confidence_threshold=0.5, window=6),
    }
    results = compare_policies(cfg, factories, n_steps=2000, n_reps=150, seed0=0)

    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metrics = [
        ("throughput_mbps", "delivered throughput [Mb/s]", axes[0]),
        ("outage_fraction", "outage fraction", axes[1]),
        ("switch_count", "switches per episode", axes[2]),
    ]
    colors = ["#57606a", "#1f6feb", "#2da44e"]
    for metric_key, ylabel, ax in metrics:
        means = [results[n][metric_key].mean for n in names]
        los = [results[n][metric_key].mean - results[n][metric_key].ci_low for n in names]
        his = [results[n][metric_key].ci_high - results[n][metric_key].mean for n in names]
        ax.bar(names, means, yerr=[los, his], capsize=4, color=colors)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Policy comparison, 95% CI over {results[names[0]]['throughput_mbps'].n} "
                 "paired Monte Carlo episodes (SIMULATED telemetry)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = Path(__file__).resolve().parents[1] / "screenshots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "policy_comparison.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")

    print()
    for name in names:
        t = results[name]["throughput_mbps"]
        o = results[name]["outage_fraction"]
        s = results[name]["switch_count"]
        print(f"{name:16s} throughput={t.mean:8.3f} [{t.ci_low:.3f},{t.ci_high:.3f}]  "
              f"outage={o.mean:.4f} [{o.ci_low:.4f},{o.ci_high:.4f}]  "
              f"switches={s.mean:.2f}")

    best = max(names, key=lambda n: results[n]["throughput_mbps"].mean)
    print(f"\nHighest mean throughput: {best}")
    assert np.isfinite(results[best]["throughput_mbps"].mean)


if __name__ == "__main__":
    main()
