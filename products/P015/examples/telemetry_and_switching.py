"""Example 1: visualise one telemetry episode and how each policy switches.

Simulated data only (see package docstring / README). Saves
``../screenshots/telemetry_and_switching.png``.
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
from linkswitch.optical import OpticalParams  # noqa: E402
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy  # noqa: E402
from linkswitch.rf import RFParams  # noqa: E402
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry  # noqa: E402


def main() -> None:
    opt = OpticalParams(sigma_i2=0.4, coherence_steps=4.0, margin_db=4.0, rate_mbps=1000.0)
    rf = RFParams(rate_mbps=150.0)
    cfg = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=1))
    tau = opt.tau_phys

    n_steps = 400
    tel = generate_telemetry(cfg, n_steps=n_steps, seed=17)

    train_tels = [generate_telemetry(cfg, 500, seed=200 + i) for i in range(10)]
    model = train_outage_predictor(train_tels, tau_phys=tau, horizon=5, window=6, random_state=0)

    policies = {
        "fixed_threshold": FixedThresholdPolicy(tau=tau),
        "hysteresis": HysteresisPolicy(tau_low=tau * 0.85, tau_high=tau * 1.15),
        "learned": LearnedPolicy(model, tau_phys=tau, confidence_threshold=0.5, window=6),
    }

    t = np.arange(n_steps)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True,
                              height_ratios=[3, 1, 1, 1])

    ax0 = axes[0]
    ax0.plot(t, tel.irradiance, color="#1f6feb", lw=0.9, label="optical irradiance I(t)")
    ax0.axhline(tau, color="#d1242f", ls="--", lw=1.2, label=r"physical outage $\tau_{phys}$")
    ax0.fill_between(t, 0, tau, color="#d1242f", alpha=0.08)
    ax0.set_ylabel("mean-norm. irradiance")
    ax0.set_title("LinkSwitch: one simulated episode (SIMULATED fading, not measured)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_ylim(bottom=0)

    for ax, (name, policy) in zip(axes[1:], policies.items(), strict=True):
        select = policy.select_channels(tel)
        ax.imshow(
            select[np.newaxis, :], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
            extent=[0, n_steps, 0, 1], interpolation="nearest",
        )
        n_switches = int(np.sum(select[1:] != select[:-1]))
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=9)
        ax.text(1.01, 0.5, f"{n_switches} switches", transform=ax.transAxes,
                va="center", fontsize=8)

    axes[-1].set_xlabel("time step")
    fig.text(0.13, 0.005, "green = optical selected, red = RF selected", fontsize=8,
              color="#57606a")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    out_dir = Path(__file__).resolve().parents[1] / "screenshots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "telemetry_and_switching.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
