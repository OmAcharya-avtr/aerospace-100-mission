"""V1 — Analytic optimal fixed-threshold check.

Two independent things are checked against each other:

(a) Two independent numerical searches over the SAME closed-form objective
    J(z_th) (bounded scalar optimisation vs. a fine grid search) must agree
    with each other -- a self-consistency check on the optimizer.

(b) The closed-form-derived optimum must agree with an INDEPENDENT
    Monte-Carlo argmax: actually simulate the fixed-threshold policy (via
    ``simulate.py``, a completely separate code path from ``analytic.py``)
    across a grid of tau values on freshly generated telemetry, and find
    the empirical throughput-maximising tau. This is the real cross-check
    requested by the product spec: "the optimal fixed switching threshold
    found numerically matches the value derived from the channel
    statistics."

Two regimes are checked:
  - Zero switch cost (frictionless): the closed form predicts the optimum
    equals the physical outage threshold tau_phys exactly (a provable
    identity -- see analytic.py docstring). This is the primary, most
    rigorous check.
  - Non-zero switch cost (downtime_steps=1, matching the package default):
    the closed form predicts the objective becomes monotonically
    non-increasing as tau approaches tau_phys from below, i.e. NO proactive
    switching (deep-tail tau) becomes optimal for the fixed-threshold
    family, once realistic switch downtime is included. This is verified
    directionally against the Monte Carlo sweep, and is exactly the honest
    finding motivating the hysteresis policy (see MODEL_CARD.md / README).

Run: ``python3 validation/analytic_threshold_check.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkswitch.analytic import (  # noqa: E402
    optimal_threshold_analytic,
    optimal_threshold_grid,
)
from linkswitch.optical import OpticalParams  # noqa: E402
from linkswitch.policies import FixedThresholdPolicy  # noqa: E402
from linkswitch.rf import RFParams  # noqa: E402
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry  # noqa: E402
from linkswitch.simulate import simulate_policy  # noqa: E402


def mc_argmax_tau(config: ScenarioConfig, taus: np.ndarray, n_steps: int, n_reps: int,
                   seed0: int) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean throughput at each tau in `taus`, over n_reps seeded episodes."""
    means = np.empty_like(taus)
    for i, tau in enumerate(taus):
        policy = FixedThresholdPolicy(tau=float(tau))
        totals = []
        for r in range(n_reps):
            tel = generate_telemetry(config, n_steps=n_steps, seed=seed0 + r)
            select = policy.select_channels(tel)
            m = simulate_policy(tel, select, config)
            totals.append(m.throughput_mbps)
        means[i] = float(np.mean(totals))
    return means, taus


def main() -> None:
    t_start = time.perf_counter()
    opt = OpticalParams(sigma_i2=0.3, coherence_steps=4.0, margin_db=6.0, rate_mbps=1000.0)
    rf = RFParams(rate_mbps=150.0)

    print("=" * 78)
    print("V1a: Zero switch-cost limit -- closed form vs grid search")
    print("=" * 78)
    cfg0 = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=0))
    ana0 = optimal_threshold_analytic(opt, rf, downtime_steps=0.0)
    grid0 = optimal_threshold_grid(opt, rf, downtime_steps=0.0, n_points=20001)
    print(f"z_phys (physical outage threshold, standardised) = {ana0.z_phys:.8f}")
    print(f"rho (AR(1) lag-1 correlation)                     = {ana0.rho:.8f}")
    print(f"bounded optimizer: z_th* = {ana0.z_th:.8f}  tau* = {ana0.tau:.8f}")
    print(f"grid search:       z_th* = {grid0.z_th:.8f}  tau* = {grid0.tau:.8f}")
    print(f"|z_th_optimizer - z_th_grid|  = {abs(ana0.z_th - grid0.z_th):.3e}")
    print(f"|z_th_optimizer - z_phys|     = {abs(ana0.z_th - ana0.z_phys):.3e}")
    pass_a = abs(ana0.z_th - ana0.z_phys) < 1e-4 and abs(ana0.z_th - grid0.z_th) < 1e-2
    print(f"PASS (tolerance 1e-4 vs z_phys, 1e-2 vs grid): {pass_a}")

    print()
    print("=" * 78)
    print("V1b: Zero switch-cost -- Monte Carlo argmax cross-check (independent code path)")
    print("=" * 78)
    taus = np.array([
        opt.tau_phys * f for f in (0.5, 0.7, 0.85, 0.93, 1.0, 1.07, 1.15, 1.3, 1.5, 2.0)
    ])
    n_steps_mc, n_reps_mc = 3000, 40
    means, _ = mc_argmax_tau(cfg0, taus, n_steps=n_steps_mc, n_reps=n_reps_mc, seed0=5000)
    for tau_i, mean_i in zip(taus, means, strict=True):
        marker = "  <-- tau_phys" if abs(tau_i - opt.tau_phys) < 1e-9 else ""
        print(f"  tau={tau_i:.6f}  mean_throughput={mean_i:.4f}{marker}")
    i_best = int(np.argmax(means))
    tau_mc_best = float(taus[i_best])
    print(f"MC argmax tau = {tau_mc_best:.6f}  (tau_phys = {opt.tau_phys:.6f})")
    rel_diff = abs(tau_mc_best - opt.tau_phys) / opt.tau_phys
    print(f"relative difference vs tau_phys = {rel_diff:.4f}")
    # Coarse grid -> loose tolerance; the analytic curve is very flat near
    # its unique maximum (see V1a), so several grid points are statistically
    # indistinguishable at this n_reps. Require the MC argmax to fall within
    # one grid step either side of tau_phys.
    idx_phys = int(np.argmin(np.abs(taus - opt.tau_phys)))
    pass_b = abs(i_best - idx_phys) <= 1
    print(f"PASS (MC argmax within 1 grid point of tau_phys): {pass_b}")

    print()
    print("=" * 78)
    print("V1c: Realistic switch cost (downtime_steps=1) -- direction of the shift")
    print("=" * 78)
    ana1 = optimal_threshold_analytic(opt, rf, downtime_steps=1.0)
    print(f"closed-form optimum with downtime=1: z_th* = {ana1.z_th:.6f}  tau* = {ana1.tau:.6f}")
    print(f"(compare to zero-cost optimum tau* = {ana0.tau:.6f} = tau_phys)")
    print("closed form predicts the optimum moves to tau* << tau_phys "
          "(deep-tail, i.e. near 'never proactively switch')")

    cfg1 = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=1))
    taus_c = np.array([opt.tau_phys * f for f in (0.05, 0.2, 0.5, 0.8, 1.0, 1.2)])
    means_c, _ = mc_argmax_tau(cfg1, taus_c, n_steps=3000, n_reps=40, seed0=6000)
    for tau_i, mean_i in zip(taus_c, means_c, strict=True):
        print(f"  tau={tau_i:.6f}  mean_throughput={mean_i:.4f}")
    monotone_decreasing_near_phys = means_c[-2] >= means_c[-1] and means_c[0] >= means_c[1]
    print(f"MC throughput is highest at the smallest tau tested: "
          f"{int(np.argmax(means_c)) == 0}")
    print(f"MC throughput decreases (or is flat) approaching tau_phys from below: "
          f"{monotone_decreasing_near_phys}")
    pass_c = int(np.argmax(means_c)) == 0
    print(f"PASS (direction matches closed-form prediction): {pass_c}")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Wall time: {elapsed:.1f} s")
    print(f"OVERALL: {'PASS' if (pass_a and pass_b and pass_c) else 'FAIL'}")


if __name__ == "__main__":
    main()
