"""V7 - regenerate the pinned regression baseline used by tests/test_regression.py.

Prints the exact dictionary of pinned values. If a deliberate numerical
change is made to the package, run this script and paste the output into
``PINNED`` in tests/test_regression.py, recording the reason in CHANGELOG.md.

Run: python validation/v7_regression_baseline.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackbench.control import (  # noqa: E402
    LQRController,
    PIDController,
    lqr_weights_from_bandwidth,
    pid_gains_from_bandwidth,
    step_response,
)
from trackbench.dynamics import GimbalAxis, JitterPSD, synthesize_jitter  # noqa: E402
from trackbench.reacq import (  # noqa: E402
    AlwaysFullPolicy,
    AlwaysLocalPolicy,
    ReacqConfig,
    evaluate_policy,
    train_q_learning,
)
from trackbench.scan import (  # noqa: E402
    GaussianUncertainty,
    coverage_fraction,
    expected_acquisition_time_spiral,
    raster_scan,
    simulate_acquisition,
    spiral_scan,
)
from trackbench.sim import DEFAULT_SCENARIO, run_episode  # noqa: E402


def collect() -> dict:
    """Compute every pinned regression value."""
    out: dict = {}
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, overlap=0.25, containment=0.995, dwell_time=1e-3)
    out["spiral_n_points"] = p.n_points
    out["spiral_max_radius"] = p.max_radius
    out["spiral_track_spacing"] = p.track_spacing
    out["spiral_scan_speed"] = p.scan_speed
    out["spiral_coverage"] = coverage_fraction(
        p, u, n_samples=20000, rng=np.random.default_rng(123)
    )
    out["raster_n_points"] = raster_scan(u, 2e-5).n_points
    out["acq_time_seed7"] = simulate_acquisition(
        p, np.array([1.5e-4, -2.0e-4]), p_dwell=0.9, rng=np.random.default_rng(7)
    )
    out["expected_acq_time"] = expected_acquisition_time_spiral(
        u, 2e-5, 0.25, p.scan_speed, containment=0.995, p_pass=0.9
    )

    x = synthesize_jitter(JitterPSD(1e-12, 3.0, 2.0), 4096, 5000.0,
                          np.random.default_rng(31))
    out["jitter_std"] = float(np.std(x))
    out["jitter_first"] = float(x[0])
    out["jitter_last"] = float(x[-1])

    kp, _, kd = pid_gains_from_bandwidth(0.05, 2 * math.pi * 5, 0.707)
    _, _, m = step_response(
        GimbalAxis(0.05, 0.02, 2.0, 1.0), PIDController(kp, 0.0, kd, 2.0), 1e-4, 1e-4, 1.0
    )
    out["pd_overshoot"] = m.overshoot
    out["pd_rise"] = m.rise_time
    out["pd_peak"] = m.peak_time
    q, qr, r = lqr_weights_from_bandwidth(0.05, 2 * math.pi * 5)
    lqr = LQRController(GimbalAxis(0.05, 0.02, 2.0, 1.0), q_angle=q, q_rate=qr, r_torque=r)
    out["lqr_gain"] = tuple(float(g) for g in lqr.gain)

    res = run_episode(DEFAULT_SCENARIO, seed=99, keep_series=False)
    out["episode99_acq"] = res.acquisition_time_s
    out["episode99_rms"] = res.track_rms_rad
    out["episode99_peak"] = res.track_peak_rad
    out["episode99_loss"] = res.loss_time_s
    out["episode99_reacq"] = res.reacq_time_s
    out["episode99_attempts"] = res.reacq_attempts
    out["episode99_total"] = res.total_time_s

    pol = train_q_learning(ReacqConfig(), episodes=2000, seed=777)
    out["q_sum_2000ep_seed777"] = float(pol.q.sum())
    out["q_visits_2000ep_seed777"] = int(pol.visits.sum())
    ef = evaluate_policy(AlwaysFullPolicy(), ReacqConfig(), n_episodes=300, seed=555)
    out["full_mean_300ep_seed555"] = ef["mean_time_s"]
    out["full_success_300ep_seed555"] = ef["success_rate"]
    el = evaluate_policy(AlwaysLocalPolicy(), ReacqConfig(), n_episodes=300, seed=555)
    out["local_mean_300ep_seed555"] = el["mean_time_s"]
    out["local_success_300ep_seed555"] = el["success_rate"]
    return out


def main() -> int:
    """Print the pinned baseline and compare with the committed values."""
    print("V7 - regression baseline regeneration")
    got = collect()
    print("PINNED = {")
    for k, v in got.items():
        print(f"    {k!r}: {v!r},")
    print("}")
    print()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    try:
        from test_regression import PINNED as committed
    except ImportError:  # pragma: no cover
        print("could not import committed baseline; skipping comparison")
        return 0

    print(f"{'key':>30} {'committed':>22} {'current':>22} {'match':>7}")
    print("-" * 84)
    all_ok = True
    for k, want in committed.items():
        have = got.get(k)
        if isinstance(want, tuple):
            ok = have is not None and all(
                abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(have, want)
            )
            shown_w, shown_h = f"{want[0]:.6g},...", f"{have[0]:.6g},..."
        elif isinstance(want, (int, float)):
            ok = have is not None and abs(have - want) <= 1e-9 * max(1.0, abs(want))
            shown_w, shown_h = f"{want:.10g}", f"{have:.10g}"
        else:  # pragma: no cover
            ok = have == want
            shown_w, shown_h = str(want), str(have)
        all_ok &= ok
        print(f"{k:>30} {shown_w:>22} {shown_h:>22} {'ok' if ok else 'DIFF':>7}")
    print()
    print("RESULT:", "PASS" if all_ok else "FAIL (baseline drift - investigate)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
