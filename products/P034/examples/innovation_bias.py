"""Measured innovation shift from an injected sensor bias, against its closed form.

Left: the biased-minus-nominal innovation sequence for a 1.0 m position bias
and for a 1.0 m/s velocity bias, each with the closed-form prediction

    Delta e_k = b - H (I - M)^-1 (I - M^k) A K b,     M = A (I - K H)

overlaid. Right: the same quantity divided by the bias magnitude for five
magnitudes, which collapse onto one curve because the shift is exactly linear
in the bias.

The derivation, its assumptions and the tolerances are in
validation/validate_analytic_bias.py.

Writes ../screenshots/innovation_bias.png.
Runtime on the 1-core build container: under 2 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.faults import Injection  # noqa: E402
from faultinject.harness import nominal_trace, run_case  # noqa: E402
from faultinject.target import kalman_gain, plant_matrices  # noqa: E402
from faultinject.taxonomy import FaultKind  # noqa: E402

N = 150
SEED = 11
I2 = np.eye(2)
A, _ = plant_matrices()
K = kalman_gain()
H = np.eye(2)
M = A @ (I2 - K @ H)
G = I2 - H @ np.linalg.inv(I2 - M) @ A @ K


def analytic(b: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(I2 - M)
    out = np.zeros((N, 2))
    mk = I2.copy()
    for k in range(N):
        out[k] = b - H @ inv @ (I2 - mk) @ A @ K @ b
        mk = mk @ M
    return out


def measured(channel: str, offset: float) -> np.ndarray:
    inj = Injection.create(FaultKind.SENSOR_BIAS, channel, {"offset": offset}, 0, N)
    f = run_case([inj], SEED, N)
    nom = nominal_trace(SEED, N)
    return np.array(
        [[a[1] - c[1], a[2] - c[2]] for a, c in zip(f.innovations, nom.innovations, strict=True)]
    )


fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.6, 5.0))

for channel, bvec, colour in (
    ("pos", np.array([1.0, 0.0]), "#4c72b0"),
    ("vel", np.array([0.0, 1.0]), "#c44e52"),
):
    meas = measured(channel, 1.0)
    exact = analytic(bvec)
    err = float(np.max(np.abs(meas - exact)))
    ss = float((G @ bvec)[0])
    print(f"bias on {channel}: max |measured - closed form| = {err:.3e}, "
          f"steady-state position-innovation shift = {ss:.9f}")
    ax_left.plot(meas[:, 0], color=colour, linewidth=2.2, alpha=0.5,
                 label=f"measured, bias on {channel}")
    ax_left.plot(exact[:, 0], color=colour, linewidth=1.0, linestyle="--",
                 label=f"closed form, bias on {channel}")
    ax_left.axhline(ss, color=colour, linewidth=0.8, linestyle=":")

ax_left.set_xlabel("step")
ax_left.set_ylabel("position-innovation shift (m)")
ax_left.set_title("Biased minus nominal innovation, b = 1.0")
ax_left.grid(alpha=0.3)
ax_left.legend(fontsize=8)

ref = measured("pos", 1.0)
worst = 0.0
for offset in (0.1, 0.25, 1.0, 4.0, 10.0):
    scaled = measured("pos", offset) / offset
    worst = max(worst, float(np.max(np.abs(scaled - ref))))
    ax_right.plot(scaled[:, 0], linewidth=1.2, label=f"b = {offset:g} m")
print(f"linearity: worst deviation of Delta e / b across five magnitudes = {worst:.3e}")
ax_right.set_xlabel("step")
ax_right.set_ylabel("position-innovation shift per metre of bias")
ax_right.set_title(f"Linearity in the bias: curves agree to {worst:.1e}")
ax_right.grid(alpha=0.3)
ax_right.legend(fontsize=8)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "innovation_bias.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
