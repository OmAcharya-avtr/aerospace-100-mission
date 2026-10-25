"""Integrator invariants and convergence: which residual is numerical and which is physical.

What to notice: the energy residual falls by a factor of about 32 per halving
of the step, which is ``h^5``, so it is the integrator. The angular momentum
magnitude residual does not move at all across four step sizes, so it is not
the integrator: it is the ``J2`` torque, which precesses the angular momentum
vector about the polar axis at constant ``h_z`` while its magnitude oscillates.
No reference solution is needed to tell the two apart, which is what makes the
distinction solid.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, is_fast, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from orbitdrift import integrate  # noqa: E402
from orbitdrift.constants import WGS72  # noqa: E402
from orbitdrift.forces import ForceModel  # noqa: E402
from orbitdrift.kepler import (  # noqa: E402
    ClassicalElements,
    elements_to_rv,
    orbital_period_s,
    propagate_two_body,
)

TWO_BODY = ForceModel(zonal=False, drag=False, third_body=False)
ZONAL = ForceModel(zonal=True, drag=False, third_body=False)
LADDER = (90, 180, 360) if is_fast() else (90, 180, 360, 720, 1440)
REVOLUTIONS = 1 if is_fast() else 4


def main() -> None:
    state = elements_to_rv(
        ClassicalElements(
            WGS72.radius_km + 500.0, 0.001, float(np.deg2rad(51.6)), 0.3, 0.7, 0.2
        )
    )
    period = orbital_period_s(*state)

    energy = []
    magnitude = []
    polar = []
    position_error = []
    for steps_per_rev in LADDER:
        trajectory = integrate.propagate(
            np.atleast_2d(state[0]),
            np.atleast_2d(state[1]),
            2451545.0,
            np.array([0.0]),
            n_steps=steps_per_rev * REVOLUTIONS,
            step_s=period / steps_per_rev,
            force=ZONAL,
            sample_every=max(1, steps_per_rev // 6),
        )
        residuals = integrate.invariant_residuals(trajectory, zonal=True)
        energy.append(float(residuals["energy"][0]))
        magnitude.append(float(residuals["h_magnitude"][0]))
        polar.append(float(residuals["h_z"][0]))

        two_body = integrate.propagate(
            np.atleast_2d(state[0]),
            np.atleast_2d(state[1]),
            2451545.0,
            np.array([0.0]),
            n_steps=steps_per_rev * REVOLUTIONS,
            step_s=period / steps_per_rev,
            force=TWO_BODY,
            sample_every=steps_per_rev * REVOLUTIONS,
        )
        reference, _ = propagate_two_body(state[0], state[1], two_body.times_s)
        position_error.append(
            float(np.max(np.linalg.norm(two_body.positions_km[0] - reference, axis=-1)))
        )

    steps = np.array([period / value for value in LADDER])

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    axes[0].loglog(steps, energy, "o-", color=SERIES[0], linewidth=1.3, label="energy")
    axes[0].loglog(steps, polar, "s-", color=SERIES[2], linewidth=1.3, label=r"$h_z$")
    axes[0].loglog(
        steps, magnitude, "^-", color=SERIES[1], linewidth=1.3, label=r"$|h|$ (physical)"
    )
    guide = energy[-1] * (steps / steps[-1]) ** 5
    axes[0].loglog(steps, guide, ":", color="#777777", linewidth=1.1, label=r"$h^5$ guide")
    axes[0].set_xlabel("integration step, s")
    axes[0].set_ylabel("relative residual over 4 revolutions")
    axes[0].set_title("two-body plus J2-J4: energy and $h_z$ shrink, $|h|$ does not")
    axes[0].legend(fontsize=8)

    axes[1].loglog(
        steps, position_error, "o-", color=SERIES[0], linewidth=1.3, label="measured"
    )
    guide = position_error[-1] * (steps / steps[-1]) ** 4
    axes[1].loglog(steps, guide, ":", color="#777777", linewidth=1.1, label=r"$h^4$ guide")
    orders = [
        float(np.log2(position_error[index] / position_error[index + 1]))
        for index in range(len(LADDER) - 1)
    ]
    axes[1].set_xlabel("integration step, s")
    axes[1].set_ylabel("worst position error against the closed form, km")
    axes[1].set_title(
        "RK4 against the exact two-body solution: order "
        f"{min(orders):.2f} to {max(orders):.2f}"
    )
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "orbitdrift: the integrator's error is measured, and separated from the physics",
        fontsize=10,
    )
    figure.tight_layout()
    print(f"wrote {save(figure, 'conservation_and_convergence')}")
    print("  steps/rev   step s    energy       h_z          |h|          |dr| km")
    for index, steps_per_rev in enumerate(LADDER):
        print(
            f"  {steps_per_rev:9d}  {steps[index]:7.3f}   {energy[index]:.4e}   "
            f"{polar[index]:.4e}   {magnitude[index]:.4e}   {position_error[index]:.4e}"
        )
    print(f"  measured convergence order against the closed form: "
          f"{min(orders):.4f} to {max(orders):.4f}")
    print(f"  |h| residual relative spread across the ladder: "
          f"{float(np.ptp(magnitude) / np.mean(magnitude)):.3e}")


if __name__ == "__main__":
    main()
