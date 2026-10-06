"""Dead-time forward maps, the paralyzable maximum, and both inverse branches.

Writes ``../screenshots/deadtime_curves.png``.

What to notice. Left: the non-paralyzable curve rises monotonically to 1/tau; the
paralyzable curve peaks at n tau = 1 and then **falls**. A detector past that peak
reports a lower count rate as the source gets brighter, which reads as a fading
link. Right: the inverse. Below the peak every observed rate has two true rates,
one on each branch, and the amplification of a relative error in the observed rate
(the dashed line, right axis) diverges as the peak is approached. That divergence
is why the preflight check in ``photoncount.ops`` refuses to start a run at or
above 1/tau.

Runtime: about 5 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from photoncount.deadtime import (  # noqa: E402
    nonparalyzable_observed,
    paralyzable_maximum,
    paralyzable_observed,
    paralyzable_true,
)

TAU = 1e-7
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "deadtime_curves.png"


def main() -> int:
    n_max, m_max = paralyzable_maximum(TAU)
    x = np.linspace(1e-4, 4.0, 4001)
    n = x / TAU

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax_left.plot(x, nonparalyzable_observed(n, TAU) * TAU, color="#1b4965", lw=2.0,
                 label=r"non-paralyzable  $m=n/(1+n\tau)$")
    ax_left.plot(x, paralyzable_observed(n, TAU) * TAU, color="#c1121f", lw=2.0,
                 label=r"paralyzable  $m=n e^{-n\tau}$")
    ax_left.plot(x, x, "k:", lw=1.2, label="ideal counter")
    ax_left.axvline(1.0, color="grey", ls="--", lw=1.0)
    ax_left.axhline(1.0 / np.e, color="grey", ls="--", lw=1.0)
    ax_left.annotate(
        f"peak at $n\\tau=1$\n$m\\tau=1/e={1 / np.e:.4f}$\n({m_max:.3e} counts/s)",
        (1.0, 1 / np.e),
        textcoords="offset points",
        xytext=(18, -34),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax_left.set_xlabel(r"true loading $n\tau$")
    ax_left.set_ylabel(r"observed loading $m\tau$")
    ax_left.set_title(f"Forward maps, tau = {TAU * 1e9:.0f} ns")
    ax_left.legend(fontsize=8, loc="upper left")
    ax_left.grid(alpha=0.3)

    frac = np.linspace(0.005, 0.9995, 2001)
    m = frac * m_max
    lower = paralyzable_true(m, TAU, "lower")
    upper = paralyzable_true(m, TAU, "upper")
    ax_right.plot(frac, lower * TAU, color="#1b4965", lw=2.0, label="lower branch $W_0$")
    ax_right.plot(frac, upper * TAU, color="#c1121f", lw=2.0, label=r"upper branch $W_{-1}$")
    ax_right.axhline(1.0, color="grey", ls="--", lw=1.0)
    ax_right.set_xlabel(r"observed rate as a fraction of $m_{max}=1/(e\tau)$")
    ax_right.set_ylabel(r"recovered $n\tau$")
    ax_right.set_yscale("log")
    ax_right.set_title("The inverse is two-valued, and ill-conditioned near the peak")
    ax_right.grid(alpha=0.3)

    twin = ax_right.twinx()
    dm = m * 1e-6
    n0 = paralyzable_true(m, TAU, "lower")
    n1 = paralyzable_true(m + dm, TAU, "lower")
    amplification = (n1 - n0) / dm * m / n0
    twin.plot(frac, amplification, "k--", lw=1.4,
              label="error amplification (lower branch)")
    twin.set_ylabel("relative error amplification $d\\ln n / d\\ln m$")
    twin.set_yscale("log")
    handles = ax_right.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    ax_right.legend(handles[0] + extra[0], handles[1] + extra[1], fontsize=8, loc="center left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    print("deadtime_curves.py")
    print(f"   wrote screenshots/{OUT.name}")
    print(f"   tau = {TAU:.3e} s; peak at n = {n_max:.4e} counts/s, "
          f"m_max = {m_max:.4e} counts/s")
    print(f"   {'m/m_max':>9} {'n tau lower':>13} {'n tau upper':>13} "
          f"{'error amplification':>21}")
    for f in (0.1, 0.5, 0.9, 0.99, 0.999):
        mm = f * m_max
        lo = float(paralyzable_true(mm, TAU, "lower")[0]) * TAU
        hi = float(paralyzable_true(mm, TAU, "upper")[0]) * TAU
        d = mm * 1e-6
        a = (
            float(paralyzable_true(mm + d, TAU, "lower")[0])
            - float(paralyzable_true(mm, TAU, "lower")[0])
        ) / d * mm / float(paralyzable_true(mm, TAU, "lower")[0])
        print(f"   {f:9.4f} {lo:13.5f} {hi:13.5f} {a:21.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
