"""Power-law capacity-fade baseline: recovery, and the cost of a fixed exponent.

The shipped physics-shaped model holds the fade exponent z at 0.55, the value
Wang et al. (2011) fit for graphite-LiFePO4 cells. The generator draws z per
trajectory from [0.45, 0.65], bracketing that value and the z = 0.5 of the
diffusion-limited calendar-ageing regime described by Broussely et al. (2005).
This script measures three things:

1. That the fit recovers the shape exactly when z is right.
2. How far the crossing estimate and its interval move when z is wrong, which
   is the error the interval does not contain because z is declared, not fitted.
3. What profiling z over a grid buys and what it costs.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

ROOT = add_src_to_path()

from degradepredict.baselines import DegradationFitError  # noqa: E402
from degradepredict.limits import LimitSpec  # noqa: E402
from degradepredict.physics import (  # noqa: E402
    DEFAULT_FADE_EXPONENT,
    FADE_EXPONENT_GRID,
    powerlaw_fade_baseline,
)

OUT = ROOT / "validation" / "validate_physics_output.txt"
LIMIT = LimitSpec(value=80.0, direction="below", unit="percent of BOL", atol=1e-9)
SEED = 20261016


def _fade_curve(t: np.ndarray, z: float, crossing: float, y0: float = 100.0) -> np.ndarray:
    """y = y0 (1 - B t**z) with B solved so the mean crosses 80 at ``crossing``."""
    b = (1.0 - 80.0 / y0) / crossing**z
    return y0 * (1.0 - b * t**z)


def main() -> int:
    say = Tee(OUT)
    say("validate_physics.py -- power-law capacity-fade baseline")
    for line in environment_lines():
        say(line)
    say(f"seed             : {SEED}")
    say(f"declared exponent: z = {DEFAULT_FADE_EXPONENT}  "
        f"(Wang et al. 2011, J. Power Sources 196(8):3942-3948)")
    say(f"profiling grid   : {np.array2string(FADE_EXPONENT_GRID, precision=2)}")

    say.rule("1. recovery when the exponent is correct")
    say("Mean curve y = 100 (1 - B t**z) with B solved to put the crossing of 80")
    say("percent at a declared time. Noise is additive Gaussian at 1e-6 percent,")
    say("so what is measured is the algebra, not the statistics.")
    say("")
    say(f"{'z':>6}{'crossing truth d':>18}{'estimate d':>14}{'rel err':>12}"
        f"{'C0 est':>10}{'B est':>14}{'B truth':>14}")
    worst = 0.0
    rng = np.random.default_rng(SEED)
    for z in (0.45, 0.50, 0.55, 0.65):
        for crossing in (600.0, 1500.0):
            t = np.arange(60, dtype=float) * 7.0
            y = _fade_curve(t, z, crossing) + rng.normal(0.0, 1e-6, size=t.size)
            fit = powerlaw_fade_baseline(t, y, LIMIT, z=z)
            b_true = 0.2 / crossing**z
            b_est = -fit.slope / fit.intercept
            rel = abs(fit.crossing_days - crossing) / crossing
            worst = max(worst, rel)
            say(f"{z:>6.2f}{crossing:>18.1f}{fit.crossing_days:>14.3f}{rel:>12.2e}"
                f"{fit.intercept:>10.4f}{b_est:>14.6e}{b_true:>14.6e}")
    say("")
    say(f"worst relative crossing error with the correct exponent: {worst:.3e}")
    say("PASS" if worst < 1e-4 else "FAIL")

    say.rule("2. the cost of holding z fixed when the truth is different")
    say("Mean curve generated at z_true, fitted at z = 0.55. Noise 0.3 percent,")
    say("600 replicates per row. 'cov' is the fraction of replicates whose 90")
    say("percent Fieller interval contains the true crossing; the interval knows")
    say("nothing about z being wrong, so it cannot be expected to cover.")
    say("")
    say(f"{'z true':>8}{'truth d':>10}{'mean est d':>13}{'bias d':>11}"
        f"{'rel bias':>11}{'mean width d':>14}{'cov':>8}")
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for z_true in (0.45, 0.50, 0.55, 0.60, 0.65):
        crossing = 1200.0
        t = np.arange(60, dtype=float) * 7.0
        ests, widths, hits = [], [], 0
        for _ in range(600):
            y = _fade_curve(t, z_true, crossing) + rng.normal(0.0, 0.3, size=t.size)
            try:
                fit = powerlaw_fade_baseline(t, y, LIMIT, z=DEFAULT_FADE_EXPONENT)
            except DegradationFitError:
                continue
            ests.append(fit.crossing_days)
            lo, hi = fit.fieller_interval_days
            widths.append(hi - lo)
            hits += int(lo <= crossing <= hi)
        mean_est = float(np.mean(ests))
        rows.append((z_true, mean_est - crossing, hits / 600.0))
        say(f"{z_true:>8.2f}{crossing:>10.1f}{mean_est:>13.2f}"
            f"{mean_est - crossing:>11.2f}{(mean_est - crossing) / crossing:>11.4f}"
            f"{float(np.mean(widths)):>14.2f}{hits / 600.0:>8.4f}")
    say("")
    say("The interval width is of order tens of days while the bias from a")
    say("0.05 error in z is of order hundreds. A declared exponent that is")
    say("slightly wrong therefore produces a confident and wrong date, and the")
    say("reported interval gives no warning. This is the single most important")
    say("limitation of the physics-shaped baseline and it is why the shipped")
    say("benchmark measures the model on a population whose z varies.")

    say.rule("3. profiling z over the grid")
    say("Same generator, 400 replicates per row, z profiled by residual sum of")
    say("squares over the declared grid against z held at 0.55.")
    say("")
    say(f"{'z true':>8}{'fixed bias d':>14}{'profiled bias d':>17}"
        f"{'fixed |bias|':>14}{'profiled |bias|':>17}{'mode z':>8}")
    rng = np.random.default_rng(SEED + 2)
    for z_true in (0.45, 0.50, 0.55, 0.65):
        crossing = 1200.0
        t = np.arange(60, dtype=float) * 7.0
        fixed, prof, picked = [], [], []
        for _ in range(400):
            y = _fade_curve(t, z_true, crossing) + rng.normal(0.0, 0.3, size=t.size)
            try:
                a = powerlaw_fade_baseline(t, y, LIMIT, z=DEFAULT_FADE_EXPONENT)
                b = powerlaw_fade_baseline(t, y, LIMIT, z=None)
            except DegradationFitError:
                continue
            fixed.append(a.crossing_days - crossing)
            prof.append(b.crossing_days - crossing)
            picked.append(b.z)
        vals, counts = np.unique(np.round(picked, 2), return_counts=True)
        mode_z = float(vals[int(np.argmax(counts))])
        say(f"{z_true:>8.2f}{float(np.mean(fixed)):>14.2f}{float(np.mean(prof)):>17.2f}"
            f"{float(np.mean(np.abs(fixed))):>14.2f}"
            f"{float(np.mean(np.abs(prof))):>17.2f}{mode_z:>8.2f}")
    say("")
    say("Profiling reduces the bias where z is badly wrong and costs accuracy")
    say("where it is right, because the selected exponent is itself noisy. The")
    say("interval returned after profiling is conditional on the selected z and")
    say("does not include the selection, so it is narrower than it should be.")
    say("The shipped default is the fixed, cited exponent.")

    say.rule("summary")
    say(f"recovery with the correct exponent, worst relative error : {worst:.3e}")
    say(f"bias at z_true = 0.45, z fitted 0.55                     : "
        f"{rows[0][1]:+.1f} d on a 1200 d crossing")
    say(f"bias at z_true = 0.65, z fitted 0.55                     : "
        f"{rows[-1][1]:+.1f} d on a 1200 d crossing")
    say(f"Fieller coverage at z_true = 0.45                        : {rows[0][2]:.4f} "
        f"(nominal 0.90)")
    say(f"Fieller coverage at z_true = 0.55 (correct)              : {rows[2][2]:.4f}")
    path = say.save()
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
