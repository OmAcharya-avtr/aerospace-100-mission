"""Validation: Gaussian-beam pointing loss closed form vs Monte Carlo.

Claim under test
----------------
For a far-field Gaussian beam with 1/e^2 half-angle divergence
theta_div and zero-mean Gaussian boresight jitter of per-axis standard
deviation sigma_theta on two independent axes, the average normalized
received power is

    <L> = 1 / (1 + 4 (sigma_theta / theta_div)^2)

Derivation: instantaneous loss L(theta) = exp(-2 theta^2 / theta_div^2)
(Siegman 1986, "Lasers", ch. 17); the radial error theta is Rayleigh
distributed with p(theta) = (theta/sigma^2) exp(-theta^2/(2 sigma^2)),
and the resulting Gaussian integral gives the expression above. This
is the point-receiver limit of the Farid & Hranilovic 2007
(J. Lightwave Technol. 25(7):1702-1710) pointing-error model; see also
Andrews & Phillips 2005, "Laser Beam Propagation through Random
Media", 2nd ed., ch. 12.

Method: independent seeded Monte Carlo draws of (theta_x, theta_y),
averaging exp(-2 (theta_x^2 + theta_y^2)/theta_div^2). MC standard
error of the mean is reported alongside the relative error so the
agreement can be judged against sampling noise.

Rerun:  python validation/val_pointing.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitterscope import pointing_loss_avg  # noqa: E402

THETA_DIV = 12.0e-6  # rad, 1/e^2 half-angle
N_MC = 1_000_000
SEED = 20260806
RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

lines: list[str] = []


def log(msg: str = "") -> None:
    lines.append(msg)


log("=== jitterscope pointing-loss validation: closed form vs Monte Carlo ===")
log("<L> = 1 / (1 + 4 (sigma_theta/theta_div)^2)   [point receiver, far field,")
log("       zero-mean isotropic Gaussian jitter on two axes]")
log(f"theta_div = {THETA_DIV * 1e6:.1f} urad, N_MC = {N_MC} draws/axis, base seed {SEED}")
log()
log(f"{'sigma/theta_div':>15} {'sigma [urad]':>13} {'closed form':>13} {'Monte Carlo':>13} "
    f"{'MC std err':>11} {'rel err':>10} {'|err|/SE':>9}")

max_rel = 0.0
for i, ratio in enumerate(RATIOS):
    sigma = ratio * THETA_DIV
    closed = pointing_loss_avg(sigma, THETA_DIV)
    rng = np.random.default_rng(SEED + i)
    tx = rng.normal(0.0, sigma, N_MC)
    ty = rng.normal(0.0, sigma, N_MC)
    samples = np.exp(-2.0 * (tx**2 + ty**2) / THETA_DIV**2)
    mc = float(np.mean(samples))
    se = float(np.std(samples) / np.sqrt(N_MC))
    rel = abs(mc - closed) / closed
    max_rel = max(max_rel, rel)
    n_sigma = abs(mc - closed) / se if se > 0 else 0.0
    log(f"{ratio:15.2f} {sigma * 1e6:13.3f} {closed:13.6f} {mc:13.6f} {se:11.2e} "
        f"{rel:10.3e} {n_sigma:9.2f}")

log()
log(f"maximum relative error across cases: {max_rel:.3e}")
log("PASS criterion: max relative error < 1e-2 (MC standard error ~1e-3 relative)")
log(f"RESULT: {'PASS' if max_rel < 1e-2 else 'FAIL'}")
log()

# Worked engineering example tying jitter budget to link budget.
log("Worked example (dB form): -10 log10(<L>)")
for ratio in (0.1, 0.25, 0.5, 1.0):
    closed = pointing_loss_avg(ratio * THETA_DIV, THETA_DIV)
    log(f"  sigma/theta_div = {ratio:4.2f}  ->  <L> = {closed:.4f}  "
        f"= {-10 * np.log10(closed):6.2f} dB loss")
log()
log("Validity: no static boresight bias, isotropic jitter, unobscured TEM00 beam,")
log("point receiver in the far field, no atmospheric scintillation.")

out = Path(__file__).resolve().parent / "val_pointing_output.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"saved {out}")
