"""Zernike mode gallery: the first 21 Noll modes, labelled in both conventions.

Every panel is titled with the Noll index, the OSA/ANSI index and ``(n, m)``,
which is the whole point -- the two single-index orderings are visibly not the
same sequence of pictures.

Run from the product root:

    python examples/mode_gallery.py

Writes ``screenshots/zernike_mode_gallery.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zernkit import (  # noqa: E402
    mode_name,
    nm_to_osa,
    noll_to_nm,
    unit_disc_grid,
    zernike_cartesian,
)

N_MODES = 21  # Noll j = 1..21, radial orders 0..5
N_PIX = 256


def main() -> Path:
    x, y, mask = unit_disc_grid(N_PIX)

    n_cols = 6
    n_rows = int(np.ceil(N_MODES / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.15 * n_cols, 2.75 * n_rows), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()

    for k, ax in enumerate(axes):
        if k >= N_MODES:
            ax.axis("off")
            continue
        j_noll = k + 1
        n, m = noll_to_nm(j_noll)
        field = np.full(x.shape, np.nan)
        field[mask] = zernike_cartesian(n, m, x[mask], y[mask])
        vmax = float(np.nanmax(np.abs(field)))
        ax.imshow(
            field,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=0.6, color="0.3"))
        ax.set_xticks([])
        ax.set_yticks([])
        name = mode_name(n, m)
        if len(name) > 22:
            head, _, tail = name.rpartition(" ")
            name = f"{head}\n{tail}"
        ax.set_title(
            f"Noll {j_noll} / OSA {nm_to_osa(n, m)}\n"
            f"$Z_{{{n}}}^{{{m:+d}}}$\n{name}",
            fontsize=7.5,
            pad=3,
        )

    fig.suptitle(
        "ZernKit mode gallery -- normalised Zernike modes on the unit disc\n"
        "Noll (1-based) and OSA/ANSI (0-based) indices shown together; "
        "$m>0\\rightarrow\\cos m\\theta$, $m<0\\rightarrow\\sin|m|\\theta$",
        fontsize=10,
    )
    out_dir = Path(__file__).resolve().parents[1] / "screenshots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "zernike_mode_gallery.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"wrote {main()}")
