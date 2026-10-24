"""The negative control as a picture: a violation injected, caught and located.

A certificate that cannot fail certifies nothing. The left panel is a clean
encode, the right the same reconstruction with one sample moved to three times
the declared bound away from the truth. The certificate's verdict and the
index it names are printed under each.
"""

from __future__ import annotations

from _common import BOUND, LINEAR, SHADE, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.certificate import certify  # noqa: E402
from boundcompress.codec import encode  # noqa: E402

style()

INDEX = 2048


def main() -> None:
    signal = dataset.generate("battery_voltage", samples=4096, seed=0)
    bound = 5e-4
    result = encode(signal.values, "abs", bound, predictor="linear")
    clean = result.certificate

    corrupted = result.reconstruction.copy()
    corrupted[INDEX] = signal.values[INDEX] + 3.0 * bound
    failed = certify(signal.values, corrupted, "abs", bound)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    for axis, errors, certificate, title in (
        (axes[0], signal.values - result.reconstruction, clean, "clean encode"),
        (axes[1], signal.values - corrupted, failed, "one sample at 3 x the bound"),
    ):
        axis.axhspan(-bound, bound, color=SHADE, zorder=0)
        axis.plot(errors, linewidth=0.5, color=LINEAR, zorder=2)
        axis.axhline(bound, color=BOUND, linewidth=1.2, zorder=3)
        axis.axhline(-bound, color=BOUND, linewidth=1.2, zorder=3)
        verdict = "HOLDS" if certificate.bound_holds else "FAILED"
        axis.set_title(
            f"{title}\nverdict {verdict}, "
            f"{certificate.violations} violation(s), "
            f"indices {list(certificate.violation_indices)}",
            fontsize=9,
        )
        axis.set_xlim(0, signal.samples)
        axis.set_xlabel("sample index")
    axes[0].set_ylabel(f"error [{signal.unit}]")

    figure.subplots_adjust(wspace=0.12)
    print(save(figure, "negative_control"))
    print(f"clean verdict  : {'HOLDS' if clean.bound_holds else 'FAILED'}")
    print(
        f"injected verdict: {'HOLDS' if failed.bound_holds else 'FAILED'}, "
        f"violations {failed.violations}, indices {list(failed.violation_indices)}, "
        f"max |error| {failed.max_absolute_error:.6e} against bound {bound:.6e}"
    )
    assert np.isclose(failed.max_absolute_error, 3.0 * bound, rtol=1e-9)


if __name__ == "__main__":
    main()
