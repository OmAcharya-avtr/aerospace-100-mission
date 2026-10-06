"""What a correlated fading channel looks like, and why the kernel choice matters.

Writes ``../screenshots/channel_and_fades.png``.

Left: a slice of the amplitude path with the threshold and the fades marked, so the
reader sees that a fade is an interval and not an instant. Middle: the measured
autocorrelation of both kernels against their models. Right: the measured
level-crossing rate against sample rate for both kernels, with Rice's (1945)
continuous-time result for the Gaussian kernel -- the Gauss-Markov rate does not
converge, because the Ornstein-Uhlenbeck process has no finite mean-square
derivative.

Runtime: about 60 s on one core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from codedfade.channel import (  # noqa: E402
    ChannelConfig,
    autocorrelation,
    correlated_gaussian,
    generate_amplitude,
)
from codedfade.fade import (  # noqa: E402
    fade_runs,
    fade_statistics,
    lognormal_standard_level,
    markov_crossing_rate,
    rice_crossing_rate_gauss_kernel,
)

SI = 0.6
TAU = 2.0e-4
FS = 1.0e6
THRESHOLD = 0.6


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    cfg = ChannelConfig(SI, TAU, FS, "lognormal", "exp", seed=41)
    a = generate_amplitude(cfg, 200_000)
    window = slice(0, 6000)
    t_us = np.arange(window.stop - window.start) / FS * 1e6
    axes[0].plot(t_us, a[window], lw=0.7, color="#1f77b4")
    axes[0].axhline(THRESHOLD, color="#d62728", ls="--", lw=1.2, label="threshold 0.6")
    below = a[window] < THRESHOLD
    starts, lengths = fade_runs(below)
    for s, length in zip(starts.tolist(), lengths.tolist(), strict=True):
        axes[0].axvspan(
            t_us[s], t_us[min(s + length, t_us.size - 1)], color="#d62728", alpha=0.16
        )
    st = fade_statistics(a, THRESHOLD, FS)
    axes[0].set_xlabel("time, us")
    axes[0].set_ylabel("amplitude (E[a^2] = 1)")
    axes[0].set_title(
        f"lognormal, SI={SI}, tau={TAU * 1e6:.0f} us\n"
        f"MFD {st.mean_fade_duration_s * 1e6:.1f} us, "
        f"worst {st.max_fade_duration_s * 1e6:.0f} us, "
        f"outage {st.outage_fraction:.3f}"
    )
    axes[0].legend(fontsize=8)

    lags = np.arange(81)
    for kernel, colour, model in (
        ("exp", "#1f77b4", np.exp(-lags / 20.0)),
        ("gauss", "#2ca02c", np.exp(-((lags / 20.0) ** 2))),
    ):
        g = correlated_gaussian(400_000, 20.0, np.random.default_rng(12), kernel)  # type: ignore[arg-type]
        acf = autocorrelation(g, 80)
        axes[1].plot(lags, acf, color=colour, lw=1.4, label=f"{kernel} measured")
        axes[1].plot(lags, model, color=colour, ls="--", lw=1.0, label=f"{kernel} model")
    axes[1].axhline(np.exp(-1.0), color="0.4", ls=":", lw=1.0, label="1/e")
    axes[1].axvline(20.0, color="0.4", ls=":", lw=1.0)
    axes[1].set_xlabel("lag, samples (Lc = 20)")
    axes[1].set_ylabel("autocorrelation of the latent Gaussian")
    axes[1].set_title("Both kernels hit 1/e at Lc, and differ at the origin")
    axes[1].legend(fontsize=8)

    u = lognormal_standard_level(THRESHOLD, SI)
    rates = [1e5, 2e5, 5e5, 1e6, 2e6]
    for kernel, colour in (("exp", "#1f77b4"), ("gauss", "#2ca02c")):
        measured = []
        for fs in rates:
            c = ChannelConfig(SI, TAU, fs, "lognormal", kernel, seed=32)  # type: ignore[arg-type]
            n = int(min(2_000_000, max(400_000, 400 * TAU * fs)))
            measured.append(
                fade_statistics(generate_amplitude(c, n), THRESHOLD, fs).level_crossing_rate_hz
            )
        axes[2].plot(rates, measured, marker="o", color=colour, label=f"{kernel} measured")
    axes[2].plot(
        rates,
        [markov_crossing_rate(u, TAU, fs) for fs in rates],
        ls="--",
        color="#1f77b4",
        label="exp: eq (14), exact for the sampled path",
    )
    axes[2].axhline(
        rice_crossing_rate_gauss_kernel(u, TAU),
        ls="--",
        color="#2ca02c",
        label="gauss: Rice (1945), eq (12)",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("sample rate, Hz")
    axes[2].set_ylabel("level-crossing rate, s^-1")
    axes[2].set_title("Gauss-Markov does not converge; Gaussian kernel does")
    axes[2].legend(fontsize=7.5)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join("..", "screenshots", "channel_and_fades.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {os.path.basename(out)} to the screenshots directory")


if __name__ == "__main__":
    main()
