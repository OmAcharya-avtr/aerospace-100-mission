"""The committed recording, as the four timestamps actually arrived.

Writes ``screenshots/exchange_timeline.png``. What to notice: the forward and
reverse dwell times differ by a constant amount -- that constant is the
asymmetry, and it is what biases the offset -- the round trip is free of the
clock offset by construction, and the gap in the middle is the recorded outage
window that the recovery procedure is written for.
"""

from __future__ import annotations

import numpy as np
from _common import SERIES, bootstrap, save, style

ROOT = bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from timetransfer import ExchangeSet, two_way_estimate  # noqa: E402
from timetransfer.asym import centred_dwells_s  # noqa: E402
from timetransfer.timebase import NS_PER_S  # noqa: E402

FIXTURE = ROOT / "fixtures" / "recorded_exchange.json"


def main() -> None:
    recorded = ExchangeSet.read_json(FIXTURE)
    estimate = two_way_estimate(recorded)
    t = recorded.midpoint_s()
    fwd = recorded.forward_dwell_ns().astype(np.float64) / NS_PER_S
    rev = recorded.reverse_dwell_ns().astype(np.float64) / NS_PER_S
    rtt = recorded.round_trip_ns().astype(np.float64) / NS_PER_S
    fwd_c, rev_c = centred_dwells_s(recorded)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.4), sharex=True)
    top, middle, bottom = axes

    top.plot(t, fwd * 1e3, ".", ms=2.6, color=SERIES[0], label="forward dwell t2 - t1")
    top.plot(t, rev * 1e3, ".", ms=2.6, color=SERIES[1], label="reverse dwell t4 - t3")
    top.plot(t, rtt * 1e3, ".", ms=2.0, color=SERIES[2], alpha=0.6,
             label="round trip (free of the offset)")
    top.set_ylabel("milliseconds")
    top.set_title(
        f"fixtures/recorded_exchange.json: {recorded.n} of {recorded.n_attempted} "
        f"exchanges completed, {recorded.n_lost} lost"
    )
    top.legend(loc="center right", fontsize=7.6)

    middle.plot(t, estimate.offset_s * 1e3, ".", ms=2.6, color=SERIES[3],
                label="symmetric offset estimate")
    middle.plot(t, estimate.delay_s * 1e3, ".", ms=2.6, color=SERIES[4],
                label="mean path delay estimate")
    middle.set_ylabel("milliseconds")
    middle.set_title(
        f"baseline 1 over the recording: median offset "
        f"{estimate.offset_median_s * 1e3:.4f} ms, minimum delay "
        f"{estimate.delay_min_s * 1e3:.4f} ms"
    )
    middle.legend(loc="center right", fontsize=7.6)

    bottom.plot(t, fwd_c * 1e6, "-", lw=0.7, color=SERIES[0],
                label="forward queueing excursion")
    bottom.plot(t, -rev_c * 1e6, "-", lw=0.7, color=SERIES[1],
                label="reverse queueing excursion (plotted negative)")
    gaps = np.diff(recorded.seq)
    if int(np.max(gaps)) > 1:
        index = int(np.argmax(gaps))
        bottom.axvspan(t[index], t[index + 1], color="#dddddd",
                       label=f"recorded outage, {int(np.max(gaps)) - 1} exchanges lost")
    bottom.set_xlabel("local clock time [s]")
    bottom.set_ylabel("microseconds")
    bottom.set_title(
        "queueing excursions above each direction's own window minimum -- "
        "the only asymmetry signal a corrector can see"
    )
    bottom.legend(loc="upper right", fontsize=7.6)

    fig.tight_layout()
    print(save(fig, "exchange_timeline"))
    print(f"exchanges              : {recorded.n} completed, {recorded.n_lost} lost")
    print(f"longest sequence gap   : {int(np.max(gaps))}")
    print(f"span                   : {recorded.span_s():.3f} s")
    print(f"median forward dwell   : {float(np.median(fwd)) * 1e3:.6f} ms")
    print(f"median reverse dwell   : {float(np.median(rev)) * 1e3:.6f} ms")
    print(f"median round trip      : {float(np.median(rtt)) * 1e3:.6f} ms")
    print(f"median offset estimate : {estimate.offset_median_s * 1e3:.6f} ms")
    print(f"is_hardware            : {recorded.is_hardware}")


if __name__ == "__main__":
    main()
