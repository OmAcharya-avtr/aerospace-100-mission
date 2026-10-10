"""Plot the four declared change types plus the transient negative control.

Writes ../screenshots/change_types.png. Every panel regenerates bit-for-bit
from the seed printed in its label.
"""

from __future__ import annotations

from pathlib import Path

from telemdrift.plotting import plot_stream_panel
from telemdrift.streams import ChangeSpec, change_stream

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots" / "change_types.png"

PANELS = (
    ("mean step\n+1.0 sigma", ChangeSpec("mean_step", 1.0), 58_501),
    ("variance step\nsigma x 2.0", ChangeSpec("variance_step", 2.0), 58_502),
    ("drift ramp\n0.02 sigma/sample", ChangeSpec("drift_ramp", 0.02), 58_503),
    ("transient\n+4.0 sigma, 20 samples\n(NOT a change)", ChangeSpec("transient", 4.0, 20),
     58_504),
)


def main() -> None:
    streams = {}
    for label, spec, seed in PANELS:
        stream, idx = change_stream(300, 500, spec, seed)
        streams[label] = (stream, idx)
        print(f"{spec.kind:14s} magnitude={spec.magnitude:<6g} seed={seed} "
              f"change_index={idx} length={stream.size} "
              f"pre_mean={stream[:idx].mean():+.4f} post_mean={stream[idx:].mean():+.4f} "
              f"pre_sd={stream[:idx].std():.4f} post_sd={stream[idx:].std():.4f}")
    path = plot_stream_panel(streams, OUT)
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
