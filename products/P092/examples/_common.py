"""Shared setup for every example: paths, the Agg backend and a figure saver.

Each example writes its PNG into ``screenshots/`` and prints what a reader
should notice. The backend is forced to Agg and ``show()`` is never called, so
an example runs headless.

Every example uses the MEASURED campaign. There is no synthetic fallback: an
example that fell back would publish a figure that looks like a measurement and
is not one, which is the one failure mode this product exists to prevent. An
example run without the measured traces prints the refusal and exits 1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where every example writes its PNG. Relative, always.
SCREENSHOTS = "screenshots"


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, chdir to the repository root, return it."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def measured_campaign():
    """Load the MEASURED campaign, or raise naming what is absent.

    Raises
    ------
    deadlinerisk.trace.TraceDataAbsentError
        With the names of the files looked for. No synthetic fallback.
    """
    from deadlinerisk.provenance import Origin
    from deadlinerisk.trace import (
        TraceDataAbsentError,
        campaign_is_complete,
        load_campaign,
    )

    directory = ROOT / "data" / "traces"
    required = (
        "gt_long",
        "mix_c0",
        "mix_c1",
        "mix_c2",
        "mix_c3",
        "mix_c4",
        "mix_c5",
        "mix_c6",
        "mix_mem2",
        "mix_c2mem1",
    )
    if not directory.is_dir() or not any(directory.glob("jobs_*.csv")):
        raise TraceDataAbsentError(
            "MEASURED TRACES ABSENT under data/traces/. Expected "
            + ", ".join(f"jobs_{label}.csv" for label in required)
            + ", manifest.json, CAMPAIGN_COMPLETE. This example uses measured data "
            "and nothing was substituted."
        )
    if not campaign_is_complete(directory):
        raise TraceDataAbsentError(
            "MEASURED CAMPAIGN INCOMPLETE: data/traces/CAMPAIGN_COMPLETE is absent, "
            "and the campaign writes that marker last of all. This example draws no "
            "figure from a partial campaign."
        )
    return load_campaign(
        directory,
        origin=Origin.MEASURED,
        require=required,
        directory_label="data/traces",
    )


def figure(name: str):
    """Create a matplotlib figure and return ``(figure, save)``.

    ``save`` writes the PNG into ``screenshots/`` and returns its relative path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    handle = plt.figure(figsize=(11.0, 6.5), dpi=130)

    def save() -> str:
        target = ROOT / SCREENSHOTS / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        handle.tight_layout()
        handle.savefig(target)
        plt.close(handle)
        return relative(target)

    return handle, save
