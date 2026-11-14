"""Shared setup for every example: paths, the Agg backend, origin labelling, output.

Each example writes its PNG into ``screenshots/`` and its printed output beside
itself as ``<example>_output.txt``, so every figure and every number in the
README traces to a script that was actually run.

**Origin labelling, which is the part that matters here.** An example prefers
the MEASURED campaign under ``data/traces/``. When it is absent the example uses
the committed SYNTHETIC generator instead, and then:

* the figure carries ``SYNTHETIC DATA -- NOT A MEASUREMENT`` as a banner across
  the top, in the figure itself, so that an embedded PNG cannot be mistaken for
  a measurement when it is viewed outside this repository;
* the first line of the example's printed output says the same;
* every axis label and the figure's caption say so too.

This is not the silent fallback the validation scripts refuse. The validation
scripts report MEASURED figures and have no fallback at all: they refuse and
exit non-zero. An example's job is to show the API working, and a labelled
synthetic figure does that; an unlabelled one would not, which is why the label
is applied by this module rather than left to each example.

The backend is forced to Agg and ``show()`` is never called, so an example runs
headless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where every example writes its PNG. Relative, always.
SCREENSHOTS = "screenshots"

#: Where the measured campaign would be. Relative, always.
MEASURED_TRACES = "data/traces"

#: The banner a synthetic figure carries.
SYNTHETIC_BANNER = "SYNTHETIC DATA -- NOT A MEASUREMENT"

#: Cycles per run for the synthetic campaign an example falls back to.
SYNTHETIC_CYCLES = 38


def bootstrap() -> Path:
    """Put ``src`` and ``tests`` on ``sys.path``, chdir to the root, return it."""
    for directory in (ROOT / "src", ROOT / "tests"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
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


def campaign():
    """``(campaign, is_measured)``: the measured campaign if present, else synthetic.

    The returned campaign's own
    :class:`~powerenvelope.provenance.Provenance` records which it is, so a
    caller that forgets to check can still not print one while meaning the
    other.
    """
    import tempfile

    from powerenvelope.provenance import Origin
    from powerenvelope.trace import campaign_is_complete, load_campaign

    directory = ROOT / MEASURED_TRACES
    if directory.is_dir() and any(directory.glob("power_*.csv")) and campaign_is_complete(
        directory
    ):
        loaded = load_campaign(
            directory, origin=Origin.MEASURED, directory_label=MEASURED_TRACES
        )
        if not loaded.manifest.get("synthetic"):
            return loaded, True
    from powerenvelope.synthetic import write_campaign

    # The directory label is fixed rather than taken from the temporary
    # directory's name, so that a committed output file is reproducible: a
    # random mkdtemp suffix would change on every run and make the example's
    # own output differ from the one in the repository for no reason.
    target = Path(tempfile.mkdtemp(prefix="powerenvelope-example-"))
    write_campaign(target, cycles=SYNTHETIC_CYCLES)
    return (
        load_campaign(
            target,
            origin=Origin.SYNTHETIC,
            directory_label="synthetic (a temporary directory, not a measurement)",
        ),
        False,
    )


class Output:
    """Collect printed lines, echo them, and write them beside the example."""

    def __init__(self, name: str, is_measured: bool) -> None:
        self.path = ROOT / "examples" / f"{name}_output.txt"
        self.lines: list[str] = []
        self.is_measured = is_measured
        self(
            "INPUT ORIGIN: MEASURED hardware campaign from "
            f"{MEASURED_TRACES}/"
            if is_measured
            else f"INPUT ORIGIN: {SYNTHETIC_BANNER}. The measured campaign is not on "
            "disk, so this example ran on the committed synthetic generator in "
            "src/powerenvelope/synthetic.py. Every number and the figure below are "
            "synthetic."
        )
        self("")

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def save(self) -> str:
        """Write the collected lines and return the relative path."""
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


def figure(name: str, is_measured: bool, caption: str):
    """``(figure, save)``. ``save`` writes the PNG and returns its relative path.

    A synthetic figure is stamped with :data:`SYNTHETIC_BANNER` across the top
    of the canvas, so the PNG carries its own provenance wherever it is viewed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    handle = plt.figure(figsize=(11.0, 6.5), dpi=130)

    def save() -> str:
        target = ROOT / SCREENSHOTS / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        handle.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        origin = (
            "INPUT ORIGIN: MEASURED hardware campaign"
            if is_measured
            else SYNTHETIC_BANNER
        )
        handle.suptitle(
            f"{caption}\n{origin}",
            fontsize=11,
            color="black" if is_measured else "#b00000",
            y=0.985,
        )
        handle.savefig(target)
        plt.close(handle)
        return relative(target)

    return handle, save
