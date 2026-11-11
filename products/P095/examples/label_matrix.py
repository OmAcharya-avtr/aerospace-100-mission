"""What label a benchmark earns, per backend pair, and every route that was refused.

Left: the label the harness earns for each combination of clock backend and
environment backend. The pattern is the lattice: the overall label is the meet
of the two, and only the live-and-live cell could ever be a measurement -- and
on this host even that reads MEASURED OFF-TARGET, because the platform
attestation does not match the declared Jetson Orin Nano.

Right: the fifteen routes ``validation/validate_adversarial_label.py`` attacks,
grouped by what each attacks, with what refused it. Two of them succeeded when
the check was first written and the design was changed in response; they are
marked.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    LABEL_COLOURS,
    NEUTRAL,
    REFUSED,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORDER,
    ORIN_NANO,
    LiveSource,
    RecordedSource,
    SimulatedSource,
    Trace,
    benchmark,
    label_for,
    reference_payload,
)

#: The route table, mirroring validation/validate_adversarial_label.py. The
#: third column is what refused the route; it is written here so the figure can
#: be read without the output file beside it.
ROUTES = (
    ("1 direct label", "labelling", "LabelViolation"),
    ("2 aggregate then label", "labelling", "LabelViolation"),
    ("3 dilute 999 live with 1", "labelling", "LabelViolation"),
    ("4 second-order derive", "labelling", "LabelViolation"),
    ("5 replay then label", "replay", "LabelViolation"),
    ("6 document claims live", "replay", "TraceError"),
    ("7 serialise, edit, reload", "replay", "LabelViolation"),
    ("8 object.__setattr__", "value object", "TamperError"),
    ("9 Reading subclass", "value object", "TamperError"),
    ("10 dataclasses.replace", "value object", "TypeError (was OPEN)"),
    ("11 override read()", "backend", "TypeError"),
    ("12 subclass LiveSource", "backend", "TypeError"),
    ("13 rebind class attribute", "backend", "RuntimeError (was OPEN)"),
    ("14 empty-fold identity", "lattice", "ProvenanceError"),
    ("15 forge a Certificate", "certificate", "TypeError"),
)


def main() -> int:
    style()
    replay = Trace.from_file(ROOT / "fixtures/replay_path_recorded_trace.json")
    payload = reference_payload(1000)

    clocks = (
        ("live", lambda: LiveSource(ORIN_NANO)),
        ("recorded", lambda: RecordedSource(ORIN_NANO, _clock_trace())),
        ("simulated", lambda: SimulatedSource(ORIN_NANO, seed=5)),
    )
    environments = (
        ("live", lambda: LiveSource(ORIN_NANO), None),
        ("recorded", lambda: RecordedSource(ORIN_NANO, replay), replay.channels),
        ("simulated", lambda: SimulatedSource(ORIN_NANO, seed=6), None),
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    grid = np.zeros((len(clocks), len(environments)))
    text = [["" for _ in environments] for _ in clocks]
    for row, (_clock_name, make_clock) in enumerate(clocks):
        for column, (_environment_name, make_environment, channels) in enumerate(
            environments
        ):
            environment = make_environment()
            try:
                result = benchmark(
                    payload, "cell", make_clock(), repeats=8, warmup=1,
                    environment=environment, environment_channels=channels,
                )
                label = result.label.value
            except Exception:  # noqa: BLE001 - a refusal is a cell value
                label = "REFUSED"
            text[row][column] = label
            grid[row, column] = (
                -1.0 if label == "REFUSED"
                else float(ORDER.index(_provenance_of(label)))
            )
    left.imshow(grid, cmap="cividis", vmin=-1, vmax=2)
    left.set_xticks(range(len(environments)),
                    [name for name, _, _ in environments], rotation=15)
    left.set_yticks(range(len(clocks)), [name for name, _ in clocks])
    left.set_xlabel("environment backend")
    left.set_ylabel("clock backend")
    for row in range(len(clocks)):
        for column in range(len(environments)):
            label = text[row][column]
            left.text(
                column, row, label, ha="center", va="center", fontsize=8.5,
                weight="bold",
                color=REFUSED if label == "REFUSED" else LABEL_COLOURS.get(label, "white"),
            )
    left.set_title("the label a benchmark earns, per backend pair")
    left.grid(False)

    groups = sorted({group for _, group, _ in ROUTES})
    palette = dict(zip(groups, plt.cm.cividis(np.linspace(0.1, 0.85, len(groups))),
                       strict=True))
    positions = np.arange(len(ROUTES))
    colours = [palette[group] for _, group, _ in ROUTES]
    right.barh(positions, np.ones(len(ROUTES)), color=colours, height=0.7)
    right.set_yticks(positions, [name for name, _, _ in ROUTES], fontsize=7.5)
    right.invert_yaxis()
    right.set_xlim(0, 1.0)
    right.set_xticks([])
    for index, (_name, _group, refused_by) in enumerate(ROUTES):
        right.text(
            0.02, index, refused_by, va="center", fontsize=7.5,
            color="white" if "OPEN" not in refused_by else "#ffd166",
            weight="bold" if "OPEN" in refused_by else "normal",
        )
    right.set_title(
        f"all {len(ROUTES)} adversarial routes, and what refused each\n"
        "(two were OPEN until the design was changed)"
    )
    right.grid(False)
    right.text(
        0.5, len(ROUTES) + 0.1,
        "groups: " + ", ".join(groups), fontsize=7.5, ha="center", color=NEUTRAL,
    )

    figure.suptitle(
        "boardprobe: the label follows the weakest input, and no route gets past it "
        "(Level 3, hardware-pending)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = save(figure, "label_matrix")
    print(f"written: {path}")
    print(f"cells computed: {grid.size}")
    for row, (name, _) in enumerate(clocks):
        print(f"clock={name:<10s} -> {text[row]}")
    print(f"adversarial routes shown: {len(ROUTES)}")
    print(f"routes refused: {sum(1 for _, _, by in ROUTES if by)}")
    return 0


def _clock_trace() -> Trace:
    """A tiny increasing clock trace, so the recorded clock row is reachable."""
    from boardprobe import Provenance, TraceSample

    return Trace(
        board_model=ORIN_NANO.model,
        origin=(
            "built in memory by examples/label_matrix.py so the recorded-clock row "
            "of the matrix is reachable; not a recording of anything"
        ),
        content_provenance=Provenance.RECORDED,
        channels=("monotonic_clock",),
        samples=tuple(
            TraceSample("monotonic_clock", 100.0 + index * 1e-4, "s", index)
            for index in range(64)
        ),
    )


def _provenance_of(label: str):
    """The provenance a label corresponds to, for the colour scale."""
    for provenance in ORDER:
        if label_for(provenance).value == label:
            return provenance
    raise KeyError(label)


if __name__ == "__main__":
    raise SystemExit(main())
