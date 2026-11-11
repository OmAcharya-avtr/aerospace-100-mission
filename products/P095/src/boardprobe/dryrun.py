"""Dry-run mode: exercise every code path, touch no board, write no file.

A dry run is only worth as much as the evidence that it was dry. This module
produces that evidence rather than asserting it:

- every device access in this package goes through ``boardprobe.sysio``, which
  appends to a ledger; the dry run runs inside a fresh
  :func:`~boardprobe.sysio.ledger_window` and reports the ledger's contents,
  which must be empty;
- the one part of the walk that does call into the device funnel -- the live
  backend's refusal path -- runs inside a NESTED ledger window against a
  filesystem root the dry run first asserts does not exist, and its attempts
  are reported in their own count. Hiding those attempts in the main ledger
  would be a self-reporting scanner; leaving them in it would make the main
  count non-zero for accesses that provably cannot reach a device. They are
  separated and both numbers are printed;
- the dry run takes a directory to watch and reports a before-and-after
  snapshot of it, so a file written anywhere inside it is detected rather than
  promised against;
- it records which code paths it exercised, by name, so "every path" is a
  countable claim and not a figure of speech.

What a dry run exercises
------------------------
Every channel of the declared table through the simulated and the recorded
backend; the live backend's refusal on every board channel; the statistics
layer; the derivation path; the choke point in both directions -- a label that
is granted and a label that is refused; the trace writer, to a string and not
to a file; and the report renderer. The live backend is NOT exercised in a way
that could reach a device: it is asked for a board channel and the refusal is
the exercised path. The clock channel is deliberately excluded from the dry
run, because reading a clock IS a system call and a dry run that made one
would have to report it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .certify import Label, LabelViolation, certify, label_for
from .channels import Access, BoardDeclaration
from .platform_attest import PlatformAttestation, attest_platform
from .provenance import ORDER, Provenance, ProvenanceError, meet, meet_all
from .readings import Attribution, Reading, TamperError, attribute, combine, mint
from .report import render_readings, render_statistics
from .sources import LiveSource, RecordedSource, SimulatedSource, read_available
from .statistics import derive, rail_power, summarise
from .sysio import ledger_window
from .trace import Trace


@dataclass
class DryRunReport:
    """What a dry run exercised, and the evidence that it stayed dry."""

    paths: list[str] = field(default_factory=list)
    device_accesses: tuple[tuple[str, str], ...] = ()
    files_before: tuple[str, ...] = ()
    files_after: tuple[str, ...] = ()
    absent_root_accesses: tuple[tuple[str, str], ...] = ()
    absent_root: str = ""
    refusals: tuple[str, ...] = ()
    labels_granted: tuple[str, ...] = ()
    labels_refused: tuple[str, ...] = ()
    watched: str = ""

    @property
    def path_count(self) -> int:
        """How many named code paths were exercised."""
        return len(self.paths)

    @property
    def files_added(self) -> tuple[str, ...]:
        """Files that appeared under the watched directory while the dry run ran."""
        return tuple(sorted(set(self.files_after) - set(self.files_before)))

    @property
    def side_effect_free(self) -> bool:
        """``True`` when no device was accessed and no file appeared."""
        return not self.device_accesses and not self.files_added

    def exercise(self, name: str) -> None:
        """Record that a named code path ran."""
        self.paths.append(name)


def _snapshot(directory: Path) -> tuple[str, ...]:
    """Every file under ``directory``, as paths relative to it, sorted."""
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        )
    )


def dry_run(
    declaration: BoardDeclaration,
    trace: Trace,
    *,
    seed: int = 0,
    watch: Path | str | None = None,
    platform: PlatformAttestation | None = None,
    absent_root: Path | str | None = None,
) -> DryRunReport:
    """Walk every code path without touching a board or writing a file.

    Parameters
    ----------
    declaration:
        The board table to walk.
    trace:
        A trace for the recorded backend to replay.
    seed:
        Seed for the simulated backend.
    watch:
        Directory to snapshot before and after. Defaults to the package's own
        repository root if it can be found from ``__file__``, else the current
        directory.
    platform:
        Attestation to use. Defaults to attesting now; attesting reads one
        device-tree node, so it is done OUTSIDE the ledger window and is
        reported as such.
    absent_root:
        Filesystem root for the live backend's refusal path. Defaults to a
        directory that does not exist, so the refusal is reached without any
        possibility of reading a real device node.
    """
    attestation = attest_platform() if platform is None else platform
    watched = Path(watch) if watch is not None else Path(__file__).resolve().parents[2]
    report = DryRunReport(watched=watched.name)
    report.files_before = _snapshot(watched)
    refusals: list[str] = []
    granted: list[str] = []
    refused_labels: list[str] = []

    with ledger_window() as ledger:
        simulated = SimulatedSource(declaration, seed=seed)
        recorded = RecordedSource(declaration, trace)
        # A root that cannot exist, so the live backend's refusal path is
        # reached without the possibility of opening a real device node.
        missing = (
            Path(absent_root)
            if absent_root is not None
            else watched / "no-such-root-for-the-dry-run"
        )
        if missing.exists():
            raise RuntimeError(
                f"the dry run's absent root {missing.name!r} exists; a dry run must "
                "exercise the live backend's refusal against a root that cannot "
                "contain a device node, and this one could"
            )
        live = LiveSource(declaration, root=missing)
        report.absent_root = missing.name
        report.exercise("construct all three sealed backends")

        # Every channel but the monotonic clock. The clock is excluded because
        # reading it IS a system call: a dry run that made one would have to
        # report it. Every other channel, including the sysfs cpufreq clock, is
        # walked.
        board_channels = tuple(
            channel.name
            for channel in declaration.channels
            if channel.access is not Access.CLOCK_SYSCALL
        )
        simulated_readings, simulated_refusals = read_available(simulated, board_channels)
        report.exercise(f"simulated backend over {len(board_channels)} board channels")
        refusals.extend(f"simulated:{name}" for name, _ in simulated_refusals)

        recorded_readings, recorded_refusals = read_available(recorded, trace.channels)
        report.exercise(f"recorded backend over {len(trace.channels)} trace channels")
        refusals.extend(f"recorded:{name}" for name, _ in recorded_refusals)

        # Nested window: these attempts go into their own ledger, against a
        # root asserted above not to exist. See the module docstring.
        with ledger_window() as absent_ledger:
            _, live_refusals = read_available(live, board_channels)
            report.exercise("live backend refusal on every board channel")
            report.exercise("the live backend's perf adapter refusal")
            report.absent_root_accesses = tuple(absent_ledger.events)
        refusals.extend(f"live:{name}" for name, _ in live_refusals)
        if len(live_refusals) != len(board_channels):
            raise RuntimeError(
                f"the dry run's live backend served {len(board_channels) - len(live_refusals)} "
                "board channel(s) from a root that does not exist; a dry run must not reach "
                "a device and this one did"
            )

        report.exercise("summarise() over a simulated channel")
        thermal = [r for r in simulated_readings if r.channel == "die_temp_cpu"]
        if len(thermal) < 1:
            thermal = [simulated.read("die_temp_cpu") for _ in range(4)]
        statistics = summarise(thermal + [simulated.read("die_temp_cpu") for _ in range(3)])

        report.exercise("rail_power() derivation across two channels")
        power = rail_power(
            simulated.read("rail_vdd_in_voltage"),
            simulated.read("rail_vdd_in_current"),
            "rail_vdd_in_power",
        )

        report.exercise("derive() a second-order figure from statistics")
        second_order = derive(
            "thermal headroom to 85 degC",
            85.0 - statistics[1].value,
            "degC",
            [statistics[1], power],
        )

        report.exercise("meet over every pair in the lattice")
        for left in ORDER:
            for right in ORDER:
                _ = meet(left, right)

        report.exercise("meet_all refuses an empty collection")
        try:
            meet_all([])
        except ProvenanceError:
            refused_labels.append("meet_all([]) refused")

        report.exercise("certify grants the label the provenance admits")
        simulated_attribution = attribute(simulated_readings or thermal)
        certificate = certify(
            "dry run: simulated board channels",
            simulated_attribution,
            label_for(simulated_attribution.provenance),
            attestation,
        )
        granted.append(f"{certificate.label.value} for simulated inputs")

        recorded_attribution = attribute(recorded_readings)
        recorded_certificate = certify(
            "dry run: replayed trace",
            recorded_attribution,
            label_for(recorded_attribution.provenance),
            attestation,
        )
        granted.append(f"{recorded_certificate.label.value} for the replayed trace")

        report.exercise("certify refuses MEASURED on simulated inputs")
        try:
            certify(
                "dry run: deliberate mislabelling",
                simulated_attribution,
                Label.MEASURED,
                attestation,
            )
        except LabelViolation:
            refused_labels.append("MEASURED on simulated inputs refused")

        report.exercise("certify refuses a tampered reading")
        tampered = mint(
            channel="die_temp_cpu",
            value=41.0,
            unit="degC",
            timestamp_ns=1,
            provenance=Provenance.SIMULATED,
            source_id="dry-run",
        )
        object.__setattr__(tampered, "provenance", Provenance.LIVE)
        try:
            attribute([tampered])
        except TamperError:
            refused_labels.append("a tampered reading refused by its seal")

        report.exercise("combine() across attributions of mixed provenance")
        mixed = combine([simulated_attribution, recorded_attribution])
        if mixed.provenance is not meet(
            simulated_attribution.provenance, recorded_attribution.provenance
        ):
            raise RuntimeError("combine() did not meet the provenances it was given")

        report.exercise("render readings, statistics and the trace document")
        _ = render_readings(simulated_readings[:4] or thermal[:4])
        _ = render_statistics([*statistics, power, second_order])
        # To a string, never to a file: a dry run writes nothing.
        document = trace.to_json()
        _ = json.loads(document)

        report.exercise("Trace.from_mapping round trip in memory")
        reloaded = Trace.from_mapping(json.loads(document))
        if reloaded.content_provenance is not trace.content_provenance:
            raise RuntimeError("a trace round trip changed its content provenance")

        report.device_accesses = tuple(ledger.events)

    report.files_after = _snapshot(watched)
    report.refusals = tuple(refusals)
    report.labels_granted = tuple(granted)
    report.labels_refused = tuple(refused_labels)
    return report


def declared_paths(report: DryRunReport) -> Iterable[str]:
    """The named code paths a dry run exercised, in order."""
    return tuple(report.paths)


def attribution_of(readings: Iterable[Reading]) -> Attribution:
    """Convenience wrapper so a caller need not import two modules for one call."""
    return attribute(readings)
