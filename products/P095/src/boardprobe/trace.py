"""Trace documents: what a recorded backend replays, and what a document may claim.

A trace is a JSON document of samples plus the metadata that decides what
replaying it entitles a reader to believe. Two fields carry that weight and
both are mandatory:

``origin``
    Free text saying where the samples came from, printed in every report
    built from the trace. Not machine-checked, because no schema can check
    honesty; printed everywhere, because a reader who sees it cannot miss it.

``content_provenance``
    Machine-checked, and the load-bearing one. It is the provenance of the
    SAMPLES, as distinct from the provenance of the act of replaying them.
    ``simulated`` for a document whose numbers a generator made up;
    ``recorded`` for a document whose numbers a device produced.

    **A trace document may not declare ``live``.** Replaying a file is not a
    live reading, whatever the file says, and a document that could claim
    ``live`` would be a one-line route from a text editor to a figure labelled
    a measurement. :func:`Trace.from_mapping` refuses it.

The replay backend then stamps ``meet(RECORDED, content_provenance)``, so:

- a trace recorded from hardware replays as ``RECORDED``;
- a trace of generated numbers replays as ``SIMULATED``, because
  ``meet(RECORDED, SIMULATED) = SIMULATED``.

Both fixtures shipped with this package declare their origin loudly. Neither
contains a reading from a Jetson Orin Nano, because no board was reachable
when they were made: ``fixtures/orin_nano_idle_simulated_trace.json`` declares
``simulated`` content and replays as ``SIMULATED``, and
``fixtures/replay_path_recorded_trace.json`` declares ``recorded`` content so
that the ``RECORDED`` path has something to exercise, and says in its own
``origin`` field, in the file, that its numbers were synthesised in a container
and are not measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import Provenance

#: Schema identifier written into and required of every trace document.
TRACE_SCHEMA = "boardprobe/trace/1"

#: The provenances a trace document may declare for its own samples.
ALLOWED_CONTENT_PROVENANCE = (Provenance.RECORDED, Provenance.SIMULATED)


class TraceError(ValueError):
    """A trace document is malformed, truncated, or claims something it may not."""


@dataclass(frozen=True, slots=True)
class TraceSample:
    """One sample in a trace: a channel, a value, its unit and its timestamp."""

    channel: str
    value: float
    unit: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class Trace:
    """A replayable trace document."""

    board_model: str
    origin: str
    content_provenance: Provenance
    channels: tuple[str, ...]
    samples: tuple[TraceSample, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.samples:
            raise TraceError(
                "a trace with no samples cannot be replayed; a zero-sample document is "
                "the empty-derivation case and is refused rather than replayed as an "
                "aggregate with no inputs"
            )
        if self.content_provenance not in ALLOWED_CONTENT_PROVENANCE:
            raise TraceError(
                f"content_provenance {self.content_provenance.value!r} is not allowed in a "
                "trace: replaying a file is not a live reading, so a document may declare "
                "only "
                + " or ".join(p.value for p in ALLOWED_CONTENT_PROVENANCE)
            )
        if not self.origin.strip():
            raise TraceError(
                "a trace must declare a non-empty origin saying where its samples came "
                "from; it is printed in every report built from the trace"
            )
        declared = set(self.channels)
        unknown = sorted({sample.channel for sample in self.samples} - declared)
        if unknown:
            raise TraceError(
                f"the trace carries samples on channel(s) {', '.join(unknown)} that its own "
                "channels list does not declare; the document is inconsistent with itself"
            )

    def samples_for(self, channel: str) -> tuple[TraceSample, ...]:
        """Every sample on ``channel``, in document order."""
        found = tuple(sample for sample in self.samples if sample.channel == channel)
        if not found:
            raise TraceError(
                f"the trace carries no samples on channel {channel!r}; channels present "
                "are " + ", ".join(sorted({sample.channel for sample in self.samples}))
            )
        return found

    @property
    def sample_count(self) -> int:
        """Total samples across all channels."""
        return len(self.samples)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> Trace:
        """Build a trace from a parsed document, refusing what it may not claim."""
        schema = payload.get("schema")
        if schema != TRACE_SCHEMA:
            raise TraceError(
                f"trace schema {schema!r} is not {TRACE_SCHEMA!r}; refusing to guess at a "
                "document this version does not understand"
            )
        for key in ("board_model", "origin", "content_provenance", "channels", "samples"):
            if key not in payload:
                raise TraceError(f"trace document is missing the required key {key!r}")
        try:
            content = Provenance(str(payload["content_provenance"]))
        except ValueError as error:
            raise TraceError(
                f"content_provenance {payload['content_provenance']!r} is not a provenance; "
                "allowed values are "
                + " or ".join(p.value for p in ALLOWED_CONTENT_PROVENANCE)
            ) from error
        samples = []
        for index, raw in enumerate(payload["samples"]):
            try:
                samples.append(
                    TraceSample(
                        channel=str(raw["channel"]),
                        value=float(raw["value"]),
                        unit=str(raw["unit"]),
                        timestamp_ns=int(raw["timestamp_ns"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise TraceError(
                    f"sample {index} of the trace is malformed ({error}); each sample needs "
                    "channel, value, unit and timestamp_ns"
                ) from error
        return cls(
            board_model=str(payload["board_model"]),
            origin=str(payload["origin"]),
            content_provenance=content,
            channels=tuple(str(name) for name in payload["channels"]),
            samples=tuple(samples),
            notes=str(payload.get("notes", "")),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> Trace:
        """Load a trace from a JSON file, naming the file by its final component.

        A partially written document -- the failure mode
        ``docs/RECOVERY.md`` calls "a partially written trace" -- fails
        ``json.loads`` and is reported as what it is rather than as a missing
        key twelve frames later.
        """
        candidate = Path(path)
        try:
            text = candidate.read_text()
        except OSError as error:
            raise TraceError(
                f"cannot read the trace {candidate.name!r}: {error.strerror or error}"
            ) from error
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise TraceError(
                f"the trace {candidate.name!r} is not valid JSON at line {error.lineno} "
                f"column {error.colno}: {error.msg}. A trace truncated by an interrupted "
                "run is the expected cause; see docs/RECOVERY.md, 'a partially written "
                "trace'."
            ) from error
        if not isinstance(payload, dict):
            raise TraceError(
                f"the trace {candidate.name!r} is a {type(payload).__name__} at the top "
                "level; a trace document is a JSON object"
            )
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        """A JSON-ready mapping, schema first."""
        return {
            "schema": TRACE_SCHEMA,
            "board_model": self.board_model,
            "origin": self.origin,
            "content_provenance": self.content_provenance.value,
            "notes": self.notes,
            "channels": list(self.channels),
            "samples": [
                {
                    "channel": sample.channel,
                    "value": sample.value,
                    "unit": sample.unit,
                    "timestamp_ns": sample.timestamp_ns,
                }
                for sample in self.samples
            ],
        }

    def to_json(self) -> str:
        """The document as JSON text, two-space indented and newline-terminated."""
        return json.dumps(self.to_mapping(), indent=2, sort_keys=False) + "\n"
