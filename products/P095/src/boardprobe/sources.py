"""Three backends behind one interface, with provenance stamped by the base class.

The structural claim of this product lives here. A backend does **not** decide
the provenance of the readings it produces; :meth:`Source.read` does, and
:meth:`Source.read` is final.

How that is enforced, rather than requested
-------------------------------------------
1. :meth:`Source.read` is the only place a :class:`~boardprobe.readings.Reading`
   is constructed in this package. A subclass supplies ``_raw(channel) ->
   float`` and nothing else.
2. :meth:`Source.__init_subclass__` **refuses** a subclass that defines
   ``read``, ``read_all``, ``provenance`` or ``_stamp_ns``. The refusal happens
   at class-creation time, so an attempt to override the stamping path is a
   ``TypeError`` at import and not a wrong number at run time.
3. The three concrete backends are **sealed**: once this module finishes
   importing, :meth:`Source.__init_subclass__` refuses any further subclass of
   :class:`LiveSource`, :class:`RecordedSource` or :class:`SimulatedSource`.
   The route "subclass ``LiveSource``, serve it trace data, and collect
   ``LIVE`` readings" is closed at the language level.
4. The three backends' declared provenances are captured in a module-private
   registry when this module finishes importing, and :attr:`Source.provenance`
   compares the class attribute against the registry on every read. Rebinding
   ``SimulatedSource.backend_provenance = Provenance.LIVE`` -- an ordinary
   Python assignment, and a route the adversarial check tries -- is refused
   with a ``RuntimeError`` naming the class. (Rebinding the registry as well is
   possible for code running in this process; see the note at the end of this
   docstring.)
5. Provenance is ``meet(class backend provenance, instance floor)``. ``meet``
   has no inverse and is monotone, so an instance's floor can only ever
   WEAKEN its class's provenance. ``RecordedSource`` sets its floor from the
   trace document's ``content_provenance``, which is how a replay of generated
   numbers comes out ``SIMULATED`` rather than ``RECORDED``. There is no
   mechanism, anywhere, that strengthens a provenance.

The three backends
------------------
``LiveSource``
    Reads the declared interfaces. Structurally complete, and in the build
    environment it cannot produce a reading on any board channel, because the
    nodes do not exist and the PMU is absent -- which is the correct outcome
    and the one the test suite asserts. It CAN serve ``monotonic_clock``,
    because ``clock_gettime(2)`` is not board-specific; those readings are
    genuinely ``LIVE`` and genuinely not readings of a Jetson, which is what
    the separate platform attestation in every certificate exists to say.

``RecordedSource``
    Replays a trace document. Cycles the samples of each channel, so a replay
    longer than the trace repeats rather than running out; the sample index is
    in each reading's ``detail``.

``SimulatedSource``
    A deterministic generator from a fixed seed. Same seed, same readings, bit
    for bit, which ``validation/validate_backends.py`` measures by replaying a
    sequence twice and comparing exactly -- the one place an exact float
    comparison is correct, because the claim is bit-reproducibility of a
    pseudo-random stream and not the value of a physical quantity.

The limit of all of this, stated rather than left implied
---------------------------------------------------------
Every defence above is enforced against the public API: subclassing,
overriding, rebinding a class attribute, serialising, mutating a field. None of
them is a defence against code in the same process that reaches into
module-private state -- ``_SEALED``, ``_DECLARED``, ``boardprobe.readings._KEY``
-- and no design in Python can be. The threat this product is built against is
the realistic one: a figure derived from simulated or recorded data reaching a
report that calls it a measurement, through ordinary use, ordinary refactoring
or ordinary serialisation. That is closed, and
``validation/validate_adversarial_label.py`` attacks it from every direction
the API exposes and reports what happened to each.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from .channels import Access, BoardDeclaration, Channel
from .provenance import Provenance, meet
from .readings import Reading, mint
from .sysio import DeviceUnreachableError, open_perf_counter, read_integer
from .trace import Trace, TraceSample

#: Filled in at the end of this module. Read at subclass-creation time, so it
#: is empty while the three concrete backends are themselves being defined and
#: populated immediately afterwards. See point 3 of the module docstring.
_SEALED: tuple[type, ...] = ()

#: Methods a subclass may not define. ``read`` stamps provenance; ``provenance``
#: computes it; ``_stamp_ns`` timestamps it; ``read_all`` fans out to ``read``.
_FINAL_MEMBERS = ("read", "read_all", "provenance", "_stamp_ns")

#: Each sealed backend's declared provenance, captured at the end of this
#: module and compared on every read. See point 4 of the module docstring.
_DECLARED: dict[type, Provenance] = {}


class Source(abc.ABC):
    """A uniform interface over one board's channels, for one backend.

    Subclasses implement :meth:`_raw` and may define ``__init__``, ``close``
    and their own helpers. They may not touch the stamping path.
    """

    #: Provenance of this backend CLASS. Combined with the instance floor by
    #: :attr:`provenance`, which can only weaken it.
    backend_provenance: ClassVar[Provenance]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        overridden = [name for name in _FINAL_MEMBERS if name in cls.__dict__]
        if overridden:
            raise TypeError(
                f"{cls.__name__} may not define {', '.join(overridden)}: provenance is "
                "stamped by boardprobe.sources.Source.read and a backend that could "
                "override it would be able to choose its own provenance, which is the one "
                "thing this package exists to prevent. Implement _raw(channel) instead."
            )
        for base in cls.__mro__[1:]:
            if base in _SEALED:
                raise TypeError(
                    f"{cls.__name__} may not subclass {base.__name__}: the three backends "
                    "are sealed once boardprobe.sources has finished importing. Subclassing "
                    "LiveSource and serving it recorded or generated values is a route from "
                    "synthetic data to a reading stamped live, and it is closed here rather "
                    "than discouraged in a docstring."
                )

    def __init__(
        self,
        declaration: BoardDeclaration,
        source_id: str,
        floor: Provenance = Provenance.LIVE,
    ) -> None:
        if type(declaration) is not BoardDeclaration:
            raise TypeError(
                "declaration must be a boardprobe.BoardDeclaration, got "
                f"{type(declaration).__name__}"
            )
        if not source_id:
            raise ValueError("source_id must name this backend instance")
        if type(floor) is not Provenance:
            raise TypeError(f"floor must be a Provenance member, got {type(floor).__name__}")
        self._declaration = declaration
        self._source_id = source_id
        self._floor = floor

    @property
    def declaration(self) -> BoardDeclaration:
        """The board table this backend serves."""
        return self._declaration

    @property
    def source_id(self) -> str:
        """Identity of this backend instance, recorded in every reading."""
        return self._source_id

    @property
    def provenance(self) -> Provenance:
        """``meet`` of this backend class's provenance and this instance's floor.

        FINAL. A subclass that defines this raises ``TypeError`` at class
        creation. Because ``meet`` is the lattice's greatest lower bound, the
        floor can only weaken the class's provenance: there is no value of
        ``floor`` for which a ``RecordedSource`` or a ``SimulatedSource``
        becomes ``LIVE``.
        """
        declared = type(self).backend_provenance
        expected = _DECLARED.get(type(self))
        if expected is not None and declared is not expected:
            raise RuntimeError(
                f"{type(self).__name__}.backend_provenance has been rebound from "
                f"{expected.value!r} to {getattr(declared, 'value', declared)!r}. A "
                "backend's provenance is fixed when boardprobe.sources is imported and "
                "is compared against the registry on every read, because rebinding a "
                "class attribute is a one-line route from a simulated backend to a "
                "reading stamped live. Restore it, or take readings from the backend "
                "whose provenance you actually have."
            )
        return meet(declared, self._floor)

    def _stamp_ns(self) -> int:
        """Timestamp for a reading, in nanoseconds. FINAL.

        ``CLOCK_MONOTONIC_RAW`` so that a timestamp cannot move backwards under
        an NTP step in the middle of a benchmark.
        """
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    @abc.abstractmethod
    def _raw(self, channel: Channel) -> float:
        """Raw value for ``channel``, in the channel's ``raw_unit``.

        A backend that cannot serve a channel raises
        :class:`~boardprobe.sysio.DeviceUnreachableError` with an actionable
        message. It must never return a substitute value.
        """

    def _detail(self, channel: Channel) -> str:
        """Where this backend got the value, as a relative string."""
        return f"{type(self).__name__.lower()}:{channel.locator}"

    def read(self, channel_name: str) -> Reading:
        """One reading from one channel. FINAL: this is where provenance is stamped.

        The sequence is fixed and short: resolve the declared channel, ask the
        backend for a raw value, apply the channel's declared scale, and
        construct the reading with ``self.provenance``. A backend never sees
        the :class:`~boardprobe.readings.Reading` constructor.
        """
        channel = self._declaration.channel(channel_name)
        raw = self._raw(channel)
        return mint(
            channel=channel.name,
            value=float(raw) * channel.scale,
            unit=channel.unit,
            timestamp_ns=self._stamp_ns(),
            provenance=self.provenance,
            source_id=self._source_id,
            detail=self._detail(channel),
        )

    def read_all(self, channel_names: Iterable[str] | None = None) -> tuple[Reading, ...]:
        """Readings from every named channel, skipping none and substituting nothing.

        FINAL. A channel this backend cannot serve propagates its
        :class:`~boardprobe.sysio.DeviceUnreachableError`; see
        :func:`read_available` for the caller who wants partial results and the
        refusals reported separately.
        """
        names = tuple(self._declaration.names if channel_names is None else channel_names)
        return tuple(self.read(name) for name in names)


def read_available(
    source: Source, channel_names: Iterable[str] | None = None
) -> tuple[tuple[Reading, ...], tuple[tuple[str, str], ...]]:
    """Read what ``source`` can serve; return the readings and the refusals.

    The refusals are returned, not swallowed: a harness that silently dropped
    an unreadable channel would report a narrower measurement than it claimed.
    Each refusal is ``(channel name, the first line of the error)``.
    """
    names = tuple(source.declaration.names if channel_names is None else channel_names)
    readings: list[Reading] = []
    refusals: list[tuple[str, str]] = []
    for name in names:
        try:
            readings.append(source.read(name))
        except (DeviceUnreachableError, KeyError) as error:
            refusals.append((name, str(error).splitlines()[0]))
    return tuple(readings), tuple(refusals)


class LiveSource(Source):
    """Reads the declared interfaces on the machine this process runs on.

    Parameters
    ----------
    declaration:
        The board table.
    root:
        Filesystem root the sysfs locators are resolved under. The default is
        the real root. A test passes a temporary directory; doing so does not
        change the provenance, because the reading IS a live read of whatever
        is at that path -- what it is a reading OF is the platform
        attestation's question, not the lattice's.
    """

    backend_provenance: ClassVar[Provenance] = Provenance.LIVE

    def __init__(
        self,
        declaration: BoardDeclaration,
        root: Path | str = Path("/"),
        source_id: str | None = None,
    ) -> None:
        super().__init__(declaration, source_id or "live:filesystem-root")
        self._root = Path(root)

    def _detail(self, channel: Channel) -> str:
        return f"live:{channel.locator}"

    def _raw(self, channel: Channel) -> float:
        if channel.access is Access.CLOCK_SYSCALL:
            return float(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        if channel.access is Access.SYSFS_INTEGER:
            return float(read_integer(self._root, channel.locator))
        if channel.access is Access.PERF_EVENT_OPEN:
            return float(self._read_counter(channel))
        raise DeviceUnreachableError(
            f"channel {channel.name!r} declares access {channel.access.value!r}, which the "
            "live backend has no adapter for"
        )

    def _read_counter(self, channel: Channel) -> int:
        """Open, read and close one perf counter. UNVERIFIED on the target."""
        import os

        try:
            perf_type, config = (int(part) for part in channel.locator.split(":", 1))
        except ValueError as error:
            raise DeviceUnreachableError(
                f"channel {channel.name!r} declares the perf locator {channel.locator!r}, "
                "which is not a 'type:config' pair"
            ) from error
        descriptor = open_perf_counter(perf_type, config)
        try:
            payload = os.read(descriptor, 8)
        finally:
            os.close(descriptor)
        if len(payload) != 8:
            raise DeviceUnreachableError(
                f"channel {channel.name!r}: a perf counter read returned {len(payload)} "
                "bytes rather than the 8 of a u64 counter value"
            )
        return int.from_bytes(payload, byteorder="little", signed=False)


class RecordedSource(Source):
    """Replays a trace document, stamping ``meet(RECORDED, trace content)``.

    A trace of generated numbers therefore replays as ``SIMULATED``. See
    ``boardprobe.trace`` for why a document may not declare ``live``.
    """

    backend_provenance: ClassVar[Provenance] = Provenance.RECORDED

    def __init__(
        self, declaration: BoardDeclaration, trace: Trace, source_id: str | None = None
    ) -> None:
        if type(trace) is not Trace:
            raise TypeError(f"trace must be a boardprobe.Trace, got {type(trace).__name__}")
        super().__init__(
            declaration,
            source_id or f"recorded:{trace.content_provenance.value}",
            floor=trace.content_provenance,
        )
        self._trace = trace
        self._cursor: dict[str, int] = {}

    @property
    def trace(self) -> Trace:
        """The trace being replayed."""
        return self._trace

    def _detail(self, channel: Channel) -> str:
        return f"recorded:{channel.name}[{self._cursor.get(channel.name, 0)}]"

    def _raw(self, channel: Channel) -> float:
        try:
            samples = self._trace.samples_for(channel.name)
        except Exception as error:
            raise DeviceUnreachableError(
                f"the replayed trace carries no samples on channel {channel.name!r}; "
                f"{error}"
            ) from error
        index = self._cursor.get(channel.name, 0) % len(samples)
        sample = samples[index]
        self._cursor[channel.name] = index + 1
        if sample.unit != channel.unit:
            raise DeviceUnreachableError(
                f"channel {channel.name!r}: the trace records unit {sample.unit!r} and the "
                f"declaration says {channel.unit!r}; a unit mismatch between a trace and a "
                "board table is a wrong number, not a formatting difference"
            )
        # The trace stores the SCALED value in the channel's unit; read()
        # multiplies by the channel scale, so divide it back out here rather
        # than letting the scale be applied twice.
        return sample.value / channel.scale


class SimulatedSource(Source):
    """A deterministic synthetic generator. Same seed, same readings, bit for bit.

    The generator is a declared fiction with a plausible shape, not a thermal
    or electrical model of anything: a per-channel baseline and amplitude, a
    slow sinusoid in the sample index, and Gaussian noise from a
    ``numpy.random.Generator`` seeded per channel. It exists so that every code
    path can be exercised without a board, and every reading it produces is
    stamped ``SIMULATED`` so that nothing derived from it can be called a
    measurement.
    """

    backend_provenance: ClassVar[Provenance] = Provenance.SIMULATED

    #: Baseline, amplitude and noise sigma per channel kind, in the channel's
    #: RAW unit. Declared data, not a model: the numbers are chosen to be
    #: recognisable in a plot and are not claimed to resemble any board.
    PROFILE: ClassVar[dict[str, tuple[float, float, float]]] = {
        "clock": (0.0, 0.0, 0.0),
        "die-temperature": (42_000.0, 6_000.0, 450.0),
        "rail-power": (4_800.0, 900.0, 60.0),
        "performance-counter": (2.5e8, 4.0e7, 2.0e6),
    }

    def __init__(
        self,
        declaration: BoardDeclaration,
        seed: int = 0,
        source_id: str | None = None,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        super().__init__(declaration, source_id or f"simulated:seed={seed}")
        self._seed = int(seed)
        self._step: dict[str, int] = {}
        self._generators: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        """The declared seed. Reproducibility rests on it alone."""
        return self._seed

    def _detail(self, channel: Channel) -> str:
        return f"simulated:seed={self._seed},step={self._step.get(channel.name, 0)}"

    def _generator(self, channel: Channel) -> np.random.Generator:
        """One generator per channel, seeded from the instance seed and the name.

        Per channel rather than one shared stream, so that reading a subset of
        the channels produces the same values for those channels as reading all
        of them. A shared stream would make a reading depend on which other
        channels were read, which is a reproducibility trap.
        """
        if channel.name not in self._generators:
            stable = sum(
                byte * (131 ** index % 1_000_003)
                for index, byte in enumerate(channel.name.encode("utf-8"))
            )
            self._generators[channel.name] = np.random.default_rng(
                [self._seed, stable % 2_147_483_647]
            )
        return self._generators[channel.name]

    def _raw(self, channel: Channel) -> float:
        step = self._step.get(channel.name, 0)
        self._step[channel.name] = step + 1
        if channel.access is Access.CLOCK_SYSCALL:
            # A synthetic monotonic clock: a fixed 1 ms per step from a fixed
            # origin, so a simulated duration is exactly reproducible.
            return float(1_000_000_000 + step * 1_000_000)
        baseline, amplitude, sigma = self.PROFILE[channel.kind.value]
        generator = self._generator(channel)
        wave = amplitude * float(np.sin(step / 11.0))
        noise = sigma * float(generator.standard_normal())
        return baseline + wave + noise


# Seal the three concrete backends. Any further subclass of them now raises at
# class-creation time; see Source.__init_subclass__ and point 3 of the module
# docstring.
_SEALED = (LiveSource, RecordedSource, SimulatedSource)
_DECLARED.update(
    {backend: backend.backend_provenance for backend in _SEALED}
)


def backend_names() -> tuple[str, ...]:
    """The three sealed backend names, in lattice order from weakest."""
    return ("simulated", "recorded", "live")


def make_source(
    name: str,
    declaration: BoardDeclaration,
    *,
    seed: int = 0,
    trace: Trace | None = None,
    root: Path | str = Path("/"),
) -> Source:
    """Build one of the three sealed backends by name.

    The CLI and the dry run go through this so there is one list of backends.
    """
    if name == "simulated":
        return SimulatedSource(declaration, seed=seed)
    if name == "recorded":
        if trace is None:
            raise ValueError(
                "the recorded backend needs a trace; pass trace=Trace.from_file(path)"
            )
        return RecordedSource(declaration, trace)
    if name == "live":
        return LiveSource(declaration, root=root)
    raise ValueError(
        f"unknown backend {name!r}; the three sealed backends are "
        + ", ".join(backend_names())
    )


def sealed_backends() -> tuple[type, ...]:
    """The three backend classes that may not be subclassed."""
    return _SEALED


__all__ = [
    "LiveSource",
    "RecordedSource",
    "SimulatedSource",
    "Source",
    "backend_names",
    "make_source",
    "read_available",
    "sealed_backends",
    "synthesise_trace",
]


def synthesise_trace(
    declaration: BoardDeclaration,
    channel_names: Sequence[str],
    samples_per_channel: int,
    *,
    seed: int,
    origin: str,
    content_provenance: Provenance,
    notes: str = "",
) -> Trace:
    """Build a trace document from the simulated backend, deterministically.

    This is how the two committed fixtures under ``fixtures/`` were made, and
    ``tests/test_fixtures.py`` re-synthesises each of them from the declared
    seed and compares, so a committed fixture cannot drift from the generator
    that is said to have produced it.

    ``content_provenance`` is what the resulting DOCUMENT will declare about
    its own samples, and it is the caller's statement, not a derived fact. The
    two fixtures use it differently and both say so in their ``origin``: the
    idle fixture declares ``simulated``, which is the truth about its numbers;
    the replay-path fixture declares ``recorded`` so that the ``RECORDED``
    branch of the replay backend has something to exercise, and its ``origin``
    field states in the file that its numbers were synthesised in a container
    and are not measurements of anything.

    ``live`` is rejected by :class:`~boardprobe.trace.Trace` itself, so there
    is no argument to this function that produces a document a replay could
    stamp ``LIVE``.
    """
    if samples_per_channel < 1:
        raise ValueError(
            f"samples_per_channel must be at least 1, got {samples_per_channel}"
        )
    source = SimulatedSource(declaration, seed=seed)
    samples = []
    for name in channel_names:
        channel = declaration.channel(name)
        for index in range(samples_per_channel):
            reading = source.read(name)
            samples.append(
                TraceSample(
                    channel=channel.name,
                    value=reading.value,
                    unit=reading.unit,
                    timestamp_ns=index * 100_000_000,
                )
            )
    return Trace(
        board_model=declaration.model,
        origin=origin,
        content_provenance=content_provenance,
        channels=tuple(channel_names),
        samples=tuple(samples),
        notes=notes,
    )
