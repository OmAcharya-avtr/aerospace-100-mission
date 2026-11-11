"""The reading value object, its tamper seal, and the attribution of a derivation.

A :class:`Reading` is the only thing a backend can produce and the only thing a
derivation can consume. It carries its value, its unit, the channel it came
from, when it was taken, which backend instance took it, and -- the point of
the product -- its :class:`~boardprobe.provenance.Provenance`.

Minting, and the laundering route that made it necessary
-------------------------------------------------------
A :class:`Reading` cannot be constructed by an ordinary call. Its first field
is a ``dataclasses.InitVar`` holding a module-private token, so
``Reading(channel=..., provenance=LIVE, ...)`` raises, and the single minting
entry point is :func:`mint`, whose only in-package callers are
``boardprobe.sources.Source.read`` and :func:`from_mapping`.

That is not fastidiousness. An earlier revision of this module had an ordinary
public constructor, and the adversarial check in
``validation/validate_adversarial_label.py`` found the resulting route and it
SUCCEEDED:

    laundered = dataclasses.replace(simulated_reading, provenance=Provenance.LIVE)
    certify("figure", attribute([laundered]), Label.MEASURED, platform)

``dataclasses.replace`` is public API on every dataclass. It re-ran
``__post_init__``, which recomputed the seal, so the laundered reading was
*validly sealed* with the simulated reading's own channel and value and
``LIVE`` provenance, and the choke point granted ``MEASURED``. One line, no
private state touched. The token closes it: ``dataclasses.replace`` and
``copy.replace`` both raise ``TypeError: InitVar 'mint' must be specified with
replace()``, and neither caller has the token.

What :func:`mint` does and does not promise is stated on the function. In
short: this package guarantees that a provenance once claimed cannot be
strengthened, laundered through a derivation, or lost. It cannot stop a caller
from making a FALSE claim at the source -- no type system can -- but that claim
now goes through one named, auditable function instead of a constructor, and
the reading records which source made it.

Why a seal and not just ``frozen=True``
---------------------------------------
``dataclasses.dataclass(frozen=True, slots=True)`` refuses attribute
assignment, and that is where most designs stop. It is not enough, and the
reason was measured during this build rather than assumed:

    >>> import dataclasses
    >>> @dataclasses.dataclass(frozen=True, slots=True)
    ... class R:
    ...     a: int
    >>> r = R(1)
    >>> r.a = 2                      # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'a'
    >>> object.__setattr__(r, "a", 2)
    >>> r.a
    2

``object.__setattr__`` goes straight past the frozen guard, on this
interpreter, today. So "frozen" means "not assignable by accident", not
"immutable", and a provenance field protected only by ``frozen=True`` can be
rewritten from ``SIMULATED`` to ``LIVE`` in one line.

Every :class:`Reading` therefore carries a ``seal``: a keyed BLAKE2b digest
over its own fields, computed in ``__post_init__`` under a per-process random
key. :func:`verify` recomputes it. A field rewritten by any means -- the
documented API, ``object.__setattr__``, a pickle round trip, a hand-written
JSON document -- leaves the seal disagreeing with the fields, and
:func:`verify` refuses the reading by name. The choke point in
``boardprobe.certify`` runs :func:`verify` over every reading behind every
figure, so a tampered reading cannot reach a label.

What the seal is and is not
---------------------------
It is an integrity check against mutation, mislabelling and forged documents.
**It is not a security boundary.** The key lives in this module's private
state, so code running in this process can import ``boardprobe.readings._KEY``
and forge a seal. No design in Python can prevent that, and claiming otherwise
would be the kind of unearned assurance this product exists to argue against.
The threat this product defends against is the realistic one: a figure derived
from simulated or recorded data reaching a report that calls it a measurement,
through ordinary use of the API, ordinary refactoring, or ordinary
serialisation. That route is closed, and
``validation/validate_adversarial_label.py`` attacks it from twelve
directions and reports what happened.

The key is per-process, which has a consequence worth stating: a reading
cannot be serialised with its seal and deserialised in another process. That
is deliberate. A reading reconstituted from a document is not the reading that
was taken; see :func:`from_mapping`, which re-seals and forces the provenance
down the lattice.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

from .provenance import Provenance, ProvenanceError, meet, meet_all

#: Per-process seal key. Private, regenerated on every import, never written
#: to disk. See the module docstring for what this does and does not promise.
_KEY = secrets.token_bytes(32)

_DIGEST_SIZE = 16


class TamperError(ValueError):
    """Raised when a reading's seal disagrees with its fields, or its type is wrong."""


class _Mint:
    """Private minting token for :class:`Reading`. Never exported."""

    __slots__ = ()


_MINT = _Mint()


def _seal_of(
    channel: str,
    value: float,
    unit: str,
    timestamp_ns: int,
    provenance: Provenance,
    source_id: str,
    detail: str,
) -> str:
    """Keyed digest over a reading's fields. ``value`` is hashed by exact bit pattern."""
    payload = "\x1f".join(
        (
            channel,
            float(value).hex(),
            unit,
            str(int(timestamp_ns)),
            provenance.value,
            source_id,
            detail,
        )
    )
    return hashlib.blake2b(
        payload.encode("utf-8"), key=_KEY, digest_size=_DIGEST_SIZE
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Reading:
    """One value from one channel, with its provenance, units and timestamp.

    Parameters
    ----------
    channel:
        Declared channel name.
    value:
        The value, in ``unit``, after the channel's scale has been applied.
    unit:
        Unit string from the channel declaration, e.g. ``degC``, ``W``, ``s``.
    timestamp_ns:
        Integer nanoseconds from the taking backend's clock. Monotonic within
        one backend instance; not comparable between instances.
    provenance:
        Set by the backend's base class and never by the caller of a backend.
    source_id:
        Identity of the backend instance, e.g. ``simulated:seed=7``.
    detail:
        Where the value came from, as the backend saw it: a relative sysfs
        locator, a trace sample index, or a generator description. Never an
        absolute path.
    """

    mint: InitVar[Any]
    channel: str
    value: float
    unit: str
    timestamp_ns: int
    provenance: Provenance
    source_id: str
    detail: str = ""
    seal: str = field(default="", compare=False)

    def __post_init__(self, mint: Any) -> None:
        if mint is not _MINT:
            raise TypeError(
                "boardprobe.Reading cannot be constructed directly. Readings are minted "
                "by boardprobe.readings.mint(), whose only in-package callers are a "
                "backend's Source.read() and from_mapping(). An ordinary constructor "
                "allowed a simulated reading to be laundered into a figure labelled "
                "MEASURED by dataclasses.replace(reading, provenance=LIVE), which "
                "recomputed the seal and was granted the label; see the module "
                "docstring. Take readings from SimulatedSource, RecordedSource or "
                "LiveSource."
            )
        if not self.channel:
            raise ValueError("reading channel must be a non-empty string")
        if not self.unit:
            raise ValueError(f"reading on channel {self.channel!r}: unit must be non-empty")
        if type(self.provenance) is not Provenance:
            raise ProvenanceError(
                f"reading on channel {self.channel!r}: provenance must be a "
                f"boardprobe.Provenance member, got {type(self.provenance).__name__}"
            )
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError(
                f"reading on channel {self.channel!r}: value must be a real number, got "
                f"{type(self.value).__name__}"
            )
        if not isinstance(self.timestamp_ns, int) or isinstance(self.timestamp_ns, bool):
            raise TypeError(
                f"reading on channel {self.channel!r}: timestamp_ns must be an int of "
                f"nanoseconds, got {type(self.timestamp_ns).__name__}"
            )
        if not self.source_id:
            raise ValueError(
                f"reading on channel {self.channel!r}: source_id must name the backend "
                "instance that took it; an anonymous reading cannot be audited"
            )
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(
            self,
            "seal",
            _seal_of(
                self.channel,
                self.value,
                self.unit,
                self.timestamp_ns,
                self.provenance,
                self.source_id,
                self.detail,
            ),
        )

    @property
    def intact(self) -> bool:
        """``True`` when the seal still agrees with the fields."""
        return self.seal == _seal_of(
            self.channel,
            self.value,
            self.unit,
            self.timestamp_ns,
            self.provenance,
            self.source_id,
            self.detail,
        )

    def to_mapping(self) -> dict[str, Any]:
        """A JSON-ready mapping. The seal is deliberately NOT included.

        A seal is keyed to the process that made it, so writing it out would
        produce a token that can never verify again and would invite a reader
        to treat a document as sealed when it is not.
        """
        return {
            "channel": self.channel,
            "value": self.value,
            "unit": self.unit,
            "timestamp_ns": self.timestamp_ns,
            "provenance": self.provenance.value,
            "source_id": self.source_id,
            "detail": self.detail,
        }


def mint(
    channel: str,
    value: float,
    unit: str,
    timestamp_ns: int,
    provenance: Provenance,
    source_id: str,
    detail: str = "",
) -> Reading:
    """Mint a sealed :class:`Reading`. The single entry point, by design.

    WHAT THIS GUARANTEES. A reading that exists cannot have its provenance
    strengthened, cannot be laundered through a derivation, and cannot be
    mutated without the seal noticing. Those are the three failure modes a
    hardware-measured validation level actually has to survive, and they are
    closed by construction: ``meet`` has no inverse, every derivation goes
    through :func:`attribute` or :func:`combine`, and every label goes through
    ``boardprobe.certify.certify``.

    WHAT THIS DOES NOT GUARANTEE, stated plainly. The caller of this function
    chooses the ``provenance`` argument. A caller who passes
    ``Provenance.LIVE`` for a number a generator produced has made a false
    statement, and nothing in Python can prevent that; a hardware abstraction
    layer is, in the end, a program that believes what the kernel tells it. The
    difference between this design and an ordinary public constructor is that
    the false statement is now made in ONE named function, is attributable to a
    ``source_id`` that travels with the reading, and cannot be made by
    accident, by a refactor, or by ``dataclasses.replace`` on a reading that
    already exists.

    The three sealed backends in ``boardprobe.sources`` are the only callers of
    this function inside the package, and a backend cannot choose a reading's
    provenance: ``Source.read`` stamps it from the backend's class, and
    ``Source.__init_subclass__`` refuses a subclass that tries to override
    that.
    """
    return Reading(
        _MINT,
        channel=channel,
        value=value,
        unit=unit,
        timestamp_ns=timestamp_ns,
        provenance=provenance,
        source_id=source_id,
        detail=detail,
    )


def verify(reading: object) -> Reading:
    """Return ``reading`` if it is an intact :class:`Reading`, else raise.

    Two checks, and the first is the one that matters. ``type(reading) is
    Reading`` is an EXACT type check: a subclass of :class:`Reading` that
    overrides ``provenance`` with a property, or overrides ``intact`` to return
    ``True``, is refused here rather than trusted. ``isinstance`` would admit
    it, which is why it is not used.
    """
    if type(reading) is not Reading:
        raise TamperError(
            "a derivation accepts boardprobe.readings.Reading and nothing else; got "
            f"{type(reading).__module__}.{type(reading).__name__}. A subclass is refused "
            "deliberately: it can override provenance or the seal check, which is exactly "
            "the route this check exists to close."
        )
    if not reading.intact:
        raise TamperError(
            f"reading on channel {reading.channel!r} from source {reading.source_id!r} "
            "fails its seal: at least one field has been modified since the backend "
            "produced it. A modified reading is refused rather than used, because the "
            "field most worth modifying is provenance."
        )
    return reading


def from_mapping(payload: Mapping[str, Any], floor: Provenance) -> Reading:
    """Rebuild a reading from a document, forcing its provenance DOWN to ``floor``.

    A reading reconstituted from a file is not the reading that was taken, and
    the document's own provenance claim is not evidence. The returned reading's
    provenance is ``meet(document claim, floor)``, so a document claiming
    ``live`` cannot raise a reading above the floor the caller is entitled to.
    Because ``meet`` is monotone and has no inverse, there is no value of the
    document that can strengthen the result.
    """
    try:
        channel = str(payload["channel"])
        value = float(payload["value"])
        unit = str(payload["unit"])
        timestamp_ns = int(payload["timestamp_ns"])
        claimed = Provenance(str(payload["provenance"]))
    except KeyError as error:
        raise ValueError(
            f"reading document is missing the required key {error.args[0]!r}; required keys "
            "are channel, value, unit, timestamp_ns, provenance"
        ) from error
    except ValueError as error:
        raise ValueError(f"reading document is not a reading: {error}") from error
    if type(floor) is not Provenance:
        raise ProvenanceError(
            f"floor must be a boardprobe.Provenance member, got {type(floor).__name__}"
        )
    return mint(
        channel=channel,
        value=value,
        unit=unit,
        timestamp_ns=timestamp_ns,
        provenance=meet(claimed, floor),
        source_id=str(payload.get("source_id", "document")),
        detail=str(payload.get("detail", "")),
    )


@dataclass(frozen=True, slots=True)
class Attribution:
    """What a derived figure rests on: provenance, and the evidence for it.

    ``provenance`` is the meet over every contributing reading. ``weakest``
    names the channels that are AT that provenance, which is what an error
    message has to say to be actionable: "this is not a measurement" is useless
    without "because ``die_temp_cpu`` is simulated".
    """

    provenance: Provenance
    reading_count: int
    channels: tuple[str, ...]
    source_ids: tuple[str, ...]
    weakest: tuple[str, ...]
    weakest_sources: tuple[str, ...]
    first_timestamp_ns: int
    last_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.reading_count < 1:
            raise ProvenanceError(
                "an attribution must rest on at least one reading; a derivation with no "
                "inputs has no provenance and must abort rather than be downgraded"
            )
        if not self.weakest:
            raise ProvenanceError(
                "an attribution must name at least one channel at its own provenance"
            )

    @property
    def span_s(self) -> float:
        """Wall-clock span of the contributing readings, in seconds."""
        return (self.last_timestamp_ns - self.first_timestamp_ns) * 1e-9

    def describe(self) -> str:
        """One line: the provenance and why."""
        return (
            f"{self.provenance.value} over {self.reading_count} readings on "
            f"{len(self.channels)} channels; weakest channel(s): "
            f"{', '.join(self.weakest)}"
        )


def attribute(readings: Iterable[object]) -> Attribution:
    """Verify every reading and return the :class:`Attribution` of a derivation.

    This is the only way an :class:`Attribution` is built from readings, and it
    is where :func:`verify` is applied. Raises
    :class:`~boardprobe.provenance.ProvenanceError` on an empty collection:
    see :func:`~boardprobe.provenance.meet_all`.
    """
    verified = [verify(reading) for reading in readings]
    if not verified:
        raise ProvenanceError(
            "no readings were supplied, so provenance cannot be established. The meet of "
            "an empty collection is the lattice identity (live), which would let a figure "
            "with no inputs be labelled a measurement; this aborts instead."
        )
    provenance = meet_all([reading.provenance for reading in verified])
    channels: list[str] = []
    for reading in verified:
        if reading.channel not in channels:
            channels.append(reading.channel)
    source_ids: list[str] = []
    for reading in verified:
        if reading.source_id not in source_ids:
            source_ids.append(reading.source_id)
    weakest: list[str] = []
    weakest_sources: list[str] = []
    for reading in verified:
        if reading.provenance is provenance:
            if reading.channel not in weakest:
                weakest.append(reading.channel)
            if reading.source_id not in weakest_sources:
                weakest_sources.append(reading.source_id)
    stamps = [reading.timestamp_ns for reading in verified]
    return Attribution(
        provenance=provenance,
        reading_count=len(verified),
        channels=tuple(channels),
        source_ids=tuple(source_ids),
        weakest=tuple(weakest),
        weakest_sources=tuple(weakest_sources),
        first_timestamp_ns=min(stamps),
        last_timestamp_ns=max(stamps),
    )


def combine(attributions: Iterable[Attribution]) -> Attribution:
    """Meet of several attributions, for a figure derived from other figures.

    Derivation is transitive and so is the propagation: a report built from
    statistics built from readings carries the meet of the whole tree. Nothing
    in this package reaches past an :class:`Attribution` to the readings
    underneath it, so a second level of derivation cannot lose the first
    level's provenance.
    """
    collected = tuple(attributions)
    if not collected:
        raise ProvenanceError(
            "no attributions were supplied, so provenance cannot be established; see "
            "boardprobe.provenance.meet_all for why the identity is refused here"
        )
    for index, item in enumerate(collected):
        if type(item) is not Attribution:
            raise TamperError(
                f"attributions[{index}] must be a boardprobe.readings.Attribution, got "
                f"{type(item).__module__}.{type(item).__name__}; a subclass is refused "
                "because it can override provenance"
            )
    provenance = meet_all([item.provenance for item in collected])
    channels: list[str] = []
    source_ids: list[str] = []
    weakest: list[str] = []
    weakest_sources: list[str] = []
    for item in collected:
        for name in item.channels:
            if name not in channels:
                channels.append(name)
        for name in item.source_ids:
            if name not in source_ids:
                source_ids.append(name)
        if item.provenance is provenance:
            for name in item.weakest:
                if name not in weakest:
                    weakest.append(name)
            for name in item.weakest_sources:
                if name not in weakest_sources:
                    weakest_sources.append(name)
    return Attribution(
        provenance=provenance,
        reading_count=sum(item.reading_count for item in collected),
        channels=tuple(channels),
        source_ids=tuple(source_ids),
        weakest=tuple(weakest),
        weakest_sources=tuple(weakest_sources),
        first_timestamp_ns=min(item.first_timestamp_ns for item in collected),
        last_timestamp_ns=max(item.last_timestamp_ns for item in collected),
    )
