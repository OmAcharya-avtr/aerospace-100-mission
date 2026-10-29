"""The change-class taxonomy, the severity tiers, and the ranking rule.

This module is the part of the package that is not available off the shelf. A
general structured differ reports *that* a field changed; it has no way to know
that a changed ``scale`` on an unchanged mnemonic is the single most dangerous
thing that can happen to a telemetry dictionary, while a removed mnemonic is
comparatively harmless because every consumer of it fails immediately and
visibly.

Three tiers, and why they are ordered this way
----------------------------------------------

``SILENT`` (rank 0)
    The name a procedure or script uses still resolves, the call still runs,
    and the number it produces now means something else. Nothing fails. There
    is no error message, no exception, and no log line. Detection requires
    somebody to compare the two dictionaries, which is what this package does.

``LOUD`` (rank 1)
    The reference no longer resolves. The procedure fails at load or at first
    use, with the missing name in the message. This is a real break and it is
    ranked *below* silent change because it announces itself.

``ADVISORY`` (rank 2)
    A change that adds capability or alters prose. No existing reference
    changes meaning or stops resolving.

The ordering of tiers is the product's claim and it is a judgement about
failure visibility, not a measurement. The ordering *within* the ``SILENT``
tier is a declared convention and is labelled as such everywhere it appears;
no measurement in this repository ranks a rescale against a re-enumeration.

What is deliberately *not* in the taxonomy: a severity for an unmodelled field.
When the two versions disagree on a field this model does not represent, the
honest output is that the comparison could not be made, and that is reported
as an incomparable field in the coverage verdict rather than as a change with
an invented severity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """Failure visibility of a change. Lower :attr:`rank` is more dangerous."""

    SILENT = "silent"
    LOUD = "loud"
    ADVISORY = "advisory"

    @property
    def rank(self) -> int:
        """0 for silent, 1 for loud, 2 for advisory."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.SILENT: 0,
    Severity.LOUD: 1,
    Severity.ADVISORY: 2,
}


class ChangeClass(StrEnum):
    """Every change class this package detects."""

    # --- silent: the name still resolves, the meaning moved ------------------
    RESCALED = "rescaled"
    UNIT_CHANGED = "unit_changed"
    REENUMERATED = "reenumerated"
    BIT_SHIFTED = "bit_shifted"
    RETYPED = "retyped"
    ENDIANNESS_CHANGED = "endianness_changed"
    PACKET_MOVED = "packet_moved"
    LIMITS_CHANGED = "limits_changed"
    ARGUMENT_REORDERED = "argument_reordered"
    ARGUMENT_RETYPED = "argument_retyped"
    ARGUMENT_RESCALED = "argument_rescaled"
    # --- loud: the reference stops resolving --------------------------------
    REMOVED = "removed"
    RENAMED = "renamed"
    PATH_MOVED = "path_moved"
    COMMAND_REMOVED = "command_removed"
    COMMAND_RENAMED = "command_renamed"
    ARGUMENT_REMOVED = "argument_removed"
    ARGUMENT_ADDED = "argument_added"
    # --- advisory: new capability or prose ----------------------------------
    ADDED = "added"
    COMMAND_ADDED = "command_added"
    DESCRIPTION_CHANGED = "description_changed"

    @property
    def severity(self) -> Severity:
        """The tier this class belongs to."""
        return CLASS_SEVERITY[self]

    @property
    def is_silent(self) -> bool:
        """True when the class belongs to the silent tier."""
        return CLASS_SEVERITY[self] is Severity.SILENT


#: Tier of each change class.
CLASS_SEVERITY: dict[ChangeClass, Severity] = {
    ChangeClass.RESCALED: Severity.SILENT,
    ChangeClass.UNIT_CHANGED: Severity.SILENT,
    ChangeClass.REENUMERATED: Severity.SILENT,
    ChangeClass.BIT_SHIFTED: Severity.SILENT,
    ChangeClass.RETYPED: Severity.SILENT,
    ChangeClass.ENDIANNESS_CHANGED: Severity.SILENT,
    ChangeClass.PACKET_MOVED: Severity.SILENT,
    ChangeClass.LIMITS_CHANGED: Severity.SILENT,
    ChangeClass.ARGUMENT_REORDERED: Severity.SILENT,
    ChangeClass.ARGUMENT_RETYPED: Severity.SILENT,
    ChangeClass.ARGUMENT_RESCALED: Severity.SILENT,
    ChangeClass.REMOVED: Severity.LOUD,
    ChangeClass.RENAMED: Severity.LOUD,
    ChangeClass.PATH_MOVED: Severity.LOUD,
    ChangeClass.COMMAND_REMOVED: Severity.LOUD,
    ChangeClass.COMMAND_RENAMED: Severity.LOUD,
    ChangeClass.ARGUMENT_REMOVED: Severity.LOUD,
    ChangeClass.ARGUMENT_ADDED: Severity.LOUD,
    ChangeClass.ADDED: Severity.ADVISORY,
    ChangeClass.COMMAND_ADDED: Severity.ADVISORY,
    ChangeClass.DESCRIPTION_CHANGED: Severity.ADVISORY,
}

#: Classes in the silent tier, in the declared within-tier order. The order is
#: a convention, not a measurement, and is labelled as such wherever printed.
SILENT_CLASSES: tuple[ChangeClass, ...] = (
    ChangeClass.RESCALED,
    ChangeClass.UNIT_CHANGED,
    ChangeClass.REENUMERATED,
    ChangeClass.BIT_SHIFTED,
    ChangeClass.RETYPED,
    ChangeClass.ENDIANNESS_CHANGED,
    ChangeClass.ARGUMENT_REORDERED,
    ChangeClass.ARGUMENT_RETYPED,
    ChangeClass.ARGUMENT_RESCALED,
    ChangeClass.PACKET_MOVED,
    ChangeClass.LIMITS_CHANGED,
)

#: Classes in the loud tier, in the declared within-tier order.
LOUD_CLASSES: tuple[ChangeClass, ...] = (
    ChangeClass.REMOVED,
    ChangeClass.RENAMED,
    ChangeClass.COMMAND_REMOVED,
    ChangeClass.COMMAND_RENAMED,
    ChangeClass.ARGUMENT_REMOVED,
    ChangeClass.ARGUMENT_ADDED,
    ChangeClass.PATH_MOVED,
)

#: Classes in the advisory tier, in the declared within-tier order.
ADVISORY_CLASSES: tuple[ChangeClass, ...] = (
    ChangeClass.ADDED,
    ChangeClass.COMMAND_ADDED,
    ChangeClass.DESCRIPTION_CHANGED,
)

#: Declared order of every class, silent tier first.
CLASS_ORDER: tuple[ChangeClass, ...] = (*SILENT_CLASSES, *LOUD_CLASSES, *ADVISORY_CLASSES)

_CLASS_INDEX: dict[ChangeClass, int] = {cls: index for index, cls in enumerate(CLASS_ORDER)}

#: One line per class, for the CLI's taxonomy listing and for the README.
CLASS_NOTES: dict[ChangeClass, str] = {
    ChangeClass.RESCALED: "scale or offset changed; every engineering value moves",
    ChangeClass.UNIT_CHANGED: "unit string changed; the number may or may not have moved with it",
    ChangeClass.REENUMERATED: "a raw code now carries a different label, or a label a "
    "different code",
    ChangeClass.BIT_SHIFTED: "bit offset or bit width changed; extraction reads other bits",
    ChangeClass.RETYPED: "data type changed; the same bits are reinterpreted",
    ChangeClass.ENDIANNESS_CHANGED: "byte order changed; multi-byte fields decode differently",
    ChangeClass.PACKET_MOVED: "parameter now lives in a different packet or container",
    ChangeClass.LIMITS_CHANGED: "soft limits changed; alarm behaviour moves, the value does not",
    ChangeClass.ARGUMENT_REORDERED: "command argument positions changed; the same written "
    "call encodes different bits",
    ChangeClass.ARGUMENT_RETYPED: "command argument type or width changed",
    ChangeClass.ARGUMENT_RESCALED: "command argument calibration changed",
    ChangeClass.REMOVED: "parameter gone; references fail to resolve",
    ChangeClass.RENAMED: "parameter name changed, structure matched to a new name",
    ChangeClass.PATH_MOVED: "item path changed; by-path references fail, by-name ones do not",
    ChangeClass.COMMAND_REMOVED: "command gone; references fail to resolve",
    ChangeClass.COMMAND_RENAMED: "command name changed, structure matched to a new name",
    ChangeClass.ARGUMENT_REMOVED: "command argument gone; calls passing it fail",
    ChangeClass.ARGUMENT_ADDED: "command gained an argument; existing calls are incomplete",
    ChangeClass.ADDED: "new parameter; no existing reference is affected",
    ChangeClass.COMMAND_ADDED: "new command; no existing reference is affected",
    ChangeClass.DESCRIPTION_CHANGED: "description text changed only",
}


@dataclass(frozen=True)
class Change:
    """One detected change.

    ``name`` is the name the change is filed under: for a rename it is the
    *old* name, because that is the name an existing procedure holds.
    """

    change_class: ChangeClass
    kind: str
    name: str
    field: str = ""
    old_value: Any = None
    new_value: Any = None
    new_name: str = ""
    similarity: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("parameter", "command"):
            raise ValueError(f"Change.kind must be 'parameter' or 'command', got {self.kind!r}")
        if not isinstance(self.change_class, ChangeClass):
            raise TypeError("Change.change_class must be a ChangeClass")

    @property
    def severity(self) -> Severity:
        """Tier of this change's class."""
        return self.change_class.severity

    @property
    def is_silent(self) -> bool:
        """True when this change does not announce itself to a consumer."""
        return self.change_class.is_silent

    @property
    def sort_key(self) -> tuple[int, int, str, str]:
        """Ranking key: tier, then declared within-tier class order, then name, then field."""
        return (
            self.severity.rank,
            _CLASS_INDEX[self.change_class],
            self.name,
            self.field,
        )

    def one_line(self) -> str:
        """A single line describing the change, with old and new values when present."""
        head = f"[{self.severity.value:8s}] {self.change_class.value:<20s} {self.name}"
        if self.new_name:
            head += f" -> {self.new_name}"
        if self.field:
            head += f"  {self.field}: {self.old_value!r} -> {self.new_value!r}"
        if self.similarity is not None:
            head += f"  (similarity {self.similarity:.4f})"
        if self.detail:
            head += f"  {self.detail}"
        return head


def rank_changes(changes: tuple[Change, ...]) -> tuple[Change, ...]:
    """Sort changes with the silent tier first. This is the product's ranking rule."""
    return tuple(sorted(changes, key=lambda change: change.sort_key))
