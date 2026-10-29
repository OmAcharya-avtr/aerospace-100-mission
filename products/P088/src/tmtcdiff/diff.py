"""Semantic diff of two dictionary versions, partitioned by failure visibility.

The walk itself is unremarkable: pair items by name, infer renames over the
unpaired remainder, compare field by field. What this module adds over a
general structured differ is that each field comparison emits a *classified*
change from :mod:`tmtcdiff.changes`, so the result can be partitioned into the
silent tier and the loud tier and ranked with the silent tier first.

Three things this module reports about its own limits, as returned data rather
than as documentation:

- **Incomparable fields.** A field present in one version and not the other,
  or differing but not represented in this model, is counted and named. The
  tool does not know what it means and says so.
- **Ambiguous renames.** A rename score inside the band around the threshold
  is counted whichever side of the threshold it fell on.
- **The numeric detection floor.** Scale and offset comparisons use
  :func:`math.isclose` with the configured tolerances, so a scale change
  smaller than the relative tolerance is not reported. The floor is in the
  returned :class:`DiffLimits` and the smallest change actually detected is
  measured in ``validation/validate_change_classes.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .changes import Change, ChangeClass, Severity, rank_changes
from .model import Command, Dictionary, Parameter
from .rename import AMBIGUOUS_BAND, RenameCandidate, match_renames

#: Rename acceptance threshold. **Chosen from the measured curve, not picked.**
#: Criterion, declared before the sweep was run: the lowest threshold on the
#: swept grid whose measured false-match rate is zero at every swept dictionary
#: size, which among thresholds meeting that condition is the one with the
#: highest measured recall. The sweep is
#: ``validation/validate_rename_sweep.py``; its committed output carries the
#: full grid, the criterion and the selected value, and
#: ``tests/test_regressions.py`` pins this constant against that output.
DEFAULT_RENAME_THRESHOLD = 0.80

#: Default relative tolerance for comparing ``scale`` and ``offset``. A change
#: below it is not reported; the floor is returned in :class:`DiffLimits`.
DEFAULT_NUMERIC_REL_TOL = 1e-12

#: Default absolute tolerance. Zero, so the relative bound is the one that
#: acts. ``math.isclose`` passes when either bound is met, exactly as
#: ``pytest.approx`` does, and a non-zero default here would silently replace
#: the relative bound for small scale factors.
DEFAULT_NUMERIC_ABS_TOL = 0.0


@dataclass(frozen=True)
class DiffConfig:
    """Options for :func:`diff_dictionaries`."""

    rename_threshold: float = DEFAULT_RENAME_THRESHOLD
    strategy: str = "optimal"
    weights: tuple[tuple[str, float], ...] = ()
    numeric_rel_tol: float = DEFAULT_NUMERIC_REL_TOL
    numeric_abs_tol: float = DEFAULT_NUMERIC_ABS_TOL
    detect_renames: bool = True
    ambiguous_band: float = AMBIGUOUS_BAND

    def __post_init__(self) -> None:
        if not 0.0 <= self.rename_threshold <= 1.0:
            raise ValueError(
                f"rename_threshold must lie in [0, 1], got {self.rename_threshold!r}"
            )
        if self.numeric_rel_tol < 0.0 or self.numeric_abs_tol < 0.0:
            raise ValueError("numeric tolerances must be non-negative")
        if not math.isfinite(self.numeric_rel_tol) or not math.isfinite(self.numeric_abs_tol):
            raise ValueError("numeric tolerances must be finite")

    @property
    def weight_map(self) -> dict[str, float] | None:
        """Score weights as a mapping, or ``None`` for the module default."""
        return dict(self.weights) if self.weights else None


@dataclass(frozen=True)
class IncomparableField:
    """A field the tool cannot compare, named rather than skipped."""

    kind: str
    name: str
    field: str
    presence: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.presence not in ("old_only", "new_only", "unmodelled_differs"):
            raise ValueError(
                "IncomparableField.presence must be 'old_only', 'new_only' or "
                f"'unmodelled_differs', got {self.presence!r}"
            )

    def one_line(self) -> str:
        """A single line naming the item, the field and why it could not be compared."""
        return f"{self.kind:<9s} {self.name:<28s} {self.field:<20s} {self.presence}"


@dataclass(frozen=True)
class DiffLimits:
    """What the diff could not determine. A returned value, not a caveat."""

    incomparable_fields: int
    ambiguous_renames: int
    unmatched_removed: int
    unmatched_added: int
    numeric_rel_tol: float
    rename_threshold: float
    reasons: tuple[str, ...] = ()

    @property
    def determinable(self) -> bool:
        """True when nothing in this diff was left undetermined."""
        return not self.reasons


@dataclass(frozen=True)
class DiffResult:
    """The ranked semantic diff of two dictionary versions."""

    old_version: str
    new_version: str
    changes: tuple[Change, ...] = ()
    renames: tuple[RenameCandidate, ...] = ()
    rename_candidates: tuple[RenameCandidate, ...] = ()
    incomparable: tuple[IncomparableField, ...] = ()
    limits: DiffLimits = field(
        default_factory=lambda: DiffLimits(0, 0, 0, 0, DEFAULT_NUMERIC_REL_TOL, 0.0)
    )
    old_item_count: int = 0
    new_item_count: int = 0

    @property
    def silent(self) -> tuple[Change, ...]:
        """Silent-tier changes, in ranked order. Reported first, always."""
        return tuple(c for c in self.changes if c.severity is Severity.SILENT)

    @property
    def loud(self) -> tuple[Change, ...]:
        """Loud-tier changes, in ranked order."""
        return tuple(c for c in self.changes if c.severity is Severity.LOUD)

    @property
    def advisory(self) -> tuple[Change, ...]:
        """Advisory-tier changes, in ranked order."""
        return tuple(c for c in self.changes if c.severity is Severity.ADVISORY)

    @property
    def semantic(self) -> tuple[Change, ...]:
        """Silent and loud changes: every change that can alter an existing reference."""
        return tuple(c for c in self.changes if c.severity is not Severity.ADVISORY)

    @property
    def class_counts(self) -> dict[ChangeClass, int]:
        """Count of changes per class, classes with no changes omitted."""
        counts: dict[ChangeClass, int] = {}
        for change in self.changes:
            counts[change.change_class] = counts.get(change.change_class, 0) + 1
        return counts

    @property
    def rename_map(self) -> dict[str, str]:
        """Accepted renames, old name to new name."""
        return {candidate.old_name: candidate.new_name for candidate in self.renames}

    def changes_for(self, name: str) -> tuple[Change, ...]:
        """Every change filed under ``name``, which for a rename is the old name."""
        return tuple(change for change in self.changes if change.name == name)

    @property
    def changed_item_names(self) -> tuple[str, ...]:
        """Names carrying at least one silent or loud change."""
        return tuple(sorted({change.name for change in self.semantic}))


def _close(left: float, right: float, config: DiffConfig) -> bool:
    return math.isclose(
        left, right, rel_tol=config.numeric_rel_tol, abs_tol=config.numeric_abs_tol
    )


def _enum_detail(old: Parameter | Any, new: Parameter | Any) -> tuple[bool, str]:
    """Compare two enumerations. Returns ``(changed, detail)``.

    Four sub-cases are distinguished in the detail text, because they fail
    differently: a relabelled code and a recoded label both change what an
    existing comparison means, a removed label makes an equality test that
    never fires, and an added label changes nothing that already existed.
    """
    old_map = dict(old.enum)
    new_map = dict(new.enum)
    old_labels = {label: code for code, label in old.enum}
    new_labels = {label: code for code, label in new.enum}
    relabelled = sorted(c for c in set(old_map) & set(new_map) if old_map[c] != new_map[c])
    recoded = sorted(
        label
        for label in set(old_labels) & set(new_labels)
        if old_labels[label] != new_labels[label]
    )
    removed = sorted(set(old_labels) - set(new_labels))
    added = sorted(set(new_labels) - set(old_labels))
    if not (relabelled or recoded or removed or added):
        return (False, "")
    parts: list[str] = []
    if relabelled:
        parts.append(f"codes relabelled {relabelled}")
    if recoded:
        parts.append(f"labels recoded {recoded}")
    if removed:
        parts.append(f"labels removed {removed} (an equality test on these never fires)")
    if added:
        parts.append(f"labels added {added}")
    if not (relabelled or recoded or removed):
        parts.append("labels added only; no existing code or label changed meaning")
    return (True, "; ".join(parts))


def compare_parameters(
    old: Parameter, new: Parameter, config: DiffConfig
) -> tuple[tuple[Change, ...], tuple[IncomparableField, ...]]:
    """Field-by-field comparison of two parameter definitions.

    ``new.name`` may differ from ``old.name``: this function is used for
    matched renames as well as for same-name pairs, and every change it emits
    is filed under ``old.name`` with ``new_name`` set when the names differ.
    """
    renamed = old.name != new.name
    new_name = new.name if renamed else ""
    changes: list[Change] = []

    def emit(
        change_class: ChangeClass,
        field_name: str,
        old_value: Any,
        new_value: Any,
        detail: str = "",
    ) -> None:
        changes.append(
            Change(
                change_class=change_class,
                kind="parameter",
                name=old.name,
                field=field_name,
                old_value=old_value,
                new_value=new_value,
                new_name=new_name,
                detail=detail,
            )
        )

    if old.data_type != new.data_type:
        emit(ChangeClass.RETYPED, "data_type", old.data_type, new.data_type)
    if old.bit_offset != new.bit_offset:
        emit(ChangeClass.BIT_SHIFTED, "bit_offset", old.bit_offset, new.bit_offset)
    if old.bit_width != new.bit_width:
        emit(ChangeClass.BIT_SHIFTED, "bit_width", old.bit_width, new.bit_width)
    if old.byte_order != new.byte_order:
        emit(ChangeClass.ENDIANNESS_CHANGED, "byte_order", old.byte_order, new.byte_order)
    if not _close(old.scale, new.scale, config):
        ratio = new.scale / old.scale if old.scale else math.inf
        emit(
            ChangeClass.RESCALED,
            "scale",
            old.scale,
            new.scale,
            f"every engineering value scales by {ratio:.6g}",
        )
    if not _close(old.offset, new.offset, config):
        emit(
            ChangeClass.RESCALED,
            "offset",
            old.offset,
            new.offset,
            f"every engineering value shifts by {new.offset - old.offset:+.6g}",
        )
    if old.unit != new.unit:
        emit(ChangeClass.UNIT_CHANGED, "unit", old.unit, new.unit)
    enum_changed, enum_detail = _enum_detail(old, new)
    if enum_changed:
        emit(ChangeClass.REENUMERATED, "enum", old.enum_map, new.enum_map, enum_detail)
    if (old.low_limit, old.high_limit) != (new.low_limit, new.high_limit):
        emit(
            ChangeClass.LIMITS_CHANGED,
            "limits",
            (old.low_limit, old.high_limit),
            (new.low_limit, new.high_limit),
        )
    if old.packet != new.packet:
        emit(ChangeClass.PACKET_MOVED, "packet", old.packet, new.packet)
    if old.path != new.path:
        emit(ChangeClass.PATH_MOVED, "path", old.path, new.path)
    if old.description != new.description:
        emit(ChangeClass.DESCRIPTION_CHANGED, "description", old.description, new.description)

    incomparable = _extra_fields("parameter", old.name, old.extra_map, new.extra_map)
    return (tuple(changes), incomparable)


def _extra_fields(
    kind: str, name: str, old_extra: dict[str, str], new_extra: dict[str, str]
) -> tuple[IncomparableField, ...]:
    out: list[IncomparableField] = []
    for key in sorted(set(old_extra) - set(new_extra)):
        out.append(IncomparableField(kind, name, key, "old_only"))
    for key in sorted(set(new_extra) - set(old_extra)):
        out.append(IncomparableField(kind, name, key, "new_only"))
    for key in sorted(set(old_extra) & set(new_extra)):
        if old_extra[key] != new_extra[key]:
            out.append(
                IncomparableField(
                    kind,
                    name,
                    key,
                    "unmodelled_differs",
                    f"{old_extra[key]!r} -> {new_extra[key]!r}; this tool does not model "
                    "this field and cannot say whether the change is semantic",
                )
            )
    return tuple(out)


def compare_commands(
    old: Command, new: Command, config: DiffConfig
) -> tuple[tuple[Change, ...], tuple[IncomparableField, ...]]:
    """Field-by-field comparison of two command definitions, arguments included."""
    renamed = old.name != new.name
    new_name = new.name if renamed else ""
    changes: list[Change] = []

    def emit(
        change_class: ChangeClass,
        field_name: str,
        old_value: Any,
        new_value: Any,
        detail: str = "",
    ) -> None:
        changes.append(
            Change(
                change_class=change_class,
                kind="command",
                name=old.name,
                field=field_name,
                old_value=old_value,
                new_value=new_value,
                new_name=new_name,
                detail=detail,
            )
        )

    old_arguments = old.argument_map
    new_arguments = new.argument_map
    for argument_name in sorted(set(old_arguments) - set(new_arguments)):
        emit(ChangeClass.ARGUMENT_REMOVED, f"argument:{argument_name}", argument_name, None)
    for argument_name in sorted(set(new_arguments) - set(old_arguments)):
        emit(ChangeClass.ARGUMENT_ADDED, f"argument:{argument_name}", None, argument_name)
    for argument_name in sorted(set(old_arguments) & set(new_arguments)):
        old_argument = old_arguments[argument_name]
        new_argument = new_arguments[argument_name]
        if old_argument.position != new_argument.position:
            emit(
                ChangeClass.ARGUMENT_REORDERED,
                f"argument:{argument_name}.position",
                old_argument.position,
                new_argument.position,
                "the same written call encodes different bits",
            )
        if old_argument.data_type != new_argument.data_type:
            emit(
                ChangeClass.ARGUMENT_RETYPED,
                f"argument:{argument_name}.data_type",
                old_argument.data_type,
                new_argument.data_type,
            )
        if old_argument.bit_width != new_argument.bit_width:
            emit(
                ChangeClass.ARGUMENT_RETYPED,
                f"argument:{argument_name}.bit_width",
                old_argument.bit_width,
                new_argument.bit_width,
            )
        if not _close(old_argument.scale, new_argument.scale, config):
            emit(
                ChangeClass.ARGUMENT_RESCALED,
                f"argument:{argument_name}.scale",
                old_argument.scale,
                new_argument.scale,
            )
        if not _close(old_argument.offset, new_argument.offset, config):
            emit(
                ChangeClass.ARGUMENT_RESCALED,
                f"argument:{argument_name}.offset",
                old_argument.offset,
                new_argument.offset,
            )
        if old_argument.unit != new_argument.unit:
            emit(
                ChangeClass.UNIT_CHANGED,
                f"argument:{argument_name}.unit",
                old_argument.unit,
                new_argument.unit,
            )
        enum_changed, enum_detail = _enum_detail(old_argument, new_argument)
        if enum_changed:
            emit(
                ChangeClass.REENUMERATED,
                f"argument:{argument_name}.enum",
                dict(old_argument.enum),
                dict(new_argument.enum),
                enum_detail,
            )

    if old.path != new.path:
        emit(ChangeClass.PATH_MOVED, "path", old.path, new.path)
    if old.description != new.description:
        emit(ChangeClass.DESCRIPTION_CHANGED, "description", old.description, new.description)

    incomparable = _extra_fields("command", old.name, old.extra_map, new.extra_map)
    return (tuple(changes), incomparable)


def diff_dictionaries(
    old: Dictionary, new: Dictionary, config: DiffConfig | None = None
) -> DiffResult:
    """Compare two dictionary versions and return a ranked semantic diff.

    Parameters and commands are matched by name; the unmatched remainder is
    offered to the rename matcher when ``config.detect_renames`` is set. Every
    change that survives is classified, and the result is ranked with the
    silent tier first.
    """
    if not isinstance(old, Dictionary) or not isinstance(new, Dictionary):
        raise TypeError("diff_dictionaries compares two Dictionary instances")
    active = DiffConfig() if config is None else config

    changes: list[Change] = []
    incomparable: list[IncomparableField] = []
    accepted: list[RenameCandidate] = []
    all_candidates: list[RenameCandidate] = []

    old_parameters = old.parameter_map
    new_parameters = new.parameter_map
    for name in sorted(set(old_parameters) & set(new_parameters)):
        item_changes, item_incomparable = compare_parameters(
            old_parameters[name], new_parameters[name], active
        )
        changes.extend(item_changes)
        incomparable.extend(item_incomparable)

    old_commands = old.command_map
    new_commands = new.command_map
    for name in sorted(set(old_commands) & set(new_commands)):
        item_changes, item_incomparable = compare_commands(
            old_commands[name], new_commands[name], active
        )
        changes.extend(item_changes)
        incomparable.extend(item_incomparable)

    removed_parameters = [
        old_parameters[n] for n in sorted(set(old_parameters) - set(new_parameters))
    ]
    added_parameters = [
        new_parameters[n] for n in sorted(set(new_parameters) - set(old_parameters))
    ]
    removed_commands = [old_commands[n] for n in sorted(set(old_commands) - set(new_commands))]
    added_commands = [new_commands[n] for n in sorted(set(new_commands) - set(old_commands))]

    for kind, removed_items, added_items, renamed_class, removed_class, added_class in (
        (
            "parameter",
            removed_parameters,
            added_parameters,
            ChangeClass.RENAMED,
            ChangeClass.REMOVED,
            ChangeClass.ADDED,
        ),
        (
            "command",
            removed_commands,
            added_commands,
            ChangeClass.COMMAND_RENAMED,
            ChangeClass.COMMAND_REMOVED,
            ChangeClass.COMMAND_ADDED,
        ),
    ):
        candidates: tuple[RenameCandidate, ...] = ()
        if active.detect_renames:
            candidates = match_renames(
                removed_items,  # type: ignore[arg-type]
                added_items,  # type: ignore[arg-type]
                threshold=active.rename_threshold,
                strategy=active.strategy,
                weights=active.weight_map,
                ambiguous_band=active.ambiguous_band,
            )
        all_candidates.extend(candidates)
        matched_old = {c.old_name for c in candidates if c.accepted}
        matched_new = {c.new_name for c in candidates if c.accepted}
        accepted.extend(c for c in candidates if c.accepted)

        removed_by_name = {item.name: item for item in removed_items}
        added_by_name = {item.name: item for item in added_items}
        for candidate in candidates:
            if not candidate.accepted:
                continue
            old_item = removed_by_name[candidate.old_name]
            new_item = added_by_name[candidate.new_name]
            changes.append(
                Change(
                    change_class=renamed_class,
                    kind=kind,
                    name=candidate.old_name,
                    field="name",
                    old_value=candidate.old_name,
                    new_value=candidate.new_name,
                    new_name=candidate.new_name,
                    similarity=candidate.score,
                    detail="structural match; a by-name reference to the old name fails to resolve",
                )
            )
            if kind == "parameter":
                item_changes, item_incomparable = compare_parameters(
                    old_item, new_item, active  # type: ignore[arg-type]
                )
            else:
                item_changes, item_incomparable = compare_commands(
                    old_item, new_item, active  # type: ignore[arg-type]
                )
            changes.extend(
                change
                for change in item_changes
                if change.change_class is not ChangeClass.PATH_MOVED
            )
            incomparable.extend(item_incomparable)

        for item in removed_items:
            if item.name not in matched_old:
                changes.append(
                    Change(
                        change_class=removed_class,
                        kind=kind,
                        name=item.name,
                        field="name",
                        old_value=item.name,
                        new_value=None,
                        detail="references fail to resolve, visibly",
                    )
                )
        for item in added_items:
            if item.name not in matched_new:
                changes.append(
                    Change(
                        change_class=added_class,
                        kind=kind,
                        name=item.name,
                        field="name",
                        old_value=None,
                        new_value=item.name,
                    )
                )

    ambiguous = tuple(c for c in all_candidates if c.ambiguous)
    unmatched_removed = len(removed_parameters) + len(removed_commands) - len(accepted)
    unmatched_added = len(added_parameters) + len(added_commands) - len(accepted)

    reasons: list[str] = []
    if incomparable:
        reasons.append(
            f"{len(incomparable)} field(s) on {len({f.name for f in incomparable})} item(s) are "
            "not modelled by this tool or are present in only one version; their semantics are "
            "undetermined"
        )
    if ambiguous:
        reasons.append(
            f"{len(ambiguous)} rename candidate(s) scored within {active.ambiguous_band} of the "
            f"acceptance threshold {active.rename_threshold}; their classification is a "
            "threshold artefact"
        )
    if active.numeric_rel_tol > 0.0:
        reasons.append(
            f"scale and offset changes below a relative {active.numeric_rel_tol:g} are not "
            "reported by construction"
        )
    if not active.detect_renames:
        reasons.append("rename detection was disabled, so every rename is reported as a removal")

    limits = DiffLimits(
        incomparable_fields=len(incomparable),
        ambiguous_renames=len(ambiguous),
        unmatched_removed=unmatched_removed,
        unmatched_added=unmatched_added,
        numeric_rel_tol=active.numeric_rel_tol,
        rename_threshold=active.rename_threshold,
        reasons=tuple(reasons),
    )

    return DiffResult(
        old_version=old.version,
        new_version=new.version,
        changes=rank_changes(tuple(changes)),
        renames=tuple(sorted(accepted, key=lambda c: (-c.score, c.old_name))),
        rename_candidates=tuple(sorted(all_candidates, key=lambda c: (-c.score, c.old_name))),
        incomparable=tuple(incomparable),
        limits=limits,
        old_item_count=len(old),
        new_item_count=len(new),
    )
