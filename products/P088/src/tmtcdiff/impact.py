"""Impact of a dictionary change on an existing corpus, with the coverage verdict.

Findings are ranked with the silent tier first, for the same reason the diff
is: a reference that still resolves and now means something else will not
announce itself, and a reference that stopped resolving will.

The coverage verdict is the point of this module
------------------------------------------------

:class:`CoverageVerdict` is computed and returned by every call to
:func:`assess_impact`. It states what the tool could **not** determine about
the corpus it was given: references it could not resolve statically,
dictionary items the corpus never mentions, sources it could not read, rename
candidates that landed on the threshold, and fields it does not model. It is a
value, not a section of prose, because a caller has to be able to act on it.

An empty finding list does not mean an upgrade is safe, and this module will
not say that it does. :attr:`ImpactReport.interpretation` renders the only
statement the evidence supports: no finding was produced *for the references
this tool could resolve in the sources it was given*, with the uncovered
fraction quantified beside it. When coverage is partial — and it almost always
is, because a corpus contains run-time-constructed references — the verdict
says the question was not answered rather than answering it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .changes import Change, ChangeClass, Severity
from .corpus import Corpus, Reference, extract_references
from .diff import DiffResult
from .model import Dictionary


class ImpactClass(StrEnum):
    """What happened to one reference. :attr:`rank` orders the report."""

    SILENTLY_CHANGED = "silently_changed"
    BROKEN = "broken"
    RENAMED_AWAY = "renamed_away"
    UNDECIDABLE = "undecidable"
    UNAFFECTED = "unaffected"

    @property
    def rank(self) -> int:
        """0 for silently changed, then broken, renamed away, undecidable, unaffected."""
        return _IMPACT_RANK[self]


_IMPACT_RANK: dict[ImpactClass, int] = {
    ImpactClass.SILENTLY_CHANGED: 0,
    ImpactClass.BROKEN: 1,
    ImpactClass.RENAMED_AWAY: 2,
    ImpactClass.UNDECIDABLE: 3,
    ImpactClass.UNAFFECTED: 4,
}

#: One line per impact class, for the report header and the README.
IMPACT_NOTES: dict[ImpactClass, str] = {
    ImpactClass.SILENTLY_CHANGED: "resolves, runs, and now means something else",
    ImpactClass.BROKEN: "no longer resolves; fails visibly at load or first use",
    ImpactClass.RENAMED_AWAY: "no longer resolves; a renamed item matched structurally",
    ImpactClass.UNDECIDABLE: "constructed at run time; this tool cannot resolve it",
    ImpactClass.UNAFFECTED: "resolves to an item with no silent or loud change",
}


@dataclass(frozen=True)
class Finding:
    """One classified reference, with the changes that explain its classification."""

    reference: Reference
    impact_class: ImpactClass
    target: str = ""
    change_classes: tuple[ChangeClass, ...] = ()
    suggestion: str = ""
    detail: str = ""

    @property
    def rank(self) -> int:
        """Rank of this finding's impact class."""
        return self.impact_class.rank

    def one_line(self) -> str:
        """A single line: class, source location, target and explanation."""
        head = (
            f"[{self.impact_class.value:<16s}] {self.reference.source}:"
            f"{self.reference.line:<4d} {self.reference.target}"
        )
        if self.change_classes:
            head += "  " + ",".join(sorted(c.value for c in self.change_classes))
        if self.suggestion:
            head += f"  suggestion: {self.suggestion}"
        if self.detail:
            head += f"  {self.detail}"
        return head


@dataclass(frozen=True)
class CoverageVerdict:
    """What the impact assessment could not determine. Computed, returned, not a caveat."""

    corpus_sources: int
    unreadable_sources: tuple[str, ...]
    references_total: int
    references_undecidable: int
    dictionary_items: int
    items_referenced: int
    items_unreferenced: int
    changed_items_total: int
    changed_items_referenced: int
    changed_items_unreferenced: int
    ambiguous_renames: int
    incomparable_fields: int
    #: Enumeration labels the corpus references that belong to more than one
    #: parameter, so a bare label reference cannot be attributed to one of them.
    ambiguous_enum_labels: tuple[str, ...]
    #: Why the assessment is incomplete. Empty means nothing was left open.
    reasons: tuple[str, ...]

    @property
    def coverage_fraction(self) -> float:
        """Fraction of dictionary items the corpus mentions. 0.0 for an empty dictionary."""
        if self.dictionary_items == 0:
            return 0.0
        return self.items_referenced / self.dictionary_items

    @property
    def changed_item_coverage(self) -> float:
        """Fraction of changed items the corpus mentions. 1.0 when nothing changed."""
        if self.changed_items_total == 0:
            return 1.0
        return self.changed_items_referenced / self.changed_items_total

    @property
    def decidable_fraction(self) -> float:
        """Fraction of extracted references this tool could resolve statically."""
        if self.references_total == 0:
            return 0.0
        return (self.references_total - self.references_undecidable) / self.references_total

    @property
    def determinable(self) -> bool:
        """True only when nothing was left undetermined. Usually false, and that is the point."""
        return not self.reasons

    def report_lines(self) -> list[str]:
        """The verdict as text, one fact per line."""
        lines = [
            f"corpus sources                 : {self.corpus_sources}",
            f"unreadable sources             : {len(self.unreadable_sources)}",
            f"references extracted           : {self.references_total}",
            f"references undecidable         : {self.references_undecidable}"
            f" ({1.0 - self.decidable_fraction:.4f} of extracted)",
            f"dictionary items               : {self.dictionary_items}",
            f"items referenced by the corpus : {self.items_referenced}"
            f" ({self.coverage_fraction:.4f})",
            f"items never referenced         : {self.items_unreferenced}",
            f"changed items                  : {self.changed_items_total}",
            f"changed items referenced       : {self.changed_items_referenced}"
            f" ({self.changed_item_coverage:.4f})",
            f"changed items not referenced   : {self.changed_items_unreferenced}",
            f"ambiguous rename candidates    : {self.ambiguous_renames}",
            f"incomparable fields            : {self.incomparable_fields}",
            f"ambiguous enum labels          : {len(self.ambiguous_enum_labels)}",
            f"fully determinable             : {self.determinable}",
        ]
        if self.reasons:
            lines.append("undetermined because:")
            lines.extend(f"  - {reason}" for reason in self.reasons)
        return lines


@dataclass(frozen=True)
class ImpactReport:
    """Ranked findings plus the coverage verdict."""

    old_version: str
    new_version: str
    findings: tuple[Finding, ...]
    verdict: CoverageVerdict

    def of_class(self, impact_class: ImpactClass) -> tuple[Finding, ...]:
        """Findings of one impact class, in ranked order."""
        return tuple(f for f in self.findings if f.impact_class is impact_class)

    @property
    def silently_changed(self) -> tuple[Finding, ...]:
        """References that still resolve and now mean something else. Reported first."""
        return self.of_class(ImpactClass.SILENTLY_CHANGED)

    @property
    def broken(self) -> tuple[Finding, ...]:
        """References that no longer resolve."""
        return self.of_class(ImpactClass.BROKEN)

    @property
    def renamed_away(self) -> tuple[Finding, ...]:
        """References whose target was matched to a renamed item."""
        return self.of_class(ImpactClass.RENAMED_AWAY)

    @property
    def undecidable(self) -> tuple[Finding, ...]:
        """References this tool could not resolve statically."""
        return self.of_class(ImpactClass.UNDECIDABLE)

    @property
    def unaffected(self) -> tuple[Finding, ...]:
        """References resolving to items with no silent or loud change."""
        return self.of_class(ImpactClass.UNAFFECTED)

    @property
    def actionable(self) -> tuple[Finding, ...]:
        """Silently changed, broken and renamed-away findings, in ranked order."""
        return tuple(
            f
            for f in self.findings
            if f.impact_class
            in (ImpactClass.SILENTLY_CHANGED, ImpactClass.BROKEN, ImpactClass.RENAMED_AWAY)
        )

    @property
    def class_counts(self) -> dict[ImpactClass, int]:
        """Count of findings per impact class, every class present."""
        return {
            impact_class: len(self.of_class(impact_class))
            for impact_class in sorted(ImpactClass, key=lambda c: c.rank)
        }

    @property
    def interpretation(self) -> str:
        """The strongest statement the evidence supports, and no stronger.

        This property exists so that the conclusion is computed from the
        verdict rather than written by the reader. In particular, an empty
        finding list never renders as a statement that the upgrade is safe:
        when coverage is partial the rendered sentence says the question was
        not answered.
        """
        if self.actionable:
            return (
                f"{len(self.actionable)} reference(s) are affected: "
                f"{len(self.silently_changed)} silently changed, {len(self.broken)} broken, "
                f"{len(self.renamed_away)} renamed away. The silently changed ones are listed "
                "first because nothing downstream of them fails."
            )
        if self.verdict.determinable:
            return (
                "No affected reference was found, and this assessment determined everything it "
                "was asked about: every extracted reference resolved statically, every source "
                "was readable, every changed item is referenced by the corpus, and no field was "
                "incomparable. That is a statement about this corpus and these two dictionary "
                "versions only."
            )
        return (
            "No affected reference was found among the references this tool could resolve, and "
            "the assessment is NOT complete: "
            + "; ".join(self.verdict.reasons)
            + ". An empty finding list under partial coverage is an absence of evidence, not "
            "evidence of absence."
        )


def _silent_classes(changes: tuple[Change, ...]) -> tuple[ChangeClass, ...]:
    return tuple(sorted({c.change_class for c in changes if c.is_silent}, key=lambda c: c.value))


def _loud_classes(changes: tuple[Change, ...]) -> tuple[ChangeClass, ...]:
    return tuple(
        sorted(
            {c.change_class for c in changes if c.severity is Severity.LOUD}, key=lambda c: c.value
        )
    )


def assess_impact(
    diff: DiffResult,
    corpus: Corpus,
    old: Dictionary,
    new: Dictionary,
    *,
    references: tuple[Reference, ...] | None = None,
) -> ImpactReport:
    """Classify every reference in ``corpus`` against ``diff`` and return a ranked report.

    ``old`` and ``new`` are the two dictionaries ``diff`` was computed from;
    they are needed to resolve references by path and by enumeration label,
    which the diff alone does not carry.
    """
    if not isinstance(diff, DiffResult):
        raise TypeError(f"assess_impact expects a DiffResult, got {type(diff).__name__}")
    if not isinstance(corpus, Corpus):
        raise TypeError(f"assess_impact expects a Corpus, got {type(corpus).__name__}")
    for name, value in (("old", old), ("new", new)):
        if not isinstance(value, Dictionary):
            raise TypeError(f"assess_impact expects a Dictionary for {name}")

    extracted = extract_references(corpus) if references is None else tuple(references)

    rename_map = diff.rename_map
    old_labels = old.enum_labels
    new_labels = new.enum_labels
    label_owners: dict[str, int] = {}
    for parameter in old.parameters:
        for _, label in parameter.enum:
            label_owners[label] = label_owners.get(label, 0) + 1
    # Only labels the corpus actually references are reported as ambiguous. A
    # label shared by two parameters is a latent ambiguity in the dictionary;
    # it becomes something this assessment could not determine only when a
    # bare label reference exists that has to be attributed to one of them.
    referenced_labels = {
        reference.target
        for reference in extracted
        if reference.kind == "enum_label" and reference.decidable
    }
    ambiguous_labels = tuple(
        sorted(
            label
            for label, count in label_owners.items()
            if count > 1 and label in referenced_labels
        )
    )

    findings: list[Finding] = []
    referenced_items: set[str] = set()

    for reference in extracted:
        if not reference.decidable:
            findings.append(
                Finding(
                    reference=reference,
                    impact_class=ImpactClass.UNDECIDABLE,
                    detail=reference.reason,
                )
            )
            continue

        if reference.kind == "enum_label":
            finding = _assess_enum_label(reference, diff, old_labels, new_labels)
            if finding.target:
                referenced_items.add(finding.target)
            findings.append(finding)
            continue

        if reference.kind == "argument":
            command_name, _, argument_name = reference.target.rpartition(".")
            resolved = old.resolve(command_name)
            if resolved is not None:
                referenced_items.add(resolved)
            findings.append(
                _assess_argument(reference, diff, old, new, command_name, argument_name)
            )
            continue

        resolved_old = old.resolve(reference.target)
        if resolved_old is not None:
            referenced_items.add(resolved_old)
        resolved_new = new.resolve(reference.target)

        if resolved_new is not None:
            changes = diff.changes_for(resolved_new) if resolved_old is None else diff.changes_for(
                resolved_old
            )
            silent = _silent_classes(changes)
            if silent:
                findings.append(
                    Finding(
                        reference=reference,
                        impact_class=ImpactClass.SILENTLY_CHANGED,
                        target=resolved_new,
                        change_classes=silent,
                        detail="resolves in both versions; the meaning moved",
                    )
                )
            else:
                findings.append(
                    Finding(
                        reference=reference,
                        impact_class=ImpactClass.UNAFFECTED,
                        target=resolved_new,
                        change_classes=_loud_classes(changes),
                    )
                )
            continue

        # The reference does not resolve in the new version.
        name_for_changes = resolved_old if resolved_old is not None else reference.target
        changes = diff.changes_for(name_for_changes)
        new_name = rename_map.get(name_for_changes, "")
        if new_name:
            silent = _silent_classes(changes)
            findings.append(
                Finding(
                    reference=reference,
                    impact_class=ImpactClass.RENAMED_AWAY,
                    target=name_for_changes,
                    change_classes=(ChangeClass.RENAMED, *silent),
                    suggestion=new_name,
                    detail=(
                        "the renamed item also carries a silent change"
                        if silent
                        else "structural match only; verify before applying"
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    reference=reference,
                    impact_class=ImpactClass.BROKEN,
                    target=name_for_changes,
                    change_classes=_loud_classes(changes),
                    detail=(
                        "not present in either version; the reference was already unresolved"
                        if resolved_old is None
                        else "present in the old version only"
                    ),
                )
            )

    findings.sort(
        key=lambda f: (f.rank, f.reference.source, f.reference.line, f.reference.target)
    )

    changed_names = set(diff.changed_item_names)
    verdict = _build_verdict(
        diff=diff,
        corpus=corpus,
        old=old,
        extracted=extracted,
        referenced_items=referenced_items,
        changed_names=changed_names,
        ambiguous_labels=ambiguous_labels,
    )
    return ImpactReport(
        old_version=diff.old_version,
        new_version=diff.new_version,
        findings=tuple(findings),
        verdict=verdict,
    )


def _assess_enum_label(
    reference: Reference,
    diff: DiffResult,
    old_labels: dict[str, tuple[str, int]],
    new_labels: dict[str, tuple[str, int]],
) -> Finding:
    """Classify a reference to an enumeration label."""
    label = reference.target
    in_old = old_labels.get(label)
    in_new = new_labels.get(label)
    if in_old is None and in_new is None:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.UNAFFECTED,
            detail="not an enumeration label in either version",
        )
    if in_old is not None and in_new is None:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.SILENTLY_CHANGED,
            target=in_old[0],
            change_classes=(ChangeClass.REENUMERATED,),
            detail=(
                f"label removed from {in_old[0]}; an equality test against it never fires "
                "and raises nothing"
            ),
        )
    if in_old is None and in_new is not None:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.UNAFFECTED,
            target=in_new[0],
            detail="label is new in this version",
        )
    assert in_old is not None and in_new is not None
    if in_old != in_new:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.SILENTLY_CHANGED,
            target=in_old[0],
            change_classes=(ChangeClass.REENUMERATED,),
            detail=f"label moved from {in_old[0]} code {in_old[1]} to {in_new[0]} code {in_new[1]}",
        )
    changes = diff.changes_for(in_old[0])
    silent = _silent_classes(changes)
    if silent:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.SILENTLY_CHANGED,
            target=in_old[0],
            change_classes=silent,
            detail="the label's own code is unchanged; its parameter carries a silent change",
        )
    return Finding(reference=reference, impact_class=ImpactClass.UNAFFECTED, target=in_old[0])


def _assess_argument(
    reference: Reference,
    diff: DiffResult,
    old: Dictionary,
    new: Dictionary,
    command_name: str,
    argument_name: str,
) -> Finding:
    """Classify a reference to one argument of one command."""
    resolved_new = new.resolve(command_name)
    resolved_old = old.resolve(command_name)
    if resolved_new is None:
        new_name = diff.rename_map.get(resolved_old or command_name, "")
        if new_name:
            return Finding(
                reference=reference,
                impact_class=ImpactClass.RENAMED_AWAY,
                target=resolved_old or command_name,
                change_classes=(ChangeClass.COMMAND_RENAMED,),
                suggestion=f"{new_name}.{argument_name}",
            )
        return Finding(
            reference=reference,
            impact_class=ImpactClass.BROKEN,
            target=resolved_old or command_name,
            change_classes=(ChangeClass.COMMAND_REMOVED,),
            detail="the command no longer resolves",
        )
    command = new.command_map.get(resolved_new)
    if command is None or argument_name not in command.argument_map:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.BROKEN,
            target=f"{resolved_new}.{argument_name}",
            change_classes=(ChangeClass.ARGUMENT_REMOVED,),
            detail="the command resolves but the argument does not",
        )
    silent = tuple(
        sorted(
            {
                change.change_class
                for change in diff.changes_for(resolved_old or resolved_new)
                if change.is_silent and change.field.startswith(f"argument:{argument_name}")
            },
            key=lambda c: c.value,
        )
    )
    if silent:
        return Finding(
            reference=reference,
            impact_class=ImpactClass.SILENTLY_CHANGED,
            target=f"{resolved_new}.{argument_name}",
            change_classes=silent,
            detail="the call still parses and encodes different bits",
        )
    return Finding(
        reference=reference,
        impact_class=ImpactClass.UNAFFECTED,
        target=f"{resolved_new}.{argument_name}",
    )


def _build_verdict(
    *,
    diff: DiffResult,
    corpus: Corpus,
    old: Dictionary,
    extracted: tuple[Reference, ...],
    referenced_items: set[str],
    changed_names: set[str],
    ambiguous_labels: tuple[str, ...],
) -> CoverageVerdict:
    undecidable = sum(1 for reference in extracted if not reference.decidable)
    items = len(old)
    referenced = len(referenced_items)
    changed_referenced = len(changed_names & referenced_items)
    reasons: list[str] = []
    if corpus.unreadable:
        reasons.append(
            f"{len(corpus.unreadable)} corpus source(s) could not be read: "
            + ", ".join(corpus.unreadable)
        )
    if undecidable:
        reasons.append(
            f"{undecidable} of {len(extracted)} extracted references are constructed at run "
            "time and cannot be resolved statically, so their impact is unknown"
        )
    if items - referenced > 0:
        reasons.append(
            f"{items - referenced} of {items} dictionary items are never referenced by this "
            "corpus, so this assessment says nothing about consumers outside it"
        )
    if changed_names - referenced_items:
        reasons.append(
            f"{len(changed_names - referenced_items)} changed item(s) are not referenced by "
            "this corpus: "
            + ", ".join(sorted(changed_names - referenced_items)[:8])
            + ("..." if len(changed_names - referenced_items) > 8 else "")
        )
    if diff.limits.ambiguous_renames:
        reasons.append(
            f"{diff.limits.ambiguous_renames} rename candidate(s) scored within the ambiguous "
            f"band of the threshold {diff.limits.rename_threshold}"
        )
    if diff.limits.incomparable_fields:
        reasons.append(
            f"{diff.limits.incomparable_fields} field(s) are not modelled by this tool or are "
            "present in one version only; whether they carry semantics is undetermined"
        )
    if ambiguous_labels:
        reasons.append(
            f"{len(ambiguous_labels)} enumeration label(s) are used by more than one parameter, "
            "so a bare label reference cannot be attributed to one parameter: "
            + ", ".join(ambiguous_labels[:8])
        )
    if not corpus.sources:
        reasons.append("the corpus is empty, so no reference was assessed at all")

    return CoverageVerdict(
        corpus_sources=len(corpus.sources),
        unreadable_sources=corpus.unreadable,
        references_total=len(extracted),
        references_undecidable=undecidable,
        dictionary_items=items,
        items_referenced=referenced,
        items_unreferenced=items - referenced,
        changed_items_total=len(changed_names),
        changed_items_referenced=changed_referenced,
        changed_items_unreferenced=len(changed_names) - changed_referenced,
        ambiguous_renames=diff.limits.ambiguous_renames,
        incomparable_fields=diff.limits.incomparable_fields,
        ambiguous_enum_labels=ambiguous_labels,
        reasons=tuple(reasons),
    )
