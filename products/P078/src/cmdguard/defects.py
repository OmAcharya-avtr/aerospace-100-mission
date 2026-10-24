"""Seeded injection of ordering defects into a clean command sequence.

The static-lint miss rate is only meaningful against a population of defects
that is described rather than asserted, so the population is defined here, in
code, with a name per family and a seed per instance. Nine families are
implemented. Eight of them are reachable only by a check that walks the
sequence; the ninth, :func:`undeclared_typo`, is deliberately one a per-command
lint finds every time, so that the measured miss rate is a miss rate over a
mixed population and not over a population selected to embarrass the baseline.

Every injection is a *textual* edit of the candidate sequence or of the
dictionary -- a step moved, a step deleted, a step inserted, a time changed, a
mnemonic mistyped. None of them edits the violation report, and none of them is
told what the forward executor will conclude. An injection that fails to create
a new violation is recorded as a no-op and counted, rather than retried until it
works: the no-op count is part of the measurement and is printed in
``validation/validate_lint_missrate_output.txt``.

Families
--------
``swap_enabler``
    A step that satisfies a later step's precondition is moved to after it, the
    time column untouched. This is the defect in the specification's own
    example: command 12 and command 47 in the wrong order.
``delete_enabler``
    The step that satisfies a later step's precondition is removed, which is
    what happens when a block is cut from a sequence during a replan.
``insert_invalidator``
    A command that sets the needed predicate false, or leaves it unverified, is
    inserted between the enabler and the consumer.
``mode_drop``
    A command that transitions out of the mode a later gated command needs is
    inserted before it.
``compress_gap``
    A step's time is moved earlier so that a declared ``min_gap_after_s`` or
    ``min_interval_s`` is no longer met.
``overlap_exclusion``
    A step is moved inside the hold window of another member of its
    mutual-exclusion group.
``resource_spike``
    A step is moved inside the hold windows of other drawing commands so that
    the accumulated draw crosses a declared ceiling.
``irreversible_early``
    An irreversible command is moved to before the step that verifies its
    precondition. This is the family the gate exists for.
``undeclared_typo``
    A mnemonic in the sequence is mistyped. A per-command lint finds this one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .model import Case, CommandSequence, SequenceStep, Verity

__all__ = ["DefectRecord", "FAMILIES", "inject", "inject_random"]

#: Family names, in the order they are reported. ``undeclared_typo`` is last
#: because it is the control: a per-command lint finds it every time.
FAMILIES: tuple[str, ...] = (
    "swap_enabler",
    "delete_enabler",
    "insert_invalidator",
    "mode_drop",
    "compress_gap",
    "overlap_exclusion",
    "resource_spike",
    "irreversible_early",
    "undeclared_typo",
)


@dataclass(frozen=True)
class DefectRecord:
    """What was injected, where, and whether the edit changed anything."""

    family: str
    description: str
    target_steps: tuple[int, ...]
    applied: bool

    def render(self) -> str:
        """One-line rendering for the validation transcripts."""
        status = "applied" if self.applied else "no-op"
        steps = ",".join(str(s) for s in self.target_steps) or "-"
        return f"{self.family:<20s} {status:<8s} steps[{steps}] {self.description}"


def _resequence(case: Case, steps: list[SequenceStep]) -> Case:
    """Rebuild ``case`` with ``steps``, re-sorting by time to keep monotonicity.

    Sorting is stable on ``t_s``, so two steps at the same time keep the order
    the caller put them in. That is what makes :func:`swap_enabler` work: the
    swap exchanges positions without touching the time column, and a stable
    sort preserves the exchange.
    """
    ordered = sorted(steps, key=lambda s: s.t_s)
    return replace(case, sequence=CommandSequence(tuple(ordered)))


def _enabler_pairs(case: Case) -> list[tuple[int, int, str]]:
    """``(enabler_index, consumer_index, predicate)`` triples in the sequence.

    An enabler is the last step before the consumer whose declared effect sets
    a predicate to the value the consumer requires. Only the last one matters,
    which is why the scan runs backwards from the consumer.
    """
    pairs: list[tuple[int, int, str]] = []
    steps = case.sequence.steps
    for consumer_index, consumer in enumerate(steps):
        spec = case.dictionary.get(consumer.command)
        if spec is None:
            continue
        for name in sorted(spec.preconditions):
            required = spec.preconditions[name]
            for enabler_index in range(consumer_index - 1, -1, -1):
                enabler_spec = case.dictionary.get(steps[enabler_index].command)
                if enabler_spec is None or name not in enabler_spec.effects:
                    continue
                if enabler_spec.effects[name] is required:
                    pairs.append((enabler_index, consumer_index, name))
                break
    return pairs


def swap_enabler(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Move an enabling step to after the step that depends on it."""
    pairs = _enabler_pairs(case)
    if not pairs:
        return case, DefectRecord("swap_enabler", "no enabler/consumer pair in sequence", (), False)
    enabler_index, consumer_index, name = rng.choice(pairs)
    steps = list(case.sequence.steps)
    enabler, consumer = steps[enabler_index], steps[consumer_index]
    steps[enabler_index] = replace(consumer, t_s=enabler.t_s)
    steps[consumer_index] = replace(enabler, t_s=consumer.t_s)
    return _resequence(case, steps), DefectRecord(
        "swap_enabler",
        f"{enabler.command} (sets {name}) swapped with {consumer.command} (needs {name})",
        (enabler_index, consumer_index),
        True,
    )


def delete_enabler(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Remove the step that satisfies a later step's precondition."""
    pairs = _enabler_pairs(case)
    if not pairs:
        return case, DefectRecord("delete_enabler", "no enabler/consumer pair", (), False)
    enabler_index, consumer_index, name = rng.choice(pairs)
    steps = list(case.sequence.steps)
    removed = steps.pop(enabler_index)
    return _resequence(case, steps), DefectRecord(
        "delete_enabler",
        f"{removed.command} (sets {name}) deleted ahead of "
        f"{case.sequence.steps[consumer_index].command}",
        (enabler_index,),
        True,
    )


def insert_invalidator(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Insert a command that falsifies or unverifies a needed predicate."""
    pairs = _enabler_pairs(case)
    rng.shuffle(pairs)
    for enabler_index, consumer_index, name in pairs:
        if consumer_index - enabler_index < 1:
            continue
        candidates = sorted(
            spec.identifier
            for spec in case.dictionary.commands.values()
            if name in spec.effects and spec.effects[name] is not case.dictionary.commands[
                case.sequence.steps[consumer_index].command
            ].preconditions[name]
        )
        if not candidates:
            continue
        chosen = candidates[rng.randrange(len(candidates))]
        steps = list(case.sequence.steps)
        t_before = steps[consumer_index - 1].t_s
        t_at = steps[consumer_index].t_s
        t_insert = t_before + 0.5 * (t_at - t_before) if t_at > t_before else t_at
        steps.insert(consumer_index, SequenceStep(t_s=t_insert, command=chosen))
        return _resequence(case, steps), DefectRecord(
            "insert_invalidator",
            f"{chosen} inserted before {case.sequence.steps[consumer_index].command}, "
            f"changing {name}",
            (consumer_index,),
            True,
        )
    return case, DefectRecord("insert_invalidator", "no invalidating command declared", (), False)


def mode_drop(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Insert a mode transition that makes a later gated command illegal."""
    steps = list(case.sequence.steps)
    gated = [
        (index, step)
        for index, step in enumerate(steps)
        if index > 0
        and case.dictionary.get(step.command) is not None
        and case.dictionary.commands[step.command].required_modes
    ]
    rng.shuffle(gated)
    for index, step in gated:
        needed = set(case.dictionary.commands[step.command].required_modes)
        movers = sorted(
            spec.identifier
            for spec in case.dictionary.commands.values()
            if spec.mode_effect is not None and spec.mode_effect not in needed
        )
        if not movers:
            continue
        chosen = movers[rng.randrange(len(movers))]
        t_before = steps[index - 1].t_s
        t_at = step.t_s
        t_insert = t_before + 0.5 * (t_at - t_before) if t_at > t_before else t_at
        new_steps = list(steps)
        new_steps.insert(index, SequenceStep(t_s=t_insert, command=chosen))
        return _resequence(case, new_steps), DefectRecord(
            "mode_drop",
            f"{chosen} inserted before {step.command}, which needs "
            f"[{', '.join(sorted(needed))}]",
            (index,),
            True,
        )
    return case, DefectRecord("mode_drop", "no mode-gated step to strand", (), False)


def compress_gap(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Move a step earlier so the previous command's declared gap is not met.

    Only ``min_gap_after_s`` is targeted, and only where the previous command
    declares a positive one. A first version of this function also took the
    moved command's own ``min_interval_s`` into account, which made 18 of 40
    injections no-ops because that interval is measured from the command's own
    previous execution and not from the previous step; the arithmetic error is
    recorded here because the no-op count is what exposed it.
    """
    steps = list(case.sequence.steps)
    candidates = []
    for index in range(1, len(steps)):
        previous = case.dictionary.get(steps[index - 1].command)
        if previous is None or previous.min_gap_after_s <= 0.0:
            continue
        if steps[index].t_s - steps[index - 1].t_s > 0.0:
            candidates.append((index, float(previous.min_gap_after_s)))
    if not candidates:
        return case, DefectRecord("compress_gap", "no declared minimum gap to break", (), False)
    index, required = rng.choice(candidates)
    base = steps[index - 1].t_s
    new_t = round(base + 0.25 * required, 6)
    steps[index] = replace(steps[index], t_s=new_t)
    return _resequence(case, steps), DefectRecord(
        "compress_gap",
        f"{steps[index].command} moved to t={new_t:.3f} s, {0.25 * required:.3f} s after the "
        f"previous command against a required {required:.3f} s",
        (index,),
        True,
    )


def overlap_exclusion(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Move a step inside the hold window of another exclusion-group member."""
    steps = list(case.sequence.steps)
    pairs: list[tuple[int, int, str]] = []
    for later in range(1, len(steps)):
        spec_later = case.dictionary.get(steps[later].command)
        if spec_later is None or spec_later.exclusion_group is None:
            continue
        for earlier in range(later - 1, -1, -1):
            spec_earlier = case.dictionary.get(steps[earlier].command)
            if spec_earlier is None or spec_earlier.exclusion_group != spec_later.exclusion_group:
                continue
            if spec_earlier.hold_s > 0.0:
                pairs.append((earlier, later, spec_earlier.exclusion_group))
            break
    if not pairs:
        return case, DefectRecord("overlap_exclusion", "no exclusion pair with a hold", (), False)
    earlier, later, group = rng.choice(pairs)
    spec_earlier = case.dictionary.commands[steps[earlier].command]
    new_t = steps[earlier].t_s + 0.5 * spec_earlier.hold_s
    steps[later] = replace(steps[later], t_s=new_t)
    return _resequence(case, steps), DefectRecord(
        "overlap_exclusion",
        f"{steps[later].command} moved to t={new_t:.3f} s, inside the {group!r} hold taken by "
        f"{steps[earlier].command} at t={steps[earlier].t_s:.3f} s",
        (earlier, later),
        True,
    )


def resource_spike(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Move a drawing step inside other holds so a ceiling is crossed."""
    steps = list(case.sequence.steps)
    limits = case.dictionary.resource_limits
    options: list[tuple[int, int, str, float]] = []
    for later in range(1, len(steps)):
        spec_later = case.dictionary.get(steps[later].command)
        if spec_later is None or not spec_later.resource_draw:
            continue
        for key in spec_later.draw_keys:
            limit = limits.limit_for(key)
            if limit is None:
                continue
            for earlier in range(later):
                spec_earlier = case.dictionary.get(steps[earlier].command)
                if spec_earlier is None or spec_earlier.hold_s <= 0.0:
                    continue
                earlier_draw = float(spec_earlier.resource_draw.get(key, 0.0))
                if earlier_draw <= 0.0:
                    continue
                if earlier_draw + float(spec_later.resource_draw[key]) > limit:
                    options.append((earlier, later, key, spec_earlier.hold_s))
    if not options:
        return case, DefectRecord("resource_spike", "no pair of draws exceeds a ceiling", (), False)
    earlier, later, key, hold = rng.choice(options)
    new_t = steps[earlier].t_s + 0.5 * hold
    steps[later] = replace(steps[later], t_s=new_t)
    return _resequence(case, steps), DefectRecord(
        "resource_spike",
        f"{steps[later].command} moved to t={new_t:.3f} s, inside the {key} hold taken by "
        f"{steps[earlier].command}",
        (earlier, later),
        True,
    )


def irreversible_early(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Move an irreversible command ahead of the step that verifies it.

    The tail of the sequence is shifted later by the moved command's own
    ``min_gap_after_s`` so that the edit is a reordering and not also a timing
    defect. Without the shift, 36 of 36 applications of this family also
    produced a ``timing_gap`` violation, which buried the finding the family
    exists to produce.
    """
    steps = list(case.sequence.steps)
    options: list[tuple[int, int, str]] = []
    for index, step in enumerate(steps):
        spec = case.dictionary.get(step.command)
        if spec is None or not spec.irreversible or not spec.preconditions:
            continue
        for name in sorted(spec.preconditions):
            for enabler in range(index):
                enabler_spec = case.dictionary.get(steps[enabler].command)
                if enabler_spec is not None and name in enabler_spec.effects:
                    options.append((enabler, index, name))
                    break
    if not options:
        return case, DefectRecord(
            "irreversible_early", "no irreversible command with an in-sequence enabler", (), False
        )
    enabler, index, name = rng.choice(options)
    moved = steps.pop(index)
    moved_spec = case.dictionary.commands[moved.command]
    shift = float(moved_spec.min_gap_after_s)
    tail = [replace(s, t_s=round(s.t_s + shift, 6)) for s in steps[enabler:]]
    steps = steps[:enabler] + [replace(moved, t_s=steps[enabler].t_s)] + tail
    return _resequence(case, steps), DefectRecord(
        "irreversible_early",
        f"irreversible {moved.command} moved ahead of {case.sequence.steps[enabler].command}, "
        f"which verifies {name}; tail shifted {shift:.3f} s later",
        (enabler, index),
        True,
    )


def undeclared_typo(case: Case, rng: random.Random) -> tuple[Case, DefectRecord]:
    """Mistype a mnemonic. The control family: a per-command lint finds this."""
    steps = list(case.sequence.steps)
    declared = [i for i, s in enumerate(steps) if s.command in case.dictionary]
    if not declared:
        return case, DefectRecord("undeclared_typo", "no declared command to mistype", (), False)
    index = rng.choice(declared)
    original = steps[index].command
    mistyped = original[:-1] + "0" if len(original) > 1 else original + "0"
    steps[index] = replace(steps[index], command=mistyped)
    return _resequence(case, steps), DefectRecord(
        "undeclared_typo",
        f"{original} mistyped as {mistyped}",
        (index,),
        True,
    )


_INJECTORS = {
    "swap_enabler": swap_enabler,
    "delete_enabler": delete_enabler,
    "insert_invalidator": insert_invalidator,
    "mode_drop": mode_drop,
    "compress_gap": compress_gap,
    "overlap_exclusion": overlap_exclusion,
    "resource_spike": resource_spike,
    "irreversible_early": irreversible_early,
    "undeclared_typo": undeclared_typo,
}


def inject(case: Case, family: str, seed: int) -> tuple[Case, DefectRecord]:
    """Inject one defect of ``family`` into ``case`` under ``seed``.

    Parameters
    ----------
    case:
        A clean case. Injecting into an already-defective case is permitted but
        makes the attribution of the resulting violations ambiguous.
    family:
        One of :data:`FAMILIES`.
    seed:
        Seed for the injection's own ``random.Random``, so one family and one
        seed always produce one edit.

    Returns
    -------
    tuple
        The edited case and the :class:`DefectRecord` describing the edit.

    Raises
    ------
    KeyError
        When ``family`` is not one of :data:`FAMILIES`.
    """
    if family not in _INJECTORS:
        raise KeyError(f"unknown defect family {family!r}; known: {', '.join(FAMILIES)}")
    rng = random.Random(seed)
    edited, record = _INJECTORS[family](case, rng)
    if record.applied:
        edited = replace(edited, name=f"{case.name}+{family}+{seed}")
    return edited, record


def inject_random(case: Case, seed: int) -> tuple[Case, DefectRecord]:
    """Inject one defect of a seed-chosen family."""
    rng = random.Random(seed)
    family = FAMILIES[rng.randrange(len(FAMILIES))]
    return inject(case, family, seed)


def predicate_effect_names(case: Case) -> tuple[str, ...]:
    """Every predicate any declared command writes, sorted. Used by the tests."""
    names: set[str] = set()
    for spec in case.dictionary.commands.values():
        names.update(spec.effects)
    return tuple(sorted(names))


def unverified_predicates(case: Case) -> tuple[str, ...]:
    """Predicates the initial state leaves unverified, sorted."""
    return tuple(
        sorted(name for name, value in case.initial.predicates.items() if value is Verity.UNKNOWN)
    )
