"""The declared limits of the measurement campaign, and what may not be claimed.

Four limits were recorded in ``docs/SPEC.md`` BEFORE the campaign ran, which is
the only time such a record is worth anything. They are reproduced here as data
so that the README, ``validation/VALIDATION.md``, ``MODEL_CARD.md`` and the
package's own report layer all state the same thing, and so that a test can
assert they do.

Beside them sits :data:`UNMEASURED_BY_NAME`: the claims this campaign cannot
support, each named individually rather than covered by a general disclaimer.
:func:`refuse_unmeasured` is the function that enforces it. A caller asking for
a figure at a different power mode, a different ambient or different hardware
gets an exception naming the claim, not a number with a caveat attached.

The distinction this module exists to keep sharp:

- A figure measured on the board at MAXN_SUPER in the campaign's one ambient is
  a MEASURED figure and may be labelled so.
- Everything else is UNMEASURED, by name, including every other power mode the
  board has, every ambient other than the one in the room, and every piece of
  hardware the campaign did not touch.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The board, as ``/proc/device-tree/model`` reports it.
BOARD = "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"

#: The one power mode reachable without root on the as-found board.
POWER_MODE = "MAXN_SUPER"

#: Software and kernel the campaign ran under, from SPEC.md.
PLATFORM = "aarch64, JetPack R36.4.7, kernel 5.15.148-tegra, 6 cores, 7619 MB, schedutil"


@dataclass(frozen=True)
class Limit:
    """One declared limit of the campaign.

    Attributes
    ----------
    key:
        Short identifier, stable across documents.
    headline:
        One line naming the limit.
    detail:
        Why it exists and what was observed, as recorded before the campaign ran.
    consequence:
        What the product may therefore not claim.
    """

    key: str
    headline: str
    detail: str
    consequence: str


#: The four limits, in the order SPEC.md records them.
DECLARED_LIMITS: tuple[Limit, ...] = (
    Limit(
        key="power_mode",
        headline="Power mode could not be varied.",
        detail=(
            "`sudo -n true` on the board returns `sudo: a password is required`, and both "
            "`nvpmodel` and `jetson_clocks` need root, so the board was characterised at "
            f"its as-found mode, {POWER_MODE}, only."
        ),
        consequence=(
            f"Every figure in this product is a {POWER_MODE} figure. No figure at any "
            "other power mode exists, and none may be inferred from these."
        ),
    ),
    Limit(
        key="ambient",
        headline="Ambient could not be varied.",
        detail=(
            "There is one ambient and no way to change it without physical access to the "
            "board. The pre-run idle temperature of every run is recorded so that ambient "
            "DRIFT across the campaign is at least visible."
        ),
        consequence=(
            "The held-out-AMBIENT evaluation axis is STRUCTURALLY UNMEASURABLE in this "
            "campaign. It is reported as unmeasurable with this reason. It is not "
            "silently dropped and it is not approximated by the duty-cycle split."
        ),
    ),
    Limit(
        key="shared_board",
        headline="The board is shared.",
        detail=(
            "`foreign_cpu_pct` -- the total CPU percentage of every process that is not "
            "the campaign -- is sampled every second alongside the temperatures, so "
            "contamination is visible in the data rather than assumed absent. At campaign "
            "start it sat near 20--25 per cent, mostly `avahi-daemon`."
        ),
        consequence=(
            "Throughput figures are figures for this payload on a board with that "
            "co-resident load, not for an otherwise idle board. The measured foreign CPU "
            "percentage is reported beside them."
        ),
    ),
    Limit(
        key="unreadable_zones",
        headline="Three thermal zones expose no reading.",
        detail=(
            "`thermal_zone2/3/4` (`cv0/cv1/cv2-thermal`) return no bytes at all: "
            "`pathlib.read_text` raises `TypeError: can't concat NoneType to bytes`, while "
            "a shell `cat` prints nothing and exits 0. Six zones are readable: "
            "`cpu-thermal`, `gpu-thermal`, `soc0/1/2-thermal` and `tj-thermal`."
        ),
        consequence=(
            "The die-temperature channel is `tj-thermal`. No figure in this product "
            "depends on the three unreadable zones, and their absence from a trace header "
            "is expected rather than a damaged file."
        ),
    ),
)


@dataclass(frozen=True)
class UnmeasuredClaim:
    """A claim this campaign cannot support, named rather than disclaimed."""

    key: str
    claim: str
    reason: str


#: Claims refused by name. Each is a thing a reader might reasonably want and
#: that this campaign did not measure.
UNMEASURED_BY_NAME: tuple[UnmeasuredClaim, ...] = (
    UnmeasuredClaim(
        key="power_mode_other",
        claim="any figure at a power mode other than MAXN_SUPER, by name 10W, 15W, 25W, "
        "MAXN or any custom nvpmodel profile",
        reason="nvpmodel needs root and the campaign account has no passwordless sudo; "
        "no run at any other mode exists",
    ),
    UnmeasuredClaim(
        key="ambient_other",
        claim="any figure at an ambient temperature other than the single ambient of the "
        "room the board sat in during the campaign",
        reason="ambient could not be varied without physical access; there is one ambient "
        "and no second point to interpolate or extrapolate from",
    ),
    UnmeasuredClaim(
        key="held_out_ambient",
        claim="a held-out-AMBIENT score for the learned predictor",
        reason="STRUCTURALLY UNMEASURABLE: scoring on held-out ambient requires at least "
        "two ambients and the campaign has one. The duty-cycle split is a different axis "
        "and is not a substitute for it",
    ),
    UnmeasuredClaim(
        key="other_hardware",
        claim="any figure for hardware the campaign did not touch, by name any other "
        "Jetson module, any Orin AGX, any x86 host and any board without this one's "
        "heatsink and fan",
        reason="one board was measured; thermal behaviour is a property of the whole "
        "assembly including its cooling, and nothing here transfers",
    ),
    UnmeasuredClaim(
        key="other_payload",
        claim="any figure for a payload other than the declared EKF-plus-GEMM kernel",
        reason="the thermal load, and therefore the steady state, is a property of the "
        "payload; a memory-bound or GPU-bound payload would dissipate differently and was "
        "not run",
    ),
    UnmeasuredClaim(
        key="no_fan",
        claim="any figure with the fan absent, stalled or under a different fan curve",
        reason="the board was characterised as found, with its active cooling running; no "
        "run with modified cooling exists",
    ),
    UnmeasuredClaim(
        key="long_duration",
        claim="any figure for a soak longer than the longest measured run",
        reason="the longest soak in the campaign plan is 720 s; a multi-hour drift, dust "
        "accumulation or enclosure heat-soak effect would not appear in it",
    ),
    UnmeasuredClaim(
        key="flight",
        claim="any flight, certification, qualification or operational-use claim",
        reason="this is research-grade software and a single-board bench measurement; it "
        "is not flight-qualified, not certified and not approved for operational "
        "aerospace use",
    ),
)


class UnmeasuredClaimError(ValueError):
    """Raised when a caller asks for a figure the campaign did not measure."""


def limit(key: str) -> Limit:
    """The declared limit with this key.

    Raises
    ------
    KeyError
        If no limit has that key.
    """
    for item in DECLARED_LIMITS:
        if item.key == key:
            return item
    raise KeyError(f"no declared limit named {key!r}; known: {[x.key for x in DECLARED_LIMITS]}")


def unmeasured(key: str) -> UnmeasuredClaim:
    """The named unmeasured claim with this key.

    Raises
    ------
    KeyError
        If no claim has that key.
    """
    for item in UNMEASURED_BY_NAME:
        if item.key == key:
            return item
    raise KeyError(
        f"no unmeasured claim named {key!r}; known: {[x.key for x in UNMEASURED_BY_NAME]}"
    )


def refuse_unmeasured(key: str) -> None:
    """Raise :class:`UnmeasuredClaimError` naming the claim and its reason.

    This is the enforcement point. It exists as a function so that a code path
    which would otherwise return a plausible number returns an exception
    instead, and so that a test can assert every named claim is refused.
    """
    item = unmeasured(key)
    raise UnmeasuredClaimError(
        f"UNMEASURED ({item.key}): {item.claim}. Reason: {item.reason}. This campaign "
        f"measured one board at {POWER_MODE} in one ambient and nothing else."
    )


def power_mode_figure(power_mode: str) -> str:
    """Return the label for a figure at ``power_mode``, or refuse.

    Raises
    ------
    UnmeasuredClaimError
        For any power mode other than :data:`POWER_MODE`.
    """
    if power_mode != POWER_MODE:
        refuse_unmeasured("power_mode_other")
    return f"MEASURED at {POWER_MODE} on {BOARD}"


def ambient_figure(ambient_c: float | None) -> str:
    """Return the label for a figure at a named ambient, or refuse.

    Parameters
    ----------
    ambient_c:
        ``None`` means the campaign's own single ambient, which is the only one
        that can be labelled. Any number is a request for an ambient the
        campaign did not establish, and is refused: the campaign did not
        instrument room temperature, so even its own ambient has no measured
        value in degrees, only an idle die temperature that stands in for it.

    Raises
    ------
    UnmeasuredClaimError
        For any explicit ambient.
    """
    if ambient_c is not None:
        refuse_unmeasured("ambient_other")
    return (
        "MEASURED in the campaign's single, uninstrumented ambient; the idle-run die "
        "temperature is the only proxy for it and ambient drift is visible only as drift "
        "in that proxy"
    )


def structurally_unmeasurable_axes() -> tuple[UnmeasuredClaim, ...]:
    """Evaluation axes that cannot be measured in this campaign at all.

    Returned as data so that the README, MODEL_CARD and VALIDATION documents
    cannot disagree with the package about which axes these are.
    """
    return (unmeasured("held_out_ambient"),)
