"""Declared inputs: activities, resources and the planning problem.

Everything in this module is a *declaration*. Nothing here is estimated from a
spacecraft, and no numerical default is taken from any vehicle. The units are
fixed and stated on every field because the whole product is an accounting
exercise and an accounting exercise with ambiguous units is worthless.

Unit convention used throughout ``activityplan``
------------------------------------------------
======================  =========================================
quantity                unit
======================  =========================================
time                    s (seconds, from the start of the horizon)
power                   W
energy                  W*h  (1 W*h = 3600 J)
data volume             Mbit (1 Mbit = 1e6 bit)
data rate               Mbit/s
depth of discharge      dimensionless fraction in [0, 1]
priority                dimensionless, strictly positive weight
======================  =========================================

Energy is carried in watt-hours rather than joules because battery capacity and
depth-of-discharge limits are declared in watt-hours in every spacecraft power
budget the references below describe. The conversion used everywhere is
``energy_wh = power_w * time_s / 3600``.

References for the *structure* of the resource model
----------------------------------------------------
- Wertz, J. R. and Larson, W. J. (eds), *Space Mission Analysis and Design*,
  3rd edition, Microcosm Press / Kluwer, 1999, chapter 11 (Spacecraft
  Subsystems: Power). Source for the orbit-average power budget formulation:
  average load, eclipse fraction, battery sized by depth of discharge.
- Patel, M. R., *Spacecraft Power Systems*, CRC Press, 2004, chapters 6 and 8.
  Source for the depth-of-discharge constraint as the operative battery limit
  and for the shunt treatment of surplus array power.
- Chien, S., Rabideau, G., Knight, R., Sherwood, R., Engelhardt, B., Mutz, D.,
  Estlin, T., Smith, B., Fisher, F., Barrett, T., Stebbins, G. and Tran, D.,
  "ASPEN -- Automating Space Mission Operations using Automated Planning and
  Scheduling", SpaceOps 2000. Source for the oversubscribed-activity framing:
  more requested activities than resources allow, each with a priority, and a
  timeline that selects among them.

Assumptions and validity range, stated once and repeated on the public API
-------------------------------------------------------------------------
1. All profiles are declared, not predicted. The solar input profile is an
   analytic shape with a declared peak and eclipse fraction; it is not an
   ephemeris computation and nothing here propagates an orbit.
2. The battery is a single energy reservoir with no voltage model, no internal
   resistance, no temperature dependence and no round-trip efficiency loss
   unless one is declared through ``charge_efficiency``.
3. The recorder is a single pool of Mbit with no file structure, no packet
   overhead and no retransmission.
4. Activities are non-preemptable: once started, an activity runs to the end of
   its declared duration.
5. Activities may overlap unless an exclusion pair forbids it. There is no
   implicit single-execution-string constraint; if a spacecraft has one, declare
   the exclusions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "Activity",
    "Contact",
    "PowerSegment",
    "Problem",
    "ResourceModel",
    "SolarProfile",
    "WH_PER_J",
]

WH_PER_J = 1.0 / 3600.0
"""Watt-hours per joule. ``energy_wh = power_w * seconds / 3600``."""


def _require_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _require_positive(value: float, name: str) -> float:
    number = _require_finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {number!r}")
    return number


def _require_non_negative(value: float, name: str) -> float:
    number = _require_finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {number!r}")
    return number


@dataclass(frozen=True)
class PowerSegment:
    """One constant-power segment of an activity's declared power profile.

    Parameters
    ----------
    duration_s
        Segment length [s]. Strictly positive.
    power_w
        Constant electrical load during the segment [W]. Non-negative. A segment
        cannot generate power; array output is declared on
        :class:`SolarProfile`.
    """

    duration_s: float
    power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _require_positive(self.duration_s, "duration_s"))
        object.__setattr__(self, "power_w", _require_non_negative(self.power_w, "power_w"))

    @property
    def energy_wh(self) -> float:
        """Exact energy of the segment [W*h] = ``power_w * duration_s / 3600``."""
        return self.power_w * self.duration_s * WH_PER_J


@dataclass(frozen=True)
class Activity:
    """A requested activity with its declared resource demand and constraints.

    The duration is *derived* from the power profile rather than declared
    separately, so a profile and a duration cannot disagree.

    Parameters
    ----------
    name
        Unique identifier within a problem.
    power_profile
        One or more :class:`PowerSegment`, in execution order. The activity's
        duration [s] is the sum of the segment durations and its energy [W*h]
        is the exact sum of the segment energies.
    data_mbit
        Data written to the recorder over the activity [Mbit], non-negative,
        produced at a constant rate ``data_mbit / duration_s`` [Mbit/s] over the
        activity. Assumption: uniform production. A burst at the end would be
        more forgiving of the recorder cap, so uniform production is the
        conservative choice only when the recorder is draining; it is stated
        rather than claimed to be conservative.
    priority
        Strictly positive dimensionless weight. The objective maximised by both
        planners is the sum of the priorities of the activities placed on the
        timeline (Chien et al. 2000 oversubscribed framing).
    earliest_start_s, latest_finish_s
        Declared time window [s] from the start of the horizon. The activity, if
        placed, must start at or after ``earliest_start_s`` and finish at or
        before ``latest_finish_s``.
    """

    name: str
    power_profile: tuple[PowerSegment, ...]
    data_mbit: float = 0.0
    priority: float = 1.0
    earliest_start_s: float = 0.0
    latest_finish_s: float = math.inf

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Activity name must be a non-empty string")
        profile = tuple(self.power_profile)
        if not profile:
            raise ValueError(f"activity {self.name!r} has an empty power profile")
        for segment in profile:
            if not isinstance(segment, PowerSegment):
                raise TypeError(
                    f"activity {self.name!r} power_profile must contain PowerSegment, "
                    f"got {type(segment).__name__}"
                )
        object.__setattr__(self, "power_profile", profile)
        object.__setattr__(self, "data_mbit", _require_non_negative(self.data_mbit, "data_mbit"))
        object.__setattr__(self, "priority", _require_positive(self.priority, "priority"))
        object.__setattr__(
            self,
            "earliest_start_s",
            _require_non_negative(self.earliest_start_s, "earliest_start_s"),
        )
        latest = float(self.latest_finish_s)
        if math.isnan(latest):
            raise ValueError("latest_finish_s must not be NaN")
        object.__setattr__(self, "latest_finish_s", latest)
        if latest < self.earliest_start_s + self.duration_s:
            raise ValueError(
                f"activity {self.name!r} window is too narrow for its duration: "
                f"[{self.earliest_start_s}, {latest}] s cannot hold {self.duration_s} s"
            )

    @classmethod
    def uniform(
        cls,
        name: str,
        duration_s: float,
        power_w: float,
        *,
        data_mbit: float = 0.0,
        priority: float = 1.0,
        earliest_start_s: float = 0.0,
        latest_finish_s: float = math.inf,
    ) -> Activity:
        """Build an activity with a single constant-power segment.

        ``duration_s`` [s] and ``power_w`` [W] define one
        :class:`PowerSegment`; the remaining arguments are as on
        :class:`Activity`.
        """
        return cls(
            name=name,
            power_profile=(PowerSegment(duration_s, power_w),),
            data_mbit=data_mbit,
            priority=priority,
            earliest_start_s=earliest_start_s,
            latest_finish_s=latest_finish_s,
        )

    @property
    def duration_s(self) -> float:
        """Total duration [s], the sum of the power-profile segment durations."""
        return math.fsum(segment.duration_s for segment in self.power_profile)

    @property
    def energy_wh(self) -> float:
        """Exact energy demand [W*h], the sum of the segment energies."""
        return math.fsum(segment.energy_wh for segment in self.power_profile)

    @property
    def mean_power_w(self) -> float:
        """Energy-equivalent mean load [W] = ``energy_wh * 3600 / duration_s``."""
        return self.energy_wh / (self.duration_s * WH_PER_J)

    @property
    def peak_power_w(self) -> float:
        """Largest segment power [W]."""
        return max(segment.power_w for segment in self.power_profile)

    @property
    def data_rate_mbit_per_s(self) -> float:
        """Constant production rate [Mbit/s] = ``data_mbit / duration_s``."""
        return self.data_mbit / self.duration_s

    def power_at(self, offset_s: float) -> float:
        """Declared load [W] at ``offset_s`` [s] after the activity starts.

        Returns 0.0 outside ``[0, duration_s)``. Segment boundaries are
        half-open: the value at a boundary belongs to the later segment.
        """
        if offset_s < 0.0:
            return 0.0
        cursor = 0.0
        for segment in self.power_profile:
            cursor += segment.duration_s
            if offset_s < cursor:
                return segment.power_w
        return 0.0

    def energy_between_wh(self, t0_s: float, t1_s: float) -> float:
        """Exact energy drawn [W*h] over ``[t0_s, t1_s)`` after the start.

        Closed form: the profile is piecewise constant, so this is the sum of
        ``power_w * overlap_s / 3600`` over segments. Valid for any real
        ``t0_s <= t1_s``; the interval is clipped to ``[0, duration_s]``.
        """
        if t1_s < t0_s:
            raise ValueError(f"t1_s ({t1_s}) must not be less than t0_s ({t0_s})")
        total = 0.0
        cursor = 0.0
        for segment in self.power_profile:
            lo = cursor
            hi = cursor + segment.duration_s
            overlap = min(hi, t1_s) - max(lo, t0_s)
            if overlap > 0.0:
                total += segment.power_w * overlap * WH_PER_J
            cursor = hi
        return total


@dataclass(frozen=True)
class SolarProfile:
    """Declared array input power as an analytic function of time.

    Model, stated in full because it is a declaration and not a prediction::

        sunlit_s = (1 - eclipse_fraction) * orbit_period_s
        tau      = (t + phase_s) mod orbit_period_s
        p_in(t)  = peak_power_w * sin(pi * tau / sunlit_s)   if tau < sunlit_s
                 = 0                                          otherwise

    The sine arc over the sunlit portion is a declared approximation to the
    cosine incidence variation a body-mounted array sees as it traverses the
    sunlit arc. The orbit-average power and eclipse-fraction *formulation* is
    the standard power-budget one (Wertz & Larson 1999, chapter 11; Patel 2004,
    chapter 6). **The sine shape itself is this package's declared input, not a
    value taken from either source, and no parameter below was identified from
    any spacecraft.**

    Validity range: ``0 <= eclipse_fraction < 1``, ``orbit_period_s > 0``. The
    model has no seasonal variation, no beta-angle dependence, no array
    degradation and no off-pointing loss. For a plan longer than a few orbits
    those omissions dominate, which is why the horizon in every shipped instance
    is a small number of orbits.

    Parameters
    ----------
    peak_power_w
        Array output [W] at the middle of the sunlit arc.
    orbit_period_s
        Repeat period [s] of the profile.
    eclipse_fraction
        Fraction of the period in eclipse, dimensionless, in ``[0, 1)``.
    phase_s
        Offset [s] added to ``t`` before the modulo, so the horizon can start
        anywhere in the orbit.
    """

    peak_power_w: float
    orbit_period_s: float
    eclipse_fraction: float = 0.0
    phase_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "peak_power_w", _require_non_negative(self.peak_power_w, "peak_power_w")
        )
        object.__setattr__(
            self, "orbit_period_s", _require_positive(self.orbit_period_s, "orbit_period_s")
        )
        fraction = _require_non_negative(self.eclipse_fraction, "eclipse_fraction")
        if fraction >= 1.0:
            raise ValueError(f"eclipse_fraction must be < 1, got {fraction!r}")
        object.__setattr__(self, "eclipse_fraction", fraction)
        object.__setattr__(self, "phase_s", _require_finite(self.phase_s, "phase_s"))

    @property
    def sunlit_s(self) -> float:
        """Length of the sunlit arc [s] = ``(1 - eclipse_fraction) * period``."""
        return (1.0 - self.eclipse_fraction) * self.orbit_period_s

    def power_at(self, t_s: float) -> float:
        """Array input power [W] at time ``t_s`` [s]. Exact, no interpolation."""
        sunlit = self.sunlit_s
        tau = math.fmod(float(t_s) + self.phase_s, self.orbit_period_s)
        if tau < 0.0:
            tau += self.orbit_period_s
        if tau >= sunlit:
            return 0.0
        return self.peak_power_w * math.sin(math.pi * tau / sunlit)

    def energy_wh(self, t0_s: float, t1_s: float) -> float:
        """Exact closed-form energy generated [W*h] over ``[t0_s, t1_s]``.

        Derivation. On a sunlit arc the antiderivative of
        ``peak * sin(pi * tau / sunlit)`` with respect to ``tau`` is
        ``-peak * (sunlit / pi) * cos(pi * tau / sunlit)``, so the energy over a
        sub-interval ``[a, b]`` of one sunlit arc is

            peak * (sunlit / pi) * (cos(pi a / sunlit) - cos(pi b / sunlit))

        in watt-seconds, divided by 3600 for watt-hours. A full sunlit arc gives
        ``2 * peak * sunlit / pi`` watt-seconds, the familiar
        ``2/pi`` form factor of a half sine. Whole orbits in the interior are
        accumulated in closed form, so cost is O(1) in the number of orbits.

        This function is the reference the discretised simulation in
        :mod:`activityplan.simulate` is validated against
        (``validation/validate_energy.py``).
        """
        t0 = _require_finite(t0_s, "t0_s")
        t1 = _require_finite(t1_s, "t1_s")
        if t1 < t0:
            raise ValueError(f"t1_s ({t1}) must not be less than t0_s ({t0})")
        if self.peak_power_w == 0.0 or t1 == t0:
            return 0.0
        period = self.orbit_period_s
        a = t0 + self.phase_s
        b = t1 + self.phase_s
        n_a = math.floor(a / period)
        n_b = math.floor(b / period)
        if n_a == n_b:
            return self._arc_energy_wh(a - n_a * period, b - n_a * period)
        total = self._arc_energy_wh(a - n_a * period, period)
        whole = n_b - n_a - 1
        if whole > 0:
            total += whole * self._arc_energy_wh(0.0, period)
        total += self._arc_energy_wh(0.0, b - n_b * period)
        return total

    def _arc_energy_wh(self, a: float, b: float) -> float:
        """Energy [W*h] over ``[a, b]`` within a single period, ``0 <= a <= b <= period``."""
        sunlit = self.sunlit_s
        lo = max(0.0, min(a, sunlit))
        hi = max(0.0, min(b, sunlit))
        if hi <= lo:
            return 0.0
        scale = self.peak_power_w * sunlit / math.pi
        watt_seconds = scale * (
            math.cos(math.pi * lo / sunlit) - math.cos(math.pi * hi / sunlit)
        )
        return watt_seconds * WH_PER_J


@dataclass(frozen=True)
class Contact:
    """A declared ground-station contact window.

    Parameters
    ----------
    start_s, end_s
        Window edges [s] from the start of the horizon, ``start_s < end_s``.
    rate_mbit_per_s
        Downlink rate available throughout the window [Mbit/s], non-negative.
        No link budget is computed here; the rate is declared. Elevation-
        dependent rate variation is not modelled, which overstates the capacity
        of a low-elevation pass.
    power_w
        Additional spacecraft load while the contact is active [W], covering the
        transmitter and its drive chain. Non-negative.
    name
        Identifier used in the feasibility report.
    """

    start_s: float
    end_s: float
    rate_mbit_per_s: float
    power_w: float = 0.0
    name: str = "contact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_s", _require_non_negative(self.start_s, "start_s"))
        object.__setattr__(self, "end_s", _require_finite(self.end_s, "end_s"))
        if self.end_s <= self.start_s:
            raise ValueError(
                f"contact {self.name!r} must have end_s > start_s, got "
                f"({self.start_s}, {self.end_s})"
            )
        object.__setattr__(
            self,
            "rate_mbit_per_s",
            _require_non_negative(self.rate_mbit_per_s, "rate_mbit_per_s"),
        )
        object.__setattr__(self, "power_w", _require_non_negative(self.power_w, "power_w"))

    @property
    def duration_s(self) -> float:
        """Window length [s]."""
        return self.end_s - self.start_s

    @property
    def capacity_mbit(self) -> float:
        """Maximum data downlinkable in the window [Mbit] = ``rate * duration``."""
        return self.rate_mbit_per_s * self.duration_s


@dataclass(frozen=True)
class ResourceModel:
    """The declared spacecraft resource envelope.

    Parameters
    ----------
    battery_capacity_wh
        Nameplate battery energy [W*h], strictly positive.
    max_depth_of_discharge
        Largest permitted depth of discharge, dimensionless in ``(0, 1]``. The
        usable energy is ``battery_capacity_wh * max_depth_of_discharge`` and the
        floor state of charge is ``battery_capacity_wh * (1 - dod)`` [W*h]
        (Patel 2004, chapter 8: the depth-of-discharge limit, not the nameplate
        capacity, is the operative constraint on a cycled battery).
    initial_state_of_charge
        State of charge at ``t = 0`` as a fraction of nameplate capacity,
        dimensionless in ``[1 - dod, 1]``.
    bus_load_w
        Constant housekeeping load [W] present over the whole horizon.
    solar
        Declared array input profile.
    recorder_capacity_mbit
        Solid-state recorder size [Mbit], strictly positive.
    initial_recorder_mbit
        Recorder content at ``t = 0`` [Mbit], in ``[0, recorder_capacity_mbit]``.
    contacts
        Declared contact windows. Overlapping windows are rejected because the
        downlink rate of overlapping passes is not additive in any way this
        package models.
    charge_efficiency
        Dimensionless factor in ``(0, 1]`` applied to *surplus* array power
        before it enters the battery, covering charge-path losses. Discharge is
        modelled at unit efficiency; the asymmetry is a declared simplification.
    """

    battery_capacity_wh: float
    max_depth_of_discharge: float
    solar: SolarProfile
    recorder_capacity_mbit: float
    initial_state_of_charge: float = 1.0
    bus_load_w: float = 0.0
    initial_recorder_mbit: float = 0.0
    contacts: tuple[Contact, ...] = ()
    charge_efficiency: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "battery_capacity_wh",
            _require_positive(self.battery_capacity_wh, "battery_capacity_wh"),
        )
        dod = _require_positive(self.max_depth_of_discharge, "max_depth_of_discharge")
        if dod > 1.0:
            raise ValueError(f"max_depth_of_discharge must be <= 1, got {dod!r}")
        object.__setattr__(self, "max_depth_of_discharge", dod)
        soc = _require_non_negative(self.initial_state_of_charge, "initial_state_of_charge")
        if soc > 1.0:
            raise ValueError(f"initial_state_of_charge must be <= 1, got {soc!r}")
        if soc < 1.0 - dod - 1e-12:
            raise ValueError(
                f"initial_state_of_charge {soc!r} already violates the declared "
                f"depth-of-discharge floor {1.0 - dod!r}"
            )
        object.__setattr__(self, "initial_state_of_charge", soc)
        object.__setattr__(self, "bus_load_w", _require_non_negative(self.bus_load_w, "bus_load_w"))
        if not isinstance(self.solar, SolarProfile):
            raise TypeError("solar must be a SolarProfile")
        object.__setattr__(
            self,
            "recorder_capacity_mbit",
            _require_positive(self.recorder_capacity_mbit, "recorder_capacity_mbit"),
        )
        stored = _require_non_negative(self.initial_recorder_mbit, "initial_recorder_mbit")
        if stored > self.recorder_capacity_mbit:
            raise ValueError(
                f"initial_recorder_mbit {stored!r} exceeds recorder_capacity_mbit "
                f"{self.recorder_capacity_mbit!r}"
            )
        object.__setattr__(self, "initial_recorder_mbit", stored)
        contacts = tuple(sorted(self.contacts, key=lambda c: c.start_s))
        for first, second in zip(contacts, contacts[1:], strict=False):
            if second.start_s < first.end_s:
                raise ValueError(
                    f"contacts {first.name!r} and {second.name!r} overlap: "
                    f"[{first.start_s}, {first.end_s}] and [{second.start_s}, {second.end_s}]"
                )
        object.__setattr__(self, "contacts", contacts)
        efficiency = _require_positive(self.charge_efficiency, "charge_efficiency")
        if efficiency > 1.0:
            raise ValueError(f"charge_efficiency must be <= 1, got {efficiency!r}")
        object.__setattr__(self, "charge_efficiency", efficiency)

    @property
    def floor_wh(self) -> float:
        """Lowest permitted battery energy [W*h] = ``capacity * (1 - dod)``."""
        return self.battery_capacity_wh * (1.0 - self.max_depth_of_discharge)

    @property
    def usable_wh(self) -> float:
        """Usable energy [W*h] = ``capacity * dod``. The normaliser for battery slack."""
        return self.battery_capacity_wh * self.max_depth_of_discharge

    @property
    def initial_energy_wh(self) -> float:
        """Battery energy at ``t = 0`` [W*h]."""
        return self.battery_capacity_wh * self.initial_state_of_charge

    @property
    def total_contact_capacity_mbit(self) -> float:
        """Sum of ``rate * duration`` over all declared contacts [Mbit]."""
        return math.fsum(contact.capacity_mbit for contact in self.contacts)

    def contact_at(self, t_s: float) -> Contact | None:
        """The contact active at ``t_s`` [s], or ``None``. Windows are half-open."""
        for contact in self.contacts:
            if contact.start_s <= t_s < contact.end_s:
                return contact
        return None

    def downlink_rate_at(self, t_s: float) -> float:
        """Downlink rate [Mbit/s] at ``t_s`` [s]; 0.0 outside every contact."""
        contact = self.contact_at(t_s)
        return 0.0 if contact is None else contact.rate_mbit_per_s

    def contact_power_at(self, t_s: float) -> float:
        """Transmitter load [W] at ``t_s`` [s]; 0.0 outside every contact."""
        contact = self.contact_at(t_s)
        return 0.0 if contact is None else contact.power_w


@dataclass(frozen=True)
class Problem:
    """A complete planning problem: activities, resources, horizon, structure.

    Parameters
    ----------
    activities
        Requested activities. Names must be unique.
    resources
        The declared resource envelope.
    horizon_s
        Planning horizon [s], strictly positive. Every placed activity must
        finish at or before the horizon.
    precedences
        Pairs ``(before, after)`` of activity names. Semantics, stated because
        several are defensible: if ``after`` is placed then ``before`` must also
        be placed and must finish at or before ``after`` starts. A precedence
        does not force either activity to be placed.
    exclusions
        Unordered pairs ``(a, b)`` of activity names that must not overlap in
        time. Touching is allowed: ``a`` finishing exactly when ``b`` starts is
        legal in continuous time, and on the discretised timeline the two
        occupy disjoint step sets.
    require_empty_recorder
        When ``True``, the recorder must hold no data at the end of the horizon:
        every bit produced must be downlinked inside the declared contacts. This
        is an operations-concept choice and not a physical law, which is why it
        is declared rather than assumed. The default is ``False``, under which
        data left onboard at the horizon is reported as a diagnostic
        (``downlink-volume``) and is not a violation.
    name
        Identifier used in reports.
    """

    activities: tuple[Activity, ...]
    resources: ResourceModel
    horizon_s: float
    precedences: tuple[tuple[str, str], ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()
    require_empty_recorder: bool = False
    name: str = "problem"
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        activities = tuple(self.activities)
        if not activities:
            raise ValueError("a problem needs at least one activity")
        for activity in activities:
            if not isinstance(activity, Activity):
                raise TypeError(f"activities must be Activity, got {type(activity).__name__}")
        names = [activity.name for activity in activities]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate activity names: {duplicates}")
        object.__setattr__(self, "activities", activities)
        if not isinstance(self.resources, ResourceModel):
            raise TypeError("resources must be a ResourceModel")
        object.__setattr__(self, "horizon_s", _require_positive(self.horizon_s, "horizon_s"))
        object.__setattr__(self, "require_empty_recorder", bool(self.require_empty_recorder))
        index = {name: position for position, name in enumerate(names)}
        object.__setattr__(self, "_index", index)

        precedences = tuple((str(a), str(b)) for a, b in self.precedences)
        for before, after in precedences:
            for label in (before, after):
                if label not in index:
                    raise ValueError(f"precedence names unknown activity {label!r}")
            if before == after:
                raise ValueError(f"precedence {before!r} -> {after!r} is self-referential")
        object.__setattr__(self, "precedences", precedences)

        exclusions = []
        seen: set[tuple[str, str]] = set()
        for a, b in self.exclusions:
            a, b = str(a), str(b)
            for label in (a, b):
                if label not in index:
                    raise ValueError(f"exclusion names unknown activity {label!r}")
            if a == b:
                raise ValueError(f"exclusion ({a!r}, {b!r}) is self-referential")
            key = (a, b) if a <= b else (b, a)
            if key not in seen:
                seen.add(key)
                exclusions.append(key)
        object.__setattr__(self, "exclusions", tuple(exclusions))

        for activity in activities:
            if activity.earliest_start_s + activity.duration_s > self.horizon_s + 1e-9:
                raise ValueError(
                    f"activity {activity.name!r} cannot fit before the horizon: "
                    f"earliest start {activity.earliest_start_s} s + duration "
                    f"{activity.duration_s} s > horizon {self.horizon_s} s"
                )
        self._check_precedence_acyclic()

    def _check_precedence_acyclic(self) -> None:
        successors: dict[str, list[str]] = {name: [] for name in self._index}
        for before, after in self.precedences:
            successors[before].append(after)
        state: dict[str, int] = dict.fromkeys(self._index, 0)

        def visit(node: str, trail: tuple[str, ...]) -> None:
            if state[node] == 1:
                cycle = " -> ".join(trail[trail.index(node):] + (node,))
                raise ValueError(f"precedence graph has a cycle: {cycle}")
            if state[node] == 2:
                return
            state[node] = 1
            for child in successors[node]:
                visit(child, trail + (node,))
            state[node] = 2

        for node in list(self._index):
            visit(node, ())

    @property
    def n_activities(self) -> int:
        """Number of requested activities."""
        return len(self.activities)

    @property
    def total_priority(self) -> float:
        """Sum of all priorities, the objective value of placing everything."""
        return math.fsum(activity.priority for activity in self.activities)

    @property
    def total_data_mbit(self) -> float:
        """Data produced if every activity is placed [Mbit]."""
        return math.fsum(activity.data_mbit for activity in self.activities)

    @property
    def total_activity_energy_wh(self) -> float:
        """Energy demanded if every activity is placed [W*h], activity load only."""
        return math.fsum(activity.energy_wh for activity in self.activities)

    def index(self, name: str) -> int:
        """Position of the named activity. Raises ``KeyError`` if unknown."""
        if name not in self._index:
            raise KeyError(f"unknown activity {name!r}")
        return self._index[name]

    def activity(self, name: str) -> Activity:
        """The named activity. Raises ``KeyError`` if unknown."""
        return self.activities[self.index(name)]

    def fixed_load_w(self, t_s: float) -> float:
        """Load present regardless of the plan [W]: bus plus any active transmitter."""
        return self.resources.bus_load_w + self.resources.contact_power_at(t_s)

    def baseline_energy_balance_wh(self) -> tuple[float, float]:
        """``(generated_wh, fixed_consumption_wh)`` over the horizon, closed form.

        ``generated_wh`` is the exact solar integral from
        :meth:`SolarProfile.energy_wh`. ``fixed_consumption_wh`` is the bus load
        over the whole horizon plus the transmitter load over every contact
        window, both exact because both profiles are piecewise constant.
        """
        generated = self.resources.solar.energy_wh(0.0, self.horizon_s)
        fixed = self.resources.bus_load_w * self.horizon_s * WH_PER_J
        for contact in self.resources.contacts:
            overlap = min(contact.end_s, self.horizon_s) - min(contact.start_s, self.horizon_s)
            if overlap > 0.0:
                fixed += contact.power_w * overlap * WH_PER_J
        return generated, fixed
