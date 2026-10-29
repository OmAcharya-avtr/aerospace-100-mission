"""Observer state, atmospheric state and the layered profile the ray trace integrates.

Units are SI throughout the module boundary with two deliberate exceptions that
are named in every signature: pressure is accepted in hectopascals because that
is the unit a station barometer reports, and wavelength is accepted in
micrometres because that is the unit every published dispersion formula is
written in.

The profile
-----------
Equation (1), the hydrostatic equation for a layer of constant lapse rate
``L = dT/dz``:

    T(z) = T_b + L (z - z_b)
    p(z) = p_b (T(z) / T_b) ** (-g0 / (R_d L))          for L != 0
    p(z) = p_b exp(-g0 (z - z_b) / (R_d T_b))           for L == 0

Source: the standard hydrostatic solution, as given in US Standard Atmosphere
1976, NOAA/NASA/USAF, equations (33) and (33a). Units: T in K, p in Pa, z in
metres of geometric height, L in K/m, ``g0 = 9.80665`` m/s^2 and
``R_d = 287.0528`` J/(kg K). Assumptions: hydrostatic equilibrium, constant
gravity, dry-air gas constant, geometric height used in place of geopotential
height. Validity: 0 to 86 km.

Equation (2), the layer structure. The lapse rates and layer base heights are
those of US Standard Atmosphere 1976:

    0 to 11 km      -6.5 K/km
    11 to 20 km       0.0 K/km
    20 to 32 km      +1.0 K/km
    32 to 47 km      +2.8 K/km
    47 to 51 km       0.0 K/km
    51 to 71 km      -2.8 K/km
    71 to 84.852 km  -2.0 K/km
    above 84.852 km   0.0 K/km   (this product's extension, declared below)

**The profile is anchored on the observer, not on mean sea level.** The
observer's measured pressure and temperature are taken as the base values at
the observer's own height and the 1976 lapse-rate structure is applied above
them, with the layer boundaries measured from the observer. This is a modelling
choice and it is the single largest uncontrolled input to a low-elevation
refraction calculation. Measured in ``validation/validate_profile_sensitivity.py``:
holding everything else fixed, replacing the 1976 surface lapse rate of
-6.5 K/km by the dry adiabatic -9.8 K/km or by a +30 K/km surface inversion
moves the refraction at zero apparent elevation by amounts that are reported
there in arcminutes. At 45 degrees elevation the same substitution moves
nothing at the arcsecond level. The library reports this as a structural
limit in :class:`refractpath.verdict.ValidityVerdict`, never as a caveat.

Equation (3), the water vapour profile:

    e(z) = min(e_0 exp(-z / H_w), 0.9 p(z))

with ``H_w = 2000`` m. This is a declared default, not a published profile: a
vapour scale height of about 2 km is the conventional first approximation for
the lower troposphere, and the 0.9 p cap exists only to keep the mixing ratio
physical in a profile that was never meant to be realistic above the boundary
layer. The humidity contribution to optical refractivity is small -- the size
is measured in ``validation/validate_dispersion.py`` -- so the crudeness of
equation (3) is tolerable in the optical band and would not be in the radio
band, which this product does not cover.

Equation (4), saturation vapour pressure over liquid water, Buck (1981),
equation (8) of that paper in its ``e_w3`` form:

    e_s(t) = 611.21 exp((18.678 - t / 234.5) t / (257.14 + t))      [Pa, t in C]

Source: Buck, A. L. (1981), "New equations for computing vapor pressure and
enhancement factor", Journal of Applied Meteorology 20, 1527-1532. Validity:
-20 to +50 C over water. Used only to turn a relative humidity into a partial
pressure; :mod:`refractpath.refractivity` uses the Ciddor/IAPWS form of
equation (12) internally where its own formulation requires it, and the two
are compared in ``validation/validate_dispersion.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Standard gravity, m/s^2. US Standard Atmosphere 1976.
G0 = 9.80665

#: Specific gas constant for dry air, J/(kg K). US Standard Atmosphere 1976.
R_DRY = 287.0528

#: WGS84 equatorial radius, m. Used as the single spherical Earth radius; the
#: ray trace is spherically symmetric and does not model the ellipsoid. The
#: difference between the equatorial and polar radius is 0.34 per cent, which
#: enters the refraction only through the curvature term; the size of that
#: sensitivity is measured in ``validation/validate_profile_sensitivity.py``.
EARTH_RADIUS_M = 6378137.0

#: Lapse rates of US Standard Atmosphere 1976, as (base height above the
#: observer in metres, dT/dz in K/m). The final entry is this product's
#: isothermal extension above the 1976 model's 84.852 km ceiling; the
#: refraction contributed above 84.852 km is measured in
#: ``validation/validate_raytrace_convergence.py``.
US1976_LAYERS: tuple[tuple[float, float], ...] = (
    (0.0, -0.0065),
    (11000.0, 0.0),
    (20000.0, 0.0010),
    (32000.0, 0.0028),
    (47000.0, 0.0),
    (51000.0, -0.0028),
    (71000.0, -0.0020),
    (84852.0, 0.0),
)

#: Default vapour scale height, m. A declared default, not a published profile.
VAPOUR_SCALE_HEIGHT_M = 2000.0

#: Declared wavelength domain of this product, micrometres. Ciddor (1996) is
#: published for the visible and near infrared; the Edlen-family formulae were
#: fitted over a narrower visible range and are applied over the same domain
#: here with that stated. Outside this interval the library still computes but
#: :class:`refractpath.verdict.ValidityVerdict` reports the wavelength as
#: outside the declared domain.
WAVELENGTH_DOMAIN_UM = (0.30, 1.69)

#: Declared pressure domain, hPa. The lower end is roughly the 300 hPa level,
#: below which an observer is not on the ground and the anchored-profile model
#: of equation (2) stops meaning anything.
PRESSURE_DOMAIN_HPA = (300.0, 1100.0)

#: Declared temperature domain, degrees Celsius. The upper end is Buck (1981)'s
#: stated validity for equation (4); the lower end is below it and is declared
#: as this product's own limit.
TEMPERATURE_DOMAIN_C = (-80.0, 50.0)


def saturation_vapour_pressure_pa(temperature_c: float) -> float:
    """Saturation vapour pressure over liquid water, Pa, from equation (4).

    Args:
        temperature_c: Air temperature, degrees Celsius.

    Returns:
        Saturation vapour pressure in pascals.

    Raises:
        ValueError: If ``temperature_c`` is not finite.
    """
    if not math.isfinite(temperature_c):
        raise ValueError(f"temperature_c must be finite, got {temperature_c!r}")
    t = float(temperature_c)
    return 611.21 * math.exp((18.678 - t / 234.5) * t / (257.14 + t))


@dataclass(frozen=True)
class AtmosphericState:
    """Atmospheric state at the observer, plus the observing wavelength.

    Attributes:
        pressure_hpa: Station pressure at the observer, hectopascals.
        temperature_c: Air temperature at the observer, degrees Celsius.
        relative_humidity: Relative humidity over liquid water, 0 to 1.
        wavelength_um: Vacuum wavelength of the observation, micrometres.
        co2_ppm: Carbon dioxide mole fraction, parts per million. Used only by
            :func:`refractpath.refractivity.ciddor_1996`, which is the only
            formulation here that takes it; the others carry a fixed value in
            their published coefficients (300 ppm for Edlen 1966, 450 ppm for
            Birch and Downs 1994).
        surface_lapse_rate_k_per_m: Lapse rate of the lowest layer, K/m.
            Defaults to the US Standard Atmosphere 1976 value of -0.0065. This
            is exposed because it is the dominant uncontrolled input near the
            horizon and because a product that hides it cannot report its own
            validity floor honestly.
        surface_layer_top_m: Height above the observer, metres, up to which
            ``surface_lapse_rate_k_per_m`` applies. ``None`` applies it to the
            whole lowest 1976 layer, 0 to 11 km. A finite value inserts one
            extra layer so that a shallow surface inversion can be represented
            at all; above it the 1976 tropospheric lapse rate of -6.5 K/km
            resumes. Must be strictly between 0 and 11000 when set.
        vapour_scale_height_m: Scale height of equation (3), metres.
    """

    pressure_hpa: float = 1013.25
    temperature_c: float = 10.0
    relative_humidity: float = 0.0
    wavelength_um: float = 0.55
    co2_ppm: float = 450.0
    surface_lapse_rate_k_per_m: float = -0.0065
    surface_layer_top_m: float | None = None
    vapour_scale_height_m: float = VAPOUR_SCALE_HEIGHT_M

    def __post_init__(self) -> None:
        for name in (
            "pressure_hpa",
            "temperature_c",
            "relative_humidity",
            "wavelength_um",
            "co2_ppm",
            "surface_lapse_rate_k_per_m",
            "vapour_scale_height_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.pressure_hpa <= 0.0:
            raise ValueError(
                f"pressure_hpa must be positive, got {self.pressure_hpa!r}; "
                "a refractivity is proportional to density and a non-positive "
                "pressure has no physical meaning"
            )
        if self.temperature_c <= -273.15:
            raise ValueError(
                f"temperature_c must be above absolute zero, got {self.temperature_c!r}"
            )
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError(
                f"relative_humidity must lie in [0, 1] as a fraction, got "
                f"{self.relative_humidity!r}; pass 0.43 rather than 43"
            )
        if self.wavelength_um <= 0.0:
            raise ValueError(f"wavelength_um must be positive, got {self.wavelength_um!r}")
        if self.co2_ppm < 0.0:
            raise ValueError(f"co2_ppm must be non-negative, got {self.co2_ppm!r}")
        if self.vapour_scale_height_m <= 0.0:
            raise ValueError(
                f"vapour_scale_height_m must be positive, got {self.vapour_scale_height_m!r}"
            )
        if self.surface_layer_top_m is not None:
            top = self.surface_layer_top_m
            if not isinstance(top, (int, float)) or isinstance(top, bool):
                raise TypeError(
                    f"surface_layer_top_m must be a real number or None, got "
                    f"{type(top).__name__}"
                )
            if not math.isfinite(float(top)):
                raise ValueError(f"surface_layer_top_m must be finite, got {top!r}")
            if not 0.0 < float(top) < US1976_LAYERS[1][0]:
                raise ValueError(
                    f"surface_layer_top_m must lie strictly between 0 and "
                    f"{US1976_LAYERS[1][0]!r} m, got {top!r}"
                )

    @property
    def temperature_k(self) -> float:
        """Air temperature at the observer, kelvin."""
        return self.temperature_c + 273.15

    @property
    def pressure_pa(self) -> float:
        """Station pressure at the observer, pascals."""
        return self.pressure_hpa * 100.0

    @property
    def vapour_pressure_pa(self) -> float:
        """Water vapour partial pressure at the observer, Pa, from equation (4)."""
        return self.relative_humidity * saturation_vapour_pressure_pa(self.temperature_c)

    def inside_declared_domains(self) -> dict[str, bool]:
        """Per-quantity report of whether this state is inside the declared domains.

        Returns:
            Mapping from quantity name to whether it lies inside the interval
            this product declares. Nothing is raised: a state outside a domain
            is computable and the verdict says so.
        """
        lo_w, hi_w = WAVELENGTH_DOMAIN_UM
        lo_p, hi_p = PRESSURE_DOMAIN_HPA
        lo_t, hi_t = TEMPERATURE_DOMAIN_C
        return {
            "wavelength_um": lo_w <= self.wavelength_um <= hi_w,
            "pressure_hpa": lo_p <= self.pressure_hpa <= hi_p,
            "temperature_c": lo_t <= self.temperature_c <= hi_t,
            "relative_humidity": 0.0 <= self.relative_humidity <= 1.0,
        }


@dataclass(frozen=True)
class Observer:
    """Observer location.

    Only ``height_m`` enters the spherically symmetric ray trace. Latitude is
    carried because it is what a user has and because it is used to report,
    rather than to apply, the Earth-radius sensitivity: the ray trace uses the
    single radius :data:`EARTH_RADIUS_M` for every latitude.

    Attributes:
        height_m: Geometric height of the observer above the reference sphere,
            metres.
        latitude_deg: Geodetic latitude, degrees. Recorded, not used by the
            ray trace.
    """

    height_m: float = 0.0
    latitude_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("height_m", "latitude_deg"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.height_m < -500.0:
            raise ValueError(
                f"height_m must be at least -500 m, got {self.height_m!r}; "
                "below that the anchored profile of equation (2) is not defined"
            )
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(f"latitude_deg must lie in [-90, 90], got {self.latitude_deg!r}")

    @property
    def radius_m(self) -> float:
        """Distance of the observer from the centre of the reference sphere, m."""
        return EARTH_RADIUS_M + self.height_m


@dataclass(frozen=True)
class LayeredProfile:
    """Temperature, pressure and vapour pressure on a height grid above the observer.

    Attributes:
        height_m: Geometric height above the observer, metres, ascending.
        temperature_k: Air temperature on that grid, kelvin.
        pressure_pa: Air pressure on that grid, pascals.
        vapour_pressure_pa: Water vapour partial pressure on that grid, pascals.
        layer_tops_m: Heights above the observer at which the lapse rate
            changes, metres. Recorded so that a caller can place grid points
            on the kinks.
    """

    height_m: np.ndarray
    temperature_k: np.ndarray
    pressure_pa: np.ndarray
    vapour_pressure_pa: np.ndarray
    layer_tops_m: tuple[float, ...] = field(default=())

    def __len__(self) -> int:
        return int(self.height_m.size)


def _layer_bases(
    state: AtmosphericState, top_m: float
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Base height, base temperature, base pressure, lapse rate and top of each layer.

    Integrates equation (1) upward from the observer, layer by layer, so that
    every layer's base values are consistent with the layer below it.
    """
    layers = list(US1976_LAYERS)
    layers[0] = (0.0, float(state.surface_lapse_rate_k_per_m))
    if state.surface_layer_top_m is not None:
        layers.insert(1, (float(state.surface_layer_top_m), US1976_LAYERS[0][1]))
    tops = [*[base for base, _ in layers][1:], float("inf")]
    out: list[tuple[float, float, float, float, float]] = []
    t_base, p_base, z_base = state.temperature_k, state.pressure_pa, 0.0
    for (_, lapse), layer_top in zip(layers, tops, strict=True):
        ceiling = min(layer_top, top_m) if math.isfinite(layer_top) else top_m
        out.append((z_base, t_base, p_base, lapse, layer_top))
        if ceiling <= z_base:
            continue
        dz = ceiling - z_base
        t_next = t_base + lapse * dz
        if t_next <= 0.0:
            raise ValueError(
                f"the profile reaches a non-positive temperature {t_next:.3f} K at "
                f"{ceiling / 1000.0:.3f} km above the observer with a surface lapse "
                f"rate of {state.surface_lapse_rate_k_per_m * 1000.0:+.3f} K/km; "
                "equation (1) has no solution there"
            )
        if lapse == 0.0:
            p_next = p_base * math.exp(-G0 * dz / (R_DRY * t_base))
        else:
            p_next = p_base * (t_next / t_base) ** (-G0 / (R_DRY * lapse))
        t_base, p_base, z_base = t_next, p_next, ceiling
    return tuple(out)


def build_profile(
    state: AtmosphericState, height_m: np.ndarray, top_m: float | None = None
) -> LayeredProfile:
    """Evaluate equations (1) to (4) on ``height_m``.

    Args:
        state: Atmospheric state at the observer.
        height_m: Geometric heights above the observer, metres, ascending and
            non-negative.
        top_m: Height to integrate the layer bases to, metres. Defaults to the
            largest value in ``height_m``.

    Returns:
        The profile on that grid.

    Raises:
        ValueError: If ``height_m`` is empty, negative, non-finite or not
            ascending, or if the profile reaches a non-positive temperature.
    """
    grid = np.asarray(height_m, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"height_m must be a non-empty 1-D array, got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("height_m must be finite")
    if grid[0] < 0.0:
        raise ValueError(f"height_m must be non-negative, got {grid.min()!r}")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("height_m must be ascending")

    ceiling = float(grid[-1]) if top_m is None else float(top_m)
    bases = _layer_bases(state, max(ceiling, float(grid[-1])))
    temperature = np.empty_like(grid)
    pressure = np.empty_like(grid)
    assigned = np.zeros(grid.shape, dtype=bool)
    for z_base, t_base, p_base, lapse, layer_top in bases:
        inside = (grid >= z_base) & (grid <= layer_top) & ~assigned
        if not inside.any():
            continue
        dz = grid[inside] - z_base
        t_here = t_base + lapse * dz
        temperature[inside] = t_here
        if lapse == 0.0:
            pressure[inside] = p_base * np.exp(-G0 * dz / (R_DRY * t_base))
        else:
            pressure[inside] = p_base * (t_here / t_base) ** (-G0 / (R_DRY * lapse))
        assigned |= inside
    if not assigned.all():
        raise ValueError("height_m contains points above the modelled atmosphere")

    vapour = state.vapour_pressure_pa * np.exp(-grid / state.vapour_scale_height_m)
    vapour = np.minimum(vapour, 0.9 * pressure)
    return LayeredProfile(
        height_m=grid,
        temperature_k=temperature,
        pressure_pa=pressure,
        vapour_pressure_pa=vapour,
        layer_tops_m=tuple(entry[0] for entry in bases[1:]),
    )
