"""Kolmogorov phase screens and frozen-flow atmospheric time series.

This module is a self-contained implementation. It does not import from, and
shares no code with, any other product in the portfolio.

Theory
------
**Kolmogorov phase power spectral density** (Roddier, F. 1981, "The effects of
atmospheric turbulence in optical astronomy", *Progress in Optics* **19**,
281-376, eq. 3.42; also Hardy 1998 eq. 3.30):

```
Phi_phi(f) = 0.023 * r0^(-5/3) * f^(-11/3)        [rad^2 m^2]
```

``f = |f_vec|`` is the **spatial frequency in cycles per metre** and ``r0`` the
Fried parameter [m] at the wavelength of interest. *Assumptions:* Kolmogorov
inertial range, infinite outer scale, near-field (no scintillation), thin
phase screen, isotropy. *Validity:* ``1/L0 << f << 1/l0``; the ``f -> 0``
divergence is why piston (and, on a finite grid, part of tip/tilt) must be
handled separately.

**Phase structure function** (Fried, D. L. 1966, "Optical resolution through a
randomly inhomogeneous medium for very long and very short exposures",
*JOSA* **56**, 1372-1379; Hardy 1998 eq. 3.35):

```
D_phi(r) = <|phi(x+r) - phi(x)|^2> = 6.88 * (r / r0)^(5/3)      [rad^2]
```

This is the reference used by :func:`structure_function` to validate a screen.

**Screen synthesis** follows the standard FFT method: filter unit complex
Gaussian noise by ``sqrt(Phi_phi)`` and inverse-transform (Schmidt, J. D. 2010,
*Numerical Simulation of Optical Wave Propagation with Examples in MATLAB*,
SPIE Press, sec. 9.4, eqs. 9.25-9.26):

```
phi(x, y) = Re sum_{n,m} (a_nm + i b_nm) sqrt(Phi_phi(f_nm)) df
            * exp(2 pi i (f_xn x + f_ym y))
```

with ``a, b ~ N(0, 1)`` i.i.d., ``df = 1 / (N dx)``. The lowest representable
frequency is ``df``, so an FFT screen is deficient in low-order power. Three
levels of **subharmonics** on a 3x3 stencil are added (Lane, R. G., Glindemann,
A. & Dainty, J. C. 1992, "Simulation of a Kolmogorov phase screen", *Waves in
Random Media* **2**, 209-224), which is the standard remedy; the residual
error is measured in ``validation/validate_atmosphere.py``.

**Fried parameter from Cn2** (Fried 1966; Hardy 1998 eq. 3.29), plane wave:

```
r0 = [ 0.423 * k^2 * integral Cn2(h) dh ]^(-3/5)        [m],  k = 2 pi / lambda
```

**Greenwood frequency** (Greenwood, D. P. 1977, "Bandwidth specification for
adaptive optics systems", *JOSA* **67**, 390-393), single layer of speed ``v``:

```
f_G = 0.426 * v / r0                                    [Hz]
```

**Atmospheric coherence time** (Hardy 1998 eq. 3.51; Fried 1990):

```
tau0 = 0.314 * r0 / v                                   [s]
```

so ``f_G * tau0 = 0.426 * 0.314 = 0.1338`` exactly, a relation checked in the
tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KOLMOGOROV_PSD_COEFF",
    "STRUCTURE_FUNCTION_COEFF",
    "PhaseScreen",
    "FrozenFlow",
    "fried_parameter",
    "greenwood_frequency",
    "coherence_time",
    "kolmogorov_psd",
    "structure_function",
    "theoretical_structure_function",
    "discrete_structure_function",
]

# Roddier (1981) eq. 3.42 / Hardy (1998) eq. 3.30, f in cycles/m.
KOLMOGOROV_PSD_COEFF: float = 0.023
# Fried (1966); Hardy (1998) eq. 3.35.
STRUCTURE_FUNCTION_COEFF: float = 6.88


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


def fried_parameter(cn2_integral: float, wavelength: float) -> float:
    """Fried parameter ``r0`` [m] for a plane wave.

    Parameters
    ----------
    cn2_integral:
        Path integral ``int Cn2(h) dh`` [m^(1/3)].
    wavelength:
        Optical wavelength [m].

    Notes
    -----
    ``r0 = (0.423 k^2 int Cn2 dh)^(-3/5)``, ``k = 2 pi / lambda``
    (Fried 1966; Hardy 1998 eq. 3.29). *Assumptions:* plane wave, weak
    fluctuations, Kolmogorov spectrum, zenith. For a spherical wave the
    coefficient 0.423 is replaced by ``0.423 * 3/8``; not implemented here.
    """
    cn2_integral = _check_positive("cn2_integral", cn2_integral)
    wavelength = _check_positive("wavelength", wavelength)
    k = 2.0 * np.pi / wavelength
    return float((0.423 * k**2 * cn2_integral) ** (-3.0 / 5.0))


def greenwood_frequency(r0: float, wind_speed: float) -> float:
    """Greenwood frequency ``f_G = 0.426 v / r0`` [Hz] (Greenwood 1977).

    Single frozen layer of speed ``v`` [m/s] and Fried parameter ``r0`` [m] at
    the same wavelength. The closed-loop residual variance for a first-order
    servo of 3 dB bandwidth ``f_3dB`` is ``(f_G / f_3dB)^(5/3)`` [rad^2].
    """
    return 0.426 * _check_positive("wind_speed", wind_speed) / _check_positive("r0", r0)


def coherence_time(r0: float, wind_speed: float) -> float:
    """Atmospheric coherence time ``tau0 = 0.314 r0 / v`` [s] (Hardy 1998 eq. 3.51).

    A pure delay ``tau`` leaves residual variance ``(tau / tau0)^(5/3)`` [rad^2]
    (see :func:`waveforge.budget.temporal_error`).
    """
    return 0.314 * _check_positive("r0", r0) / _check_positive("wind_speed", wind_speed)


def kolmogorov_psd(f: NDArray[np.float64], r0: float) -> NDArray[np.float64]:
    """Kolmogorov phase PSD [rad^2 m^2] at spatial frequency ``f`` [cycles/m].

    ``Phi(f) = 0.023 r0^(-5/3) f^(-11/3)``. ``f = 0`` returns 0 (piston is
    removed rather than being infinite).
    """
    r0 = _check_positive("r0", r0)
    freq = np.asarray(f, dtype=np.float64)
    if np.any(freq < 0):
        raise ValueError("spatial frequency must be >= 0")
    out = np.zeros_like(freq)
    nz = freq > 0
    out[nz] = KOLMOGOROV_PSD_COEFF * r0 ** (-5.0 / 3.0) * freq[nz] ** (-11.0 / 3.0)
    return out


def theoretical_structure_function(r: NDArray[np.float64], r0: float) -> NDArray[np.float64]:
    """``D_phi(r) = 6.88 (r/r0)^(5/3)`` [rad^2] (Fried 1966; Hardy 1998 eq. 3.35)."""
    r0 = _check_positive("r0", r0)
    sep = np.asarray(r, dtype=np.float64)
    if np.any(sep < 0):
        raise ValueError("separation r must be >= 0")
    return STRUCTURE_FUNCTION_COEFF * (sep / r0) ** (5.0 / 3.0)


def structure_function(
    screens: NDArray[np.float64], dx: float, max_lag: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Empirical 1-D phase structure function along the array's x axis.

    Parameters
    ----------
    screens:
        ``(n_screens, ny, nx)`` or ``(ny, nx)`` array of phase [rad].
    dx:
        Grid pitch [m].
    max_lag:
        Largest lag in samples. Default ``nx // 4`` -- lags beyond about a
        quarter of the box are contaminated by the screen's periodicity and by
        the missing low-frequency power.

    Returns
    -------
    (separations [m], D_phi [rad^2])
    """
    arr = np.asarray(screens, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"screens must be 2-D or 3-D, got shape {arr.shape}")
    dx = _check_positive("dx", dx)
    nx = arr.shape[2]
    if max_lag is None:
        max_lag = max(1, nx // 4)
    max_lag = int(max_lag)
    if not (1 <= max_lag < nx):
        raise ValueError(f"max_lag must be in [1, {nx - 1}], got {max_lag}")
    lags = np.arange(1, max_lag + 1)
    d = np.empty(lags.size, dtype=np.float64)
    for i, lag in enumerate(lags):
        diff = arr[:, :, lag:] - arr[:, :, :-lag]
        d[i] = float(np.mean(diff**2))
    return lags * dx, d


@dataclass(frozen=True)
class PhaseScreen:
    """A square Kolmogorov phase screen generator.

    Parameters
    ----------
    n:
        Screen size in samples (square). Must be >= 16 and even.
    dx:
        Sample pitch [m]. Must be > 0.
    r0:
        Fried parameter [m] at the working wavelength. Must be > 0.
    n_subharmonics:
        Number of subharmonic levels added to compensate the FFT's missing
        low-frequency power (Lane et al. 1992). 0 disables. Default 3.

    Notes
    -----
    The generated screen is **periodic** with period ``n * dx`` in both axes --
    an artefact of the FFT method. :class:`FrozenFlow` wraps around it, so a
    frozen-flow sequence repeats after ``n * dx / v`` seconds.
    """

    n: int
    dx: float
    r0: float
    n_subharmonics: int = 3

    def __post_init__(self) -> None:
        n = int(self.n)
        if n < 16 or n % 2 != 0:
            raise ValueError(f"n must be an even integer >= 16, got {self.n!r}")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "dx", _check_positive("dx", self.dx))
        object.__setattr__(self, "r0", _check_positive("r0", self.r0))
        ns = int(self.n_subharmonics)
        if not (0 <= ns <= 6):
            raise ValueError(f"n_subharmonics must be in [0, 6], got {self.n_subharmonics!r}")
        object.__setattr__(self, "n_subharmonics", ns)

    @property
    def size(self) -> float:
        """Physical side length of the screen [m]."""
        return self.n * self.dx

    def generate(self, rng: np.random.Generator | int | None = None) -> NDArray[np.float64]:
        """Generate one screen, shape ``(n, n)``, piston removed [rad].

        Deterministic for a given ``numpy.random.Generator`` seed.
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        n = self.n
        df = 1.0 / (n * self.dx)
        fx = (np.arange(n) - n // 2) * df
        fxx, fyy = np.meshgrid(fx, fx, indexing="xy")
        f = np.hypot(fxx, fyy)
        psd = kolmogorov_psd(f, self.r0)

        cn = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * np.sqrt(psd) * df
        phase = np.real(np.fft.ifft2(np.fft.ifftshift(cn)) * n**2)

        if self.n_subharmonics > 0:
            phase = phase + self._subharmonics(rng)
        return phase - phase.mean()

    def _subharmonics(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Low-frequency correction on a 3x3 stencil per level (Lane et al. 1992)."""
        n = self.n
        axis = (np.arange(n) - n // 2) * self.dx
        out = np.zeros((n, n), dtype=np.float64)
        for level in range(1, self.n_subharmonics + 1):
            df = 1.0 / (3.0**level * n * self.dx)
            for ii in (-1, 0, 1):
                for jj in (-1, 0, 1):
                    if ii == 0 and jj == 0:
                        continue
                    fxl = ii * df
                    fyl = jj * df
                    psd = kolmogorov_psd(np.array(np.hypot(fxl, fyl)), self.r0)
                    c = (rng.standard_normal() + 1j * rng.standard_normal()) * np.sqrt(psd) * df
                    # exp(2i pi (fx x + fy y)) factorises into an outer product.
                    ex = np.exp(2j * np.pi * fxl * axis)
                    ey = np.exp(2j * np.pi * fyl * axis)
                    out += np.real(c * np.outer(ey, ex))
        return out


class FrozenFlow:
    """Taylor frozen-flow sampler: a pupil window sliding across a phase screen.

    Parameters
    ----------
    screen:
        Phase array ``(n, n)`` [rad], e.g. from :meth:`PhaseScreen.generate`.
    dx:
        Screen sample pitch [m].
    n_pupil:
        Pupil window size in samples. Must be <= screen size.
    wind_speed:
        Layer speed [m/s]. Must be > 0.
    wind_direction:
        Direction of travel [rad], measured from the +x axis. Default 0.
    dt:
        Frame interval [s]. Must be > 0.
    allow_wrap:
        If False (default) :meth:`frame` raises once the sliding window would
        run off the screen and wrap. Wrapping is **not** seamless: the FFT part
        of a screen is periodic but the Lane et al. subharmonics are not, so a
        wrapped frame contains a discontinuity that shows up as a large false
        wavefront excursion. :attr:`max_frames` gives the last safe index.
    boiling:
        Fraction ``b`` in ``[0, 1]`` of an independent screen mixed in per frame
        to represent non-frozen evolution ("boiling"):
        ``phi_k = sqrt(1-b^2) phi_frozen,k + b * phi_indep,k``. ``b = 0`` is pure
        Taylor advection. This is an *ad hoc* two-component model, not a
        measured atmospheric statistic -- see README Limitations.

    Notes
    -----
    Taylor's frozen-turbulence hypothesis (Taylor, G. I. 1938, "The spectrum of
    turbulence", *Proc. R. Soc. Lond. A* **164**, 476-490) assumes the phase
    pattern is advected unchanged by the wind. Sampling uses **bilinear
    interpolation with periodic wrap**, which acts as a mild low-pass filter;
    the induced variance loss is measured in ``validation/validate_atmosphere.py``.
    """

    def __init__(
        self,
        screen: NDArray[np.float64],
        dx: float,
        n_pupil: int,
        wind_speed: float,
        wind_direction: float = 0.0,
        dt: float = 1.0e-3,
        boiling: float = 0.0,
        allow_wrap: bool = False,
    ) -> None:
        arr = np.asarray(screen, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"screen must be square 2-D, got shape {arr.shape}")
        self.screen = arr
        self.dx = _check_positive("dx", dx)
        n_pupil = int(n_pupil)
        if n_pupil < 4 or n_pupil > arr.shape[0]:
            raise ValueError(f"n_pupil must be in [4, {arr.shape[0]}], got {n_pupil}")
        self.n_pupil = n_pupil
        self.wind_speed = _check_positive("wind_speed", wind_speed)
        self.wind_direction = float(wind_direction)
        self.dt = _check_positive("dt", dt)
        b = float(boiling)
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"boiling must be in [0, 1], got {boiling!r}")
        self.boiling = b
        self.allow_wrap = bool(allow_wrap)
        self._boil_screen: NDArray[np.float64] | None = None

    def set_boiling_screen(self, screen: NDArray[np.float64]) -> None:
        """Provide the independent screen used for the boiling component."""
        arr = np.asarray(screen, dtype=np.float64)
        if arr.shape != self.screen.shape:
            raise ValueError(
                f"boiling screen must have shape {self.screen.shape}, got {arr.shape}"
            )
        self._boil_screen = arr

    def _window(self, source: NDArray[np.float64], shift_x: float, shift_y: float):
        """Bilinear-interpolated, periodically wrapped window of ``source``."""
        n = self.n_pupil
        idx = np.arange(n, dtype=np.float64)
        xs = idx[None, :] + shift_x
        ys = idx[:, None] + shift_y
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        fx = xs - x0
        fy = ys - y0
        ns = source.shape[0]
        x0m = x0 % ns
        x1m = (x0 + 1) % ns
        y0m = y0 % ns
        y1m = (y0 + 1) % ns
        v00 = source[y0m, x0m]
        v01 = source[y0m, x1m]
        v10 = source[y1m, x0m]
        v11 = source[y1m, x1m]
        return (
            v00 * (1 - fx) * (1 - fy)
            + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy
            + v11 * fx * fy
        )

    @property
    def max_frames(self) -> int:
        """Number of frames before the sliding window wraps around the screen [-].

        ``floor( (n_screen - n_pupil) * dx / (v dt * max(|cos|, |sin|)) ) + 1``.
        """
        span = (self.screen.shape[0] - self.n_pupil) * self.dx
        step = self.wind_speed * self.dt
        axis = max(
            abs(np.cos(self.wind_direction)), abs(np.sin(self.wind_direction)), 1e-12
        )
        return int(np.floor(span / (step * axis))) + 1

    def frame(self, k: int) -> NDArray[np.float64]:
        """Pupil phase at frame index ``k``, shape ``(n_pupil, n_pupil)`` [rad].

        Piston is removed. The window is translated by ``v k dt`` in the wind
        direction, in units of screen samples.
        """
        k = int(k)
        if k < 0:
            raise ValueError(f"frame index must be >= 0, got {k}")
        if not self.allow_wrap and k >= self.max_frames:
            raise ValueError(
                f"frame {k} would wrap the screen (max_frames = {self.max_frames}); "
                "use a larger screen, a slower wind, or allow_wrap=True"
            )
        travel = self.wind_speed * k * self.dt / self.dx
        sx = travel * np.cos(self.wind_direction)
        sy = travel * np.sin(self.wind_direction)
        frozen = self._window(self.screen, sx, sy)
        if self.boiling > 0.0:
            if self._boil_screen is None:
                raise ValueError(
                    "boiling > 0 requires set_boiling_screen() to be called first"
                )
            # Advect the independent screen in a different direction so the two
            # components decorrelate; amplitude mixing preserves total variance.
            bx = travel * np.cos(self.wind_direction + 2.399963)
            by = travel * np.sin(self.wind_direction + 2.399963)
            boil = self._window(self._boil_screen, bx, by)
            frozen = np.sqrt(1.0 - self.boiling**2) * frozen + self.boiling * boil
        return frozen - frozen.mean()

    def sequence(self, n_frames: int) -> NDArray[np.float64]:
        """Frames ``0 .. n_frames-1`` stacked, shape ``(n_frames, n_pupil, n_pupil)`` [rad]."""
        n_frames = int(n_frames)
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        return np.stack([self.frame(k) for k in range(n_frames)])


def discrete_structure_function(
    r: NDArray[np.float64],
    dx: float,
    n: int,
    r0: float,
    n_subharmonics: int = 3,
) -> NDArray[np.float64]:
    """Exact structure function of the *band-limited* spectrum a screen actually contains.

    An FFT screen samples the Kolmogorov spectrum only on the discrete grid
    ``f = k / (n dx)`` inside the Nyquist square ``|f_x|, |f_y| <= 1/(2 dx)``,
    plus the subharmonic points. Its structure function is therefore **not**
    the infinite-band Fried law ``6.88 (r/r0)^(5/3)``; it is the exact discrete
    sum

    ```
    D(r) = 2 sum_k Phi_phi(f_k) df^2 [1 - cos(2 pi f_kx r)]      [rad^2]
    ```

    evaluated here for a separation along +x. Comparing a generated screen
    against *this* quantity separates "is the generator correct?" from "how
    much power does a finite grid lose?". Both comparisons are reported in
    ``validation/validate_atmosphere.py``.

    Parameters
    ----------
    r:
        Separations along x [m].
    dx, n, r0, n_subharmonics:
        Screen parameters, as for :class:`PhaseScreen`.

    Returns
    -------
    ndarray
        ``D(r)`` [rad^2], same shape as ``r``.
    """
    sep = np.atleast_1d(np.asarray(r, dtype=np.float64))
    dx = _check_positive("dx", dx)
    r0 = _check_positive("r0", r0)
    n = int(n)
    if n < 16 or n % 2:
        raise ValueError(f"n must be an even integer >= 16, got {n}")
    df = 1.0 / (n * dx)
    fx = (np.arange(n) - n // 2) * df
    fxx, fyy = np.meshgrid(fx, fx, indexing="xy")
    psd = kolmogorov_psd(np.hypot(fxx, fyy), r0)
    out = np.empty(sep.shape, dtype=np.float64)
    for i, rr in enumerate(sep.ravel()):
        total = 2.0 * np.sum(psd * df * df * (1.0 - np.cos(2.0 * np.pi * fxx * rr)))
        for level in range(1, int(n_subharmonics) + 1):
            dfl = df / 3.0**level
            for ii in (-1, 0, 1):
                for jj in (-1, 0, 1):
                    if ii == 0 and jj == 0:
                        continue
                    p = float(kolmogorov_psd(np.array(np.hypot(ii * dfl, jj * dfl)), r0))
                    total += 2.0 * p * dfl * dfl * (1.0 - np.cos(2.0 * np.pi * ii * dfl * rr))
        out.ravel()[i] = total
    return out.reshape(sep.shape)
