"""Kolmogorov screens: PSD, structure function, spectral gradients, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.turbulence import (
    KOLMOGOROV_PSD_COEFF,
    STRUCTURE_FUNCTION_COEFF,
    KolmogorovScreens,
    kolmogorov_psd,
    measure_structure_function,
    structure_function_theory,
)


def test_structure_function_coefficient_closed_form() -> None:
    """Known answer: 2 (24/5 Gamma(6/5))^(5/6) = 6.883877... (Fried 1965)."""
    assert STRUCTURE_FUNCTION_COEFF == pytest.approx(6.883877, abs=1e-5)


def test_psd_coefficient_and_scaling() -> None:
    """Roddier 1981 / Hardy 1998 Eq. 3.31: 0.0229 r0^(-5/3) f^(-11/3)."""
    assert KOLMOGOROV_PSD_COEFF == 0.0229
    f = np.array([1.0, 2.0])
    p = kolmogorov_psd(f, 0.1)
    assert p[0] == pytest.approx(0.0229 * 0.1 ** (-5 / 3))
    # A factor 2 in frequency is a factor 2^(-11/3) in power.
    assert p[1] / p[0] == pytest.approx(2.0 ** (-11 / 3))
    # A factor 2 in r0 is a factor 2^(-5/3) in power.
    assert kolmogorov_psd(np.array([1.0]), 0.2)[0] / p[0] == pytest.approx(2.0 ** (-5 / 3))


def test_structure_function_scaling() -> None:
    r = np.array([0.05, 0.1])
    d = structure_function_theory(r, 0.1)
    assert d[1] == pytest.approx(STRUCTURE_FUNCTION_COEFF)
    assert d[1] / d[0] == pytest.approx(2.0 ** (5 / 3))


@settings(max_examples=25, deadline=None)
@given(
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.02, max_value=0.5),
)
def test_structure_function_is_a_power_law(r: float, r0: float) -> None:
    """Property: D(2r)/D(r) = 2^(5/3) at every separation and every r0."""
    a = float(structure_function_theory(np.array([r]), r0)[0])
    b = float(structure_function_theory(np.array([2 * r]), r0)[0])
    assert b / a == pytest.approx(2.0 ** (5 / 3), rel=1e-12)


def test_screens_are_reproducible() -> None:
    kwargs = dict(n_grid=32, dx=0.05, r0=0.15, n_subharmonics=2, seed=42)
    a = KolmogorovScreens(**kwargs).generate(3)
    b = KolmogorovScreens(**kwargs).generate(3)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    c = KolmogorovScreens(**{**kwargs, "seed": 43}).generate(3)
    assert not np.allclose(a[0], c[0])


def test_screen_shapes_and_zero_mean() -> None:
    ph, gx, gy = KolmogorovScreens(n_grid=32, dx=0.05, r0=0.2, seed=1).generate(4)
    assert ph.shape == gx.shape == gy.shape == (4, 32, 32)
    assert np.max(np.abs(ph.mean(axis=(1, 2)))) < 1e-10


def test_gradient_is_the_exact_spectral_derivative() -> None:
    """Known answer: DFT(gx) = 2 pi i f_x DFT(phase), to machine precision.

    Checked with the subharmonics switched off, because the subharmonic
    components are deliberately non-periodic and so are not represented on the
    DFT grid. A finite-difference comparison is *not* used: a Kolmogorov
    gradient has variance density proportional to f^(-2/3), i.e. dominated by
    the top of the band, where a central difference errs by tens of per cent no
    matter how fine the grid is.
    """
    dx = 0.02
    n = 64
    ph, gx, gy = KolmogorovScreens(n_grid=n, dx=dx, r0=0.4, n_subharmonics=0, seed=7).generate(2)
    fx = np.fft.fftfreq(n, d=dx)
    fxx, fyy = np.meshgrid(fx, fx, indexing="xy")
    spec = np.fft.fft2(ph, axes=(-2, -1))
    # The identity holds bin by bin except on the Nyquist row and column, where
    # +f and -f alias onto the same bin and the derivative of a real signal is
    # not representable. Those two lines are excluded.
    keep = np.ones((n, n), dtype=bool)
    keep[n // 2, :] = False
    keep[:, n // 2] = False
    for g, ff in ((gx, fxx), (gy, fyy)):
        got = np.fft.fft2(g, axes=(-2, -1))
        want = 2j * np.pi * ff * spec
        scale = np.max(np.abs(got))
        assert np.max(np.abs(got - want)[:, keep]) < 1e-8 * scale


def test_gradient_mean_is_small() -> None:
    """A zero-mean screen has no net gradient bias across the ensemble."""
    n = 200
    _, gx, gy = KolmogorovScreens(n_grid=32, dx=0.05, r0=0.2, seed=9).generate(n)
    for g in (gx, gy):
        per_screen = g.mean(axis=(1, 2))
        sem = per_screen.std(ddof=1) / np.sqrt(n)
        assert abs(per_screen.mean()) < 4.0 * sem


def test_subharmonics_add_low_frequency_power() -> None:
    """Plain FFT screens lack power below 1/L; subharmonics restore it."""
    kw = dict(n_grid=64, dx=0.05, r0=0.2, seed=3)
    v0 = KolmogorovScreens(**kw, n_subharmonics=0).generate(40)[0].var()
    v3 = KolmogorovScreens(**kw, n_subharmonics=3).generate(40)[0].var()
    assert v3 > 1.5 * v0


def test_measured_structure_function_tracks_theory() -> None:
    """Regression-style check on a well-sampled screen set.

    With dx = r0/8, a 128-point grid and three subharmonic levels, the measured
    structure function is within 12 % of 6.88 (r/r0)^(5/3) over
    0.5 <= r/r0 <= 3. The residual is dominated by the grid band limit at small
    r and by the finite screen period at large r.
    """
    r0 = 0.1
    ph, _, _ = KolmogorovScreens(n_grid=128, dx=r0 / 8, r0=r0, n_subharmonics=3, seed=5).generate(
        120
    )
    r, d = measure_structure_function(ph, r0 / 8, max_lag=24)
    ratio = d / structure_function_theory(r, r0)
    sel = (r / r0 >= 0.5) & (r / r0 <= 3.0)
    assert np.all(np.abs(ratio[sel] - 1.0) < 0.12), f"ratios {ratio[sel]}"


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="r0 must be finite"):
        kolmogorov_psd(np.array([1.0]), -1.0)
    with pytest.raises(ValueError, match="non-negative"):
        kolmogorov_psd(np.array([-1.0]), 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        structure_function_theory(np.array([-1.0]), 0.1)
    with pytest.raises(ValueError, match="n_grid must be >= 8"):
        KolmogorovScreens(n_grid=4, dx=0.1, r0=0.1)
    with pytest.raises(TypeError, match="n_grid"):
        KolmogorovScreens(n_grid=16.0, dx=0.1, r0=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dx"):
        KolmogorovScreens(n_grid=16, dx=0.0, r0=0.1)
    with pytest.raises(ValueError, match="n_subharmonics"):
        KolmogorovScreens(n_grid=16, dx=0.1, r0=0.1, n_subharmonics=-1)
    gen = KolmogorovScreens(n_grid=16, dx=0.1, r0=0.1, n_subharmonics=0)
    with pytest.raises(ValueError, match="n_screens must be >= 1"):
        gen.generate(0)
    with pytest.raises(TypeError, match="n_screens"):
        gen.generate(2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_lag"):
        measure_structure_function(np.zeros((2, 8, 8)), 0.1, max_lag=8)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        measure_structure_function(np.zeros(8), 0.1)
