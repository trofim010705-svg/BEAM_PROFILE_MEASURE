import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from beam_propagation import fit_propagation


def diameters(z_mm, z0_mm, d0_um, theta_mrad):
    z_mm = np.asarray(z_mm, dtype=float)
    return np.hypot(d0_um, 2 * theta_mrad * (z_mm - z0_mm))


def test_recovers_independent_astigmatic_axes_in_display_units():
    # Two measurements at the same Z are valid replicates.
    z_mm = [-45, -22, -22, -6, 0, 13, 38]
    wavelength_nm = 780
    for z0_mm, d0_um, theta_mrad in [(-5.5, 95, 7.2), (9.25, 108, 10.4)]:
        fit = fit_propagation(z_mm, diameters(z_mm, z0_mm, d0_um, theta_mrad), wavelength_nm)

        assert fit.z0_mm == pytest.approx(z0_mm)
        assert fit.d0_um == pytest.approx(d0_um)
        assert fit.theta_mrad == pytest.approx(theta_mrad)
        # µm·mrad/nm is dimensionless, and µm²/nm is mm.
        assert fit.m_squared == pytest.approx(math.pi * d0_um * theta_mrad / (2 * wavelength_nm))
        assert fit.rayleigh_ideal_mm == pytest.approx(math.pi * (d0_um / 2) ** 2 / wavelength_nm)
        assert fit.rayleigh_fit_mm == pytest.approx(d0_um / (2 * theta_mrad))
        assert fit.r_squared == pytest.approx(1.0)
        assert fit.waist_within_range
        assert fit.predict_squared_diameter(z0_mm) == pytest.approx(d0_um ** 2)
        np.testing.assert_allclose(
            fit.predict_squared_diameter([-68, 3, 72]),
            diameters([-68, 3, 72], z0_mm, d0_um, theta_mrad) ** 2,
        )


def test_large_coordinate_offset_and_extrapolated_waist():
    offset = 1e9
    z_mm = offset + np.array([12.5, 21.0, 28.5, 43.0, 51.5])
    z0_mm = offset - 4.25
    fit = fit_propagation(z_mm, diameters(z_mm, z0_mm, 130, 5.6), 780)

    assert fit.z0_mm == pytest.approx(z0_mm, rel=0, abs=5e-7)
    assert fit.d0_um == pytest.approx(130)
    assert fit.theta_mrad == pytest.approx(5.6)
    assert not fit.waist_within_range


@pytest.mark.parametrize(
    ('z_mm', 'diameter_um', 'wavelength_nm'),
    [
        ([0, 1, 1, 0], [50, 55, 55, 50], 780),
        ([0, 1, 2], [50, 55], 780),
        ([0, 1, np.nan], [50, 55, 63], 780),
        ([0, 1, 2], [50, np.inf, 63], 780),
        ([0, 1, 2], [50, 0, 63], 780),
        ([0, 1, 2], [50, 55, 63], 0),
        ([-1, 0, 1], [50, 50, 50], 780),
        ([-1, 0, 1], [50, 55, 50], 780),
        ([5, 6, 7], np.sqrt([24, 35, 48]), 780),  # D² = Z² − 1 has no real waist diameter.
    ],
    ids=[
        'fewer-than-three-distinct-z', 'different-lengths', 'nonfinite-z',
        'nonfinite-diameter', 'nonpositive-diameter', 'nonpositive-wavelength',
        'zero-curvature', 'negative-curvature', 'nonpositive-waist-squared',
    ],
)
def test_rejects_unfit_or_nonphysical_inputs(z_mm, diameter_um, wavelength_nm):
    with pytest.raises(ValueError):
        fit_propagation(z_mm, diameter_um, wavelength_nm)
