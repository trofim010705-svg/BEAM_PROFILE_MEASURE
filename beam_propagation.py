"""Fit beam propagation from 1/e² diameters measured along the optical axis."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PropagationFit:
    z0_mm: float
    d0_um: float
    theta_mrad: float
    m_squared: float
    rayleigh_ideal_mm: float
    rayleigh_fit_mm: float
    r_squared: float
    waist_within_range: bool

    def predict_squared_diameter(self, z_mm: ArrayLike) -> np.ndarray | np.float64:
        """Evaluate D² in µm² at axial positions in mm."""
        distance_mm = np.asarray(z_mm, dtype=np.float64) - self.z0_mm
        return self.d0_um ** 2 + np.square(2 * self.theta_mrad * distance_mm)


def fit_propagation(z_mm: ArrayLike, diameter_um: ArrayLike,
                    wavelength_nm: float) -> PropagationFit:
    """Fit D²(z) by least squares. The divergence θ is the half-angle."""
    try:
        z = np.asarray(z_mm, dtype=np.float64)
        diameters = np.asarray(diameter_um, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Координаты Z и диаметры должны быть числами.") from exc

    if z.ndim != 1 or z.shape != diameters.shape:
        raise ValueError("Каждой координате Z должен соответствовать один диаметр.")
    if not np.all(np.isfinite(z)):
        raise ValueError("Координаты Z должны быть конечными числами в миллиметрах.")
    if not np.all(np.isfinite(diameters)) or np.any(diameters <= 0):
        raise ValueError("Диаметры должны быть положительными конечными числами в микронах.")
    if np.unique(z).size < 3:
        raise ValueError("Для аппроксимации нужны измерения как минимум в трёх разных положениях Z.")

    try:
        wavelength = float(wavelength_nm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Длина волны должна быть положительным числом в нанометрах.") from exc
    if not math.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("Длина волны должна быть положительным числом в нанометрах.")

    # Center and scale the inputs so a large Z origin cannot make the
    # square and linear terms numerically indistinguishable.
    z_min, z_max = float(z.min()), float(z.max())
    midpoint = z_min / 2 + z_max / 2
    z_scale = max(midpoint - z_min, z_max - midpoint)
    distance = (z - midpoint) / z_scale
    design = np.column_stack((distance ** 2, distance, np.ones_like(distance)))

    diameter_scale = float(diameters.max())
    squared = (diameters / diameter_scale) ** 2
    squared_mean = float(squared.mean())
    centered_squared = squared - squared_mean
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, centered_squared, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Не удалось аппроксимировать зависимость D² от Z.") from exc
    if rank < 3:
        raise ValueError("Положения Z расположены слишком близко для устойчивой аппроксимации.")

    curvature, linear, offset = map(float, coefficients)
    if curvature <= 8 * np.finfo(np.float64).eps:
        raise ValueError("Кривизна D² должна быть положительной и различимой в данных.")

    waist_position = -linear / (2 * curvature)
    waist_squared = offset + squared_mean + 0.5 * linear * waist_position
    if waist_squared <= 0:
        raise ValueError("Аппроксимация дала неположительный квадрат диаметра в перетяжке.")

    z0_mm = midpoint + z_scale * waist_position
    d0_um = diameter_scale * math.sqrt(waist_squared)
    # One micron per millimeter is numerically one milliradian.
    theta_mrad = (0.5 * math.sqrt(curvature) * diameter_scale) / z_scale
    if not all(math.isfinite(value) for value in (z0_mm, d0_um, theta_mrad)) or theta_mrad <= 0:
        raise ValueError("Параметры перетяжки или расходимости выходят за допустимый числовой диапазон.")

    m_squared = (math.pi / 2) * d0_um * (theta_mrad / wavelength)
    rayleigh_ideal_mm = (math.pi / 4) * d0_um * (d0_um / wavelength)
    rayleigh_fit_mm = (0.5 * d0_um) / theta_mrad
    if not all(math.isfinite(value) for value in (m_squared, rayleigh_ideal_mm, rayleigh_fit_mm)):
        raise ValueError("Вычисленные M² или длина Рэлея выходят за допустимый числовой диапазон.")

    residual = centered_squared - design @ coefficients
    r_squared = float(1 - (residual @ residual) / (centered_squared @ centered_squared))
    return PropagationFit(
        z0_mm=z0_mm,
        d0_um=d0_um,
        theta_mrad=theta_mrad,
        m_squared=m_squared,
        rayleigh_ideal_mm=rayleigh_ideal_mm,
        rayleigh_fit_mm=rayleigh_fit_mm,
        r_squared=r_squared,
        waist_within_range=z_min <= z0_mm <= z_max,
    )
