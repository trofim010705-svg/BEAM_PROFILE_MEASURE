"""Beam analysis: 1/e² Gaussian diameters from a USB camera frame.

The Gaussian fits operate on integrated horizontal and vertical profiles with
a separate linear background on each axis. Inputs should be linear,
unsaturated, unprocessed sensor data where possible. Outputs are optical
1/e² diameters, not FWHM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class AxisFit:
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    center_px: float
    radius_px: float
    diameter_px: float
    r_squared: float
    amplitude: float
    offset: float  # Linear background value at the midpoint of x.
    slope: float = 0.0

    @property
    def baseline(self) -> np.ndarray:
        midpoint = (self.x[0] + self.x[-1]) / 2
        return self.offset + self.slope * (self.x - midpoint)


@dataclass
class BeamAnalysis:
    roi: tuple[int, int, int, int]
    corrected_roi: np.ndarray
    raw_roi: np.ndarray
    saturated_mask: np.ndarray
    saturated_count: int
    saturation_percent: float
    background: float
    peak_signal: float
    x_fit: Optional[AxisFit]
    y_fit: Optional[AxisFit]
    warning: str
    fit_reliable: bool


def extract_channel(frame: np.ndarray, mode: str = "Red") -> np.ndarray:
    """Select a raw-ish video channel. OpenCV frames conventionally use BGR.

    For standard USB UVC video frames, the samples can already be subjected
    to ISP gamma, automatic gain, demosaicing or MJPEG compression.
    """
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Неподдерживаемый формат кадра")
    if mode == "Red":
        return frame[..., 2].astype(np.float32)
    if mode == "Green":
        return frame[..., 1].astype(np.float32)
    if mode == "Blue":
        return frame[..., 0].astype(np.float32)
    b, g, r = (frame[..., i].astype(np.float32) for i in range(3))
    if mode == "Mean RGB":
        return (r + g + b) / 3.0
    if mode == "Gray":
        return 0.114 * b + 0.587 * g + 0.299 * r
    raise ValueError(f"Неизвестный канал: {mode}")


def gaussian_1d(x: np.ndarray, amplitude: float, center: float,
                radius: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-2.0 * ((x - center) / radius) ** 2) + offset


def gaussian_with_baseline(x: np.ndarray, amplitude: float, center: float,
                           radius: float, offset: float, slope: float) -> np.ndarray:
    """The model uses coordinates centered at the middle of the profile."""
    return gaussian_1d(x, amplitude, center, radius, offset) + slope * x


def fit_axis(x: np.ndarray, values: np.ndarray) -> Optional[AxisFit]:
    vals = np.asarray(values, dtype=np.float64)
    pos = np.asarray(x, dtype=np.float64)
    if vals.ndim != 1 or vals.shape != pos.shape or len(vals) < 12:
        return None
    if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(pos)) or np.any(np.diff(pos) <= 0):
        return None

    midpoint = (pos[0] + pos[-1]) / 2
    centered_x = pos - midpoint
    span = pos[-1] - pos[0] + 1

    # Estimate the initial line from the wings, then seed the peak above it.
    edge_count = max(2, len(vals) // 10)
    left_x = np.median(centered_x[:edge_count])
    right_x = np.median(centered_x[-edge_count:])
    left_y = np.median(vals[:edge_count])
    right_y = np.median(vals[-edge_count:])
    slope0 = (right_y - left_y) / (right_x - left_x)
    offset0 = left_y - slope0 * left_x
    detrended = vals - (offset0 + slope0 * centered_x)
    amp0 = float(detrended.max())
    if amp0 <= 1:
        return None

    weights = np.maximum(detrended - 0.05 * amp0, 0)
    total = float(np.sum(weights))
    center0 = float(np.sum(centered_x * weights) / total)
    variance = float(np.sum((centered_x - center0) ** 2 * weights) / total)
    radius0 = min(2 * np.sqrt(max(variance, 1.)), span * 1.5)

    try:
        params, _ = curve_fit(
            gaussian_with_baseline, centered_x, vals,
            p0=[amp0, center0, radius0, offset0, slope0],
            bounds=(
                [0, centered_x[0] - span, 0.7, -np.inf, -np.inf],
                [np.inf, centered_x[-1] + span, 2.0 * span, np.inf, np.inf],
            ),
            maxfev=3000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    amp, center, radius, offset, slope = (float(v) for v in params)
    if amp <= 1:
        return None
    modeled = gaussian_with_baseline(centered_x, *params)
    ss_res = np.sum((vals - modeled) ** 2)
    # Measure fit quality on the beam signal so that a large slope cannot
    # make an inaccurate Gaussian look like a good fit.
    signal = vals - (offset + slope * centered_x)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return AxisFit(pos, vals, modeled, float(center + midpoint), radius,
                   2 * radius, r2, amp, offset, slope)


def analyze_frame(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    mode: str = "Red",
    dark_frame: Optional[np.ndarray] = None,
    saturation_threshold: float = 250,
) -> BeamAnalysis:
    channel = extract_channel(frame, mode)
    h, w = channel.shape
    if roi is None:
        roi = (0, 0, w, h)
    x0, y0, x1, y1 = (int(v) for v in roi)
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 - x0 < 12 or y1 - y0 < 12:
        raise ValueError("ROI слишком мала: выдели прямоугольник хотя бы 12×12 px")

    source = channel[y0:y1, x0:x1]
    if dark_frame is not None:
        if dark_frame.shape != frame.shape:
            raise ValueError("Размер тёмного кадра не совпадает с текущим кадром")
        # dark_frame contains the average original color/monochrome frames
        dark = extract_channel(dark_frame, mode)[y0:y1, x0:x1]
        corrected = source - dark
    else:
        corrected = source.copy()

    # Estimate residual electronic/background offset from the outer ROI rim.
    ny, nx = corrected.shape
    bx = max(2, nx // 12)
    by = max(2, ny // 12)
    rim = np.concatenate([
        corrected[:by, :].ravel(), corrected[-by:, :].ravel(),
        corrected[:, :bx].ravel(), corrected[:, -bx:].ravel(),
    ])
    bg = float(np.median(rim))
    corrected -= bg
    corrected_roi = np.maximum(corrected, 0).astype(np.float32, copy=False)
    peak = float(corrected_roi.max())

    saturated = (source >= saturation_threshold)
    satcount = int(np.count_nonzero(saturated))
    satpercent = satcount / source.size * 100.0

    # Retain signed samples in the projections. Clipping the low side of a
    # tilted background would change the line the fit has to subtract.
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    x_profile = corrected.sum(axis=0, dtype=np.float64)
    y_profile = corrected.sum(axis=1, dtype=np.float64)
    fx = fit_axis(xx, x_profile) if peak >= 8 else None
    fy = fit_axis(yy, y_profile) if peak >= 8 else None

    warnings = []
    if peak < 8:
        warnings.append("Слишком слабый сигнал")
    if satcount:
        warnings.append(f"НАСЫЩЕНИЕ: {satcount} px (порог {saturation_threshold:g})")
    if fx is None or fy is None:
        warnings.append("Нет устойчивого гауссова фита")
    else:
        if min(fx.r_squared, fy.r_squared) < .98:
            warnings.append("Плохое согласие с гауссианой (R² < 0.98)")
        for axis, fit, lo, hi in [("X", fx, x0, x1), ("Y", fy, y0, y1)]:
            # A projected profile should include most of its wings, ideally >= 1.5*w.
            if min(fit.center_px - lo, hi - fit.center_px) < fit.radius_px:
                warnings.append(f"ROI обрезает крылья пучка по {axis}")
            if fit.diameter_px < 6:
                warnings.append(f"Пятно менее 6 пикселей по {axis}")

    reliable = not warnings
    return BeamAnalysis((x0, y0, x1, y1), corrected_roi, source, saturated,
                        satcount, satpercent, bg, peak, fx, fy,
                        "; ".join(warnings), reliable)
