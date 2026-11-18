"""Beam analysis: 1/e² Gaussian diameters from a USB camera frame.

The Gaussian fits operate on *integrated* horizontal/vertical profiles rather
than only on the central row. Inputs should be linear, unsaturated, unprocessed
sensor data where possible. Outputs are optical 1/e² diameters, not FWHM.
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
    offset: float


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
    b, g, r = (frame[..., i].astype(np.float32) for i in range(3))
    if mode == "Red":
        return r
    if mode == "Green":
        return g
    if mode == "Blue":
        return b
    if mode == "Mean RGB":
        return (r + g + b) / 3.0
    if mode == "Gray":
        return 0.114 * b + 0.587 * g + 0.299 * r
    raise ValueError(f"Неизвестный канал: {mode}")


def gaussian_1d(x: np.ndarray, amplitude: float, center: float,
                radius: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-2.0 * ((x - center) / radius) ** 2) + offset


def fit_axis(x: np.ndarray, values: np.ndarray) -> Optional[AxisFit]:
    vals = np.asarray(values, dtype=np.float64)
    pos = np.asarray(x, dtype=np.float64)
    if len(vals) < 12 or not np.all(np.isfinite(vals)):
        return None

    baseline = float(np.percentile(vals, 10))
    weights = np.clip(vals - baseline, 0, None)
    total = float(np.sum(weights))
    if total <= 0 or float(vals.max() - baseline) < 1:
        return None
    center0 = float(np.sum(pos * weights) / total)
    variance = float(np.sum((pos - center0) ** 2 * weights) / total)
    radius0 = max(2., min(2 * np.sqrt(max(variance, 1.)), len(vals) * 1.5))
    amp0 = max(float(vals.max() - baseline), 1e-5)

    try:
        params, _ = curve_fit(
            gaussian_1d, pos, vals,
            p0=[amp0, center0, radius0, max(0, baseline)],
            bounds=(
                [0, pos[0] - len(pos), 0.7, 0],
                [np.inf, pos[-1] + len(pos), 2.0 * len(pos), np.inf],
            ),
            maxfev=3000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    amp, center, radius, offset = (float(v) for v in params)
    modeled = gaussian_1d(pos, *params)
    ss_res = np.sum((vals - modeled) ** 2)
    ss_tot = np.sum((vals - np.mean(vals)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return AxisFit(pos, vals, modeled, center, radius, 2 * radius,
                   r2, amp, offset)


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
    corrected = np.maximum(corrected - bg, 0).astype(np.float32)
    peak = float(corrected.max())

    saturated = (source >= saturation_threshold)
    satcount = int(np.count_nonzero(saturated))
    satpercent = satcount / source.size * 100.0

    # Axes use absolute camera pixel coordinates.
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

    reliable = (satcount == 0 and fx is not None and fy is not None and
                fx.r_squared >= .98 and fy.r_squared >= .98 and
                not any("обрезает" in s for s in warnings) and peak >= 8)
    return BeamAnalysis((x0, y0, x1, y1), corrected, source, saturated,
                        satcount, satpercent, bg, peak, fx, fy,
                        "; ".join(warnings), reliable)
