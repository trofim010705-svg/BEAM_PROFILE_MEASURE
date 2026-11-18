import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from beam_math import analyze_frame, extract_channel, fit_axis


def gaussian_frame(width=420, height=320, x0=210.0, y0=145.0,
                   radius_x=25.0, radius_y=48.0, peak=150., background=10., noise=0., seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:height, :width]
    data = background + peak * np.exp(-2 * (
        (xx - x0) ** 2 / radius_x**2 + (yy - y0) ** 2 / radius_y**2))
    if noise:
        data = data + rng.normal(0, noise, size=data.shape)
    return np.clip(np.round(data), 0, 255).astype(np.uint8)


def test_correct_gaussian_widths_and_center():
    data = gaussian_frame()
    result = analyze_frame(data, saturation_threshold=250)
    assert result.fit_reliable, result.warning
    assert abs(result.x_fit.diameter_px - 50.) < 0.8
    assert abs(result.y_fit.diameter_px - 96.) < 0.8
    assert abs(result.x_fit.center_px - 210) < 0.5
    assert abs(result.y_fit.center_px - 145) < 0.5


def test_roi_and_dark_subtraction():
    data = gaussian_frame(background=24, noise=1)
    dark = np.full_like(data, 24)
    result = analyze_frame(data, roi=(130, 60, 290, 235),
                           dark_frame=dark, saturation_threshold=250)
    assert result.fit_reliable, result.warning
    assert abs(result.x_fit.diameter_px - 50) < 1
    assert abs(result.y_fit.diameter_px - 96) < 1


def test_linear_background_fit_recovers_both_radii():
    yy, xx = np.ogrid[:320, :420]
    slope_x, slope_y = 0.22, -0.18
    background = 72 + slope_x * (xx - 210) + slope_y * (yy - 145)
    beam = 116 * np.exp(-2 * (((xx - 210) / 25) ** 2 + ((yy - 145) / 48) ** 2))
    data = np.clip(np.rint(background + beam), 0, 255).astype(np.uint8)

    roi = (60, 27, 392, 290)
    result = analyze_frame(data, roi=roi)

    assert result.fit_reliable, result.warning
    assert abs(result.x_fit.radius_px - 25) < 0.5
    assert abs(result.y_fit.radius_px - 48) < 0.5
    assert abs(result.x_fit.center_px - 210) < 0.5
    assert abs(result.y_fit.center_px - 145) < 0.5
    assert abs(result.x_fit.slope - slope_x * (roi[3] - roi[1])) < 0.15
    assert abs(result.y_fit.slope - slope_y * (roi[2] - roi[0])) < 0.15


def test_tilted_background_alone_has_no_radius():
    yy, xx = np.ogrid[:150, :200]
    data = (40 + 0.2 * xx + 0.3 * yy).astype(np.float32)

    result = analyze_frame(data)

    assert not result.fit_reliable
    assert result.x_fit is None
    assert result.y_fit is None


def test_large_background_slope_does_not_mask_bad_beam_fit():
    x = np.arange(360, dtype=float)
    y = 1000 + 10 * x + 90 * np.exp(-2 * ((x - 110) / 22) ** 2)
    y += 75 * np.exp(-2 * ((x - 190) / 30) ** 2)

    fit = fit_axis(x, y)

    assert fit is not None
    raw_r_squared = 1 - np.sum((y - fit.fitted) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert raw_r_squared > 0.98
    assert fit.r_squared < 0.98


def test_clipped_beam_flagged_not_reliable():
    data = gaussian_frame(peak=500)
    result = analyze_frame(data, saturation_threshold=250)
    assert result.saturated_count > 0
    assert not result.fit_reliable
    assert "НАСЫЩЕНИЕ" in result.warning
    assert result.x_fit is not None and result.x_fit.radius_px > 0
    assert result.y_fit is not None and result.y_fit.radius_px > 0


def test_rgb_channel_selection():
    data = gaussian_frame()
    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    rgb[..., 2] = data  # OpenCV BGR red
    assert np.all(extract_channel(rgb, "Red") == data)
    assert np.count_nonzero(extract_channel(rgb, "Blue")) == 0


def test_tiny_roi_rejected():
    data = gaussian_frame()
    try:
        analyze_frame(data, roi=(10, 10, 15, 15))
    except ValueError:
        return
    assert False
