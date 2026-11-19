import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from beam_math import analyze_frame, extract_channel


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


def test_clipped_beam_flagged_not_reliable():
    data = gaussian_frame(peak=500)
    result = analyze_frame(data, saturation_threshold=250)
    assert result.saturated_count > 0
    assert not result.fit_reliable
    assert "НАСЫЩЕНИЕ" in result.warning


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
