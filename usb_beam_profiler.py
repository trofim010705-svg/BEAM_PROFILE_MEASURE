#!/usr/bin/env python3
"""USB Laser Beam Profiler: live 2D + 1D Gaussian profiles + clipping mask.

Start: python usb_beam_profiler.py
Requires numpy scipy opencv-python Pillow matplotlib (Tkinter from Python).
"""
from __future__ import annotations

import csv
import json
import math
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beam_math import BeamAnalysis, analyze_frame, extract_channel
from propagation_tab import PropagationTab


APP_TITLE = "USB Laser Beam Profiler"
NO_CAMERA_READBACK = "Выдержка (Exposure): — · Усиление (Gain): —"


def parse_optional_float(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    return float(text) if text else None


def parse_pixel_pitch(text: str) -> float | None:
    pitch = parse_optional_float(text)
    if pitch is not None and not 0 < pitch <= 1000:
        raise ValueError("Масштаб должен быть положительным и <= 1000 мкм/px")
    return pitch


class CameraWorker(threading.Thread):
    """Camera access is entirely owned by one thread; GUI reads latest frame."""

    retry_delay = 2.0
    read_retry_delay = 0.06
    read_failure_limit = 5
    readback_interval = 1.0

    def __init__(self, index: int, backend: int, resolution: tuple[int, int],
                 out_queue: queue.Queue, cmd_queue: queue.Queue, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.index = index
        self.backend = backend
        self.resolution = resolution
        self.out_queue = out_queue
        self.cmd_queue = cmd_queue
        self.event_queue = event_queue
        self.please_stop = threading.Event()

    @staticmethod
    def put_latest(q: queue.Queue, value):
        try:
            q.put_nowait(value)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(value)

    @staticmethod
    def read_property(cap, prop):
        try:
            value = cap.get(prop)
        except cv2.error:
            return None
        return value if math.isfinite(value) else None

    def report_readback(self, cap, backend_name):
        exposure = self.read_property(cap, cv2.CAP_PROP_EXPOSURE)
        gain = self.read_property(cap, cv2.CAP_PROP_GAIN)
        if exposure is None:
            exposure_text = "недоступна"
        elif exposure in (-1, 0):
            exposure_text = f"{exposure:g} (может означать отсутствие данных)"
        else:
            exposure_text = f"{exposure:g}"
        gain_text = "—" if gain is None or gain == -1 else f"{gain:g}"
        qualifier = "значения по умолчанию" if backend_name == "MSMF" else "данные драйвера"
        self.event_queue.put(("camera_readback", f"{backend_name} · Выдержка (Exposure): {exposure_text} · Усиление (Gain): {gain_text} · {qualifier}"))

    def apply_control(self, cap, backend_name, cmd, value):
        prop = {
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
        }[cmd]
        if cmd == "auto_exposure":
            if backend_name == "MSMF" and not value:
                # MSMF would set Exposure to zero (one second) along with manual mode.
                self.event_queue.put(("control_result", "Media Foundation: для ручного режима задай Exposure"))
                return False
            request = f"Автоэкспозиция {'вкл.' if value else 'выкл.'}"
            if backend_name == "MSMF":
                # MSMF forwards this value to Set as a long and also tests it for
                # nonzero to select auto mode. Use the driver's default in range.
                setting = self.read_property(cap, cv2.CAP_PROP_EXPOSURE) or 0.5
            elif backend_name == "DSHOW":
                setting = float(value)
            else:
                setting = 0.75 if value else 0.25
        else:
            request = f"{cmd.capitalize()} {value:g}"
            setting = float(value)
        try:
            ok = cap.set(prop, setting)
        except cv2.error:
            ok = False
        response = "драйвер принял запрос" if ok else "драйвер отклонил запрос"
        self.event_queue.put(("control_result", f"{request} · {response}"))
        return ok

    def capture_frames(self, cap, controls) -> str:
        try:
            backend_name = cap.getBackendName()
        except cv2.error:
            backend_name = "Камера"
        width, height = self.resolution
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, height),
                            (cv2.CAP_PROP_BUFFERSIZE, 1)):
            # The driver can ignore unsupported mode/buffer requests.
            try:
                cap.set(prop, value)
            except cv2.error:
                pass
        for cmd, value in controls.items():
            self.apply_control(cap, backend_name, cmd, value)

        streaming = False
        failures = 0
        next_readback = 0.0
        while not self.please_stop.is_set():
            while not self.please_stop.is_set():
                try:
                    cmd, value = self.cmd_queue.get_nowait()
                except queue.Empty:
                    break
                if self.apply_control(cap, backend_name, cmd, value):
                    controls.pop(cmd, None)
                    if cmd == "auto_exposure" and value:
                        controls.pop("exposure", None)
                    controls[cmd] = value
                next_readback = 0.0
            if self.please_stop.is_set():
                break
            ok, frame = cap.read()
            if self.please_stop.is_set():
                break
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= self.read_failure_limit or self.please_stop.wait(self.read_retry_delay):
                    break
                continue
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame[:, :, 0]
            if frame.dtype != np.uint8 or not (
                frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (3, 4))
            ):
                return f"Камера {self.index} вернула кадр неподдерживаемого формата"
            if not streaming:
                height, width = frame.shape[:2]
                self.event_queue.put(("streaming", f"Камера {self.index}: получен кадр {width}×{height} px"))
                streaming = True
            failures = 0
            now = time.monotonic()
            self.put_latest(self.out_queue, (now, frame))
            if now >= next_readback:
                self.report_readback(cap, backend_name)
                next_readback = now + self.readback_interval
        return f"Нет кадров с камеры {self.index}"

    def run(self):
        controls = {}
        while not self.please_stop.is_set():
            cap = None
            reason = f"Камера {self.index} недоступна"
            try:
                cap = cv2.VideoCapture(self.index, self.backend)
                if not self.please_stop.is_set() and cap.isOpened():
                    reason = f"Нет кадров с камеры {self.index}"
                    reason = self.capture_frames(cap, controls)
            except cv2.error:
                pass
            finally:
                if not self.please_stop.is_set():
                    self.put_latest(self.out_queue, None)
                    self.event_queue.put(("retrying", f"{reason}. Переподключаю…"))
                if cap is not None:
                    try:
                        cap.release()
                    except cv2.error:
                        pass
                    cap = None
            self.please_stop.wait(self.retry_delay)


def make_synthetic_frame(t: float, w: int = 960, h: int = 600) -> np.ndarray:
    """Deliberately artificial image; used for a software check without a laser."""
    rng = np.random.default_rng(int(t * 10) % (2**32))
    yy, xx = np.ogrid[:h, :w]
    cx = w * 0.51 + 10 * np.sin(t * .48)
    cy = h * 0.48 + 7 * np.cos(t * .36)
    wx, wy = 53., 88.
    beam = 170 * np.exp(-2 * ((xx - cx) ** 2 / wx ** 2 + (yy - cy) ** 2 / wy ** 2))
    noise = rng.normal(0, 1.1, size=(h, w))
    ch_r = np.clip(9 + beam + noise, 0, 255).astype(np.uint8)
    ch_g = np.clip(8 + beam * .65 + noise, 0, 255).astype(np.uint8)
    ch_b = np.clip(9 + beam * .28 + noise, 0, 255).astype(np.uint8)
    return np.stack([ch_b, ch_g, ch_r], axis=2)


class App:
    def __init__(self, root: tk.Tk, demo: bool = False):
        self.root = root
        self.root.title(APP_TITLE + (" — ДЕМО" if demo else ""))
        self.root.geometry("1410x870")
        self.root.minsize(1060, 710)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.configure(bg="#edf1f6")

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.cmd_queue: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        self.worker: CameraWorker | None = None
        self.retiring_worker: CameraWorker | None = None
        self.camera_streaming = False
        self.camera_frame_shape: tuple[int, ...] | None = None
        self.demo = demo
        self.roi: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.last_frame: np.ndarray | None = None
        self.last_result: BeamAnalysis | None = None
        self.dark_image: np.ndarray | None = None
        self.dark_pending = 0
        self.dark_accumulator: np.ndarray | None = None
        self.image_photo = None
        self.canvas_scale = 1.0
        self.canvas_xoffset = 0
        self.canvas_yoffset = 0
        self.last_plot_time = 0.0
        self.last_processing_time = 0.0
        self.last_display_time = 0.0
        self.fps_counter = 0
        self.fps_start = time.monotonic()
        self.stream_fps = 0.0
        self.initialize_vars()
        self.build_ui()
        if demo:
            self.status("ДЕМО-РЕЖИМ: искусственный гауссов пучок, не реальные данные")
        else:
            self.status("Выбери устройство и нажми «Подключить». Для теста: Демо")
        self.root.after(50, self.tick)

    def initialize_vars(self):
        self.cam_id = tk.StringVar(value="0")
        self.backend = tk.StringVar(value="Авто")
        self.resolution = tk.StringVar(value="1280×720")
        self.channel = tk.StringVar(value="Red")
        self.pixel_pitch = tk.StringVar(value="4,18")
        self.wavelength_nm = tk.StringVar(value="780")
        self.z_mm = tk.StringVar(value="0")
        self.record_feedback = tk.StringVar(value="")
        self.sat_level = tk.IntVar(value=250)
        self.exposure = tk.StringVar(value="-12")
        self.gain = tk.StringVar(value="0")
        self.camera_readback = tk.StringVar(value=NO_CAMERA_READBACK)
        self.camera_control_result = tk.StringVar(value="")
        self.radius_x = tk.StringVar(value="—")
        self.radius_y = tk.StringVar(value="—")
        self.fit_quality = tk.StringVar(value="Фит: —")
        self.center_text = tk.StringVar(value="Центр: —")
        self.fit_text = tk.StringVar(value="R²: —")
        self.sat_text = tk.StringVar(value="Насыщение: —")
        self.status_text = tk.StringVar(value="")
        self.stream_info = tk.StringVar(value="Камера: нет данных")
        self.show_sat = tk.BooleanVar(value=True)
        self.show_roi = tk.BooleanVar(value=True)

    def build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Top.TFrame", background="#edf1f6")
        style.configure("Top.TLabel", background="#edf1f6")
        style.configure("Section.TLabelframe", padding=8)
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 17, "bold"))

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        camera_tab = ttk.Frame(self.tabs, style="Top.TFrame")
        self.tabs.add(camera_tab, text="Камера")
        self.propagation_tab = PropagationTab(self.tabs, self.wavelength_nm)
        self.tabs.add(self.propagation_tab, text="Фит пучка")

        top = ttk.Frame(camera_tab, style="Top.TFrame", padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        for text, var, width in [
            ("Камера ID", self.cam_id, 4),
        ]:
            ttk.Label(top, text=text, style="Top.TLabel").pack(side=tk.LEFT)
            ttk.Entry(top, textvariable=var, width=width).pack(side=tk.LEFT, padx=(4, 9))
        ttk.Label(top, text="Драйвер", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.backend, state="readonly", width=12,
                     values=["Авто", "DirectShow", "Media Foundation", "V4L2"]).pack(side=tk.LEFT, padx=(4, 9))
        ttk.Label(top, text="Размер", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.resolution, state="readonly", width=11,
                     values=["640×480", "1280×720", "1920×1080", "2592×1944"]).pack(side=tk.LEFT, padx=(4, 9))
        ttk.Button(top, text="▶ Подключить", command=self.connect).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="■ Остановить", command=self.disconnect).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Демо", command=self.start_demo).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Тёмный кадр", command=self.capture_dark).pack(side=tk.LEFT, padx=7)
        ttk.Button(top, text="Сохранить PNG/CSV", command=self.save_measurement).pack(side=tk.LEFT, padx=3)

        options = ttk.Frame(camera_tab, style="Top.TFrame", padding=(10, 0, 10, 10))
        options.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(options, text="Канал", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(options, textvariable=self.channel, state="readonly", width=10,
                     values=["Red", "Green", "Blue", "Gray", "Mean RGB"]).pack(side=tk.LEFT, padx=(3, 14))
        ttk.Label(options, text="Масштаб, мкм/пиксель", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Entry(options, textvariable=self.pixel_pitch, width=8).pack(side=tk.LEFT, padx=(3, 14))
        ttk.Label(options, text="Порог насыщения, 8 бит", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(options, from_=180, to=255, textvariable=self.sat_level, width=5).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Checkbutton(options, text="Красные пиксели", variable=self.show_sat).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(options, text="ROI", variable=self.show_roi).pack(side=tk.LEFT, padx=6)
        ttk.Button(options, text="Сброс ROI", command=self.reset_roi).pack(side=tk.LEFT, padx=6)

        exp_bar = ttk.Frame(camera_tab, style="Top.TFrame", padding=(10, 0, 10, 8))
        exp_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(exp_bar, text="Автоэкспозиция", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Button(exp_bar, text="Включить авто", command=lambda: self.request_auto_exposure(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(exp_bar, text="Выключить авто", command=lambda: self.request_auto_exposure(False)).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(exp_bar, text="Exposure", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Entry(exp_bar, textvariable=self.exposure, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Button(exp_bar, text="Задать", command=self.request_exposure).pack(side=tk.LEFT)
        ttk.Label(exp_bar, text="Gain", style="Top.TLabel").pack(side=tk.LEFT, padx=(12, 3))
        ttk.Entry(exp_bar, textvariable=self.gain, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Button(exp_bar, text="Задать", command=self.request_gain).pack(side=tk.LEFT)

        readback_bar = ttk.Frame(camera_tab, style="Top.TFrame", padding=(10, 0, 10, 8))
        readback_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(readback_bar, textvariable=self.camera_readback, style="Top.TLabel").pack(anchor=tk.W)
        ttk.Label(readback_bar, textvariable=self.camera_control_result, style="Top.TLabel",
                  foreground="#667085").pack(anchor=tk.W)

        record_bar = ttk.Frame(camera_tab, style="Top.TFrame", padding=(10, 0, 10, 8))
        record_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(record_bar, text="Длина волны λ, нм", style="Top.TLabel").pack(side=tk.LEFT)
        ttk.Entry(record_bar, textvariable=self.wavelength_nm, width=8).pack(side=tk.LEFT, padx=(5, 14))
        ttk.Label(record_bar, text="z, мм", style="Top.TLabel").pack(side=tk.LEFT)
        z_entry = ttk.Entry(record_bar, textvariable=self.z_mm, width=9)
        z_entry.pack(side=tk.LEFT, padx=5)
        z_entry.bind("<Return>", lambda _event: self.record_measurement())
        ttk.Button(record_bar, text="Записать", command=self.record_measurement).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(record_bar, textvariable=self.record_feedback, style="Top.TLabel").pack(side=tk.LEFT)

        content = ttk.Panedwindow(camera_tab, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=9, pady=(0, 5))
        left = ttk.LabelFrame(content, text="Камера · потяни мышью для выбора ROI", style="Section.TLabelframe")
        right = ttk.LabelFrame(content, text="Профиль лазерного пучка", style="Section.TLabelframe")
        content.add(left, weight=5)
        content.add(right, weight=5)

        self.video_canvas = tk.Canvas(left, bg="#111821", highlightthickness=0, cursor="crosshair")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<ButtonPress-1>", self.on_roi_start)
        self.video_canvas.bind("<B1-Motion>", self.on_roi_move)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_roi_end)
        self.video_canvas.bind("<Button-3>", lambda _e: self.reset_roi())
        ttk.Label(left, textvariable=self.stream_info).pack(anchor=tk.W, pady=(7, 0))

        self.figure = Figure(figsize=(6.7, 5.2), dpi=100, facecolor="#ffffff")
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.2, 1.0], hspace=.42, wspace=.35)
        self.ax_x = self.figure.add_subplot(grid[0, 0])
        self.ax_y = self.figure.add_subplot(grid[1, 0])
        self.ax_2d = self.figure.add_subplot(grid[:, 1])
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.clear_plots()

        metrics = ttk.Frame(camera_tab, style="Top.TFrame", padding=(12, 5))
        metrics.pack(fill=tk.X)
        for title, var in [("Радиус X (1/e²)", self.radius_x), ("Радиус Y (1/e²)", self.radius_y)]:
            cell = ttk.Frame(metrics, style="Top.TFrame")
            cell.pack(side=tk.LEFT, padx=(0, 25))
            ttk.Label(cell, text=title, style="Top.TLabel").pack(anchor=tk.W)
            ttk.Label(cell, textvariable=var, style="Metric.TLabel").pack(anchor=tk.W)
        ttk.Label(metrics, textvariable=self.fit_quality, style="Top.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(metrics, textvariable=self.center_text, style="Top.TLabel").pack(side=tk.LEFT, padx=12)
        ttk.Label(metrics, textvariable=self.fit_text, style="Top.TLabel").pack(side=tk.LEFT, padx=12)
        ttk.Label(metrics, textvariable=self.sat_text, style="Top.TLabel").pack(side=tk.LEFT, padx=12)
        ttk.Label(camera_tab, textvariable=self.status_text,
                  background="#e0e7ef", foreground="#253247", anchor=tk.W,
                  padding=(12, 8)).pack(side=tk.BOTTOM, fill=tk.X)

        self.video_canvas.create_text(20, 20, anchor=tk.NW, fill="white",
                                      text="Нет кадров. Подключи камеру или открой демо.")

    def status(self, message: str):
        self.status_text.set(message)

    def record_measurement(self):
        if self.last_frame is None or self.dark_pending:
            self.record_feedback.set("Дождись кадра с текущим профилем пучка")
            return
        try:
            wavelength = parse_optional_float(self.wavelength_nm.get())
            if wavelength is None or not math.isfinite(wavelength) or wavelength <= 0:
                raise ValueError()
        except ValueError:
            self.record_feedback.set("Укажи положительную длину волны в нм")
            return
        try:
            z = parse_optional_float(self.z_mm.get())
            if z is None or not math.isfinite(z):
                raise ValueError()
        except ValueError:
            self.record_feedback.set("Укажи координату z в мм")
            return
        try:
            pitch = parse_pixel_pitch(self.pixel_pitch.get())
            if pitch is None:
                raise ValueError()
        except ValueError:
            self.record_feedback.set("Укажи масштаб, чтобы записать диаметры в мкм")
            return

        self.analyze(self.last_frame)
        result = self.last_result
        if result is None or result.x_fit is None or result.y_fit is None:
            self.record_feedback.set("На текущем кадре нужны оценки обоих диаметров")
            return
        reliable = result.fit_reliable and not self.demo
        self.propagation_tab.add_measurement(z, result.x_fit.diameter_px * pitch,
                                             result.y_fit.diameter_px * pitch, reliable)
        message = "Записано во вкладку «Фит пучка»"
        if self.demo:
            message += " (демо)"
        elif not reliable:
            message += " (с предупреждением)"
        self.record_feedback.set(message)

    def clear_plots(self):
        for ax, title in ((self.ax_x, "Профиль X"), (self.ax_y, "Профиль Y"),
                          (self.ax_2d, "Карта интенсивности")):
            ax.clear()
            ax.set_title(title)
        self.plot_canvas.draw_idle()

    def clear_measurement_metrics(self):
        self.radius_x.set("—")
        self.radius_y.set("—")
        self.fit_quality.set("Фит: —")
        self.center_text.set("Центр: —")
        self.fit_text.set("R²: —")
        self.sat_text.set("Насыщение: —")

    def clear_stream(self, message: str):
        self.last_frame = None
        self.last_result = None
        self.dark_image = None
        self.dark_pending = 0
        self.dark_accumulator = None
        self.drag_start = None
        self.last_processing_time = 0.0
        self.last_display_time = 0.0
        self.last_plot_time = 0.0
        self.fps_counter = 0
        self.fps_start = time.monotonic()
        self.stream_fps = 0.0
        self.clear_measurement_metrics()
        self.stream_info.set(message)
        self.video_canvas.delete("all")
        self.image_photo = None
        self.video_canvas.create_text(20, 20, anchor=tk.NW, fill="white", text=message)
        self.clear_plots()

    def reset_roi(self):
        self.roi = None
        self.drag_start = None
        self.status("ROI сброшена: измеряется весь кадр. Выдели пятно и немного фона вокруг.")

    def connect(self):
        try:
            idx = int(self.cam_id.get().strip())
            if idx < 0:
                raise ValueError()
            width, height = [int(v) for v in self.resolution.get().split("×")]
        except (ValueError, AttributeError):
            self.status("Некорректный ID камеры или разрешение")
            return
        backend_value = {
            "Авто": cv2.CAP_ANY,
            "DirectShow": cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
            "Media Foundation": cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY,
            "V4L2": cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY,
        }[self.backend.get()]
        self.stop_camera_worker()
        self.demo = False
        self.roi = None
        self.camera_frame_shape = None
        # The previous worker can finish a native call after Stop was pressed.
        self.frame_queue = queue.Queue(maxsize=1)
        self.cmd_queue = queue.Queue()
        self.events = queue.Queue()
        self.worker = CameraWorker(idx, backend_value, (width, height),
                                   self.frame_queue, self.cmd_queue, self.events)
        message = f"Подключаю камеру {idx}…"
        self.clear_stream(message)
        self.status(message)
        self.start_camera_worker()

    def start_camera_worker(self):
        if self.retiring_worker is not None:
            if self.retiring_worker.is_alive():
                return
            self.retiring_worker = None
        if self.worker is not None and self.worker.ident is None:
            self.worker.start()

    def stop_camera_worker(self):
        if self.worker is not None:
            self.worker.please_stop.set()
            if self.worker.is_alive():
                self.retiring_worker = self.worker
            self.worker = None
        self.camera_streaming = False
        self.clear_camera_readback()

    def clear_camera_readback(self):
        self.camera_readback.set(NO_CAMERA_READBACK)
        self.camera_control_result.set("")

    def disconnect(self):
        self.stop_camera_worker()
        self.demo = False
        self.clear_stream("Камера остановлена")
        self.status("Камера остановлена")

    def start_demo(self):
        self.stop_camera_worker()
        self.demo = True
        self.roi = None
        self.clear_stream("Демо: подготовка кадров…")
        self.status("ДЕМО: модель эллиптического гауссова пятна, не физическое измерение")

    def capture_dark(self):
        if self.last_frame is None:
            self.status("Сначала получи кадр")
            return
        if self.demo:
            self.status("В демо тёмный кадр не нужен")
            return
        if not messagebox.askokcancel("Тёмный кадр", "Полностью перекрой лазер БЕЗОПАСНЫМ поглотителем, затем нажми OK.\nБудут усреднены 16 кадров."):
            return
        if self.last_frame is None or not self.camera_streaming:
            self.status("Дождись кадров с камеры и повтори запись тёмного кадра")
            return
        self.dark_pending = 16
        self.dark_accumulator = None
        self.status("Записываю 16 тёмных кадров. Лазер должен оставаться перекрытым.")

    def request_auto_exposure(self, enabled):
        if self.worker is not None:
            self.cmd_queue.put(("auto_exposure", enabled))
        else:
            self.status("Подключи камеру для управления экспозицией")

    def request_exposure(self):
        try:
            v = parse_optional_float(self.exposure.get())
            if v is None or not math.isfinite(v):
                raise ValueError()
        except ValueError:
            self.status("Некорректное значение Exposure")
            return
        if self.worker is not None:
            self.cmd_queue.put(("exposure", v))
        else:
            self.status("Подключи камеру")

    def request_gain(self):
        try:
            v = parse_optional_float(self.gain.get())
            if v is None or not math.isfinite(v):
                raise ValueError()
        except ValueError:
            self.status("Некорректное значение Gain")
            return
        if self.worker is not None:
            self.cmd_queue.put(("gain", v))
        else:
            self.status("Подключи камеру")

    def canvas_to_pixel(self, x: int, y: int) -> tuple[int, int] | None:
        if self.last_frame is None:
            return None
        h, w = self.last_frame.shape[:2]
        px = int((x - self.canvas_xoffset) / self.canvas_scale)
        py = int((y - self.canvas_yoffset) / self.canvas_scale)
        return max(0, min(px, w - 1)), max(0, min(py, h - 1))

    def on_roi_start(self, event):
        self.drag_start = self.canvas_to_pixel(event.x, event.y)

    def on_roi_move(self, event):
        if self.drag_start is None:
            return
        pt = self.canvas_to_pixel(event.x, event.y)
        if pt is None:
            return
        self.roi = (min(self.drag_start[0], pt[0]), min(self.drag_start[1], pt[1]),
                    max(self.drag_start[0], pt[0]), max(self.drag_start[1], pt[1]))
        if self.last_frame is not None:
            self.show_video(self.last_frame, self.last_result)

    def on_roi_end(self, event):
        if self.drag_start is None:
            return
        pt = self.canvas_to_pixel(event.x, event.y)
        if pt is not None:
            xa, ya = self.drag_start
            xb, yb = pt
            self.roi = (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))
            if self.roi[2] - self.roi[0] < 12 or self.roi[3] - self.roi[1] < 12:
                self.roi = None
        self.drag_start = None

    def tick(self):
        self.start_camera_worker()
        if self.worker is not None:
            while True:
                try:
                    event, msg = self.events.get_nowait()
                except queue.Empty:
                    break
                if event == "camera_readback":
                    self.camera_readback.set(msg)
                    continue
                if event == "control_result":
                    self.camera_control_result.set(msg)
                    continue
                if event == "retrying":
                    if self.camera_streaming or self.stream_info.get() != msg:
                        self.clear_stream(msg)
                    self.camera_streaming = False
                    self.clear_camera_readback()
                elif event == "streaming":
                    self.camera_streaming = True
                self.status(msg)

        now = time.monotonic()
        packet = None
        if self.demo and now - self.last_display_time > 0.075:
            packet = (now, make_synthetic_frame(now))
        elif self.camera_streaming:
            try:
                packet = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        if packet is not None:
            _, frame = packet
            if not self.demo:
                if self.camera_frame_shape is not None and self.camera_frame_shape != frame.shape:
                    self.roi = None
                    self.pixel_pitch.set("")
                    self.clear_stream("Размер или формат кадра изменился")
                self.camera_frame_shape = frame.shape
            self.last_frame = frame
            self.fps_counter += 1
            if now - self.fps_start >= 1.5:
                self.stream_fps = self.fps_counter / (now - self.fps_start)
                self.fps_start = now
                self.fps_counter = 0
            self.stream_info.set(f"{frame.shape[1]}×{frame.shape[0]} px · {self.stream_fps:.1f} кадров/с в интерфейсе · правая кнопка мыши: сброс ROI")

            if self.dark_pending > 0:
                f = frame.astype(np.float64)
                self.dark_accumulator = f if self.dark_accumulator is None else self.dark_accumulator + f
                self.dark_pending -= 1
                if self.dark_pending == 0:
                    self.dark_image = (self.dark_accumulator / 16).astype(np.float32)
                    self.dark_accumulator = None
                    self.status("Тёмный кадр сохранён. Открой лазерный пучок при безопасном ослаблении.")
            if now - self.last_processing_time >= .14 and self.dark_pending == 0:
                self.analyze(frame)
                self.last_processing_time = now
            if now - self.last_display_time >= .07:
                self.show_video(frame, self.last_result)
                self.last_display_time = now
        self.root.after(35, self.tick)

    def analyze(self, frame):
        try:
            result = analyze_frame(frame, roi=self.roi,
                                   mode=self.channel.get(), dark_frame=self.dark_image,
                                   saturation_threshold=float(self.sat_level.get()))
        except (ValueError, TypeError, tk.TclError) as exc:
            self.last_result = None
            self.clear_measurement_metrics()
            self.status(f"Измерение: {exc}")
            return
        self.last_result = result
        try:
            pitch = parse_pixel_pitch(self.pixel_pitch.get())
        except ValueError:
            pitch = None
            self.status("Неверный масштаб (мкм/px): показываю пиксели")
        unit = "мкм" if pitch is not None else "px"
        factor = pitch if pitch is not None else 1.0
        for value, fit in ((self.radius_x, result.x_fit), (self.radius_y, result.y_fit)):
            value.set(f"{fit.radius_px * factor:.1f} {unit}" if fit is not None else "—")
        if result.x_fit is None and result.y_fit is None:
            self.fit_quality.set("Фит: нет оценки")
        else:
            self.fit_quality.set("Фит: достоверно" if result.fit_reliable else "Фит: недостоверно")
        if result.x_fit and result.y_fit:
            self.center_text.set(f"Центр: ({result.x_fit.center_px:.1f}, {result.y_fit.center_px:.1f}) px")
            self.fit_text.set(f"R²: X={result.x_fit.r_squared:.4f} Y={result.y_fit.r_squared:.4f}")
        else:
            self.center_text.set("Центр: —")
            self.fit_text.set("R²: —")
        self.sat_text.set(f"≥ порога: {result.saturated_count} px ({result.saturation_percent:.3f}%)")
        if result.warning:
            self.status(result.warning)
        elif not self.demo:
            self.status("Гауссов фит прошёл проверки")
        if time.monotonic() - self.last_plot_time > .3:
            self.refresh_plots(result, pitch)
            self.last_plot_time = time.monotonic()

    def show_video(self, frame: np.ndarray, result: BeamAnalysis | None):
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = rgb.copy()
        if result is not None and self.show_sat.get():
            x0, y0, x1, y1 = result.roi
            view = rgb[y0:y1, x0:x1]
            # Solid red for pixels over clipping threshold.
            view[result.saturated_mask] = (255, 36, 38)
        h, w = rgb.shape[:2]
        cw = max(self.video_canvas.winfo_width(), 2)
        ch = max(self.video_canvas.winfo_height(), 2)
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        self.canvas_scale = scale
        self.canvas_xoffset = (cw - nw) // 2
        self.canvas_yoffset = (ch - nh) // 2
        mode = Image.Resampling.BILINEAR if scale < 1 else Image.Resampling.NEAREST
        photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((nw, nh), mode))
        self.image_photo = photo
        self.video_canvas.delete("all")
        self.video_canvas.create_image(self.canvas_xoffset, self.canvas_yoffset,
                                       anchor=tk.NW, image=photo)
        if self.roi is not None and self.show_roi.get():
            x0, y0, x1, y1 = self.roi
            self.video_canvas.create_rectangle(
                self.canvas_xoffset + x0 * scale, self.canvas_yoffset + y0 * scale,
                self.canvas_xoffset + x1 * scale, self.canvas_yoffset + y1 * scale,
                outline="#4bfcff", width=2)
        if result is not None and result.x_fit and result.y_fit:
            cx = self.canvas_xoffset + result.x_fit.center_px * scale
            cy = self.canvas_yoffset + result.y_fit.center_px * scale
            self.video_canvas.create_line(cx - 8, cy, cx + 8, cy, fill="#60f4bb", width=1)
            self.video_canvas.create_line(cx, cy - 8, cx, cy + 8, fill="#60f4bb", width=1)
        self.video_canvas.create_text(self.canvas_xoffset + 10, self.canvas_yoffset + 10,
                                     anchor=tk.NW, fill="white",
                                     text="Красный = порог насыщения · 1/e²",
                                     font=("Segoe UI", 10, "bold"))

    def refresh_plots(self, result: BeamAnalysis, pitch: float | None):
        for ax in (self.ax_x, self.ax_y, self.ax_2d):
            ax.clear()
        factor = pitch if pitch is not None else 1.0
        unit = "мкм" if pitch is not None else "px"
        for ax, fit, label in [(self.ax_x, result.x_fit, "X"),
                               (self.ax_y, result.y_fit, "Y")]:
            if fit is not None:
                xx = fit.x * factor
                ax.plot(xx, fit.y, color="#245c9e", linewidth=1.5, label="Данные")
                ax.plot(xx, fit.fitted, color="#db612a", linestyle="--", linewidth=2,
                        label="Гаусс + фон")
                ax.plot(xx, fit.baseline, color="#697386", linestyle=":", linewidth=1.5,
                        label="Линейный фон")
                title = f"{label}: D = {fit.diameter_px * factor:.1f} {unit}"
                if not result.fit_reliable:
                    title += " (недостоверно)"
                ax.set_title(title, fontsize=10)
                ax.legend(loc="upper right", fontsize=7)
            else:
                ax.set_title(f"{label}: нет фита", fontsize=10)
            ax.set_xlabel(f"Координата, {unit}", fontsize=8)
            ax.set_ylabel("Σ интенсивность, отн. ед.", fontsize=8)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=.2)

        x0, y0, x1, y1 = result.roi
        # Downsample the heatmap but preserve the spatial aspect ratio.
        data = result.corrected_roi
        step = max(1, int(max(data.shape) / 480))
        img = data[::step, ::step]
        self.ax_2d.imshow(img, cmap="inferno", origin="upper", aspect="equal",
                          extent=[x0 * factor, (x0 + img.shape[1] * step) * factor,
                                  (y0 + img.shape[0] * step) * factor, y0 * factor])
        self.ax_2d.set_title("2D интенсивность", fontsize=10)
        self.ax_2d.set_xlabel(f"X, {unit}", fontsize=8)
        self.ax_2d.set_ylabel(f"Y, {unit}", fontsize=8)
        self.ax_2d.tick_params(axis="both", labelsize=8)
        self.plot_canvas.draw_idle()

    def save_measurement(self):
        if self.last_frame is None or self.last_result is None or self.dark_pending:
            messagebox.showwarning("Нет данных", "Сначала получи кадр с камеры")
            return
        folder = filedialog.askdirectory(title="Папка для измерения")
        if not folder:
            return
        # The folder dialog can run Tk callbacks, including a camera disconnect.
        frame = self.last_frame
        if frame is None or self.dark_pending:
            self.status("Сохранение: нет текущего измерения")
            return
        self.analyze(frame)
        result = self.last_result
        if result is None:
            return
        prefix = datetime.now().strftime("beam_%Y%m%d_%H%M%S")
        base = Path(folder) / prefix
        if not cv2.imwrite(str(base) + "_camera.png", frame):
            messagebox.showerror("Сохранение", "Не удалось записать PNG с камеры")
            return
        np.save(str(base) + "_background_subtracted.npy", result.corrected_roi)
        try:
            pitch = parse_pixel_pitch(self.pixel_pitch.get())
        except ValueError:
            pitch = None
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "demo_mode": self.demo,
            "channel": self.channel.get(),
            "frame_shape": list(frame.shape),
            "roi_xyxy_px": list(result.roi),
            "pixel_scale_um_per_px": pitch,
            "pixel_scale_note": "Needed for microns; calibrated output-pixel pitch, not nominal sensor pitch if frame is resized",
            "saturation_threshold": self.sat_level.get(),
            "saturated_pixels": result.saturated_count,
            "saturated_percent": result.saturation_percent,
            "fit_reliable": result.fit_reliable,
            "warnings": result.warning,
            "background_level": result.background,
            "profile_model": "Gaussian plus linear background",
            "radius_x_px": result.x_fit.radius_px if result.x_fit else None,
            "radius_y_px": result.y_fit.radius_px if result.y_fit else None,
            "radius_x_um": result.x_fit.radius_px * pitch if pitch and result.x_fit else None,
            "radius_y_um": result.y_fit.radius_px * pitch if pitch and result.y_fit else None,
            "diameter_x_px": result.x_fit.diameter_px if result.x_fit else None,
            "diameter_y_px": result.y_fit.diameter_px if result.y_fit else None,
            "diameter_x_um": result.x_fit.diameter_px * pitch if pitch and result.x_fit else None,
            "diameter_y_um": result.y_fit.diameter_px * pitch if pitch and result.y_fit else None,
            "center_x_px": result.x_fit.center_px if result.x_fit else None,
            "center_y_px": result.y_fit.center_px if result.y_fit else None,
            "r2_x": result.x_fit.r_squared if result.x_fit else None,
            "r2_y": result.y_fit.r_squared if result.y_fit else None,
        }
        with open(str(base) + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        for tag, fit in (("x", result.x_fit), ("y", result.y_fit)):
            if fit is None:
                continue
            with open(str(base) + f"_profile_{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["coordinate_pixel", "coordinate_um", "intensity_sum", "gaussian_fit", "linear_background"])
                for x, y, yf, baseline in zip(fit.x, fit.y, fit.fitted, fit.baseline):
                    writer.writerow([float(x), float(x * pitch) if pitch else "", float(y), float(yf), float(baseline)])
        self.refresh_plots(result, pitch)
        self.last_plot_time = time.monotonic()
        self.figure.savefig(str(base) + "_plots.png", dpi=160)
        self.status(f"Измерение сохранено: {base.name} (PNG, CSV, NPY, JSON)")

    def on_close(self):
        self.stop_camera_worker()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="USB laser beam profiler")
    parser.add_argument("--demo", action="store_true", help="Synthetic test without camera")
    args = parser.parse_args()
    root = tk.Tk()
    App(root, demo=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
