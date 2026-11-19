"""Editable axial measurements and quadratic beam-propagation fits."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beam_propagation import PropagationFit, fit_propagation


class PropagationTab(ttk.Frame):
    PARAMETERS = (
        ("z0_mm", "z₀, мм"),
        ("d0_um", "D₀, мкм"),
        ("theta_mrad", "θ, мрад (полуугол)"),
        ("m_squared", "M²"),
        ("rayleigh_ideal_mm", "zᵣ (πD₀²/4λ), мм"),
        ("rayleigh_fit_mm", "zᵣ по кривизне фита, мм"),
        ("r_squared", "R² фита"),
    )

    def __init__(self, parent: tk.Misc, wavelength_var: tk.StringVar):
        super().__init__(parent, padding=8)
        self.wavelength_var = wavelength_var
        self.status = tk.StringVar(master=self)
        self._editor: ttk.Entry | None = None
        self._editing: tuple[str, int] | None = None
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_table()
        self._build_results()
        self._invalidate_fit()
        self._wavelength_trace = wavelength_var.trace_add("write", self._wavelength_changed)

    @property
    def row_count(self) -> int:
        return len(self.table.get_children())

    def _build_table(self) -> None:
        panel = ttk.LabelFrame(self, text="Точки измерения", padding=8)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(panel, columns=("z", "dx", "dy"), show="headings", height=12)
        for column, title in zip(("z", "dx", "dy"), ("z, мм", "Dₓ, мкм", "Dᵧ, мкм")):
            self.table.heading(column, text=title)
            self.table.column(column, width=102, minwidth=75, anchor="e", stretch=False)
        self.table.tag_configure("unreliable", foreground="#9a4f09")
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self._scroll_table)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind("<Double-1>", self._edit_clicked_cell)
        self.table.bind("<Return>", self._edit_selected_row)
        self.table.bind("<Delete>", self._delete_selected)
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Configure>"):
            self.table.bind(event, self._finish_editor_on_scroll, add=True)

        toolbar = ttk.Frame(panel)
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(toolbar, text="Добавить строку", command=self._add_manual_row).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Удалить", command=self._delete_selected).pack(side=tk.LEFT, padx=6)
        ttk.Label(panel, text="Двойной щелчок по ячейке для правки").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(panel, text="Построить фит пучка", command=self.fit_measurements).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

    def _build_results(self) -> None:
        panel = ttk.Frame(self)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        self.results = ttk.Treeview(
            panel, columns=("parameter", "x", "y"), show="headings",
            height=len(self.PARAMETERS), selectmode="none",
        )
        for column, text, width in (("parameter", "Параметр", 220), ("x", "X", 130), ("y", "Y", 130)):
            self.results.heading(column, text=text)
            self.results.column(column, width=width, anchor="w" if column == "parameter" else "e")
        for key, label in self.PARAMETERS:
            self.results.insert("", tk.END, iid=key, values=(label, "—", "—"))
        self.results.grid(row=0, column=0, sticky="ew")

        figure = Figure(figsize=(5.6, 3.1), dpi=100, layout="constrained", facecolor="white")
        self.axis = figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(figure, master=panel)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=8)

        footer = ttk.Label(panel, textvariable=self.status, justify="left", wraplength=560)
        footer.grid(row=2, column=0, sticky="ew")
        panel.bind("<Configure>", lambda event: footer.configure(wraplength=max(240, event.width - 8)))

    @staticmethod
    def _format_value(value: float, precision: int = 7) -> str:
        return f"{value:.{precision}g}".replace(".", ",")

    @staticmethod
    def _parse_value(text: str) -> float:
        value = float(text.strip().replace(",", ".").replace("−", "-"))
        if not math.isfinite(value):
            raise ValueError
        return value

    def add_measurement(self, z_mm: float, dx_um: float, dy_um: float, reliable: bool = True) -> int:
        """Append one measurement while keeping unreliable camera fits visible."""
        self._finish_editor(save=True)
        values = tuple(self._format_value(value, precision=12) for value in (z_mm, dx_um, dy_um))
        tags = () if reliable else ("unreliable",)
        row = self.table.insert("", tk.END, values=values, tags=tags)
        self.table.selection_set(row)
        self.table.see(row)
        self._invalidate_fit()
        return self.row_count

    def _add_manual_row(self) -> None:
        self._finish_editor(save=True)
        row = self.table.insert("", tk.END, values=("", "", ""))
        self._invalidate_fit()
        self._open_editor(row, 0)

    def _delete_selected(self, _event: tk.Event | None = None) -> str:
        self._finish_editor(save=True)
        rows = self.table.selection()
        if rows:
            self.table.delete(*rows)
            self._invalidate_fit()
        return "break"

    def _edit_clicked_cell(self, event: tk.Event) -> str | None:
        if self.table.identify_region(event.x, event.y) == "cell":
            row = self.table.identify_row(event.y)
            column = int(self.table.identify_column(event.x)[1:]) - 1
            self._open_editor(row, column)
            return "break"
        return None

    def _edit_selected_row(self, _event: tk.Event) -> str:
        rows = self.table.selection()
        if rows:
            self._open_editor(rows[0], 0)
        return "break"

    def _open_editor(self, row: str, column: int) -> None:
        self._finish_editor(save=True)
        self.table.focus(row)
        self.table.selection_set(row)
        self.table.see(row)
        self.table.update_idletasks()
        bounds = self.table.bbox(row, f"#{column + 1}")
        if not bounds:
            return
        x, y, width, height = bounds
        entry = ttk.Entry(self.table, justify="right")
        entry.insert(0, self.table.item(row, "values")[column])
        entry.place(x=x, y=y, width=width, height=height)
        self._editor = entry
        self._editing = (row, column)
        entry.selection_range(0, tk.END)
        entry.focus_set()
        entry.bind("<Return>", self._save_editor)
        entry.bind("<Escape>", self._cancel_editor)
        entry.bind("<FocusOut>", lambda _event: self._finish_editor(save=True))
        entry.bind("<Tab>", lambda _event: self._edit_adjacent_cell(1))
        entry.bind("<Shift-Tab>", lambda _event: self._edit_adjacent_cell(-1))
        entry.bind("<ISO_Left_Tab>", lambda _event: self._edit_adjacent_cell(-1))

    def _finish_editor(self, save: bool) -> None:
        if self._editor is None or self._editing is None:
            return
        entry, (row, column) = self._editor, self._editing
        self._editor = self._editing = None
        value = entry.get().strip()
        entry.destroy()
        if save and self.table.exists(row):
            values = list(self.table.item(row, "values"))
            if value != str(values[column]):
                values[column] = value
                self.table.item(row, values=values)
                self._invalidate_fit()

    def _save_editor(self, _event: tk.Event) -> str:
        self._finish_editor(save=True)
        self.table.focus_set()
        return "break"

    def _cancel_editor(self, _event: tk.Event) -> str:
        self._finish_editor(save=False)
        self.table.focus_set()
        return "break"

    def _edit_adjacent_cell(self, direction: int) -> str:
        if self._editing is None:
            return "break"
        row, column = self._editing
        rows = self.table.get_children()
        adjacent = rows.index(row) * 3 + column + direction
        self._finish_editor(save=True)
        if 0 <= adjacent < 3 * len(rows):
            next_row, next_column = divmod(adjacent, 3)
            self._open_editor(rows[next_row], next_column)
        else:
            self.table.focus_set()
        return "break"

    def _scroll_table(self, *args: str) -> None:
        self._finish_editor(save=True)
        self.table.yview(*args)

    def _finish_editor_on_scroll(self, _event: tk.Event) -> None:
        self._finish_editor(save=True)

    def _wavelength_changed(self, *_args: str) -> None:
        self._invalidate_fit()

    def _unreliable_note(self) -> str:
        count = sum("unreliable" in self.table.item(row, "tags") for row in self.table.get_children())
        return f"Строк с предупреждением, выделенных цветом — {count}." if count else ""

    def _clear_fit(self) -> None:
        for key, label in self.PARAMETERS:
            self.results.item(key, values=(label, "—", "—"))
        self.axis.clear()
        self.axis.set_xlabel("z, мм")
        self.axis.set_ylabel("D², мкм²")
        self.axis.grid(alpha=0.3)

    def _invalidate_fit(self) -> None:
        self._clear_fit()
        prompt = "Нажмите «Построить фит пучка»." if self.row_count else "Добавьте измерения при разных z."
        self.status.set(" ".join(filter(None, (prompt, self._unreliable_note()))))
        self.canvas.draw_idle()

    def _read_column(self, rows: tuple[str, ...], column: int, label: str) -> np.ndarray:
        values = []
        for index, row in enumerate(rows, start=1):
            cell = str(self.table.item(row, "values")[column])
            try:
                value = self._parse_value(cell)
            except ValueError as exc:
                raise ValueError(f"Строка {index} — введите число в столбце {label}.") from exc
            if column and value <= 0:
                raise ValueError(f"Строка {index} — диаметр {label} должен быть больше нуля.")
            values.append(value)
        return np.asarray(values)

    def fit_measurements(self) -> None:
        self._finish_editor(save=True)
        self._clear_fit()
        rows = self.table.get_children()
        try:
            try:
                wavelength = self._parse_value(self.wavelength_var.get())
            except ValueError as exc:
                raise ValueError("Введите положительную длину волны λ в нм на вкладке камеры.") from exc
            if wavelength <= 0:
                raise ValueError("Введите положительную длину волны λ в нм на вкладке камеры.")
            z_mm = self._read_column(rows, 0, "z")
            count_distinct = np.unique(z_mm).size
            if count_distinct < 3:
                raise ValueError("Для параболы нужны измерения хотя бы при трёх разных z.")
        except ValueError as exc:
            self.status.set(str(exc))
            self.canvas.draw_idle()
            return

        notes = []
        results: dict[str, PropagationFit] = {}
        for index, label, color, marker, style in (
            (1, "X", "#245c9e", "o", "-"), (2, "Y", "#b35021", "s", "--"),
        ):
            try:
                diameter = self._read_column(rows, index, f"D{label.lower()}")
                with np.errstate(over="ignore"):
                    squared = np.square(diameter)
                if not np.all(np.isfinite(squared)):
                    raise ValueError("Слишком большие значения диаметра.")
                self.axis.scatter(z_mm, squared, label=f"{label} — точки", color=color, marker=marker, s=25)
                fit = fit_propagation(z_mm, diameter, wavelength)
            except ValueError as exc:
                notes.append(f"{label} — {exc}")
                continue
            results[label] = fit
            z_curve = np.linspace(z_mm.min(), z_mm.max(), 250)
            self.axis.plot(
                z_curve, fit.predict_squared_diameter(z_curve),
                label=f"{label} — фит", color=color, linestyle=style,
            )
            if not fit.waist_within_range:
                notes.append(f"Перетяжка {label} вне диапазона измерений.")
            if fit.m_squared < 1 - 1e-8:
                notes.append(f"M² по {label} меньше 1. Проверьте измерения и масштаб.")

        for key, label in self.PARAMETERS:
            values = [
                self._format_value(getattr(results[direction], key)) if direction in results else "—"
                for direction in ("X", "Y")
            ]
            self.results.item(key, values=(label, *values))
        if self.axis.has_data():
            self.axis.legend(loc="best", ncols=2)
        if count_distinct == 3 and results:
            notes.append("Фит построен по трём разным z. Дополнительные позиции помогут проверить модель.")
        unreliable_note = self._unreliable_note()
        if unreliable_note:
            notes.append(unreliable_note)
        self.status.set(" ".join(notes) if notes else f"Точек в фите — {len(rows)}.")
        self.canvas.draw_idle()

    def destroy(self) -> None:
        self.wavelength_var.trace_remove("write", self._wavelength_trace)
        super().destroy()
