"""Reusable pyqtgraph widgets for ECG waveform display."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .ecg_grid import apply_ecg_grid
from .utils import DEFAULT_VIEW_SECONDS, DISPLAY_SIGNALS, FS


class MultiLeadECGPlotWidget(pg.PlotWidget):
    point_clicked = QtCore.Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.lead_offsets: dict[str, float] = {}
        self.signals: dict[str, np.ndarray] = {}
        self.lead_spacing = 1.2
        self.pending_marker_items: list[object] = []
        self.annotation_items: list[object] = []
        self.plotItem.setTitle("10导联同步心电图")
        self.plotItem.setLabel("bottom", "时间", units="s")
        self.plotItem.setLabel("left", "导联")
        self.plotItem.setMenuEnabled(False)
        self.setMenuEnabled(False)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def draw(
        self,
        t: np.ndarray,
        signals: dict[str, np.ndarray],
        r_peaks: dict[str, np.ndarray],
        fs: int = FS,
        reset_view: bool = True,
    ) -> None:
        previous_range = self.plotItem.vb.viewRange()
        self.plotItem.clear()
        self.pending_marker_items = []
        self.annotation_items = []
        if not signals:
            return

        self.signals = signals
        duration_seconds = float(t[-1]) if t.size else DEFAULT_VIEW_SECONDS
        display_order = [name for name in DISPLAY_SIGNALS if name in signals]
        spacing = self._lead_spacing(signals)
        self.lead_spacing = spacing
        self.lead_offsets = {
            signal_type: float((len(display_order) - 1 - index) * spacing)
            for index, signal_type in enumerate(display_order)
        }
        y_min = -spacing * 0.7
        y_max = len(display_order) * spacing - spacing * 0.3
        apply_ecg_grid(self.plotItem, 0.0, duration_seconds, y_min, y_max, DEFAULT_VIEW_SECONDS)

        axis = self.plotItem.getAxis("left")
        axis.setTicks([[(offset, name) for name, offset in self.lead_offsets.items()]])

        for signal_type in display_order:
            y = np.asarray(signals[signal_type], dtype=float)
            offset = self.lead_offsets[signal_type]
            color = self._wave_color(signal_type)
            self.plotItem.plot(t, y + offset, pen=pg.mkPen(color, width=1.25))

            peaks = np.asarray(r_peaks.get(signal_type, []), dtype=int)
            peaks = peaks[(peaks >= 0) & (peaks < y.size)]
            if peaks.size:
                peak_color = "#2563eb" if "FECG" in signal_type else "#e11d48"
                self.plotItem.plot(
                    peaks / fs,
                    y[peaks] + offset,
                    pen=None,
                    symbol="o",
                    symbolSize=7,
                    symbolBrush=pg.mkBrush(peak_color),
                    symbolPen=pg.mkPen("#ffffff", width=1),
                )
        if not reset_view:
            self.plotItem.setRange(xRange=previous_range[0], yRange=previous_range[1], padding=0)

    def reset_initial_view(self) -> None:
        x_max = DEFAULT_VIEW_SECONDS
        if self.signals:
            sample_count = max(len(values) for values in self.signals.values())
            duration = max(0.0, (sample_count - 1) / FS)
            x_max = min(DEFAULT_VIEW_SECONDS, duration) if duration > 0 else DEFAULT_VIEW_SECONDS
        self.plotItem.setXRange(0.0, x_max, padding=0)
        self.plotItem.enableAutoRange(axis="y")

    def current_view_range(self) -> list[list[float]]:
        return self.plotItem.vb.viewRange()

    def restore_view_range(self, view_range: list[list[float]]) -> None:
        self.plotItem.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    def draw_annotations(
        self,
        annotation_items,
        highlighted_index: int | None = None,
        show_labels: bool = True,
    ) -> None:
        self.clear_annotations()
        for item in annotation_items:
            if isinstance(item, tuple) and len(item) == 2:
                row_index, record = item
            else:
                row_index, record = None, item
            if record.signal_type not in self.lead_offsets:
                continue
            highlighted = row_index is not None and row_index == highlighted_index
            if record.start_time is not None and record.end_time is not None:
                self.add_time_span(
                    record.signal_type,
                    float(record.start_time),
                    float(record.end_time),
                    record.annotation_type,
                    getattr(record, "subject", ""),
                    row_index,
                    highlighted,
                    show_labels,
                )
            if (
                record.baseline_time is not None
                and record.baseline_value is not None
                and record.peak_time is not None
                and record.peak_value is not None
            ):
                self.add_amplitude_line(
                    record.signal_type,
                    float(record.baseline_time),
                    float(record.baseline_value),
                    float(record.peak_time),
                    float(record.peak_value),
                    record.annotation_type,
                    getattr(record, "subject", ""),
                    record.start_time,
                    record.end_time,
                    highlighted,
                )

    def raw_value_for_lead(self, signal_type: str, plotted_y: float) -> float:
        return plotted_y - self.lead_offsets.get(signal_type, 0.0)

    def add_time_span(
        self,
        signal_type: str,
        start_time: float,
        end_time: float,
        annotation_type: str,
        subject: str = "",
        row_index: int | None = None,
        highlighted: bool = False,
        show_label: bool = True,
    ) -> None:
        offset = self.lead_offsets.get(signal_type, 0.0)
        values = self.signals.get(signal_type)
        if values is None or len(values) == 0:
            return
        color = self._annotation_color(annotation_type, subject)
        low, high = self._segment_bounds(signal_type, start_time, end_time)
        rect = QtWidgets.QGraphicsRectItem(
            QtCore.QRectF(
                min(start_time, end_time),
                offset + low,
                abs(end_time - start_time),
                high - low,
            )
        )
        rect.setBrush(pg.mkBrush(color[0], color[1], color[2], 42))
        rect.setPen(pg.mkPen("#111827" if highlighted else color, width=2.4 if highlighted else 1.0))
        self.plotItem.addItem(rect)
        self.annotation_items.append(rect)
        top_line = self.plotItem.plot(
            [start_time, end_time],
            [offset + high, offset + high],
            pen=pg.mkPen("#111827" if highlighted else color, width=3 if highlighted else 2),
        )
        self.annotation_items.append(top_line)
        if show_label and row_index is not None:
            label = pg.TextItem(
                text=f"#{row_index + 1} {annotation_type}",
                color=pg.mkColor("#111827" if highlighted else color),
                anchor=(0.5, 1.0),
            )
            label.setPos((start_time + end_time) / 2.0, offset + high)
            self.plotItem.addItem(label)
            self.annotation_items.append(label)

    def add_annotation_line(self, signal_type: str, start_time: float, end_time: float, y_value: float) -> None:
        y = y_value + self.lead_offsets.get(signal_type, 0.0)
        self.plotItem.plot(
            [start_time, end_time],
            [y, y],
            pen=pg.mkPen("#7c3aed", width=2),
            symbol="x",
            symbolSize=8,
            symbolBrush=pg.mkBrush("#7c3aed"),
        )

    def add_amplitude_line(
        self,
        signal_type: str,
        baseline_time: float,
        baseline_value: float,
        peak_time: float,
        peak_value: float,
        annotation_type: str = "幅度",
        subject: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
        highlighted: bool = False,
    ) -> None:
        offset = self.lead_offsets.get(signal_type, 0.0)
        color = self._annotation_color(annotation_type, subject)
        pen_color = "#111827" if highlighted else color
        symbol = "p" if "P波" in annotation_type else "t" if "T波" in annotation_type else "o"
        if start_time is not None and end_time is not None:
            left, right = sorted((float(start_time), float(end_time)))
            baseline_time = float(np.clip(baseline_time, left, right))
            peak_time = float(np.clip(peak_time, left, right))
        y1 = baseline_value + offset
        y2 = peak_value + offset
        line = self.plotItem.plot(
            [baseline_time, peak_time],
            [y1, y2],
            pen=pg.mkPen(pen_color, width=3 if highlighted else 2),
            symbol=symbol,
            symbolSize=10 if highlighted else 8,
            symbolBrush=pg.mkBrush(pen_color),
        )
        self.annotation_items.append(line)

    def clear_annotations(self) -> None:
        for item in self.annotation_items:
            try:
                self.plotItem.removeItem(item)
            except Exception:
                pass
        self.annotation_items = []

    def add_pending_marker(self, signal_type: str, time_seconds: float, label_text: str = "第1点") -> None:
        self.clear_pending_marker()
        offset = self.lead_offsets.get(signal_type, 0.0)
        low, high = self._lead_bounds(signal_type)
        line = pg.PlotDataItem(
            [time_seconds, time_seconds],
            [offset + low, offset + high],
            pen=pg.mkPen("#0891b2", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        label = pg.TextItem(text=label_text, color=pg.mkColor("#0891b2"), anchor=(0.0, 1.0))
        label.setPos(time_seconds, offset + high)
        self.plotItem.addItem(line)
        self.plotItem.addItem(label)
        self.pending_marker_items = [line, label]

    def clear_pending_marker(self) -> None:
        for item in self.pending_marker_items:
            try:
                self.plotItem.removeItem(item)
            except Exception:
                pass
        self.pending_marker_items = []

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plotItem.vb.mapSceneToView(event.scenePos())
        self.point_clicked.emit(float(point.x()), float(point.y()))

    @staticmethod
    def _wave_color(signal_type: str) -> str:
        if "FECG" in signal_type:
            return "#1d4ed8"
        if "MECG" in signal_type:
            return "#111827"
        return "#047857"

    @staticmethod
    def _annotation_color(annotation_type: str, subject: str = "") -> tuple[int, int, int]:
        fetal_shift = 55 if subject == "胎儿" else 0
        if "P波" in annotation_type:
            return (126, 34 + fetal_shift, 206)
        if "T波" in annotation_type:
            return (234, 88 + fetal_shift // 2, 12)
        if "QRS" in annotation_type or "R峰" in annotation_type:
            return (220, 38, 38)
        return (5, 150, 105)

    @staticmethod
    def _local_band_height(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=float)
        if y.size == 0:
            return 0.0
        return float(np.nanpercentile(y, 85))

    def _segment_bounds(self, signal_type: str, start_time: float, end_time: float) -> tuple[float, float]:
        values = self.signals.get(signal_type)
        if values is None or len(values) == 0:
            return -0.1, 0.1
        left, right = sorted((start_time, end_time))
        start_index = max(0, int(np.floor(left * FS)))
        end_index = min(len(values), int(np.ceil(right * FS)) + 1)
        segment = np.asarray(values[start_index:end_index], dtype=float)
        if segment.size == 0:
            center = 0.0
            pad = 0.08
            return center - pad, center + pad
        low = float(np.nanmin(segment))
        high = float(np.nanmax(segment))
        span = max(high - low, 0.04)
        pad = max(span * 0.18, 0.025)
        return low - pad, high + pad

    def _lead_bounds(self, signal_type: str) -> tuple[float, float]:
        values = self.signals.get(signal_type)
        if values is None or len(values) == 0:
            return -0.5, 0.5
        y = np.asarray(values, dtype=float)
        low = float(np.nanpercentile(y, 1))
        high = float(np.nanpercentile(y, 99))
        span = max(high - low, 0.5)
        pad = max(span * 0.15, 0.12)
        return low - pad, high + pad

    @staticmethod
    def _lead_spacing(signals: dict[str, np.ndarray]) -> float:
        max_span = 1.0
        for values in signals.values():
            y = np.asarray(values, dtype=float)
            if y.size:
                span = float(np.nanpercentile(y, 99) - np.nanpercentile(y, 1))
                max_span = max(max_span, span)
        return max(1.2, max_span * 1.45)
