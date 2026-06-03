"""ECG paper style grid drawing for pyqtgraph plots."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


def apply_ecg_grid(
    plot_item: pg.PlotItem,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    initial_view_seconds: float = 10.0,
) -> None:
    """Draw clinical ECG-like grid lines and keep the initial view at 10 seconds."""
    plot_item.setClipToView(True)
    plot_item.setDownsampling(auto=True, mode="peak")
    plot_item.showGrid(x=False, y=False)
    plot_item.setLimits(xMin=x_min, xMax=x_max)
    plot_item.setRange(
        xRange=(x_min, min(x_max, x_min + initial_view_seconds)),
        yRange=(y_min, y_max),
        padding=0.02,
    )
    plot_item.getViewBox().setBackgroundColor("#fffafa")

    small_pen = pg.mkPen("#f8d7da", width=0.55)
    big_pen = pg.mkPen("#e7a6ad", width=0.9)

    plot_item.addItem(_vertical_grid_item(x_min, x_max, 0.04, y_min, y_max, small_pen))
    plot_item.addItem(_vertical_grid_item(x_min, x_max, 0.20, y_min, y_max, big_pen))
    plot_item.addItem(_horizontal_grid_item(y_min, y_max, 0.1, x_min, x_max, small_pen))
    plot_item.addItem(_horizontal_grid_item(y_min, y_max, 0.5, x_min, x_max, big_pen))


def _vertical_grid_item(x_min: float, x_max: float, step: float, y_min: float, y_max: float, pen) -> pg.PlotDataItem:
    xs = np.arange(x_min, x_max + step * 0.5, step)
    x_data = np.repeat(xs, 3)
    y_data = np.tile([y_min, y_max, np.nan], xs.size)
    x_data[2::3] = np.nan
    return pg.PlotDataItem(x_data, y_data, pen=pen)


def _horizontal_grid_item(y_min: float, y_max: float, step: float, x_min: float, x_max: float, pen) -> pg.PlotDataItem:
    start = np.floor(y_min / step) * step
    ys = np.arange(start, y_max + step * 0.5, step)
    x_data = np.tile([x_min, x_max, np.nan], ys.size)
    y_data = np.repeat(ys, 3)
    y_data[2::3] = np.nan
    return pg.PlotDataItem(x_data, y_data, pen=pen)
