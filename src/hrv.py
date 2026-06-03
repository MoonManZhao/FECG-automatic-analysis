"""Basic HRV metrics from R-peak sample indices."""

from __future__ import annotations

import numpy as np


def compute_hrv_metrics(r_peaks: np.ndarray, fs: int) -> dict[str, float | int]:
    r_peaks = np.asarray(r_peaks, dtype=float)
    count = int(r_peaks.size)

    metrics: dict[str, float | int] = {
        "R峰数量": count,
        "Mean RR (ms)": np.nan,
        "Mean HR (bpm)": np.nan,
        "SDNN (ms)": np.nan,
        "RMSSD (ms)": np.nan,
        "Min HR (bpm)": np.nan,
        "Max HR (bpm)": np.nan,
    }

    if count < 2:
        return metrics

    rr_ms = np.diff(np.unique(r_peaks)) / fs * 1000.0
    rr_ms = rr_ms[rr_ms > 0]
    if rr_ms.size == 0:
        return metrics
    hr = 60000.0 / rr_ms

    metrics["Mean RR (ms)"] = float(np.mean(rr_ms))
    metrics["Mean HR (bpm)"] = float(np.mean(hr))
    metrics["SDNN (ms)"] = float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else 0.0
    metrics["RMSSD (ms)"] = (
        float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if rr_ms.size > 1 else 0.0
    )
    metrics["Min HR (bpm)"] = float(np.min(hr))
    metrics["Max HR (bpm)"] = float(np.max(hr))
    return metrics
