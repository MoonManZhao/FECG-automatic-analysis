"""Signal processing routines for ECG-like waveforms."""

from __future__ import annotations

import numpy as np


def _import_neurokit2():
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise RuntimeError("缺少 neurokit2，请先在 ecg_draw 环境中安装 neurokit2。") from exc
    return nk


def _clean_signal(data: np.ndarray, fs: int) -> np.ndarray:
    nk = _import_neurokit2()
    x = np.asarray(data, dtype=float)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)), posinf=0.0, neginf=0.0)
    if x.size < max(3, int(0.5 * fs)):
        return x
    return nk.ecg_clean(x, sampling_rate=fs, method="neurokit")


def detect_r_peaks(x: np.ndarray, fs: int, kind: str = "adult") -> np.ndarray:
    """Detect R peaks with NeuroKit2 only."""
    nk = _import_neurokit2()
    try:
        cleaned = _clean_signal(x, fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        return np.unique(peaks[(peaks >= 0) & (peaks < len(cleaned))])
    except Exception:
        return np.array([], dtype=int)


def auto_wave_annotations(
    data: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    qrs_pre_percent: float = 4.5,
    qrs_post_percent: float = 6.5,
    p_pre_percent: float = 8.0,
    p_post_percent: float = 8.0,
    t_pre_percent: float = 10.0,
    t_post_percent: float = 10.0,
) -> list[dict[str, float | str]]:
    """Use NeuroKit2 delineation for automatic P/QRS/T annotations."""
    nk = _import_neurokit2()
    x = np.asarray(data, dtype=float)
    r_peaks = np.unique(np.asarray(r_peaks, dtype=int))
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < x.size)]
    if x.size < fs or r_peaks.size == 0:
        return []

    x = np.nan_to_num(x, nan=float(np.nanmedian(x)), posinf=0.0, neginf=0.0)
    cleaned = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
    _, waves = nk.ecg_delineate(
        cleaned,
        rpeaks={"ECG_R_Peaks": r_peaks},
        sampling_rate=fs,
        method="dwt",
        show=False,
        show_type="bounds",
    )

    return _beatwise_annotations_from_delineation(
        x,
        waves,
        r_peaks,
        fs,
        qrs_pre_percent=qrs_pre_percent,
        qrs_post_percent=qrs_post_percent,
        p_pre_percent=p_pre_percent,
        p_post_percent=p_post_percent,
        t_pre_percent=t_pre_percent,
        t_post_percent=t_post_percent,
    )


def _beatwise_annotations_from_delineation(
    data: np.ndarray,
    waves: dict[str, list[int | float] | np.ndarray],
    r_peaks: np.ndarray,
    fs: int,
    qrs_pre_percent: float,
    qrs_post_percent: float,
    p_pre_percent: float,
    p_post_percent: float,
    t_pre_percent: float,
    t_post_percent: float,
) -> list[dict[str, float | str]]:
    p_onsets = _as_float_array(waves.get("ECG_P_Onsets"))
    p_peaks = _as_float_array(waves.get("ECG_P_Peaks"))
    p_offsets = _as_float_array(waves.get("ECG_P_Offsets"))
    qrs_onsets = _as_float_array(waves.get("ECG_R_Onsets"))
    qrs_offsets = _as_float_array(waves.get("ECG_R_Offsets"))
    t_onsets = _as_float_array(waves.get("ECG_T_Onsets"))
    t_peaks = _as_float_array(waves.get("ECG_T_Peaks"))
    t_offsets = _as_float_array(waves.get("ECG_T_Offsets"))
    r_peaks = _valid_indices(r_peaks, data.size)
    count = max(
        r_peaks.size,
        p_onsets.size,
        qrs_onsets.size,
        t_onsets.size,
    )

    beats: list[dict] = []
    for index in range(count):
        if index >= r_peaks.size:
            continue
        r_peak = int(r_peaks[index])
        p_span = _wave_indices(p_onsets, p_peaks, p_offsets, index, data.size)
        if p_span is not None and p_pre_percent > 0 and p_post_percent > 0:
            p_percent = _p_percent_span(
                r_peaks,
                p_span[1],
                r_peak,
                index,
                data.size,
                p_pre_percent,
                p_post_percent,
            )
            if p_percent is not None:
                p_span = (p_percent[0], p_span[1], p_percent[1])
        qrs_span = _qrs_indices(
            data,
            qrs_onsets,
            qrs_offsets,
            r_peaks,
            r_peak,
            index,
            data.size,
            fs,
            qrs_pre_percent,
            qrs_post_percent,
        )
        t_span = _wave_indices(t_onsets, t_peaks, t_offsets, index, data.size)
        if t_span is not None and t_pre_percent > 0 and t_post_percent > 0:
            t_percent = _t_percent_span(
                r_peaks,
                t_span[1],
                r_peak,
                index,
                data.size,
                t_pre_percent,
                t_post_percent,
            )
            if t_percent is not None:
                t_span = (t_percent[0], t_span[1], t_percent[1])

        if p_span is not None and not (p_span[0] < p_span[1] < p_span[2] < r_peak):
            p_span = None
        if t_span is not None and not (r_peak < t_span[0] < t_span[1] < t_span[2]):
            t_span = None

        beat_info: dict = {"r_peak": r_peak, "index": index}
        if p_span is not None:
            beat_info["p"] = p_span
        if qrs_span is not None:
            beat_info["qrs"] = qrs_span
        if t_span is not None:
            beat_info["t"] = t_span
        beats.append(beat_info)

    stats = _compute_beat_statistics(beats)

    records: list[dict[str, float | str]] = []
    for beat in beats:
        r_peak = beat["r_peak"]
        p_span = beat.get("p")
        qrs_span = beat.get("qrs")
        t_span = beat.get("t")

        if p_span is None and stats.get("p") is not None:
            p_span = _estimate_p_wave(data, r_peak, stats["p"], data.size, fs)
        if p_span is None:
            p_span = _estimate_p_wave_from_rr(
                data,
                r_peaks,
                r_peak,
                beat["index"],
                data.size,
                p_pre_percent,
                p_post_percent,
            )
        if qrs_span is None and stats.get("qrs") is not None:
            qrs_span = _estimate_qrs_wave(data, r_peak, stats["qrs"], data.size, fs)
        if qrs_span is None:
            qrs_percent = _qrs_percent_span(
                r_peaks,
                r_peak,
                beat["index"],
                data.size,
                qrs_pre_percent,
                qrs_post_percent,
            )
            if qrs_percent is not None:
                qrs_span = _repair_qrs_span(qrs_percent[0], r_peak, qrs_percent[1], None, None, fs, data.size)
        if t_span is None and stats.get("t") is not None:
            t_span = _estimate_t_wave(data, r_peak, stats["t"], data.size, fs)
        if t_span is None:
            t_span = _estimate_t_wave_from_rr(
                data,
                r_peaks,
                r_peak,
                beat["index"],
                data.size,
                t_pre_percent,
                t_post_percent,
            )

        if qrs_span is not None:
            qrs_onset, _, qrs_offset = qrs_span
            if p_span is not None and qrs_onset <= p_span[2]:
                qrs_onset = p_span[2] + 1
            if t_span is not None and qrs_offset >= t_span[0]:
                qrs_offset = t_span[0] - 1
            qrs_span = _repair_qrs_span(qrs_onset, r_peak, qrs_offset, p_span, t_span, fs, data.size)

        if p_span is not None and qrs_span is not None and p_span[2] >= qrs_span[0]:
            p_span = None
        if t_span is not None and qrs_span is not None and qrs_span[2] >= t_span[0]:
            t_span = None

        p_from_delineation = "p" in beat
        qrs_from_delineation = "qrs" in beat
        t_from_delineation = "t" in beat

        if p_span is not None:
            note = "自动标记，建议人工复核" if p_from_delineation else "自动标记(估计)"
            records.append(_record_from_indices(data, fs, "P波", *p_span, note=note))
        if qrs_span is not None:
            note = "自动标记，建议人工复核" if qrs_from_delineation else "自动标记(估计)"
            records.append(_record_from_indices(data, fs, "QRS波", *qrs_span, note=note))
        if t_span is not None:
            note = "自动标记，建议人工复核" if t_from_delineation else "自动标记(估计)"
            records.append(_record_from_indices(data, fs, "T波", *t_span, note=note))
    return records


def _wave_indices(
    onsets: np.ndarray,
    peaks: np.ndarray,
    offsets: np.ndarray,
    index: int,
    size: int,
) -> tuple[int, int, int] | None:
    if index >= min(onsets.size, peaks.size, offsets.size):
        return None
    values = (onsets[index], peaks[index], offsets[index])
    if not all(np.isfinite(value) for value in values):
        return None
    onset, peak, offset = (int(value) for value in values)
    if not (0 <= onset < peak < offset < size):
        return None
    return onset, peak, offset


def _qrs_indices(
    data: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    r_peaks: np.ndarray,
    r_peak: int,
    index: int,
    size: int,
    fs: int,
    qrs_pre_percent: float,
    qrs_post_percent: float,
) -> tuple[int, int, int] | None:
    if index >= min(onsets.size, offsets.size):
        return None
    onset_raw, offset_raw = onsets[index], offsets[index]
    if not (np.isfinite(onset_raw) and np.isfinite(offset_raw)):
        return None
    onset, offset = int(onset_raw), int(offset_raw)
    if not (0 <= onset < r_peak < offset < size):
        return None
    percent_span = _qrs_percent_span(
        r_peaks,
        r_peak,
        index,
        size,
        qrs_pre_percent,
        qrs_post_percent,
    )
    if percent_span is not None:
        onset, offset = percent_span
    return _repair_qrs_span(onset, r_peak, offset, None, None, fs, size)


def _repair_qrs_span(
    onset: int,
    peak: int,
    offset: int,
    p_span: tuple[int, int, int] | None,
    t_span: tuple[int, int, int] | None,
    fs: int,
    size: int,
) -> tuple[int, int, int] | None:
    min_width = max(4, int(round(0.035 * fs)))
    max_width = max(min_width + 2, int(round(0.14 * fs)))
    left_limit = (p_span[2] + 1) if p_span is not None else 0
    right_limit = (t_span[0] - 1) if t_span is not None else size - 1

    onset = max(onset, left_limit)
    offset = min(offset, right_limit)
    if not (0 <= onset < peak < offset < size):
        return None

    width = offset - onset
    if width > max_width:
        half_left = int(round(0.04 * fs))
        half_right = max_width - half_left
        onset = max(left_limit, peak - half_left)
        offset = min(right_limit, peak + half_right)
        width = offset - onset
    if width < min_width:
        need = min_width - width
        grow_left = need // 2
        grow_right = need - grow_left
        onset = max(left_limit, onset - grow_left)
        offset = min(right_limit, offset + grow_right)
        width = offset - onset

    if not (min_width <= width <= max_width and onset < peak < offset):
        return None
    return onset, peak, offset


def _qrs_percent_span(
    r_peaks: np.ndarray,
    r_peak: int,
    index: int,
    size: int,
    qrs_pre_percent: float,
    qrs_post_percent: float,
) -> tuple[int, int] | None:
    if r_peaks.size < 2:
        return None
    previous_rr = r_peak - int(r_peaks[index - 1]) if index > 0 else int(r_peaks[index + 1]) - r_peak
    next_rr = int(r_peaks[index + 1]) - r_peak if index < r_peaks.size - 1 else r_peak - int(r_peaks[index - 1])
    if previous_rr <= 0 or next_rr <= 0:
        return None
    pre = max(0.0, qrs_pre_percent) / 100.0
    post = max(0.0, qrs_post_percent) / 100.0
    onset = int(round(r_peak - previous_rr * pre))
    offset = int(round(r_peak + next_rr * post))
    if not (0 <= onset < r_peak < offset < size):
        return None
    return onset, offset


def _p_percent_span(
    r_peaks: np.ndarray,
    p_peak: int,
    r_peak: int,
    index: int,
    size: int,
    p_pre_percent: float,
    p_post_percent: float,
) -> tuple[int, int] | None:
    """Compute P-wave onset/offset from P peak using RR-based percentages."""
    if r_peaks.size < 2:
        return None
    previous_rr = (
        r_peak - int(r_peaks[index - 1])
        if index > 0
        else int(r_peaks[index + 1]) - r_peak
    )
    if previous_rr <= 0:
        return None
    pre = max(0.0, p_pre_percent) / 100.0
    post = max(0.0, p_post_percent) / 100.0
    onset = int(round(p_peak - previous_rr * pre))
    offset = int(round(p_peak + previous_rr * post))
    if not (0 <= onset < p_peak < offset < r_peak):
        return None
    return onset, offset


def _t_percent_span(
    r_peaks: np.ndarray,
    t_peak: int,
    r_peak: int,
    index: int,
    size: int,
    t_pre_percent: float,
    t_post_percent: float,
) -> tuple[int, int] | None:
    """Compute T-wave onset/offset from T peak using RR-based percentages."""
    if r_peaks.size < 2:
        return None
    next_rr = (
        int(r_peaks[index + 1]) - r_peak
        if index < r_peaks.size - 1
        else r_peak - int(r_peaks[index - 1])
    )
    if next_rr <= 0:
        return None
    pre = max(0.0, t_pre_percent) / 100.0
    post = max(0.0, t_post_percent) / 100.0
    onset = int(round(t_peak - next_rr * pre))
    offset = int(round(t_peak + next_rr * post))
    if not (0 <= onset < t_peak < offset < size and onset > r_peak):
        return None
    return onset, offset


def _record_from_indices(
    data: np.ndarray,
    fs: int,
    wave_type: str,
    onset: int,
    peak: int,
    offset: int,
    note: str = "",
) -> dict[str, float | str]:
    baseline_value = float(data[onset])
    peak_value = float(data[peak])
    return {
        "wave_type": wave_type,
        "start_time": onset / fs,
        "end_time": offset / fs,
        "width_ms": (offset - onset) / fs * 1000.0,
        "baseline_time": onset / fs,
        "baseline_value": baseline_value,
        "peak_time": peak / fs,
        "peak_value": peak_value,
        "amplitude_mv": peak_value - baseline_value,
        "note": note,
    }


def _compute_beat_statistics(
    beats: list[dict],
) -> dict[str, dict[str, float] | None]:
    qrs_pre: list[float] = []
    qrs_post: list[float] = []
    p_to_r: list[float] = []
    p_widths: list[float] = []
    r_to_t: list[float] = []
    t_widths: list[float] = []

    for beat in beats:
        r_peak = beat.get("r_peak")
        if r_peak is None:
            continue
        qrs = beat.get("qrs")
        if qrs is not None:
            qrs_pre.append(float(r_peak - qrs[0]))
            qrs_post.append(float(qrs[2] - r_peak))
        p_span = beat.get("p")
        if p_span is not None:
            p_to_r.append(float(r_peak - p_span[1]))
            p_widths.append(float(p_span[2] - p_span[0]))
        t_span = beat.get("t")
        if t_span is not None:
            r_to_t.append(float(t_span[1] - r_peak))
            t_widths.append(float(t_span[2] - t_span[0]))

    stats: dict[str, dict[str, float] | None] = {"qrs": None, "p": None, "t": None}
    if qrs_pre:
        stats["qrs"] = {
            "qrs_pre": float(np.mean(qrs_pre)),
            "qrs_post": float(np.mean(qrs_post)),
        }
    if p_to_r:
        stats["p"] = {
            "p_to_r": float(np.mean(p_to_r)),
            "p_width": float(np.mean(p_widths)),
        }
    if r_to_t:
        stats["t"] = {
            "r_to_t": float(np.mean(r_to_t)),
            "t_width": float(np.mean(t_widths)),
        }
    return stats


def _estimate_qrs_wave(
    data: np.ndarray,
    r_peak: int,
    stats: dict[str, float],
    size: int,
    fs: int,
) -> tuple[int, int, int] | None:
    onset = max(0, int(round(r_peak - stats["qrs_pre"])))
    offset = min(size - 1, int(round(r_peak + stats["qrs_post"])))
    if onset >= r_peak or offset <= r_peak:
        return None
    return _repair_qrs_span(onset, r_peak, offset, None, None, fs, size)


def _estimate_p_wave(
    data: np.ndarray,
    r_peak: int,
    stats: dict[str, float],
    size: int,
    fs: int,
) -> tuple[int, int, int] | None:
    p_peak_est = max(0, int(round(r_peak - stats["p_to_r"])))
    half_width = max(1, int(round(stats["p_width"] / 2.0)))
    p_onset = max(0, p_peak_est - half_width)
    p_offset = min(size - 1, p_peak_est + half_width)
    if p_onset >= p_peak_est or p_offset <= p_peak_est or p_offset >= r_peak:
        return None
    segment = data[p_onset : p_offset + 1]
    if segment.size == 0:
        return None
    actual_peak = p_onset + int(np.argmax(segment))
    return (p_onset, actual_peak, p_offset)


def _estimate_p_wave_from_rr(
    data: np.ndarray,
    r_peaks: np.ndarray,
    r_peak: int,
    index: int,
    size: int,
    p_pre_percent: float,
    p_post_percent: float,
) -> tuple[int, int, int] | None:
    if r_peaks.size < 2:
        return None
    previous_rr = (
        r_peak - int(r_peaks[index - 1])
        if index > 0
        else int(r_peaks[index + 1]) - r_peak
    )
    if previous_rr <= 0:
        return None
    p_peak_est = int(round(r_peak - previous_rr * 0.24))
    span = _p_percent_span(r_peaks, p_peak_est, r_peak, index, size, p_pre_percent, p_post_percent)
    if span is None:
        return None
    p_onset, p_offset = span
    segment = data[p_onset : p_offset + 1]
    if segment.size == 0:
        return None
    actual_peak = p_onset + int(np.argmax(segment))
    if not (p_onset < actual_peak < p_offset < r_peak):
        actual_peak = p_peak_est
    return (p_onset, actual_peak, p_offset)


def _estimate_t_wave(
    data: np.ndarray,
    r_peak: int,
    stats: dict[str, float],
    size: int,
    fs: int,
) -> tuple[int, int, int] | None:
    t_peak_est = min(size - 1, int(round(r_peak + stats["r_to_t"])))
    half_width = max(1, int(round(stats["t_width"] / 2.0)))
    t_onset = max(0, t_peak_est - half_width)
    t_offset = min(size - 1, t_peak_est + half_width)
    if t_onset >= t_peak_est or t_offset <= t_peak_est or t_onset <= r_peak:
        return None
    segment = data[t_onset : t_offset + 1]
    if segment.size == 0:
        return None
    actual_peak = t_onset + int(np.argmax(segment))
    return (t_onset, actual_peak, t_offset)


def _estimate_t_wave_from_rr(
    data: np.ndarray,
    r_peaks: np.ndarray,
    r_peak: int,
    index: int,
    size: int,
    t_pre_percent: float,
    t_post_percent: float,
) -> tuple[int, int, int] | None:
    if r_peaks.size < 2:
        return None
    next_rr = (
        int(r_peaks[index + 1]) - r_peak
        if index < r_peaks.size - 1
        else r_peak - int(r_peaks[index - 1])
    )
    if next_rr <= 0:
        return None
    t_peak_est = int(round(r_peak + next_rr * 0.32))
    span = _t_percent_span(r_peaks, t_peak_est, r_peak, index, size, t_pre_percent, t_post_percent)
    if span is None:
        return None
    t_onset, t_offset = span
    segment = data[t_onset : t_offset + 1]
    if segment.size == 0:
        return None
    actual_peak = t_onset + int(np.argmax(segment))
    if not (r_peak < t_onset < actual_peak < t_offset):
        actual_peak = t_peak_est
    return (t_onset, actual_peak, t_offset)


def _valid_indices(values, size: int) -> np.ndarray:
    if values is None:
        return np.array([], dtype=int)
    indices = np.asarray(values, dtype=float)
    indices = indices[np.isfinite(indices)]
    indices = indices.astype(int)
    return indices[(indices >= 0) & (indices < size)]


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float)


def time_axis(sample_count: int, fs: int) -> np.ndarray:
    return np.arange(sample_count, dtype=float) / fs
