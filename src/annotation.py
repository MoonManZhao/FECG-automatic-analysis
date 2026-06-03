"""Annotation data structures and CSV export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AnnotationRecord:
    source_file: str
    batch_index: int
    group_index: int
    subject: str
    signal_type: str
    annotation_type: str
    start_time: float | None = None
    end_time: float | None = None
    width_ms: float | None = None
    baseline_time: float | None = None
    baseline_value: float | None = None
    peak_time: float | None = None
    peak_value: float | None = None
    amplitude_mv: float | None = None
    note: str = ""


class AnnotationStore:
    def __init__(self) -> None:
        self.records: list[AnnotationRecord] = []

    def add(self, record: AnnotationRecord) -> None:
        self.records.append(record)

    def undo(self) -> AnnotationRecord | None:
        if not self.records:
            return None
        return self.records.pop()

    def delete(self, index: int) -> AnnotationRecord | None:
        if index < 0 or index >= len(self.records):
            return None
        return self.records.pop(index)

    def clear(self) -> None:
        self.records.clear()

    def to_dataframe(self) -> pd.DataFrame:
        columns = list(AnnotationRecord.__dataclass_fields__.keys())
        df = pd.DataFrame([asdict(item) for item in self.records], columns=columns)
        df.rename(columns={
            "start_time": "start_time (s)",
            "end_time": "end_time (s)",
            "width_ms": "width (ms)",
            "baseline_time": "baseline_time (s)",
            "baseline_value": "baseline_value (mV)",
            "peak_time": "peak_time (s)",
            "peak_value": "peak_value (mV)",
            "amplitude_mv": "amplitude (mV)",
        }, inplace=True)
        return df

    def export_csv(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False, encoding="utf-8-sig")
