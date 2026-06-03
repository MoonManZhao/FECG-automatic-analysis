"""Loading and validating `.npy` ECG data files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import EXPECTED_CHANNELS


@dataclass(frozen=True)
class ECGDataset:
    path: Path
    data: np.ndarray

    @property
    def batch_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[2])


def load_npy_file(path: str | Path) -> ECGDataset:
    file_path = Path(path)
    data = np.load(file_path)
    validate_ecg_shape(data)
    return ECGDataset(path=file_path, data=data.astype(float, copy=False))


def validate_ecg_shape(data: np.ndarray) -> None:
    if data.ndim != 3:
        raise ValueError(f"数据必须是三维数组，当前 shape 为 {data.shape}。")

    if data.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            "数据 shape 必须为 (batch, 10, samples)，"
            f"当前 shape 为 {data.shape}。"
        )

    if data.shape[2] < 2:
        raise ValueError(f"每个通道至少需要 2 个采样点，当前 shape 为 {data.shape}。")
