"""Shared constants and small helpers."""

from __future__ import annotations

from pathlib import Path

FS = 250
SAMPLE_COUNT = 2500
DEFAULT_VIEW_SECONDS = 10.0
EXPECTED_CHANNELS = 10

CHANNEL_NAMES = {
    0: "MECG_clean",
    1: "AECG_1",
    2: "MECG_1",
    3: "FECG_1",
    4: "AECG_2",
    5: "MECG_2",
    6: "FECG_2",
    7: "AECG_3",
    8: "MECG_3",
    9: "FECG_3",
}

GROUP_CHANNELS = {
    1: {"AECG": 1, "MECG": 2, "FECG": 3},
    2: {"AECG": 4, "MECG": 5, "FECG": 6},
    3: {"AECG": 7, "MECG": 8, "FECG": 9},
}

DISPLAY_SIGNALS = tuple(CHANNEL_NAMES[index] for index in range(EXPECTED_CHANNELS))

R_PEAK_CHANNELS = {
    "MECG_clean": "adult",
    "MECG_1": "adult",
    "FECG_1": "fetal",
    "MECG_2": "adult",
    "FECG_2": "fetal",
    "MECG_3": "adult",
    "FECG_3": "fetal",
}

WIDTH_ANNOTATIONS = {
    "R峰",
    "P波宽度",
    "QRS宽度",
    "T波宽度",
    "PR间期",
    "QT间期",
    "ST偏移",
    "自定义",
}

AMPLITUDE_ANNOTATIONS = {"幅度", "P波高度", "T波高度"}
DELETE_R_PEAK_ANNOTATION = "删除错误R峰"

WAVE_OPTIONS = ("P波", "QRS波", "T波", "PR间期", "QT间期", "ST段", "自定义")
TIME_ANNOTATION_BY_WAVE = {
    "P波": "P波宽度",
    "QRS波": "QRS宽度",
    "T波": "T波宽度",
    "PR间期": "PR间期",
    "QT间期": "QT间期",
    "ST段": "ST偏移",
    "自定义": "自定义",
}
AMPLITUDE_ANNOTATION_BY_WAVE = {
    "P波": "P波高度",
    "T波": "T波高度",
    "自定义": "幅度",
}


def subject_for_signal(signal_type: str) -> str:
    if signal_type.startswith("FECG"):
        return "胎儿"
    if signal_type.startswith("MECG"):
        return "母亲"
    return "混合"


def display_name_for_file(path: str | Path | None) -> str:
    if not path:
        return ""
    return Path(path).name
