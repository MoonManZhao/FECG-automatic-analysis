"""PySide6 main window and user interaction logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

import pandas as pd

from .annotation import AnnotationRecord, AnnotationStore
from .data_loader import ECGDataset, load_npy_file
from .hrv import compute_hrv_metrics
from .plotting import MultiLeadECGPlotWidget
from .signal_processing import auto_wave_annotations, detect_r_peaks, time_axis
from .utils import (
    AMPLITUDE_ANNOTATIONS,
    AMPLITUDE_ANNOTATION_BY_WAVE,
    CHANNEL_NAMES,
    DISPLAY_SIGNALS,
    FS,
    GROUP_CHANNELS,
    R_PEAK_CHANNELS,
    TIME_ANNOTATION_BY_WAVE,
    WAVE_OPTIONS,
    WIDTH_ANNOTATIONS,
    display_name_for_file,
    subject_for_signal,
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("母胎心电信号标定与分析")
        self.resize(1500, 1000)

        self.dataset: ECGDataset | None = None
        self.current_file = ""
        self.annotations = AnnotationStore()
        self.pending_points: list[tuple[str, float, float]] = []
        self.deleted_r_peaks: dict[tuple[int, str], set[int]] = {}
        self.manual_r_peaks: dict[tuple[int, str], set[int]] = {}
        self.r_move_source: tuple[str, int] | None = None
        self.range_edit_row: int | None = None
        self.replacement_annotation: tuple[str, str, str] | None = None
        self.click_mode = "select_annotation"
        self.time_precision_spin: QtWidgets.QSpinBox | None = None
        self.voltage_precision_spin: QtWidgets.QSpinBox | None = None
        self._updating_annotation_table = False
        self.plot: MultiLeadECGPlotWidget | None = None
        self.current_signals: dict[str, np.ndarray] = {}
        self.current_r_peaks: dict[str, np.ndarray] = {}
        self.t = np.array([], dtype=float)
        self._recalc_running = False
        self._preview_auto_records: dict[tuple[int, int, str], list[AnnotationRecord]] = {}
        self._annotation_table_record_indices: list[int] = []
        self._ruler_items: list[object] = []
        self.ruler_points: list[tuple[str, float, float]] = []

        self._build_ui()

    @staticmethod
    def _make_table_columns_resizable(table: QtWidgets.QTableWidget) -> None:
        header = table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        table.resizeColumnsToContents()
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        toolbar_widget = QtWidgets.QWidget()
        toolbar_root = QtWidgets.QVBoxLayout(toolbar_widget)
        toolbar_root.setContentsMargins(0, 0, 0, 0)
        toolbar_row1 = QtWidgets.QHBoxLayout()
        toolbar_row2 = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("打开NPY")
        self.export_button = QtWidgets.QPushButton("导出CSV")
        self.undo_button = QtWidgets.QPushButton("撤销标注")
        self.reset_view_button = QtWidgets.QPushButton("返回初始视图")
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setPrefix("Batch ")
        self.batch_spin.setEnabled(False)
        self.signal_combo = QtWidgets.QComboBox()
        for label, signal_type, group_index in self._signal_options():
            self.signal_combo.addItem(label, (signal_type, group_index))
        self.signal_combo.hide()
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(WAVE_OPTIONS)
        self.wave_combo.hide()
        self.signal_buttons: dict[str, QtWidgets.QPushButton] = {}
        self.wave_buttons: dict[str, QtWidgets.QPushButton] = {}
        self.show_signal_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self.signal_button_group = QtWidgets.QButtonGroup(self)
        self.signal_button_group.setExclusive(True)
        self.wave_button_group = QtWidgets.QButtonGroup(self)
        self.wave_button_group.setExclusive(True)
        self.time_mark_button = QtWidgets.QPushButton("时间标注")
        self.amp_mark_button = QtWidgets.QPushButton("幅度标注")
        self.auto_pt_button = QtWidgets.QPushButton("显示当前结果")
        self.recalc_auto_button = QtWidgets.QPushButton("重算当前导联")
        self.joint_analysis_button = QtWidgets.QPushButton("联合分析")
        self.refresh_stats_button = QtWidgets.QPushButton("刷新统计")
        self.show_auto_mother_checkbox = QtWidgets.QCheckBox("显示母亲自动标记")
        self.show_auto_mother_checkbox.setChecked(False)
        self.show_auto_fetal_checkbox = QtWidgets.QCheckBox("显示胎儿自动标记")
        self.show_auto_fetal_checkbox.setChecked(False)
        self.show_label_checkbox = QtWidgets.QCheckBox("显示编号")
        self.show_label_checkbox.setChecked(False)
        self.show_joint_checkbox = QtWidgets.QCheckBox("显示联合结果")
        self.show_joint_checkbox.setChecked(False)
        self.qrs_pre_spin = QtWidgets.QDoubleSpinBox()
        self.qrs_pre_spin.setRange(0.5, 20.0)
        self.qrs_pre_spin.setSingleStep(0.5)
        self.qrs_pre_spin.setSuffix("%")
        self.qrs_pre_spin.setValue(4.5)
        self.qrs_pre_spin.setToolTip("QRS波起点在RR间期中的百分比")
        self.qrs_post_spin = QtWidgets.QDoubleSpinBox()
        self.qrs_post_spin.setRange(0.5, 20.0)
        self.qrs_post_spin.setSingleStep(0.5)
        self.qrs_post_spin.setSuffix("%")
        self.qrs_post_spin.setValue(6.5)
        self.qrs_post_spin.setToolTip("QRS波终点在RR间期中的百分比")
        self.joint_results: dict | None = None  # {aligned_beats, combined_beats}
        self._joint_overlay_items: list = []
        self.joint_selected_signals: list[str] = []
        self.joint_start_time = 0.0
        self.joint_end_time = 10.0
        self.joint_tolerance_ms = 20
        self.beat_split_spin = QtWidgets.QDoubleSpinBox()
        self.beat_split_spin.setRange(30.0, 80.0)
        self.beat_split_spin.setSingleStep(1.0)
        self.beat_split_spin.setSuffix("%")
        self.beat_split_spin.setValue(58.0)
        self.beat_split_spin.setToolTip("RR间期中心搏切分百分比")
        self.p_pre_spin = QtWidgets.QDoubleSpinBox()
        self.p_pre_spin.setRange(0.5, 30.0)
        self.p_pre_spin.setSingleStep(0.5)
        self.p_pre_spin.setSuffix("%")
        self.p_pre_spin.setValue(8.0)
        self.p_post_spin = QtWidgets.QDoubleSpinBox()
        self.p_post_spin.setRange(0.5, 30.0)
        self.p_post_spin.setSingleStep(0.5)
        self.p_post_spin.setSuffix("%")
        self.p_post_spin.setValue(8.0)
        self.t_pre_spin = QtWidgets.QDoubleSpinBox()
        self.t_pre_spin.setRange(0.5, 30.0)
        self.t_pre_spin.setSingleStep(0.5)
        self.t_pre_spin.setSuffix("%")
        self.t_pre_spin.setValue(10.0)
        self.t_post_spin = QtWidgets.QDoubleSpinBox()
        self.t_post_spin.setRange(0.5, 30.0)
        self.t_post_spin.setSingleStep(0.5)
        self.t_post_spin.setSuffix("%")
        self.t_post_spin.setValue(10.0)
        self.add_r_button = QtWidgets.QPushButton("添加R峰")
        self.time_precision_spin = QtWidgets.QSpinBox()
        self.time_precision_spin.setRange(0, 6)
        self.time_precision_spin.setValue(1)
        self.time_precision_spin.setToolTip("时间值显示的小数位数（如1表示0.1s）")
        self.time_precision_spin.setFixedWidth(60)
        self.voltage_precision_spin = QtWidgets.QSpinBox()
        self.voltage_precision_spin.setRange(0, 6)
        self.voltage_precision_spin.setValue(3)
        self.voltage_precision_spin.setToolTip("电压值显示的小数位数（如3表示0.123mV）")
        self.voltage_precision_spin.setFixedWidth(60)
        self.delete_r_button = QtWidgets.QPushButton("删除R峰")
        self.move_r_button = QtWidgets.QPushButton("移动R峰")
        self.select_annotation_button = QtWidgets.QPushButton("图上选标注")
        self.edit_range_button = QtWidgets.QPushButton("修改范围")
        self.ruler_button = QtWidgets.QPushButton("标尺")
        self.clear_ruler_button = QtWidgets.QPushButton("清除标尺")
        self.delete_annotation_button = QtWidgets.QPushButton("删除选中标注")
        self.note_edit = QtWidgets.QLineEdit()
        self.note_edit.setPlaceholderText("备注")

        # 联合分析控件
        self.joint_fecg1_cb = QtWidgets.QCheckBox("F1")
        self.joint_fecg1_cb.setChecked(False)
        self.joint_fecg1_cb.setToolTip("FECG_1")
        self.joint_fecg2_cb = QtWidgets.QCheckBox("F2")
        self.joint_fecg2_cb.setChecked(False)
        self.joint_fecg2_cb.setToolTip("FECG_2")
        self.joint_fecg3_cb = QtWidgets.QCheckBox("F3")
        self.joint_fecg3_cb.setChecked(False)
        self.joint_fecg3_cb.setToolTip("FECG_3")
        self.joint_mecg_cb = QtWidgets.QCheckBox("M1")
        self.joint_mecg_cb.setToolTip("MECG_1")
        self.joint_start_spin = QtWidgets.QDoubleSpinBox()
        self.joint_start_spin.setRange(0.0, 10000.0)
        self.joint_start_spin.setValue(0.0)
        self.joint_start_spin.setDecimals(1)
        self.joint_start_spin.setFixedWidth(55)
        self.joint_end_spin = QtWidgets.QDoubleSpinBox()
        self.joint_end_spin.setRange(0.0, 10000.0)
        self.joint_end_spin.setValue(10.0)
        self.joint_end_spin.setDecimals(1)
        self.joint_end_spin.setFixedWidth(55)
        self.joint_tol_spin = QtWidgets.QSpinBox()
        self.joint_tol_spin.setRange(1, 100)
        self.joint_tol_spin.setValue(20)
        self.joint_tol_spin.setFixedWidth(50)

        self.mode_buttons = [
            self.time_mark_button,
            self.amp_mark_button,
            self.add_r_button,
            self.delete_r_button,
            self.move_r_button,
            self.select_annotation_button,
            self.edit_range_button,
            self.ruler_button,
        ]
        for button in self.mode_buttons:
            button.setCheckable(True)

        toolbar_row1.addWidget(self.open_button)
        toolbar_row1.addWidget(self.export_button)
        toolbar_row1.addWidget(self.undo_button)
        toolbar_row1.addWidget(self.reset_view_button)
        toolbar_row1.addWidget(QtWidgets.QLabel("样本"))
        toolbar_row1.addWidget(self.batch_spin)
        toolbar_row1.addWidget(self.time_mark_button)
        toolbar_row1.addWidget(self.amp_mark_button)
        toolbar_row1.addWidget(self.note_edit, stretch=1)

        toolbar_row2.addWidget(self.auto_pt_button)
        toolbar_row2.addWidget(self.recalc_auto_button)
        toolbar_row2.addWidget(self.refresh_stats_button)
        toolbar_row2.addWidget(self.joint_analysis_button)
        toolbar_row2.addWidget(self.show_auto_mother_checkbox)
        toolbar_row2.addWidget(self.show_auto_fetal_checkbox)
        toolbar_row2.addWidget(self.show_label_checkbox)
        toolbar_row2.addWidget(self.show_joint_checkbox)
        toolbar_row2.addWidget(self.add_r_button)
        toolbar_row2.addWidget(self.delete_r_button)
        toolbar_row2.addWidget(self.move_r_button)
        toolbar_row2.addWidget(self.select_annotation_button)
        toolbar_row2.addWidget(self.edit_range_button)
        toolbar_row2.addWidget(self.ruler_button)
        toolbar_row2.addWidget(self.clear_ruler_button)
        toolbar_row2.addWidget(self.delete_annotation_button)
        toolbar_row2.addWidget(QtWidgets.QLabel("电压"))
        toolbar_row2.addWidget(self.voltage_precision_spin)
        toolbar_row2.addWidget(QtWidgets.QLabel("时间"))
        toolbar_row2.addWidget(self.time_precision_spin)

        toolbar_row2.addStretch(1)
        toolbar_row3 = QtWidgets.QHBoxLayout()
        # 波形参数微调
        toolbar_row3.addWidget(QtWidgets.QLabel("QRS前"))
        toolbar_row3.addWidget(self.qrs_pre_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel("后"))
        toolbar_row3.addWidget(self.qrs_post_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel(" | P前"))
        toolbar_row3.addWidget(self.p_pre_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel("后"))
        toolbar_row3.addWidget(self.p_post_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel(" | T前"))
        toolbar_row3.addWidget(self.t_pre_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel("后"))
        toolbar_row3.addWidget(self.t_post_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel(" | 切分"))
        toolbar_row3.addWidget(self.beat_split_spin)
        # 联合分析控制
        toolbar_row3.addWidget(QtWidgets.QLabel(" | 联合:"))
        toolbar_row3.addWidget(self.joint_fecg1_cb)
        toolbar_row3.addWidget(self.joint_fecg2_cb)
        toolbar_row3.addWidget(self.joint_fecg3_cb)
        toolbar_row3.addWidget(self.joint_mecg_cb)
        toolbar_row3.addWidget(QtWidgets.QLabel("窗(s)"))
        toolbar_row3.addWidget(self.joint_start_spin)
        toolbar_row3.addWidget(self.joint_end_spin)
        toolbar_row3.addWidget(QtWidgets.QLabel("容(ms)"))
        toolbar_row3.addWidget(self.joint_tol_spin)
        toolbar_row3.addStretch(1)

        toolbar_row4 = QtWidgets.QHBoxLayout()
        toolbar_row4.addWidget(QtWidgets.QLabel("操作导联:"))
        for signal_type in DISPLAY_SIGNALS:
            button = QtWidgets.QPushButton(signal_type)
            button.setCheckable(True)
            button.setFixedHeight(26)
            self.signal_button_group.addButton(button)
            self.signal_buttons[signal_type] = button
            toolbar_row4.addWidget(button)
            button.clicked.connect(lambda _checked=False, value=signal_type: self._select_signal_from_button(value))
        toolbar_row4.addStretch(1)

        toolbar_row5 = QtWidgets.QHBoxLayout()
        toolbar_row5.addWidget(QtWidgets.QLabel("标注波形:"))
        for wave_type in WAVE_OPTIONS:
            button = QtWidgets.QPushButton(wave_type)
            button.setCheckable(True)
            button.setFixedHeight(26)
            self.wave_button_group.addButton(button)
            self.wave_buttons[wave_type] = button
            toolbar_row5.addWidget(button)
            button.clicked.connect(lambda _checked=False, value=wave_type: self._select_wave_from_button(value))
        toolbar_row5.addWidget(QtWidgets.QLabel(" | 显示标注:"))
        for signal_type in DISPLAY_SIGNALS:
            checkbox = QtWidgets.QCheckBox(signal_type)
            checkbox.setChecked(False)
            self.show_signal_checkboxes[signal_type] = checkbox
            toolbar_row5.addWidget(checkbox)
            checkbox.toggled.connect(self.update_annotation_overlay)
        toolbar_row5.addStretch(1)

        toolbar_root.addLayout(toolbar_row1)
        toolbar_root.addLayout(toolbar_row2)
        toolbar_root.addLayout(toolbar_row3)
        toolbar_root.addLayout(toolbar_row4)
        toolbar_root.addLayout(toolbar_row5)
        root.addWidget(toolbar_widget)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        self.plot = MultiLeadECGPlotWidget()
        self.plot.point_clicked.connect(self._handle_plot_click)

        side_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.status_label = QtWidgets.QLabel("请打开 shape 为 (batch, 10, samples) 的 .npy 文件。")
        self.status_label.setWordWrap(True)
        side_splitter.addWidget(self.status_label)

        self.hrv_table = QtWidgets.QTableWidget(0, 5)
        self.hrv_table.setHorizontalHeaderLabels(["导联", "R峰数量", "平均RR (ms)", "平均心率 (bpm)", "SDNN/RMSSD (ms)"])
        self._make_table_columns_resizable(self.hrv_table)
        hrv_container = QtWidgets.QWidget()
        hrv_layout = QtWidgets.QVBoxLayout(hrv_container)
        hrv_layout.setContentsMargins(0, 0, 0, 0)
        hrv_layout.addWidget(QtWidgets.QLabel("HRV"))
        hrv_layout.addWidget(self.hrv_table, 1)
        side_splitter.addWidget(hrv_container)

        self.wave_stats_table = QtWidgets.QTableWidget(0, 16)
        self.wave_stats_table.setHorizontalHeaderLabels(
            [
                "导联",
                "P (ms)",
                "P幅度 (mV)",
                "PR (ms)",
                "PR最小 (ms)",
                "PR最大 (ms)",
                "QRS (ms)",
                "QRS幅度 (mV)",
                "ST (ms)",
                "ST最小 (ms)",
                "ST最大 (ms)",
                "T (ms)",
                "T幅度 (mV)",
                "RR (ms)",
                "58%RR (ms)",
                "心搏数",
            ]
        )
        self._make_table_columns_resizable(self.wave_stats_table)
        wave_container = QtWidgets.QWidget()
        wave_layout = QtWidgets.QVBoxLayout(wave_container)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        wave_layout.addWidget(QtWidgets.QLabel("波段平均时间(ms)"))
        wave_layout.addWidget(self.wave_stats_table, 1)
        side_splitter.addWidget(wave_container)

        self.annotation_table = QtWidgets.QTableWidget(0, 15)
        self.annotation_table.setHorizontalHeaderLabels(
            [
                "文件",
                "样本",
                "组",
                "对象",
                "导联",
                "类型",
                "起点(s)",
                "终点(s)",
                "宽度(ms)",
                "基线时间(s)",
                "基线值(mV)",
                "峰值时间(s)",
                "峰值(mV)",
                "幅度(mV)",
                "备注",
            ]
        )
        annotation_header = self.annotation_table.horizontalHeader()
        annotation_header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        annotation_header.setMinimumSectionSize(48)
        self._make_table_columns_resizable(self.annotation_table)
        self.annotation_table.verticalHeader().setDefaultSectionSize(24)
        self.annotation_table.setWordWrap(False)
        self.annotation_table.setAlternatingRowColors(True)
        self.annotation_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.annotation_table.setStyleSheet(
            "QHeaderView::section { padding: 4px 6px; font-weight: 600; }"
            "QTableWidget { gridline-color: #d1d5db; alternate-background-color: #f9fafb; }"
        )
        self.annotation_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.annotation_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        annotation_container = QtWidgets.QWidget()
        annotation_layout = QtWidgets.QVBoxLayout(annotation_container)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(QtWidgets.QLabel("标注结果"))
        annotation_layout.addWidget(self.annotation_table, 1)

        # Joint analysis table
        self.joint_table = QtWidgets.QTableWidget(0, 15)
        self.joint_table.setHorizontalHeaderLabels(
            [
                "来源",
                "心搏",
                "P(ms)",
                "P幅度(mV)",
                "PR(ms)",
                "PR最小(ms)",
                "PR最大(ms)",
                "QRS(ms)",
                "QRS幅度(mV)",
                "ST(ms)",
                "ST最小(ms)",
                "ST最大(ms)",
                "T(ms)",
                "T幅度(mV)",
                "RR(ms)",
            ]
        )
        self._make_table_columns_resizable(self.joint_table)
        joint_container = QtWidgets.QWidget()
        joint_layout = QtWidgets.QVBoxLayout(joint_container)
        joint_layout.setContentsMargins(0, 0, 0, 0)
        joint_layout.addWidget(QtWidgets.QLabel("联合分析（整体平均）"))
        joint_layout.addWidget(self.joint_table, 1)
        side_splitter.addWidget(joint_container)
        side_splitter.addWidget(annotation_container)

        splitter.addWidget(self.plot)
        splitter.addWidget(side_splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([900, 600])
        side_splitter.setHandleWidth(6)
        side_splitter.setChildrenCollapsible(False)
        side_splitter.setSizes([60, 250, 250, 200, 400])

        self.open_button.clicked.connect(self.open_file)
        self.export_button.clicked.connect(self.export_annotations)
        self.undo_button.clicked.connect(self.undo_annotation)
        self.reset_view_button.clicked.connect(self.reset_initial_view)
        self.delete_annotation_button.clicked.connect(self.delete_selected_annotation)
        self.time_precision_spin.valueChanged.connect(self._on_precision_changed)
        self.voltage_precision_spin.valueChanged.connect(self._on_precision_changed)
        self.time_mark_button.clicked.connect(self.start_time_annotation)
        self.amp_mark_button.clicked.connect(self.start_amplitude_annotation)
        self.auto_pt_button.clicked.connect(self.show_current_wave_analysis)
        self.recalc_auto_button.clicked.connect(self.reanalyze_current_signal)
        self.refresh_stats_button.clicked.connect(self.refresh_stats_from_annotations)
        self.joint_analysis_button.clicked.connect(self.show_joint_analysis_dialog)
        self.show_auto_mother_checkbox.toggled.connect(self.update_annotation_overlay)
        self.joint_fecg1_cb.toggled.connect(lambda: self._run_joint_analysis_if_ready())
        self.joint_fecg2_cb.toggled.connect(lambda: self._run_joint_analysis_if_ready())
        self.joint_fecg3_cb.toggled.connect(lambda: self._run_joint_analysis_if_ready())
        self.joint_mecg_cb.toggled.connect(lambda: self._run_joint_analysis_if_ready())
        self.joint_start_spin.valueChanged.connect(lambda: self._run_joint_analysis_if_ready())
        self.joint_end_spin.valueChanged.connect(lambda: self._run_joint_analysis_if_ready())
        self.joint_tol_spin.valueChanged.connect(lambda: self._run_joint_analysis_if_ready())
        self.show_auto_fetal_checkbox.toggled.connect(self.update_annotation_overlay)
        self.show_label_checkbox.toggled.connect(self.update_annotation_overlay)
        self.show_joint_checkbox.toggled.connect(self._toggle_joint_overlay)
        self.annotation_table.itemChanged.connect(self.annotation_table_item_changed)
        self.annotation_table.itemSelectionChanged.connect(self.annotation_table_selection_changed)
        self.add_r_button.clicked.connect(lambda: self.set_r_edit_mode("add_r"))
        self.delete_r_button.clicked.connect(lambda: self.set_r_edit_mode("delete_r"))
        self.move_r_button.clicked.connect(lambda: self.set_r_edit_mode("move_r"))
        self.select_annotation_button.clicked.connect(self.start_annotation_selection)
        self.edit_range_button.clicked.connect(self.start_range_edit)
        self.ruler_button.clicked.connect(self.start_ruler_measure)
        self.clear_ruler_button.clicked.connect(self.clear_ruler_measurements)
        self.batch_spin.valueChanged.connect(self.refresh_view)
        self.signal_combo.currentTextChanged.connect(self._clear_pending)
        self.wave_combo.currentTextChanged.connect(self._clear_pending)
        self.beat_split_spin.valueChanged.connect(self._refresh_wave_stats_table)
        self.qrs_pre_spin.valueChanged.connect(self._on_qrs_parameter_changed)
        self.qrs_post_spin.valueChanged.connect(self._on_qrs_parameter_changed)
        self.p_pre_spin.valueChanged.connect(self._on_qrs_parameter_changed)
        self.p_post_spin.valueChanged.connect(self._on_qrs_parameter_changed)
        self.t_pre_spin.valueChanged.connect(self._on_qrs_parameter_changed)
        self.t_post_spin.valueChanged.connect(self._on_qrs_parameter_changed)
        self._sync_signal_buttons()
        self._sync_wave_buttons()
        self._set_active_button(self.select_annotation_button)

    def open_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开母胎心电 .npy 文件",
            str(Path.cwd()),
            "NumPy files (*.npy)",
        )
        if not path:
            return
        try:
            self.dataset = load_npy_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "数据格式错误", str(exc))
            return

        self.current_file = path
        self.batch_spin.setEnabled(True)
        self.batch_spin.setRange(0, self.dataset.batch_count - 1)
        self.batch_spin.setValue(0)
        self.annotations.clear()
        self.deleted_r_peaks.clear()
        self.manual_r_peaks.clear()
        self._preview_auto_records.clear()
        self.r_move_source = None
        self._set_result_visibility_defaults()
        self._clear_pending()
        self._refresh_annotation_table()
        self.refresh_view()
        created = self._precompute_all_auto_annotations()
        self._refresh_annotation_table()
        self._refresh_all_stats()
        self._run_joint_analysis(fallback_to_fetal=True)
        self.update_annotation_overlay()
        self.status_label.setText(
            f"文件已打开并完成后台自动计算：{display_name_for_file(self.current_file)}，生成 {created} 条 P/QRS/T 自动结果。"
        )

    def _signal_options(self) -> list[tuple[str, str, int]]:
        options = [("母亲纯净 / MECG_clean", "MECG_clean", 0)]
        for group, channels in GROUP_CHANNELS.items():
            for signal_type, channel_index in channels.items():
                channel_name = CHANNEL_NAMES[channel_index]
                options.append((f"第{group}组 / {signal_type} ({channel_name})", channel_name, group))
        return options

    def _current_signal_type(self) -> str:
        data = self.signal_combo.currentData()
        if isinstance(data, tuple) and data:
            return str(data[0])
        return self.signal_combo.currentText()

    def _current_group_index(self) -> int:
        data = self.signal_combo.currentData()
        if isinstance(data, tuple) and len(data) > 1:
            return int(data[1])
        return self._group_index_for_signal(self._current_signal_type())

    @staticmethod
    def _group_index_for_signal(signal_type: str) -> int:
        if signal_type.endswith("_1"):
            return 1
        if signal_type.endswith("_2"):
            return 2
        if signal_type.endswith("_3"):
            return 3
        return 0

    def _set_current_signal_type(self, signal_type: str) -> None:
        for index in range(self.signal_combo.count()):
            data = self.signal_combo.itemData(index)
            if isinstance(data, tuple) and data and data[0] == signal_type:
                self.signal_combo.blockSignals(True)
                self.signal_combo.setCurrentIndex(index)
                self.signal_combo.blockSignals(False)
                self._sync_signal_buttons()
                return

    def _select_signal_from_button(self, signal_type: str) -> None:
        self._set_current_signal_type(signal_type)
        self._clear_pending()
        self.status_label.setText(f"当前操作导联：{signal_type}")

    def _sync_signal_buttons(self) -> None:
        current = self._current_signal_type()
        for signal_type, button in self.signal_buttons.items():
            button.blockSignals(True)
            button.setChecked(signal_type == current)
            button.blockSignals(False)

    def _select_wave_from_button(self, wave_type: str) -> None:
        index = self.wave_combo.findText(wave_type)
        if index >= 0:
            self.wave_combo.blockSignals(True)
            self.wave_combo.setCurrentIndex(index)
            self.wave_combo.blockSignals(False)
        self._sync_wave_buttons()
        self._clear_pending()
        self.status_label.setText(f"当前标注波形：{wave_type}")

    def _sync_wave_buttons(self) -> None:
        current = self.wave_combo.currentText()
        for wave_type, button in self.wave_buttons.items():
            button.blockSignals(True)
            button.setChecked(wave_type == current)
            button.blockSignals(False)

    def _set_result_visibility_defaults(self) -> None:
        for checkbox in self.show_signal_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        for checkbox in (self.show_auto_mother_checkbox, self.show_auto_fetal_checkbox, self.show_label_checkbox):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        for checkbox in (self.joint_fecg1_cb, self.joint_fecg2_cb, self.joint_fecg3_cb, self.joint_mecg_cb):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self.show_joint_checkbox.blockSignals(True)
        self.show_joint_checkbox.setChecked(False)
        self.show_joint_checkbox.blockSignals(False)

    def _set_active_button(self, active_button: QtWidgets.QPushButton | None) -> None:
        for button in getattr(self, "mode_buttons", []):
            button.setChecked(button is active_button)

    def refresh_view(self) -> None:
        if self.dataset is None:
            return

        batch = self.batch_spin.value()
        self._clear_joint_overlay()
        prev_joint = self.joint_results
        if prev_joint is not None:
            self.joint_results = prev_joint
        sample = self.dataset.data[batch]
        self.t = time_axis(sample.shape[1], FS)
        self.current_signals = {
            CHANNEL_NAMES[channel_index]: sample[channel_index]
            for channel_index in range(sample.shape[0])
        }

        self.current_r_peaks = {}
        for signal_type, kind in R_PEAK_CHANNELS.items():
            peaks = detect_r_peaks(self.current_signals[signal_type], FS, kind)
            removed = self.deleted_r_peaks.get((batch, signal_type), set())
            if removed:
                peaks = np.asarray([peak for peak in peaks if int(peak) not in removed], dtype=int)
            added = self.manual_r_peaks.get((batch, signal_type), set())
            if added:
                peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
            self.current_r_peaks[signal_type] = peaks

        if self.plot is not None:
            self.plot.draw(self.t, self.current_signals, self.current_r_peaks, FS)
            self.update_annotation_overlay()

        self._refresh_hrv_table()
        self._refresh_wave_stats_table()
        self._refresh_joint_table()
        if self.joint_results is not None:
            self._run_joint_analysis()
        self._refresh_annotation_table()
        duration = sample.shape[1] / FS
        signal_type = self._current_signal_type()
        group = self._current_group_index()
        group_text = "母亲纯净" if group == 0 else f"第{group}组"
        self.status_label.setText(
            f"文件: {display_name_for_file(self.current_file)} | 样本 {batch} | 一个大图同步显示10导联 | "
            f"总时长 {duration:.2f}s，当前窗口10s | 当前：{group_text} / {signal_type}"
        )
        self._clear_pending()

    def _handle_plot_click(self, x: float, plotted_y: float) -> None:
        if self.dataset is None:
            return
        selected_signal = self._current_signal_type()
        if self.plot is None:
            return

        if self.click_mode == "add_r":
            self._add_r_peak(selected_signal, x)
            return
        if self.click_mode == "delete_r":
            self._delete_nearest_r_peak(selected_signal, x)
            return
        if self.click_mode == "move_r":
            self._move_r_peak_click(selected_signal, x)
            return
        if self.click_mode == "edit_range":
            self._range_edit_click(x)
            return
        if self.click_mode == "replace_range":
            self._replacement_range_click(x, plotted_y)
            return
        if self.click_mode == "ruler":
            self._ruler_click(selected_signal, x, plotted_y)
            return
        if self.click_mode in {"select_annotation", "select_annotation_for_range"}:
            row = self._select_annotation_at(x, plotted_y)
            if row is None:
                self.status_label.setText("当前位置没有可选标注，请点击图中带编号的标注块或幅度线附近。")
                return
            if self.click_mode == "select_annotation_for_range":
                self._begin_range_edit(row)
            return

        annotation_type = self._current_annotation_type()
        if annotation_type is None:
            self.status_label.setText("当前波形不支持幅度标注，请选择 P波、T波或自定义。")
            return
        raw_y = self.plot.raw_value_for_lead(selected_signal, plotted_y)

        self.pending_points.append((selected_signal, x, raw_y))
        self.status_label.setText(f"{annotation_type}: 已选择 {len(self.pending_points)} / 2 个点。")

        if len(self.pending_points) < 2:
            self.plot.add_pending_marker(selected_signal, x, "第1点")
            return

        first, second = self.pending_points[:2]
        self.pending_points.clear()
        self.plot.clear_pending_marker()
        record = self._record_from_points(annotation_type, first, second)
        self.annotations.add(record)
        self._refresh_annotation_table()
        self.update_annotation_overlay()

        if annotation_type in AMPLITUDE_ANNOTATIONS:
            self.status_label.setText(f"幅度差: {self._fmt_voltage(record.amplitude_mv)} mV")
        else:
            self.status_label.setText(f"时间差: {self._fmt_time(record.width_ms)} ms")

    def start_time_annotation(self) -> None:
        self.click_mode = "time"
        self._set_active_button(self.time_mark_button)
        self._clear_pending()
        self.status_label.setText(f"时间标注：请选择 {self._current_signal_type()} 的起点和终点。")

    def start_amplitude_annotation(self) -> None:
        self.click_mode = "amplitude"
        self._set_active_button(self.amp_mark_button)
        self._clear_pending()
        self.status_label.setText(f"幅度标注：请选择 {self._current_signal_type()} 的基线点和峰值点。")

    def set_r_edit_mode(self, mode: str) -> None:
        self.click_mode = mode
        active_buttons = {
            "add_r": self.add_r_button,
            "delete_r": self.delete_r_button,
            "move_r": self.move_r_button,
        }
        self._set_active_button(active_buttons.get(mode))
        self._clear_pending()
        self.r_move_source = None
        self.range_edit_row = None
        labels = {
            "add_r": "添加R峰：点击当前导联的新R峰位置。",
            "delete_r": "删除R峰：点击当前导联错误R峰附近。",
            "move_r": "移动R峰：先点错误R峰，再点新R峰位置。",
        }
        self.status_label.setText(labels.get(mode, ""))

    def start_annotation_selection(self) -> None:
        self.click_mode = "select_annotation"
        self._set_active_button(self.select_annotation_button)
        self._clear_pending()
        self.range_edit_row = None
        self.r_move_source = None
        self.replacement_annotation = None
        self.ruler_points.clear()
        self.status_label.setText("图上选标注：请点击图中带编号的标注块或幅度线，右侧表格会同步选中。")

    def start_ruler_measure(self) -> None:
        self.click_mode = "ruler"
        self._set_active_button(self.ruler_button)
        self._clear_pending()
        self.range_edit_row = None
        self.r_move_source = None
        self.replacement_annotation = None
        self.ruler_points.clear()
        self.status_label.setText(f"标尺：请在 {self._current_signal_type()} 上点击两个点。")

    def start_range_edit(self) -> None:
        row = self.annotation_table.currentRow()
        record_index = self._record_index_for_table_row(row)
        if record_index is None:
            self.click_mode = "select_annotation_for_range"
            self._set_active_button(self.edit_range_button)
            self._clear_pending()
            self.status_label.setText("修改范围：请先在图中点击要修改的标注。")
            return
        self._begin_range_edit(record_index)

    def _ruler_click(self, signal_type: str, x: float, plotted_y: float) -> None:
        if self.plot is None:
            return
        raw_y = self.plot.raw_value_for_lead(signal_type, plotted_y)
        if self.ruler_points and self.ruler_points[0][0] != signal_type:
            self.ruler_points.clear()
        self.ruler_points.append((signal_type, x, raw_y))
        if len(self.ruler_points) < 2:
            self.plot.add_pending_marker(signal_type, x, "标尺起点")
            self.status_label.setText("标尺：已选择起点，请点击终点。")
            return

        first, second = self.ruler_points[:2]
        self.ruler_points.clear()
        self.plot.clear_pending_marker()
        self._draw_ruler_overlay(first, second)

    def _draw_ruler_overlay(
        self,
        first: tuple[str, float, float],
        second: tuple[str, float, float],
    ) -> None:
        if self.plot is None:
            return
        signal_type = first[0]
        offset = self.plot.lead_offsets.get(signal_type, 0.0)
        x1, y1 = first[1], first[2]
        x2, y2 = second[1], second[2]
        plotted_y1 = y1 + offset
        plotted_y2 = y2 + offset
        color = "#0f766e"
        line = self.plot.plotItem.plot(
            [x1, x2],
            [plotted_y1, plotted_y2],
            pen=pg.mkPen(color, width=2.2, style=QtCore.Qt.PenStyle.DashLine),
            symbol="o",
            symbolSize=8,
            symbolBrush=pg.mkBrush(color),
            symbolPen=pg.mkPen("#ffffff", width=1),
        )
        horizontal = self.plot.plotItem.plot(
            [x1, x2],
            [plotted_y1, plotted_y1],
            pen=pg.mkPen("#2563eb", width=1.8, style=QtCore.Qt.PenStyle.DotLine),
        )
        vertical = self.plot.plotItem.plot(
            [x2, x2],
            [plotted_y1, plotted_y2],
            pen=pg.mkPen("#dc2626", width=1.8, style=QtCore.Qt.PenStyle.DotLine),
        )
        dt_ms = abs(x2 - x1) * 1000.0
        dv = abs(y2 - y1)
        distance = float(np.hypot(x2 - x1, y2 - y1))
        label = pg.TextItem(
            text=f"横 {self._fmt_time(dt_ms)} ms\n竖 {self._fmt_voltage(dv)} mV\n距 {distance:.4f}",
            color=pg.mkColor(color),
            anchor=(0.5, 1.0),
        )
        label.setPos((x1 + x2) / 2.0, max(plotted_y1, plotted_y2))
        self.plot.plotItem.addItem(label)
        self._ruler_items.extend([line, horizontal, vertical, label])
        self.status_label.setText(
            f"标尺：横向 {self._fmt_time(dt_ms)} ms，竖向 {self._fmt_voltage(dv)} mV，直线距离 {distance:.4f}。"
        )

    def clear_ruler_measurements(self) -> None:
        self.ruler_points.clear()
        self._clear_pending()
        self._clear_ruler_overlay()
        self.status_label.setText("已清除所有标尺。")

    def _clear_ruler_overlay(self) -> None:
        if self.plot is None:
            self._ruler_items.clear()
            return
        for item in self._ruler_items:
            try:
                self.plot.plotItem.removeItem(item)
            except Exception:
                pass
        self._ruler_items.clear()

    def _begin_range_edit(self, row: int) -> None:
        if row < 0 or row >= len(self.annotations.records):
            self.status_label.setText("修改范围失败：未选中有效标注。")
            return
        record = self.annotations.records[row]
        if record.start_time is None or record.end_time is None:
            self.status_label.setText("这条标注没有时间范围，不能用“修改范围”。")
            return
        signal_type = record.signal_type
        annotation_type = self._manual_annotation_type(record.annotation_type)
        subject = record.subject
        self.annotations.delete(row)
        self._refresh_annotation_table()
        self.update_annotation_overlay()
        self.click_mode = "replace_range"
        self._set_active_button(self.edit_range_button)
        self.replacement_annotation = (signal_type, annotation_type, subject)
        self.range_edit_row = None
        self.pending_points.clear()
        self._set_current_signal_type(signal_type)
        self.status_label.setText(f"已删除原标注，请为 {signal_type} / {annotation_type} 重新点击起点和终点。")

    def _replacement_range_click(self, x: float, plotted_y: float) -> None:
        if self.replacement_annotation is None or self.plot is None:
            self.click_mode = "select_annotation"
            self.status_label.setText("重画范围失败：没有可替换的标注。")
            return
        signal_type, annotation_type, subject = self.replacement_annotation
        raw_y = self.plot.raw_value_for_lead(signal_type, plotted_y)
        self.pending_points.append((signal_type, x, raw_y))
        if len(self.pending_points) < 2:
            self.plot.add_pending_marker(signal_type, x, "新起点")
            self.status_label.setText("重画范围：已选择起点，请点击新的终点。")
            return

        first, second = self.pending_points[:2]
        self.pending_points.clear()
        self.plot.clear_pending_marker()
        record = self._record_from_points(annotation_type, first, second)
        record.subject = subject
        self.annotations.add(record)
        self._refresh_annotation_table()
        self.annotation_table.selectRow(len(self.annotations.records) - 1)
        self.update_annotation_overlay()
        self.click_mode = "select_annotation"
        self._set_active_button(self.select_annotation_button)
        self.replacement_annotation = None
        self.status_label.setText(f"已重画范围：{signal_type} / {annotation_type}，统计未自动重算。")

    def _range_edit_click(self, time_seconds: float) -> None:
        if self.range_edit_row is None or self.range_edit_row >= len(self.annotations.records):
            self.status_label.setText("修改范围失败：未选中有效标注。")
            self.click_mode = "select_annotation"
            return
        record = self.annotations.records[self.range_edit_row]
        self.pending_points.append((record.signal_type, time_seconds, 0.0))
        if len(self.pending_points) < 2:
            if self.plot is not None:
                self.plot.add_pending_marker(record.signal_type, time_seconds, "新起点")
            self.status_label.setText("修改范围：已选择起点，请点击新的终点。")
            return

        first, second = self.pending_points[:2]
        self.pending_points.clear()
        if self.plot is not None:
            self.plot.clear_pending_marker()
        start_time = max(0.0, min(first[1], second[1]))
        end_time = max(first[1], second[1])
        if self.dataset is not None:
            max_time = (self.dataset.sample_count - 1) / FS
            end_time = min(end_time, max_time)
        record.start_time = start_time
        record.end_time = end_time
        if record.baseline_time is not None:
            record.baseline_time = float(np.clip(record.baseline_time, start_time, end_time))
        if record.peak_time is not None:
            record.peak_time = float(np.clip(record.peak_time, start_time, end_time))
        self._recalculate_annotation(record)
        self._refresh_annotation_table()
        self._select_annotation_row(self.range_edit_row)
        self.update_annotation_overlay()
        self.click_mode = "select_annotation"
        self.range_edit_row = None
        self.status_label.setText(f"已修改范围：{record.signal_type} / {record.annotation_type}")

    def _current_annotation_type(self) -> str | None:
        wave_type = self.wave_combo.currentText()
        if self.click_mode == "amplitude":
            return AMPLITUDE_ANNOTATION_BY_WAVE.get(wave_type)
        return TIME_ANNOTATION_BY_WAVE.get(wave_type, "自定义")

    def _add_r_peak(self, signal_type: str, time_seconds: float) -> None:
        if signal_type not in R_PEAK_CHANNELS:
            self.status_label.setText(f"{signal_type} 不是母亲或胎儿分离导联，暂不编辑R峰。")
            return
        sample_index = self._bounded_sample_index(time_seconds)
        key = (self.batch_spin.value(), signal_type)
        self.manual_r_peaks.setdefault(key, set()).add(sample_index)
        self.deleted_r_peaks.get(key, set()).discard(sample_index)
        self._set_current_peaks(signal_type, np.append(self.current_r_peaks.get(signal_type, []), sample_index))
        self._after_r_peaks_changed(signal_type)
        self.status_label.setText(f"已在 {signal_type} 的 {sample_index / FS:.3f}s 添加R峰。")

    def _delete_nearest_r_peak(self, signal_type: str, time_seconds: float) -> None:
        if signal_type not in self.current_r_peaks:
            self.status_label.setText(f"{signal_type} 当前没有自动R峰，无法删除。")
            return
        peaks = self.current_r_peaks[signal_type]
        if peaks.size == 0:
            self.status_label.setText(f"{signal_type} 当前没有可删除的R峰。")
            return

        clicked_sample = int(round(time_seconds * FS))
        nearest_index = int(np.argmin(np.abs(peaks - clicked_sample)))
        nearest_peak = int(peaks[nearest_index])
        tolerance = int(0.12 * FS)
        if abs(nearest_peak - clicked_sample) > tolerance:
            self.status_label.setText("点击位置附近没有R峰，请更靠近红/蓝色R峰点。")
            return

        key = (self.batch_spin.value(), signal_type)
        self.deleted_r_peaks.setdefault(key, set()).add(nearest_peak)
        self.manual_r_peaks.get(key, set()).discard(nearest_peak)
        self._set_current_peaks(signal_type, np.delete(peaks, nearest_index))
        self._after_r_peaks_changed(signal_type)
        self.status_label.setText(f"已删除 {signal_type} 在 {nearest_peak / FS:.3f}s 附近的错误R峰。")

    def _move_r_peak_click(self, signal_type: str, time_seconds: float) -> None:
        if self.r_move_source is None:
            peak = self._nearest_r_peak(signal_type, time_seconds)
            if peak is None:
                self.status_label.setText("点击位置附近没有R峰，请先点要移动的R峰。")
                return
            self.r_move_source = (signal_type, peak)
            if self.plot is not None:
                self.plot.add_pending_marker(signal_type, peak / FS, "原R峰")
            self.status_label.setText(f"已选择 {signal_type} 的 {peak / FS:.3f}s R峰，请点击新位置。")
            return

        source_signal, old_peak = self.r_move_source
        if source_signal != signal_type:
            self.status_label.setText("移动R峰时请保持同一个导联。")
            return
        new_peak = self._bounded_sample_index(time_seconds)
        key = (self.batch_spin.value(), signal_type)
        self.deleted_r_peaks.setdefault(key, set()).add(old_peak)
        self.manual_r_peaks.setdefault(key, set()).add(new_peak)
        self.r_move_source = None
        if self.plot is not None:
            self.plot.clear_pending_marker()
        peaks = np.asarray([peak for peak in self.current_r_peaks.get(signal_type, []) if int(peak) != old_peak], dtype=int)
        self._set_current_peaks(signal_type, np.append(peaks, new_peak))
        self._after_r_peaks_changed(signal_type)
        self.status_label.setText(f"已将 {signal_type} 的R峰从 {old_peak / FS:.3f}s 移到 {new_peak / FS:.3f}s。")

    def _after_r_peaks_changed(self, signal_type: str) -> None:
        batch = self.batch_spin.value()
        group = self._group_index_for_signal(signal_type)
        self._preview_auto_records.pop((batch, group, signal_type), None)
        self._clear_joint_results()
        self._redraw_plot_without_joint()
        self._refresh_hrv_table()
        self._refresh_wave_stats_table()

    def _redraw_plot_without_joint(self) -> None:
        if self.plot is None:
            return
        view_range = self.plot.current_view_range()
        self.plot.draw(self.t, self.current_signals, self.current_r_peaks, FS, reset_view=False)
        self._ruler_items.clear()
        self.update_annotation_overlay()
        self.plot.restore_view_range(view_range)

    def _clear_joint_results(self) -> None:
        self.joint_results = None
        self._clear_joint_overlay()
        if hasattr(self, "joint_table"):
            self.joint_table.setRowCount(0)

    def _set_current_peaks(self, signal_type: str, peaks: np.ndarray) -> None:
        peaks = np.asarray(peaks, dtype=int)
        if self.dataset is not None:
            peaks = peaks[(peaks >= 0) & (peaks < self.dataset.sample_count)]
        self.current_r_peaks[signal_type] = np.unique(peaks)

    def _nearest_r_peak(self, signal_type: str, time_seconds: float) -> int | None:
        peaks = self.current_r_peaks.get(signal_type)
        if peaks is None or peaks.size == 0:
            return None
        clicked_sample = int(round(time_seconds * FS))
        nearest_index = int(np.argmin(np.abs(peaks - clicked_sample)))
        nearest_peak = int(peaks[nearest_index])
        if abs(nearest_peak - clicked_sample) > int(0.12 * FS):
            return None
        return nearest_peak

    def _bounded_sample_index(self, time_seconds: float) -> int:
        if self.dataset is None:
            return 0
        return int(np.clip(round(time_seconds * FS), 0, self.dataset.sample_count - 1))

    def auto_mark_pt(self) -> None:
        if self.dataset is None:
            return
        signal_type = self._current_signal_type()
        group = self._current_group_index()
        if signal_type not in self.current_r_peaks:
            self.status_label.setText(f"{signal_type} 没有R峰，无法自动估计P/T波。")
            return

        batch = self.batch_spin.value()
        before = len(self.annotations.records)
        self.annotations.records = [
            record
            for record in self.annotations.records
            if not (
                record.batch_index == batch
                and record.group_index == group
                and record.signal_type == signal_type
                and self._is_auto_record(record)
            )
        ]
        removed = before - len(self.annotations.records)

        created = 0
        try:
            auto_items = auto_wave_annotations(
                self.current_signals[signal_type],
                self.current_r_peaks[signal_type],
                FS,
                qrs_pre_percent=self.qrs_pre_spin.value(),
                qrs_post_percent=self.qrs_post_spin.value(),
                p_pre_percent=self.p_pre_spin.value(),
                p_post_percent=self.p_post_spin.value(),
                t_pre_percent=self.t_pre_spin.value(),
                t_post_percent=self.t_post_spin.value(),
            )
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "缺少依赖", str(exc))
            return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "自动标记失败", f"自动 P/QRS/T 运行失败：{exc}")
            return

        for item in auto_items:
            wave_type = str(item["wave_type"])
            note = str(item.get("note", ""))
            record = AnnotationRecord(
                source_file=display_name_for_file(self.current_file),
                batch_index=batch,
                group_index=group,
                subject=subject_for_signal(signal_type),
                signal_type=signal_type,
                annotation_type=f"{wave_type}自动",
                start_time=float(item["start_time"]),
                end_time=float(item["end_time"]),
                width_ms=float(item["width_ms"]),
                baseline_time=float(item["baseline_time"]),
                baseline_value=float(item["baseline_value"]),
                peak_time=float(item["peak_time"]),
                peak_value=float(item["peak_value"]),
                amplitude_mv=float(item["amplitude_mv"]),
                note=note,
            )
            if self._add_annotation_once(record):
                created += 1
        self._refresh_annotation_table()
        self.redraw_plot()
        self._refresh_wave_stats_table()
        self._refresh_joint_table()
        self._refresh_annotation_table()
        self._set_active_button(self.select_annotation_button)
        msg = f"已为 {signal_type} 自动生成 {created} 条 P/QRS/T 标注"
        if removed > 0:
            msg += f"，已清除 {removed} 条旧自动标注"
        msg += "，可直接在表格修改或删除。"
        self.status_label.setText(msg)

    def show_current_wave_analysis(self) -> None:
        if self.dataset is None:
            return
        signal_type = self._current_signal_type()
        batch = self.batch_spin.value()
        existing_count = sum(
            1
            for record in self.annotations.records
            if record.batch_index == batch
            and record.signal_type == signal_type
            and self._is_auto_record(record)
            and any(keyword in record.annotation_type for keyword in ("P波", "QRS", "T波"))
        )
        if existing_count == 0:
            created = self._compute_current_signal_auto_annotations(remove_existing=False)
            existing_count = created

        checkbox = self.show_signal_checkboxes.get(signal_type)
        if checkbox is not None and not checkbox.isChecked():
            checkbox.setChecked(True)
        else:
            self.update_annotation_overlay()
        self._refresh_wave_stats_table()
        self._refresh_joint_table()
        self._set_active_button(self.select_annotation_button)
        self.status_label.setText(f"已显示 {signal_type} 的 P/QRS/T 分析结果（{existing_count} 条）。")

    def _compute_current_signal_auto_annotations(self, remove_existing: bool) -> int:
        if self.dataset is None:
            return 0
        signal_type = self._current_signal_type()
        group = self._current_group_index()
        if signal_type not in self.current_r_peaks:
            self.status_label.setText(f"{signal_type} 没有R峰，无法进行 P/QRS/T 分析。")
            return 0

        batch = self.batch_spin.value()
        if remove_existing:
            self.annotations.records = [
                record
                for record in self.annotations.records
                if not (
                    record.batch_index == batch
                    and record.group_index == group
                    and record.signal_type == signal_type
                    and self._is_auto_record(record)
                )
            ]

        created = 0
        self.auto_pt_button.setEnabled(False)
        self.recalc_auto_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.status_label.setText(f"正在计算 {signal_type} 的 P/QRS/T，请稍候...")
        QtWidgets.QApplication.processEvents()
        try:
            auto_items = auto_wave_annotations(
                self.current_signals[signal_type],
                self.current_r_peaks[signal_type],
                FS,
                qrs_pre_percent=self.qrs_pre_spin.value(),
                qrs_post_percent=self.qrs_post_spin.value(),
                p_pre_percent=self.p_pre_spin.value(),
                p_post_percent=self.p_post_spin.value(),
                t_pre_percent=self.t_pre_spin.value(),
                t_post_percent=self.t_post_spin.value(),
            )
            for idx, item in enumerate(auto_items):
                record = self._auto_record_from_item(batch, group, signal_type, item)
                if self._add_annotation_once(record):
                    created += 1
                if idx % 20 == 0:
                    QtWidgets.QApplication.processEvents()
            self._refresh_annotation_table()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "缺少依赖", str(exc))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "P/QRS/T分析失败", f"P/QRS/T 分析运行失败：{exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.auto_pt_button.setEnabled(True)
            self.recalc_auto_button.setEnabled(True)
        return created

    def _precompute_all_auto_annotations(self) -> int:
        if self.dataset is None:
            return 0
        created = 0
        current_batch = self.batch_spin.value()
        self.status_label.setText("正在后台计算所有 batch 的 P/QRS/T 和联合分析，请稍候...")
        QtWidgets.QApplication.processEvents()
        for batch in range(self.dataset.batch_count):
            sample = self.dataset.data[batch]
            for signal_type, kind in R_PEAK_CHANNELS.items():
                channel_index = self._channel_index_for_signal(signal_type)
                signal_data = sample[channel_index]
                peaks = detect_r_peaks(signal_data, FS, kind)
                removed = self.deleted_r_peaks.get((batch, signal_type), set())
                if removed:
                    peaks = np.asarray([peak for peak in peaks if int(peak) not in removed], dtype=int)
                added = self.manual_r_peaks.get((batch, signal_type), set())
                if added:
                    peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
                if batch == current_batch:
                    self.current_r_peaks[signal_type] = peaks
                if peaks.size == 0:
                    continue
                try:
                    auto_items = auto_wave_annotations(
                        signal_data,
                        peaks,
                        FS,
                        qrs_pre_percent=self.qrs_pre_spin.value(),
                        qrs_post_percent=self.qrs_post_spin.value(),
                        p_pre_percent=self.p_pre_spin.value(),
                        p_post_percent=self.p_post_spin.value(),
                        t_pre_percent=self.t_pre_spin.value(),
                        t_post_percent=self.t_post_spin.value(),
                    )
                except RuntimeError as exc:
                    QtWidgets.QMessageBox.warning(self, "缺少依赖", str(exc))
                    return created
                except Exception:
                    continue
                group = self._group_index_for_signal(signal_type)
                for item in auto_items:
                    record = self._auto_record_from_item(batch, group, signal_type, item)
                    if self._add_annotation_once(record):
                        created += 1
            QtWidgets.QApplication.processEvents()
        return created

    def _auto_record_from_item(
        self,
        batch: int,
        group: int,
        signal_type: str,
        item: dict[str, float | str],
    ) -> AnnotationRecord:
        wave_type = str(item["wave_type"])
        return AnnotationRecord(
            source_file=display_name_for_file(self.current_file),
            batch_index=batch,
            group_index=group,
            subject=subject_for_signal(signal_type),
            signal_type=signal_type,
            annotation_type=f"{wave_type}自动",
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            width_ms=float(item["width_ms"]),
            baseline_time=float(item["baseline_time"]),
            baseline_value=float(item["baseline_value"]),
            peak_time=float(item["peak_time"]),
            peak_value=float(item["peak_value"]),
            amplitude_mv=float(item["amplitude_mv"]),
            note=str(item.get("note", "")),
        )

    @staticmethod
    def _channel_index_for_signal(signal_type: str) -> int:
        for index, name in CHANNEL_NAMES.items():
            if name == signal_type:
                return index
        return 0

    def recalculate_current_auto_annotations(self) -> None:
        if self.dataset is None:
            return
        batch = self.batch_spin.value()
        updated = 0
        for record in self.annotations.records:
            if record.batch_index != batch:
                continue
            self._recalculate_annotation(record)
            updated += 1
        self._refresh_annotation_table()
        self._refresh_all_stats()
        if self.joint_results is not None:
            self._run_joint_analysis()
        self.update_annotation_overlay()
        self.status_label.setText(f"已根据当前标注结果重算统计和联合分析，共刷新 {updated} 条标注。")

    def reanalyze_current_signal(self) -> None:
        if self.dataset is None:
            return
        if self._recalc_running:
            self.status_label.setText("正在重算中，请等待当前操作完成。")
            return
        signal_type = self._current_signal_type()
        if signal_type not in self.current_r_peaks or self.current_r_peaks[signal_type].size == 0:
            self.status_label.setText(f"{signal_type} 没有R峰，无法重算。")
            return

        self._recalc_running = True
        self.recalc_auto_button.setEnabled(False)
        self.refresh_stats_button.setEnabled(False)
        self.auto_pt_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.status_label.setText(f"正在按新参数重算 {signal_type} 的 P/QRS/T，请稍候...")
        QtWidgets.QApplication.processEvents()
        QtCore.QTimer.singleShot(10, self._do_reanalyze_current_signal)

    def _do_reanalyze_current_signal(self) -> None:
        try:
            signal_type = self._current_signal_type()
            group = self._current_group_index()
            batch = self.batch_spin.value()

            auto_items = auto_wave_annotations(
                self.current_signals[signal_type],
                self.current_r_peaks[signal_type],
                FS,
                qrs_pre_percent=self.qrs_pre_spin.value(),
                qrs_post_percent=self.qrs_post_spin.value(),
                p_pre_percent=self.p_pre_spin.value(),
                p_post_percent=self.p_post_spin.value(),
                t_pre_percent=self.t_pre_spin.value(),
                t_post_percent=self.t_post_spin.value(),
            )

            preview_records = [
                self._auto_record_from_item(batch, group, signal_type, item)
                for item in auto_items
            ]
            self._preview_auto_records[(batch, group, signal_type)] = preview_records

            checkbox = self.show_signal_checkboxes.get(signal_type)
            if checkbox is not None and not checkbox.isChecked():
                checkbox.setChecked(True)
            subject = subject_for_signal(signal_type)
            if subject == "母亲" and not self.show_auto_mother_checkbox.isChecked():
                self.show_auto_mother_checkbox.setChecked(True)
            if subject == "胎儿" and not self.show_auto_fetal_checkbox.isChecked():
                self.show_auto_fetal_checkbox.setChecked(True)

            self.redraw_plot()
            self.status_label.setText(
                f"已按当前参数重算 {signal_type} 的 P/QRS/T 预览 {len(preview_records)} 条；点击“刷新统计”后写入表格。"
            )
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "缺少依赖", str(exc))
            self.status_label.setText(f"重算失败：{exc}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "重算失败", f"P/QRS/T 分析运行失败：{exc}")
            self.status_label.setText(f"重算失败：{exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.recalc_auto_button.setEnabled(True)
            self.refresh_stats_button.setEnabled(True)
            self.auto_pt_button.setEnabled(True)
            self._recalc_running = False

    def visible_annotation_items(self) -> list[tuple[int | None, AnnotationRecord]]:
        batch = self.batch_spin.value()
        visible: list[tuple[int | None, AnnotationRecord]] = []
        enabled_signals = {
            signal_type
            for signal_type, checkbox in self.show_signal_checkboxes.items()
            if checkbox.isChecked()
        }
        preview_keys = {
            key for key, records in self._preview_auto_records.items()
            if key[0] == batch and records
        }
        for index, record in enumerate(self.annotations.records):
            if record.batch_index != batch:
                continue
            if record.signal_type in self.show_signal_checkboxes and record.signal_type not in enabled_signals:
                continue
            record_key = (record.batch_index, record.group_index, record.signal_type)
            if self._is_auto_record(record) and record_key in preview_keys:
                continue
            if self._is_auto_record(record):
                if record.subject == "母亲" and not self.show_auto_mother_checkbox.isChecked():
                    continue
                if record.subject == "胎儿" and not self.show_auto_fetal_checkbox.isChecked():
                    continue
            visible.append((index, record))
        for key, records in self._preview_auto_records.items():
            preview_batch, _group, signal_type = key
            if preview_batch != batch or not records:
                continue
            if signal_type in self.show_signal_checkboxes and signal_type not in enabled_signals:
                continue
            for record in records:
                if record.subject == "母亲" and not self.show_auto_mother_checkbox.isChecked():
                    continue
                if record.subject == "胎儿" and not self.show_auto_fetal_checkbox.isChecked():
                    continue
                visible.append((None, record))
        return visible

    def visible_annotation_records(self) -> list[AnnotationRecord]:
        return [record for _, record in self.visible_annotation_items()]

    @staticmethod
    def _is_auto_record(record: AnnotationRecord) -> bool:
        return record.annotation_type.endswith("自动") or record.note.startswith("自动")

    def _add_annotation_once(self, record: AnnotationRecord, tolerance_seconds: float | None = None) -> bool:
        if tolerance_seconds is None:
            tolerance_seconds = 1.5 / FS
        for existing in self.annotations.records:
            if (
                existing.batch_index == record.batch_index
                and existing.group_index == record.group_index
                and existing.signal_type == record.signal_type
                and existing.annotation_type == record.annotation_type
                and self._close_time(existing.start_time, record.start_time, tolerance_seconds)
                and self._close_time(existing.end_time, record.end_time, tolerance_seconds)
            ):
                return False
        self.annotations.add(record)
        return True

    def _commit_preview_auto_records(self) -> int:
        if not self._preview_auto_records:
            return 0
        preview_keys = {key for key, records in self._preview_auto_records.items() if records}
        if not preview_keys:
            self._preview_auto_records.clear()
            return 0
        self.annotations.records = [
            record
            for record in self.annotations.records
            if not (
                self._is_auto_record(record)
                and (record.batch_index, record.group_index, record.signal_type) in preview_keys
            )
        ]
        created = 0
        for records in self._preview_auto_records.values():
            for record in records:
                self.annotations.add(record)
                created += 1
        self._preview_auto_records.clear()
        return created

    @staticmethod
    def _close_time(left: float | None, right: float | None, tolerance_seconds: float) -> bool:
        if left is None or right is None:
            return left is right
        return abs(float(left) - float(right)) <= tolerance_seconds

    @staticmethod
    def _manual_annotation_type(annotation_type: str) -> str:
        if "P波" in annotation_type:
            return "P波宽度"
        if "QRS" in annotation_type:
            return "QRS宽度"
        if "T波" in annotation_type:
            return "T波宽度"
        return annotation_type.replace("自动", "")

    def redraw_plot(self) -> None:
        if self.plot is None:
            return
        view_range = self.plot.current_view_range()
        self.plot.draw(self.t, self.current_signals, self.current_r_peaks, FS, reset_view=False)
        self._ruler_items.clear()
        self.update_annotation_overlay()
        self.plot.restore_view_range(view_range)

    def update_annotation_overlay(self) -> None:
        if self.plot is None:
            return
        self.plot.draw_annotations(
            self.visible_annotation_items(),
            self._selected_annotation_row(),
            self.show_label_checkbox.isChecked(),
        )
        if self.joint_results is not None and self.show_joint_checkbox.isChecked():
            self._draw_joint_on_plot()
        else:
            self._clear_joint_overlay()

    def _toggle_joint_overlay(self) -> None:
        if self.show_joint_checkbox.isChecked():
            self._draw_joint_on_plot()
        else:
            self._clear_joint_overlay()

    def reset_initial_view(self) -> None:
        if self.plot is not None:
            self.plot.reset_initial_view()
            self.status_label.setText("已返回初始 10 秒视图。")

    def _record_from_points(
        self,
        annotation_type: str,
        first: tuple[str, float, float],
        second: tuple[str, float, float],
    ) -> AnnotationRecord:
        batch = self.batch_spin.value()
        signal_type = first[0]
        group = self._group_index_for_signal(signal_type)
        note = self.note_edit.text()

        if annotation_type in AMPLITUDE_ANNOTATIONS:
            amplitude = second[2] - first[2]
            return AnnotationRecord(
                source_file=display_name_for_file(self.current_file),
                batch_index=batch,
                group_index=group,
                subject=subject_for_signal(signal_type),
                signal_type=signal_type,
                annotation_type=annotation_type,
                baseline_time=first[1],
                baseline_value=first[2],
                peak_time=second[1],
                peak_value=second[2],
                amplitude_mv=amplitude,
                note=note,
            )

        start_time = min(first[1], second[1])
        end_time = max(first[1], second[1])
        return AnnotationRecord(
            source_file=display_name_for_file(self.current_file),
            batch_index=batch,
            group_index=group,
            subject=subject_for_signal(signal_type),
            signal_type=signal_type,
            annotation_type=annotation_type if annotation_type in WIDTH_ANNOTATIONS else "自定义",
            start_time=start_time,
            end_time=end_time,
            width_ms=(end_time - start_time) * 1000.0,
            note=note,
        )

    def _refresh_hrv_table(self) -> None:
        rows = []
        for signal_type in R_PEAK_CHANNELS:
            metrics = compute_hrv_metrics(self.current_r_peaks.get(signal_type, []), FS)
            rows.append((signal_type, metrics))

        self.hrv_table.setUpdatesEnabled(False)
        self.hrv_table.setRowCount(len(rows))
        for row, (signal_type, metrics) in enumerate(rows):
            values = [
                signal_type,
                str(metrics["R峰数量"]),
                self._fmt_time(metrics["Mean RR (ms)"]),
                self._fmt_int(metrics["Mean HR (bpm)"]),
                f'{self._fmt_time(metrics["SDNN (ms)"])} / {self._fmt_time(metrics["RMSSD (ms)"])}',
            ]
            for col, value in enumerate(values):
                self.hrv_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self.hrv_table.setUpdatesEnabled(True)

    def _refresh_wave_stats_table(self) -> None:
        if not hasattr(self, "wave_stats_table"):
            return
        self.wave_stats_table.setHorizontalHeaderItem(
            14, QtWidgets.QTableWidgetItem(f"{self.beat_split_spin.value():.0f}%RR (ms)")
        )
        current_batch = self.batch_spin.value()
        batch_records = [
            record for record in self.annotations.records
            if record.batch_index == current_batch
            and record.start_time is not None
            and record.end_time is not None
        ]
        rows = []
        for signal_type in DISPLAY_SIGNALS:
            stats = self._wave_interval_stats_from_records(
                batch_records, signal_type, "整体"
            )
            if stats["count"] > 0:
                rows.append((signal_type, stats))

        self.wave_stats_table.setUpdatesEnabled(False)
        self.wave_stats_table.setRowCount(0)
        self.wave_stats_table.clearContents()
        if not rows:
            self.wave_stats_table.setRowCount(1)
            self.wave_stats_table.setItem(
                0,
                0,
                QtWidgets.QTableWidgetItem("暂无波段平均结果，请先点击 P/QRS/T分析 或确认R峰已识别。"),
            )
            self.wave_stats_table.setUpdatesEnabled(True)
            return

        self.wave_stats_table.setRowCount(len(rows))
        for row, (signal_type, stats) in enumerate(rows):
            values = [
                signal_type,
                self._fmt_time(stats["P"]),
                self._fmt_voltage(stats["P_amp"]),
                self._fmt_time(stats["PR"]),
                self._fmt_time(stats["PR_min"]),
                self._fmt_time(stats["PR_max"]),
                self._fmt_time(stats["QRS"]),
                self._fmt_voltage(stats["QRS_amp"]),
                self._fmt_time(stats["ST"]),
                self._fmt_time(stats["ST_min"]),
                self._fmt_time(stats["ST_max"]),
                self._fmt_time(stats["T"]),
                self._fmt_voltage(stats["T_amp"]),
                self._fmt_time(stats["RR"]),
                self._fmt_time(stats["split_rr"]),
                str(stats["count"]),
            ]
            for col, value in enumerate(values):
                self.wave_stats_table.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self.wave_stats_table.setUpdatesEnabled(True)

    def _wave_interval_stats(self, signal_type: str, source: str) -> dict[str, float | int]:
        batch = self.batch_spin.value()
        records = [
            record
            for record in self.annotations.records
            if record.batch_index == batch
            and record.signal_type == signal_type
            and record.start_time is not None
            and record.end_time is not None
            and (
                source == "整体"
                or (source == "自动" and self._is_auto_record(record))
                or (source == "手动" and not self._is_auto_record(record))
            )
        ]

        def widths(keyword: str) -> list[float]:
            return [
                float(record.width_ms)
                for record in records
                if keyword in record.annotation_type and record.width_ms is not None
            ]

        def amplitudes(keyword: str) -> list[float]:
            return [
                float(record.amplitude_mv)
                for record in records
                if keyword in record.annotation_type and record.amplitude_mv is not None
            ]

        if source == "整体":
            records = self._prefer_manual_wave_records(signal_type, records)

        p_records = sorted([record for record in records if "P波" in record.annotation_type], key=lambda item: item.start_time or 0)
        qrs_records = sorted([record for record in records if "QRS" in record.annotation_type], key=lambda item: item.start_time or 0)
        t_records = sorted([record for record in records if "T波" in record.annotation_type], key=lambda item: item.start_time or 0)

        pr_values: list[float] = []
        st_values: list[float] = []
        for qrs in qrs_records:
            previous_p = [
                p
                for p in p_records
                if p.start_time is not None and p.start_time < (qrs.start_time or 0)
            ]
            next_t = [
                t
                for t in t_records
                if t.start_time is not None and t.start_time > (qrs.end_time or 0)
            ]
            if previous_p:
                p = previous_p[-1]
                pr_values.append(((qrs.start_time or 0.0) - (p.start_time or 0.0)) * 1000.0)
            if next_t:
                t = next_t[0]
                st_values.append(((t.start_time or 0.0) - (qrs.end_time or 0.0)) * 1000.0)

        peaks = np.asarray(self.current_r_peaks.get(signal_type, []), dtype=float)
        rr_values = np.diff(peaks) / FS * 1000.0 if source in {"整体", "自动"} and peaks.size > 1 else np.array([], dtype=float)
        split_ratio = self.beat_split_spin.value() / 100.0
        return {
            "P": self._mean(widths("P波")),
            "P_amp": self._mean(amplitudes("P波")),
            "PR": self._mean(pr_values),
            "PR_min": self._min(pr_values),
            "PR_max": self._max(pr_values),
            "QRS": self._mean(widths("QRS")),
            "QRS_amp": self._mean(amplitudes("QRS")),
            "ST": self._mean(st_values),
            "ST_min": self._min(st_values),
            "ST_max": self._max(st_values),
            "T": self._mean(widths("T波")),
            "T_amp": self._mean(amplitudes("T波")),
            "RR": self._mean(rr_values),
            "split_rr": self._mean(rr_values * split_ratio),
            "count": max(len(qrs_records), len(p_records), len(t_records), int(rr_values.size)),
        }

    def _wave_interval_stats_from_records(
        self, batch_records: list["AnnotationRecord"], signal_type: str, source: str
    ) -> dict[str, float | int]:
        records = [
            record
            for record in batch_records
            if record.signal_type == signal_type
            and (
                source == "整体"
                or (source == "自动" and self._is_auto_record(record))
                or (source == "手动" and not self._is_auto_record(record))
            )
        ]

        def widths(keyword: str) -> list[float]:
            return [
                float(record.width_ms)
                for record in records
                if keyword in record.annotation_type and record.width_ms is not None
            ]

        def amplitudes(keyword: str) -> list[float]:
            return [
                float(record.amplitude_mv)
                for record in records
                if keyword in record.annotation_type and record.amplitude_mv is not None
            ]

        if source == "整体":
            records = self._prefer_manual_wave_records(signal_type, records)

        p_records = sorted([record for record in records if "P波" in record.annotation_type], key=lambda item: item.start_time or 0)
        qrs_records = sorted([record for record in records if "QRS" in record.annotation_type], key=lambda item: item.start_time or 0)
        t_records = sorted([record for record in records if "T波" in record.annotation_type], key=lambda item: item.start_time or 0)

        pr_values: list[float] = []
        p_idx = 0
        for qrs in qrs_records:
            qrs_start = qrs.start_time or 0.0
            while p_idx < len(p_records) and (p_records[p_idx].start_time or 0.0) < qrs_start:
                p_idx += 1
            if p_idx > 0:
                p = p_records[p_idx - 1]
                pr_values.append((qrs_start - (p.start_time or 0.0)) * 1000.0)

        st_values: list[float] = []
        t_idx = 0
        for qrs in qrs_records:
            qrs_end = qrs.end_time or 0.0
            while t_idx < len(t_records) and (t_records[t_idx].start_time or 0.0) <= qrs_end:
                t_idx += 1
            if t_idx < len(t_records):
                t = t_records[t_idx]
                st_values.append(((t.start_time or 0.0) - qrs_end) * 1000.0)

        peaks = np.asarray(self.current_r_peaks.get(signal_type, []), dtype=float)
        rr_values = np.diff(peaks) / FS * 1000.0 if source in {"整体", "自动"} and peaks.size > 1 else np.array([], dtype=float)
        split_ratio = self.beat_split_spin.value() / 100.0
        return {
            "P": self._mean(widths("P波")),
            "P_amp": self._mean(amplitudes("P波")),
            "PR": self._mean(pr_values),
            "PR_min": self._min(pr_values),
            "PR_max": self._max(pr_values),
            "QRS": self._mean(widths("QRS")),
            "QRS_amp": self._mean(amplitudes("QRS")),
            "ST": self._mean(st_values),
            "ST_min": self._min(st_values),
            "ST_max": self._max(st_values),
            "T": self._mean(widths("T波")),
            "T_amp": self._mean(amplitudes("T波")),
            "RR": self._mean(rr_values),
            "split_rr": self._mean(rr_values * split_ratio),
            "count": max(len(qrs_records), len(p_records), len(t_records), int(rr_values.size)),
        }

    def _wave_interval_stats_for_batch(self, signal_type: str, source: str, batch: int) -> dict[str, float | int]:
        records = [
            record
            for record in self.annotations.records
            if record.batch_index == batch
            and record.signal_type == signal_type
            and record.start_time is not None
            and record.end_time is not None
            and (
                source == "整体"
                or (source == "自动" and self._is_auto_record(record))
                or (source == "手动" and not self._is_auto_record(record))
            )
        ]

        def widths(keyword: str) -> list[float]:
            return [
                float(record.width_ms)
                for record in records
                if keyword in record.annotation_type and record.width_ms is not None
            ]

        def amplitudes(keyword: str) -> list[float]:
            return [
                float(record.amplitude_mv)
                for record in records
                if keyword in record.annotation_type and record.amplitude_mv is not None
            ]

        if source == "整体":
            records = self._prefer_manual_wave_records(signal_type, records, batch=batch)

        p_records = sorted([record for record in records if "P波" in record.annotation_type], key=lambda item: item.start_time or 0)
        qrs_records = sorted([record for record in records if "QRS" in record.annotation_type], key=lambda item: item.start_time or 0)
        t_records = sorted([record for record in records if "T波" in record.annotation_type], key=lambda item: item.start_time or 0)

        pr_values: list[float] = []
        st_values: list[float] = []
        for qrs in qrs_records:
            previous_p = [
                p
                for p in p_records
                if p.start_time is not None and p.start_time < (qrs.start_time or 0)
            ]
            next_t = [
                t
                for t in t_records
                if t.start_time is not None and t.start_time > (qrs.end_time or 0)
            ]
            if previous_p:
                p = previous_p[-1]
                pr_values.append(((qrs.start_time or 0.0) - (p.start_time or 0.0)) * 1000.0)
            if next_t:
                t = next_t[0]
                st_values.append(((t.start_time or 0.0) - (qrs.end_time or 0.0)) * 1000.0)

        sample = self.dataset.data[batch]
        channel_index = 0
        for idx_name, name in CHANNEL_NAMES.items():
            if name == signal_type:
                channel_index = idx_name
                break
        signal_data = sample[channel_index]
        kind = R_PEAK_CHANNELS.get(signal_type, "adult")
        peaks = detect_r_peaks(signal_data, FS, kind)
        removed = self.deleted_r_peaks.get((batch, signal_type), set())
        if removed:
            peaks = np.asarray([p for p in peaks if int(p) not in removed], dtype=int)
        added = self.manual_r_peaks.get((batch, signal_type), set())
        if added:
            peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
        rr_values = np.diff(np.asarray(peaks, dtype=float)) / FS * 1000.0 if source in {"整体", "自动"} and peaks.size > 1 else np.array([], dtype=float)
        split_ratio = self.beat_split_spin.value() / 100.0
        return {
            "P": self._mean(widths("P波")),
            "P_amp": self._mean(amplitudes("P波")),
            "PR": self._mean(pr_values),
            "PR_min": self._min(pr_values),
            "PR_max": self._max(pr_values),
            "QRS": self._mean(widths("QRS")),
            "QRS_amp": self._mean(amplitudes("QRS")),
            "ST": self._mean(st_values),
            "ST_min": self._min(st_values),
            "ST_max": self._max(st_values),
            "T": self._mean(widths("T波")),
            "T_amp": self._mean(amplitudes("T波")),
            "RR": self._mean(rr_values),
            "split_rr": self._mean(rr_values * split_ratio),
            "count": max(len(qrs_records), len(p_records), len(t_records), int(rr_values.size)),
        }

    def _refresh_annotation_table(self) -> None:
        self._updating_annotation_table = True
        batch = self.batch_spin.value()
        self._annotation_table_record_indices = [
            index for index, record in enumerate(self.annotations.records) if record.batch_index == batch
        ]
        batch_records = [self.annotations.records[index] for index in self._annotation_table_record_indices]
        temp_store = AnnotationStore()
        temp_store.records = batch_records
        df = temp_store.to_dataframe()
        self.annotation_table.blockSignals(True)
        self.annotation_table.setUpdatesEnabled(False)
        self.annotation_table.setRowCount(len(df))
        time_fields = {"start_time", "end_time", "width_ms", "baseline_time", "peak_time"}
        voltage_fields = {"baseline_value", "peak_value", "amplitude_mv"}
        field_names = list(AnnotationRecord.__dataclass_fields__.keys())
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                if isinstance(value, float):
                    if np.isnan(value):
                        text = ""
                    elif col_idx < len(field_names) and field_names[col_idx] in voltage_fields:
                        text = self._fmt_voltage(value)
                    else:
                        text = self._fmt_time(value)
                else:
                    text = "" if value is None else str(value)
                self.annotation_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(text))
        self.annotation_table.setUpdatesEnabled(True)
        self.annotation_table.blockSignals(False)
        self._updating_annotation_table = False

    def annotation_table_selection_changed(self) -> None:
        if self._updating_annotation_table:
            return
        self.update_annotation_overlay()

    def annotation_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_annotation_table:
            return
        row = item.row()
        col = item.column()
        record_index = self._record_index_for_table_row(row)
        if record_index is None:
            return
        field_names = list(AnnotationRecord.__dataclass_fields__.keys())
        if col < 0 or col >= len(field_names):
            return

        record = self.annotations.records[record_index]
        field = field_names[col]
        text = item.text().strip()
        try:
            if field in {"batch_index", "group_index"}:
                setattr(record, field, int(text))
            elif field in {
                "start_time",
                "end_time",
                "width_ms",
                "baseline_time",
                "baseline_value",
                "peak_time",
                "peak_value",
                "amplitude_mv",
            }:
                setattr(record, field, None if text == "" else float(text))
            else:
                setattr(record, field, text)
        except ValueError:
            self.status_label.setText("表格修改失败：数值列请输入数字。")
            self._refresh_annotation_table()
            return

        self._recalculate_annotation(record)
        self._refresh_annotation_table()
        self.update_annotation_overlay()
        self.status_label.setText(f"已更新标注：{record.signal_type} / {record.annotation_type}")

    @staticmethod
    def _recalculate_annotation(record: AnnotationRecord) -> None:
        record.subject = subject_for_signal(record.signal_type)
        if record.start_time is not None and record.end_time is not None:
            start = min(record.start_time, record.end_time)
            end = max(record.start_time, record.end_time)
            record.start_time = start
            record.end_time = end
            record.width_ms = (end - start) * 1000.0
        if record.baseline_value is not None and record.peak_value is not None:
            record.amplitude_mv = record.peak_value - record.baseline_value

    def delete_selected_annotation(self) -> None:
        rows = set()
        for selected_range in self.annotation_table.selectedRanges():
            rows.update(range(selected_range.topRow(), selected_range.bottomRow() + 1))
        record_indices = [
            record_index
            for row in rows
            if (record_index := self._record_index_for_table_row(row)) is not None
        ]
        record_indices = sorted(set(record_indices), reverse=True)
        if not rows:
            row = self.annotation_table.currentRow()
            record_index = self._record_index_for_table_row(row)
            record_indices = [record_index] if record_index is not None else []
        if not record_indices:
            self.status_label.setText("请先在标注结果表中选择一条记录。")
            return
        deleted_count = 0
        for record_index in record_indices:
            if self.annotations.delete(record_index) is not None:
                deleted_count += 1
        self._refresh_annotation_table()
        self.update_annotation_overlay()
        self.status_label.setText(f"已删除 {deleted_count} 条标注。")

    def export_annotations(self) -> None:
        if self.dataset is None:
            QtWidgets.QMessageBox.information(self, "无数据", "请先打开数据文件。")
            return
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            str(Path.cwd()),
        )
        if not dir_path:
            return
        dir_path = Path(dir_path)

        # 自动为所有 batch 生成标注
        self.status_label.setText("正在为所有样本自动标记 P/QRS/T，请稍候...")
        QtWidgets.QApplication.processEvents()
        for batch in range(self.dataset.batch_count):
            sample = self.dataset.data[batch]
            for signal_type, kind in R_PEAK_CHANNELS.items():
                channel_index = 0
                for idx, name in CHANNEL_NAMES.items():
                    if name == signal_type:
                        channel_index = idx
                        break
                signal_data = sample[channel_index]
                peaks = detect_r_peaks(signal_data, FS, kind)
                removed = self.deleted_r_peaks.get((batch, signal_type), set())
                if removed:
                    peaks = np.asarray([p for p in peaks if int(p) not in removed], dtype=int)
                added = self.manual_r_peaks.get((batch, signal_type), set())
                if added:
                    peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
                if peaks.size < 2:
                    continue
                # 检查该 batch/signal 是否已有自动标注
                try:
                    auto_items = auto_wave_annotations(
                        signal_data,
                        peaks,
                        FS,
                        qrs_pre_percent=self.qrs_pre_spin.value(),
                        qrs_post_percent=self.qrs_post_spin.value(),
                        p_pre_percent=self.p_pre_spin.value(),
                        p_post_percent=self.p_post_spin.value(),
                        t_pre_percent=self.t_pre_spin.value(),
                        t_post_percent=self.t_post_spin.value(),
                    )
                except Exception:
                    continue
                group = self._group_index_for_signal(signal_type)
                for item in auto_items:
                    wave_type = str(item["wave_type"])
                    note = str(item.get("note", ""))
                    record = AnnotationRecord(
                        source_file=display_name_for_file(self.current_file),
                        batch_index=batch,
                        group_index=group,
                        subject=subject_for_signal(signal_type),
                        signal_type=signal_type,
                        annotation_type=f"{wave_type}自动",
                        start_time=float(item["start_time"]),
                        end_time=float(item["end_time"]),
                        width_ms=float(item["width_ms"]),
                        baseline_time=float(item["baseline_time"]),
                        baseline_value=float(item["baseline_value"]),
                        peak_time=float(item["peak_time"]),
                        peak_value=float(item["peak_value"]),
                        amplitude_mv=float(item["amplitude_mv"]),
                        note=note,
                    )
                    self._add_annotation_once(record)

        # 1. 导出所有标注
        if self.annotations.records:
            self.annotations.export_csv(dir_path / "annotations.csv")

        # 2. 导出所有 batch 的 HRV
        hrv_rows: list[dict] = []
        for batch in range(self.dataset.batch_count):
            sample = self.dataset.data[batch]
            for signal_type, kind in R_PEAK_CHANNELS.items():
                channel_index = 0
                for idx, name in CHANNEL_NAMES.items():
                    if name == signal_type:
                        channel_index = idx
                        break
                peaks = detect_r_peaks(sample[channel_index], FS, kind)
                removed = self.deleted_r_peaks.get((batch, signal_type), set())
                if removed:
                    peaks = np.asarray([p for p in peaks if int(p) not in removed], dtype=int)
                added = self.manual_r_peaks.get((batch, signal_type), set())
                if added:
                    peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
                metrics = compute_hrv_metrics(peaks, FS)
                row_data: dict = {"batch": batch, "signal": signal_type}
                row_data.update(metrics)
                hrv_rows.append(row_data)
        if hrv_rows:
            pd.DataFrame(hrv_rows).to_csv(dir_path / "hrv.csv", index=False, encoding="utf-8-sig")

        # 3. 导出所有 batch 的波段统计
        wave_rows: list[dict] = []
        for batch in range(self.dataset.batch_count):
            for signal_type in DISPLAY_SIGNALS:
                stats = self._wave_interval_stats_for_batch(signal_type, "整体", batch)
                if stats["count"] > 0:
                    row_data: dict = {"batch": batch, "signal": signal_type}
                    row_data.update(stats)
                    wave_rows.append(row_data)
        if wave_rows:
            wave_df = pd.DataFrame(wave_rows)
            wave_df.rename(columns={
                "P": "P (ms)",
                "P_amp": "P amplitude (mV)",
                "PR": "PR (ms)",
                "PR_min": "PR min (ms)",
                "PR_max": "PR max (ms)",
                "QRS": "QRS (ms)",
                "QRS_amp": "QRS amplitude (mV)",
                "ST": "ST (ms)",
                "ST_min": "ST min (ms)",
                "ST_max": "ST max (ms)",
                "T": "T (ms)",
                "T_amp": "T amplitude (mV)",
                "RR": "RR (ms)",
                "split_rr": "split_rr (ms)",
            }, inplace=True)
            wave_df.to_csv(dir_path / "wave_stats.csv", index=False, encoding="utf-8-sig")
        # 4. 导出所有 batch 的逐心搏详情
        beat_rows: list[dict] = []
        for batch in range(self.dataset.batch_count):
            sample = self.dataset.data[batch]
            for signal_type in R_PEAK_CHANNELS:
                channel_index = 0
                for idx, name in CHANNEL_NAMES.items():
                    if name == signal_type:
                        channel_index = idx
                        break
                signal_data = sample[channel_index]
                kind = R_PEAK_CHANNELS.get(signal_type, "adult")
                peaks = detect_r_peaks(signal_data, FS, kind)
                removed = self.deleted_r_peaks.get((batch, signal_type), set())
                if removed:
                    peaks = np.asarray([p for p in peaks if int(p) not in removed], dtype=int)
                added = self.manual_r_peaks.get((batch, signal_type), set())
                if added:
                    peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
                if peaks.size < 2:
                    continue

                batch_records = [r for r in self.annotations.records if r.batch_index == batch and r.signal_type == signal_type]
                p_records = sorted([r for r in batch_records if "P波" in r.annotation_type], key=lambda x: x.start_time or 0)
                qrs_records = sorted([r for r in batch_records if "QRS" in r.annotation_type], key=lambda x: x.start_time or 0)
                t_records = sorted([r for r in batch_records if "T波" in r.annotation_type], key=lambda x: x.start_time or 0)

                rr_intervals = np.diff(peaks.astype(float)) / FS * 1000.0

                for i, peak in enumerate(peaks):
                    beat_data = {
                        "batch": batch,
                        "signal": signal_type,
                        "beat_index": i,
                        "r_peak_time (s)": round(peak / FS, 4),
                    }

                    if i > 0:
                        beat_data["rr_interval (ms)"] = round(rr_intervals[i - 1], 2)
                    if i + 1 < len(peaks):
                        beat_data["next_rr (ms)"] = round(rr_intervals[i], 2)

                    nearest_p = next((r for r in p_records if r.start_time is not None and abs(r.start_time - peak / FS) < 0.5), None)
                    nearest_qrs = next((r for r in qrs_records if r.start_time is not None and abs(r.start_time - peak / FS) < 0.15), None)
                    nearest_t = next((r for r in t_records if r.start_time is not None and abs(r.start_time - peak / FS) < 0.6), None)

                    if nearest_p and nearest_p.start_time is not None:
                        beat_data["p_onset (s)"] = round(nearest_p.start_time, 4)
                        beat_data["p_offset (s)"] = round(nearest_p.end_time, 4) if nearest_p.end_time is not None else None
                        beat_data["p_width (ms)"] = round(nearest_p.width_ms, 2) if nearest_p.width_ms is not None else None
                        beat_data["p_amplitude (mV)"] = round(nearest_p.amplitude_mv, 4) if nearest_p.amplitude_mv is not None else None

                    if nearest_qrs and nearest_qrs.start_time is not None:
                        beat_data["qrs_onset (s)"] = round(nearest_qrs.start_time, 4)
                        beat_data["qrs_offset (s)"] = round(nearest_qrs.end_time, 4) if nearest_qrs.end_time is not None else None
                        beat_data["qrs_width (ms)"] = round(nearest_qrs.width_ms, 2) if nearest_qrs.width_ms is not None else None
                        beat_data["qrs_amplitude (mV)"] = round(nearest_qrs.amplitude_mv, 4) if nearest_qrs.amplitude_mv is not None else None

                    if nearest_t and nearest_t.start_time is not None:
                        beat_data["t_onset (s)"] = round(nearest_t.start_time, 4)
                        beat_data["t_offset (s)"] = round(nearest_t.end_time, 4) if nearest_t.end_time is not None else None
                        beat_data["t_width (ms)"] = round(nearest_t.width_ms, 2) if nearest_t.width_ms is not None else None
                        beat_data["t_amplitude (mV)"] = round(nearest_t.amplitude_mv, 4) if nearest_t.amplitude_mv is not None else None

                    beat_rows.append(beat_data)

        if beat_rows:
            beat_df = pd.DataFrame(beat_rows)
            beat_df.to_csv(dir_path / "beat_details.csv", index=False, encoding="utf-8-sig")

        msg = f"已导出到：{dir_path}"
        file_list = []
        if self.annotations.records:
            file_list.append("annotations.csv")
        if hrv_rows:
            file_list.append("hrv.csv")
        if wave_rows:
            file_list.append("wave_stats.csv")
            if beat_rows:
                file_list.append("beat_details.csv")

        # 5. 导出联合分析结果
        joint_rows: list[dict] = []
        selected = self._get_joint_selected_signals_safe()
        for batch in range(self.dataset.batch_count):
            for signal_type in selected:
                if not signal_type in R_PEAK_CHANNELS:
                    continue
                channel_index = 0
                for idx, name in CHANNEL_NAMES.items():
                    if name == signal_type:
                        channel_index = idx
                        break
                signal_data = self.dataset.data[batch][channel_index]
                kind = R_PEAK_CHANNELS.get(signal_type, "adult")
                peaks = detect_r_peaks(signal_data, FS, kind)
                removed = self.deleted_r_peaks.get((batch, signal_type), set())
                if removed:
                    peaks = np.asarray([p for p in peaks if int(p) not in removed], dtype=int)
                added = self.manual_r_peaks.get((batch, signal_type), set())
                if added:
                    peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
                if peaks.size < 2:
                    continue
                batch_records = [r for r in self.annotations.records if r.batch_index == batch and r.signal_type == signal_type]
                p_records = sorted([r for r in batch_records if "P波" in r.annotation_type], key=lambda x: x.start_time or 0)
                qrs_records = sorted([r for r in batch_records if "QRS" in r.annotation_type], key=lambda x: x.start_time or 0)
                t_records = sorted([r for r in batch_records if "T波" in r.annotation_type], key=lambda x: x.start_time or 0)
                rr_intervals = np.diff(peaks.astype(float)) / FS * 1000.0
                for i, peak in enumerate(peaks):
                    joint_row: dict = {"batch": batch, "signal": signal_type, "beat_index": i, "r_peak_time_s": round(peak / FS, 4)}
                    if i > 0:
                        joint_row["rr_interval_ms"] = round(rr_intervals[i - 1], 2)
                    nearest_p = next((r for r in p_records if r.start_time is not None and abs(r.start_time - peak / FS) < 0.5), None)
                    nearest_qrs = next((r for r in qrs_records if r.start_time is not None and abs(r.start_time - peak / FS) < 0.15), None)
                    nearest_t = next((r for r in t_records if r.start_time is not None and abs(r.start_time - peak / FS) < 0.6), None)
                    if nearest_p and nearest_p.start_time is not None:
                        joint_row["p_onset_s"] = round(nearest_p.start_time, 4)
                        joint_row["p_offset_s"] = round(nearest_p.end_time, 4) if nearest_p.end_time is not None else None
                        joint_row["p_width_ms"] = round(nearest_p.width_ms, 2) if nearest_p.width_ms is not None else None
                        joint_row["p_amplitude_mV"] = round(nearest_p.amplitude_mv, 4) if nearest_p.amplitude_mv is not None else None
                    if nearest_qrs and nearest_qrs.start_time is not None:
                        joint_row["qrs_onset_s"] = round(nearest_qrs.start_time, 4)
                        joint_row["qrs_offset_s"] = round(nearest_qrs.end_time, 4) if nearest_qrs.end_time is not None else None
                        joint_row["qrs_width_ms"] = round(nearest_qrs.width_ms, 2) if nearest_qrs.width_ms is not None else None
                    if nearest_t and nearest_t.start_time is not None:
                        joint_row["t_onset_s"] = round(nearest_t.start_time, 4)
                        joint_row["t_offset_s"] = round(nearest_t.end_time, 4) if nearest_t.end_time is not None else None
                        joint_row["t_width_ms"] = round(nearest_t.width_ms, 2) if nearest_t.width_ms is not None else None
                    joint_rows.append(joint_row)
        if joint_rows:
            joint_df = pd.DataFrame(joint_rows)
            joint_df.to_csv(dir_path / "joint_analysis_per_beat.csv", index=False, encoding="utf-8-sig")
            file_list.append("joint_analysis_per_beat.csv")
        if file_list:
            msg += f"\n文件: {', '.join(file_list)}"
        QtWidgets.QMessageBox.information(self, "导出完成", msg)
        self.status_label.setText(f"导出完成: {', '.join(file_list)}")

    def undo_annotation(self) -> None:
        self.annotations.undo()
        self._refresh_annotation_table()
        self.update_annotation_overlay()

    def _clear_pending(self) -> None:
        self.pending_points.clear()
        self.replacement_annotation = None
        if self.plot is not None:
            self.plot.clear_pending_marker()

    def _on_qrs_parameter_changed(self) -> None:
        """QRS/P/T percentage spinboxes changed — only affects future auto_wave_annotations calls."""
        self.status_label.setText("波形参数已变更，点击「重算当前导联」以应用新参数。")

    def _on_precision_changed(self) -> None:
        """Re-format all tables when precision spinbox values change."""
        self._refresh_annotation_table()
        self._refresh_all_stats()

    def _refresh_all_stats(self) -> None:
        try:
            batch = self.batch_spin.value() if self.dataset is not None else 0
            batch_records = [r for r in self.annotations.records if r.batch_index == batch]
            for record in batch_records:
                self._recalculate_annotation(record)
            self._refresh_hrv_table()
            self._refresh_wave_stats_table()
            self._refresh_joint_table()
        except Exception:
            pass

    def refresh_stats_from_annotations(self) -> None:
        if self.dataset is None:
            return
        if self._recalc_running:
            self.status_label.setText("正在重算中，请等待当前操作完成后再刷新统计。")
            return

        self._recalc_running = True
        self.recalc_auto_button.setEnabled(False)
        self.refresh_stats_button.setEnabled(False)
        self.auto_pt_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            committed, recalculated = self._refresh_stats_now()
            if committed:
                self.status_label.setText(f"已写入图上预览标记 {committed} 条，并刷新统计。")
            else:
                self.status_label.setText(f"已根据当前图上和表格结果刷新统计，共重算 {recalculated} 条标注。")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "刷新失败", f"刷新统计时出错：{exc}")
            self.status_label.setText(f"刷新失败：{exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.recalc_auto_button.setEnabled(True)
            self.refresh_stats_button.setEnabled(True)
            self.auto_pt_button.setEnabled(True)
            self._recalc_running = False

    def _refresh_stats_now(self) -> tuple[int, int]:
        committed = self._commit_preview_auto_records()
        batch = self.batch_spin.value()
        recalculated = 0
        for record in self.annotations.records:
            if record.batch_index == batch:
                self._recalculate_annotation(record)
                recalculated += 1
        self.joint_results = None
        if hasattr(self, "joint_table"):
            self.joint_table.setRowCount(0)
        self._refresh_hrv_table()
        self._refresh_wave_stats_table()
        return committed, recalculated

    def _selected_annotation_row(self) -> int | None:
        return self._record_index_for_table_row(self.annotation_table.currentRow())

    def _record_index_for_table_row(self, row: int) -> int | None:
        if row < 0 or row >= len(self._annotation_table_record_indices):
            return None
        record_index = self._annotation_table_record_indices[row]
        if record_index < 0 or record_index >= len(self.annotations.records):
            return None
        return record_index

    def _table_row_for_record_index(self, record_index: int) -> int | None:
        try:
            return self._annotation_table_record_indices.index(record_index)
        except ValueError:
            return None

    def _select_annotation_row(self, row: int) -> None:
        if row < 0 or row >= len(self.annotations.records):
            return
        table_row = self._table_row_for_record_index(row)
        if table_row is None:
            return
        self.annotation_table.selectRow(table_row)
        self.annotation_table.scrollToItem(self.annotation_table.item(table_row, 0))
        self.update_annotation_overlay()

    def _select_annotation_at(self, x: float, plotted_y: float) -> int | None:
        if self.plot is None:
            return None
        candidates: list[tuple[float, int]] = []
        for row, record in self.visible_annotation_items():
            if row is None:
                continue
            if record.signal_type not in self.plot.lead_offsets:
                continue
            offset = self.plot.lead_offsets[record.signal_type]
            if record.start_time is not None and record.end_time is not None:
                left, right = sorted((float(record.start_time), float(record.end_time)))
                if left <= x <= right:
                    low, high = self.plot._segment_bounds(record.signal_type, left, right)
                    if offset + low <= plotted_y <= offset + high:
                        center = (left + right) / 2.0
                        candidates.append((abs(x - center), row))
            if (
                record.baseline_time is not None
                and record.baseline_value is not None
                and record.peak_time is not None
                and record.peak_value is not None
            ):
                left = min(float(record.baseline_time), float(record.peak_time))
                right = max(float(record.baseline_time), float(record.peak_time))
                if left <= x <= right:
                    y1 = float(record.baseline_value) + offset
                    y2 = float(record.peak_value) + offset
                    min_y, max_y = sorted((y1, y2))
                    margin = max(0.05, self.plot.lead_spacing * 0.04)
                    if min_y - margin <= plotted_y <= max_y + margin:
                        candidates.append((abs(x - (left + right) / 2.0), row))
        if not candidates:
            return None
        _, row = min(candidates, key=lambda item: item[0])
        self._select_annotation_row(row)
        record = self.annotations.records[row]
        self.status_label.setText(f"已选中图中标注 #{row + 1}: {record.signal_type} / {record.annotation_type}")
        return row

    def _fmt_time(self, value) -> str:
        try:
            if np.isnan(value):
                return "-"
        except TypeError:
            pass
        prec = self.time_precision_spin.value() if self.time_precision_spin is not None else 1
        return f"{float(value):.{prec}f}"

    def _fmt_int(self, value) -> str:
        try:
            if np.isnan(value):
                return "-"
        except TypeError:
            pass
        return str(int(round(float(value))))

    def _fmt_voltage(self, value) -> str:
        try:
            if np.isnan(value):
                return "-"
        except TypeError:
            pass
        prec = self.voltage_precision_spin.value() if self.voltage_precision_spin is not None else 3
        return f"{float(value):.{prec}f}"

    @staticmethod
    def _mean(values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    @staticmethod
    def _min(values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.min(arr)) if arr.size > 0 else float("nan")

    @staticmethod
    def _max(values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.max(arr)) if arr.size > 0 else float("nan")

    def _prefer_manual_wave_records(
        self,
        signal_type: str,
        records: list[AnnotationRecord],
        batch: int | None = None,
    ) -> list[AnnotationRecord]:
        if not records:
            return []
        if batch is None:
            batch = self.batch_spin.value()
        current_batch = self.batch_spin.value() if self.dataset is not None else batch
        peaks = np.asarray(self.current_r_peaks.get(signal_type, []), dtype=int) if batch == current_batch else np.array([], dtype=int)

        selected: dict[tuple[str, int], AnnotationRecord] = {}
        for record in records:
            wave_key = self._wave_key(record.annotation_type)
            if wave_key is None or record.start_time is None:
                continue
            beat_key = self._nearest_record_beat_key(record, peaks)
            key = (wave_key, beat_key)
            existing = selected.get(key)
            if existing is None or (self._is_auto_record(existing) and not self._is_auto_record(record)):
                selected[key] = record
        return list(selected.values())

    @staticmethod
    def _wave_key(annotation_type: str) -> str | None:
        if "P波" in annotation_type:
            return "p"
        if "QRS" in annotation_type:
            return "qrs"
        if "T波" in annotation_type:
            return "t"
        return None

    def _nearest_record_beat_key(self, record: AnnotationRecord, peaks: np.ndarray) -> int:
        anchor = record.peak_time if record.peak_time is not None else record.start_time
        if anchor is None:
            return -1
        if peaks.size == 0:
            return int(round(float(anchor) * FS))
        sample = int(round(float(anchor) * FS))
        return int(np.argmin(np.abs(peaks - sample)))

    def _get_joint_selected_signals_safe(self) -> list[str]:
        """Get selected joint analysis signals."""
        signals = []
        if hasattr(self, 'joint_fecg1_cb') and self.joint_fecg1_cb.isChecked():
            signals.append("FECG_1")
        if hasattr(self, 'joint_fecg2_cb') and self.joint_fecg2_cb.isChecked():
            signals.append("FECG_2")
        if hasattr(self, 'joint_fecg3_cb') and self.joint_fecg3_cb.isChecked():
            signals.append("FECG_3")
        if hasattr(self, 'joint_mecg_cb') and self.joint_mecg_cb.isChecked():
            signals.append("MECG_1")
        return signals

    def _build_joint_combined_beats(
        self,
        aligned_beats: list[dict[str, int]],
        selected_signals: list[str],
        batch_records: list[AnnotationRecord],
        fs: int,
    ) -> list[dict]:
        combined_beats: list[dict] = []
        records_by_signal = {
            signal_type: [record for record in batch_records if record.signal_type == signal_type]
            for signal_type in selected_signals
        }

        for beat_index, beat_peaks in enumerate(aligned_beats):
            beat_result: dict = {
                "beat_index": beat_index,
                "r_peak_time": float(np.mean([peak / fs for peak in beat_peaks.values()])),
                "signals": {},
            }

            for signal_type in selected_signals:
                r_time = beat_peaks[signal_type] / fs
                left_bound, right_bound = self._joint_beat_bounds(aligned_beats, beat_index, signal_type, fs)
                signal_records = records_by_signal.get(signal_type, [])
                beat_result["signals"][signal_type] = {
                    "r_peak_time": r_time,
                    "p": self._find_wave_record_for_beat(signal_records, "P波", r_time, left_bound, right_bound),
                    "qrs": self._find_wave_record_for_beat(signal_records, "QRS", r_time, left_bound, right_bound),
                    "t": self._find_wave_record_for_beat(signal_records, "T波", r_time, left_bound, right_bound),
                }

            for key in ("p", "qrs", "t"):
                spans = [
                    wave
                    for signal_data in beat_result["signals"].values()
                    if (wave := signal_data.get(key)) is not None
                    and wave.start_time is not None
                    and wave.end_time is not None
                ]
                if not spans:
                    continue
                start = min(float(record.start_time) for record in spans)
                end = max(float(record.end_time) for record in spans)
                if end > start:
                    beat_result[key] = {
                        "start": start,
                        "end": end,
                        "width_ms": (end - start) * 1000.0,
                        "count": len(spans),
                    }

            p_span = beat_result.get("p")
            qrs_span = beat_result.get("qrs")
            t_span = beat_result.get("t")
            if p_span is not None and qrs_span is not None:
                beat_result["pr_ms"] = (qrs_span["start"] - p_span["end"]) * 1000.0
            if qrs_span is not None and t_span is not None:
                beat_result["st_ms"] = (t_span["start"] - qrs_span["end"]) * 1000.0
            if beat_index > 0 and combined_beats:
                beat_result["rr_ms"] = (beat_result["r_peak_time"] - combined_beats[-1]["r_peak_time"]) * 1000.0
            combined_beats.append(beat_result)

        return combined_beats

    def _joint_beat_bounds(
        self,
        aligned_beats: list[dict[str, int]],
        beat_index: int,
        signal_type: str,
        fs: int,
    ) -> tuple[float, float]:
        r_peak = aligned_beats[beat_index][signal_type]
        if beat_index > 0:
            previous_r = aligned_beats[beat_index - 1][signal_type]
            left_bound = (previous_r + r_peak) / 2.0 / fs
        else:
            left_bound = max(0.0, r_peak / fs - 0.8)
        if beat_index + 1 < len(aligned_beats):
            next_r = aligned_beats[beat_index + 1][signal_type]
            right_bound = (r_peak + next_r) / 2.0 / fs
        else:
            right_bound = r_peak / fs + 0.8
        return left_bound, right_bound

    def _find_wave_record_for_beat(
        self,
        records: list[AnnotationRecord],
        keyword: str,
        r_time: float,
        left_bound: float,
        right_bound: float,
    ) -> AnnotationRecord | None:
        candidates: list[tuple[int, float, AnnotationRecord]] = []
        for record in records:
            if keyword not in record.annotation_type or record.start_time is None or record.end_time is None:
                continue
            start = float(record.start_time)
            end = float(record.end_time)
            peak = float(record.peak_time) if record.peak_time is not None else (start + end) / 2.0
            source_rank = 1 if self._is_auto_record(record) else 0
            if "P波" in keyword:
                if left_bound <= start < end < r_time:
                    candidates.append((source_rank, abs(end - r_time), record))
            elif "QRS" in keyword:
                if start <= r_time <= end:
                    candidates.append((source_rank, abs(peak - r_time), record))
                elif left_bound <= start < end <= right_bound and abs(peak - r_time) <= 0.18:
                    candidates.append((source_rank, abs(peak - r_time) + 0.2, record))
            elif "T波" in keyword:
                max_t_end = right_bound + max(0.18, (right_bound - r_time) * 0.45)
                if r_time < start < end <= max_t_end:
                    candidates.append((source_rank, abs(start - r_time), record))
                elif r_time < start <= right_bound and end > right_bound:
                    candidates.append((source_rank, abs(start - r_time) + 0.15, record))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    def show_joint_analysis_dialog(self) -> None:
        """Run joint analysis and display results in the right-side panel and on the plot."""
        if self.dataset is None:
            QtWidgets.QMessageBox.information(self, "无数据", "请先打开数据文件。")
            return
        selected_signals = self._get_joint_selected_signals_safe()
        if not selected_signals:
            QtWidgets.QMessageBox.warning(self, "提示", "请至少选择一个联合分析通道（在第三行工具栏勾选F1/F2/F3/M1）。")
            return
        self._run_joint_analysis()


    def _run_joint_analysis_if_ready(self) -> None:
        """Run joint analysis only if results already exist (for control changes)."""
        if self.joint_results is not None:
            self._run_joint_analysis()
    def _run_joint_analysis(self, fallback_to_fetal: bool = False) -> None:
        """Execute joint analysis on selected channels."""
        if self.dataset is None:
            return

        selected_signals = self._get_joint_selected_signals_safe()
        if not selected_signals and fallback_to_fetal:
            selected_signals = ["FECG_1", "FECG_2", "FECG_3"]
        if not selected_signals:
            return

        batch = self.batch_spin.value()
        sample = self.dataset.data[batch]
        fs = FS
        try:
            start_time = self.joint_start_spin.value()
            end_time = self.joint_end_spin.value()
        except Exception:
            start_time = 0.0
            end_time = sample.shape[1] / fs
        try:
            tolerance = self.joint_tol_spin.value()
        except Exception:
            tolerance = 20

        start_sample = max(0, int(start_time * fs))
        end_sample = min(sample.shape[1], int(end_time * fs))
        tolerance_samples = int(tolerance * fs / 1000.0)

        # Collect R peaks for each selected signal
        signal_r_peaks = {}
        for signal_type in selected_signals:
            channel_index = next((idx for idx, name in CHANNEL_NAMES.items() if name == signal_type), 0)
            signal_data = sample[channel_index]
            peaks = self.current_r_peaks.get(signal_type, np.array([], dtype=int))
            if peaks.size == 0:
                kind = R_PEAK_CHANNELS.get(signal_type, "adult")
                peaks = detect_r_peaks(signal_data, fs, kind)
            key = (batch, signal_type)
            removed = self.deleted_r_peaks.get(key, set())
            if removed:
                peaks = np.asarray([p for p in peaks if int(p) not in removed], dtype=int)
            added = self.manual_r_peaks.get(key, set())
            if added:
                peaks = np.unique(np.concatenate([peaks, np.asarray(sorted(added), dtype=int)]))
            peaks = peaks[(peaks >= start_sample) & (peaks <= end_sample)]
            signal_r_peaks[signal_type] = peaks

        # Build aligned R-peak matrix (sequential matching)
        ref_signal = max(selected_signals, key=lambda s: len(signal_r_peaks[s]))
        ref_peaks = signal_r_peaks[ref_signal]
        if len(ref_peaks) < 2:
            self.status_label.setText(f"联合分析：参考通道 {ref_signal} R峰不足（{len(ref_peaks)}个）。")
            return

        last_used = {sig: -1 for sig in selected_signals}
        aligned_beats = []

        for ref_peak in ref_peaks:
            beat_peaks = {ref_signal: ref_peak}
            valid = True
            for sig in selected_signals:
                if sig == ref_signal:
                    continue
                other_peaks = signal_r_peaks[sig]
                if len(other_peaks) == 0:
                    valid = False
                    break
                start_idx = last_used[sig] + 1
                candidates = other_peaks[start_idx:]
                if len(candidates) == 0:
                    valid = False
                    break
                diffs = np.abs(candidates - ref_peak)
                min_diff = np.min(diffs)
                if min_diff <= tolerance_samples:
                    best_local_idx = int(np.argmin(diffs))
                    best_global_idx = start_idx + best_local_idx
                    beat_peaks[sig] = int(other_peaks[best_global_idx])
                    last_used[sig] = best_global_idx
                else:
                    valid = False
                    break
            if valid:
                aligned_beats.append(beat_peaks)

        if len(aligned_beats) < 2:
            self.status_label.setText(f"联合分析：对齐心搏不足（{len(aligned_beats)}个），请调整容差。")
            return

        batch_records = [r for r in self.annotations.records if r.batch_index == batch]
        combined_beats = self._build_joint_combined_beats(aligned_beats, selected_signals, batch_records, fs)

        if not any(("p" in beat or "qrs" in beat or "t" in beat) for beat in combined_beats):
            self.status_label.setText("联合分析：已对齐R峰，但当前标注中没有可匹配的P/QRS/T结果。请先点击 P/QRS/T分析 或调整窗口。")
            return

        self.joint_results = {
            "aligned_beats": aligned_beats,
            "combined_beats": combined_beats,
            "selected_signals": selected_signals,
            "signal_r_peaks": signal_r_peaks,
            "start_time": start_time,
            "end_time": end_time,
            "tolerance": tolerance,
            "batch": batch,
        }
        self.joint_aligned_beats = aligned_beats
        self.joint_selected_signals = selected_signals

        # Refresh joint table
        self._refresh_joint_table()

        # Draw joint analysis on plot
        self._draw_joint_on_plot()

        self.status_label.setText(f"联合分析完成：{len(aligned_beats)}个对齐心搏，{len(selected_signals)}个通道 ({', '.join(selected_signals)})")

    def _refresh_joint_table(self) -> None:
        """Refresh the joint analysis table with average results."""
        if not hasattr(self, "joint_table") or self.joint_results is None:
            return

        results = self.joint_results
        combined_beats = results.get("combined_beats", [])

        def mean_or_nan(values):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size > 0 else float("nan")

        result_rows = []

        selected_signals = [signal for signal in results.get("selected_signals", []) if signal.startswith("FECG")]
        for signal_type in selected_signals:
            signal_rows = [
                signal_data
                for beat in combined_beats
                if (signal_data := beat.get("signals", {}).get(signal_type)) is not None
            ]

            def wave_values(key: str, field: str) -> list[float]:
                values: list[float] = []
                for signal_data in signal_rows:
                    record = signal_data.get(key)
                    if record is None:
                        continue
                    value = getattr(record, field)
                    if value is not None:
                        values.append(float(value))
                return values

            pr_values: list[float] = []
            st_values: list[float] = []
            for signal_data in signal_rows:
                p_record = signal_data.get("p")
                qrs_record = signal_data.get("qrs")
                t_record = signal_data.get("t")
                if p_record is not None and qrs_record is not None and p_record.end_time is not None and qrs_record.start_time is not None:
                    pr_values.append((float(qrs_record.start_time) - float(p_record.end_time)) * 1000.0)
                if qrs_record is not None and t_record is not None and qrs_record.end_time is not None and t_record.start_time is not None:
                    st_values.append((float(t_record.start_time) - float(qrs_record.end_time)) * 1000.0)

            r_times = [float(signal_data["r_peak_time"]) for signal_data in signal_rows if "r_peak_time" in signal_data]
            rr_values = np.diff(np.asarray(r_times, dtype=float)) * 1000.0 if len(r_times) > 1 else np.array([], dtype=float)
            result_rows.append([
                signal_type,
                str(len(signal_rows)),
                self._fmt_time(mean_or_nan(wave_values("p", "width_ms"))),
                self._fmt_voltage(mean_or_nan(wave_values("p", "amplitude_mv"))),
                self._fmt_time(mean_or_nan(pr_values)),
                self._fmt_time(self._min(pr_values)),
                self._fmt_time(self._max(pr_values)),
                self._fmt_time(mean_or_nan(wave_values("qrs", "width_ms"))),
                self._fmt_voltage(mean_or_nan(wave_values("qrs", "amplitude_mv"))),
                self._fmt_time(mean_or_nan(st_values)),
                self._fmt_time(self._min(st_values)),
                self._fmt_time(self._max(st_values)),
                self._fmt_time(mean_or_nan(wave_values("t", "width_ms"))),
                self._fmt_voltage(mean_or_nan(wave_values("t", "amplitude_mv"))),
                self._fmt_time(mean_or_nan(rr_values)),
            ])

        if combined_beats:
            def combined_amplitudes(key: str) -> list[float]:
                values: list[float] = []
                for beat in combined_beats:
                    for signal_data in beat.get("signals", {}).values():
                        record = signal_data.get(key)
                        if record is not None and record.amplitude_mv is not None:
                            values.append(float(record.amplitude_mv))
                return values

            result_rows.append([
                "联合整体平均",
                str(len(combined_beats)),
                self._fmt_time(mean_or_nan([beat["p"]["width_ms"] for beat in combined_beats if "p" in beat])),
                self._fmt_voltage(mean_or_nan(combined_amplitudes("p"))),
                self._fmt_time(mean_or_nan([beat["pr_ms"] for beat in combined_beats if "pr_ms" in beat])),
                self._fmt_time(self._min([beat["pr_ms"] for beat in combined_beats if "pr_ms" in beat])),
                self._fmt_time(self._max([beat["pr_ms"] for beat in combined_beats if "pr_ms" in beat])),
                self._fmt_time(mean_or_nan([beat["qrs"]["width_ms"] for beat in combined_beats if "qrs" in beat])),
                self._fmt_voltage(mean_or_nan(combined_amplitudes("qrs"))),
                self._fmt_time(mean_or_nan([beat["st_ms"] for beat in combined_beats if "st_ms" in beat])),
                self._fmt_time(self._min([beat["st_ms"] for beat in combined_beats if "st_ms" in beat])),
                self._fmt_time(self._max([beat["st_ms"] for beat in combined_beats if "st_ms" in beat])),
                self._fmt_time(mean_or_nan([beat["t"]["width_ms"] for beat in combined_beats if "t" in beat])),
                self._fmt_voltage(mean_or_nan(combined_amplitudes("t"))),
                self._fmt_time(mean_or_nan([beat["rr_ms"] for beat in combined_beats if "rr_ms" in beat])),
            ])

        # Populate table
        self.joint_table.setRowCount(len(result_rows))
        for row_idx, row_data in enumerate(result_rows):
            for col_idx, value in enumerate(row_data):
                self.joint_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(str(value)))

    def _draw_joint_on_plot(self) -> None:
        """Draw joint analysis combined intervals on the ECG plot."""
        if self.plot is None or self.joint_results is None:
            return
        if not self.show_joint_checkbox.isChecked():
            self._clear_joint_overlay()
            return

        self._clear_joint_overlay()

        results = self.joint_results
        combined_beats = results.get("combined_beats", [])

        for beat in combined_beats:
            for signal_type, signal_data in beat.get("signals", {}).items():
                if signal_type not in self.plot.lead_offsets:
                    continue
                self._add_joint_r_line(
                    signal_type,
                    float(signal_data.get("r_peak_time", beat.get("r_peak_time", 0.0))),
                )
                self._add_joint_record_span(signal_data.get("p"), "联合P", "#7e22ce")
                self._add_joint_record_span(signal_data.get("qrs"), "联合QRS", "#dc2626")
                self._add_joint_record_span(signal_data.get("t"), "联合T", "#ea580c")

    def _add_joint_record_span(
        self,
        record: AnnotationRecord | None,
        label_text: str,
        color: str,
    ) -> None:
        if (
            self.plot is None
            or record is None
            or record.signal_type not in self.plot.lead_offsets
            or record.start_time is None
            or record.end_time is None
        ):
            return
        start_time = float(record.start_time)
        end_time = float(record.end_time)
        if end_time <= start_time:
            return
        signal_type = record.signal_type
        offset = self.plot.lead_offsets[signal_type]
        low, high = self.plot._segment_bounds(signal_type, start_time, end_time)
        qcolor = pg.mkColor(color)
        rect = QtWidgets.QGraphicsRectItem(
            QtCore.QRectF(start_time, offset + low, end_time - start_time, high - low)
        )
        rect.setBrush(pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 38))
        rect.setPen(pg.mkPen(color, width=1.6, style=QtCore.Qt.PenStyle.DashLine))
        self.plot.plotItem.addItem(rect)
        self._joint_overlay_items.append(rect)

        top_line = self.plot.plotItem.plot(
            [start_time, end_time],
            [offset + high, offset + high],
            pen=pg.mkPen(color, width=2.0),
        )
        self._joint_overlay_items.append(top_line)

        if self.show_label_checkbox.isChecked():
            label = pg.TextItem(text=label_text, color=pg.mkColor(color), anchor=(0.5, 1.0))
            label.setPos((start_time + end_time) / 2.0, offset + high)
            self.plot.plotItem.addItem(label)
            self._joint_overlay_items.append(label)

    def _add_joint_r_line(self, signal_type: str, r_time: float) -> None:
        if self.plot is None or signal_type not in self.plot.lead_offsets:
            return
        offset = self.plot.lead_offsets[signal_type]
        low, high = self.plot._lead_bounds(signal_type)
        line = self.plot.plotItem.plot(
            [r_time, r_time],
            [offset + low, offset + high],
            pen=pg.mkPen("#0891b2", width=1.6, style=QtCore.Qt.PenStyle.DotLine),
        )
        self._joint_overlay_items.append(line)

    def _clear_joint_overlay(self) -> None:
        """Remove all joint analysis overlay items from the plot."""
        if self.plot is None:
            return
        for item in self._joint_overlay_items:
            try:
                self.plot.plotItem.removeItem(item)
            except Exception:
                pass
        self._joint_overlay_items.clear()

def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
