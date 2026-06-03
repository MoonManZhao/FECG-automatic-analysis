# 母胎心电信号标定与分析 MVP

这是一个基于 Python、PySide6 和 pyqtgraph 的桌面程序，用于打开分离好的母胎心电 `.npy` 文件，在一个大图中同步显示 10 个导联，显示标准 ECG 格纸背景，自动检测 R 峰，计算基础 HRV 指标，并支持鼠标两点标注和 CSV 导出。

## 数据格式

程序读取 `.npy` 文件，要求 shape 为：

```text
(batch, 10, samples)
```

采样率固定为 `fs = 250 Hz`。界面初始只显示 10 秒，文件更长时可以水平拖动查看后续波形。

通道含义：

| Channel | 含义 |
| --- | --- |
| 0 | MECG_clean |
| 1 | AECG_1 |
| 2 | MECG_1 |
| 3 | FECG_1 |
| 4 | AECG_2 |
| 5 | MECG_2 |
| 6 | FECG_2 |
| 7 | AECG_3 |
| 8 | MECG_3 |
| 9 | FECG_3 |

## 安装

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用已有 conda 环境：

```bash
conda activate ecg_draw
pip install -r requirements.txt
```

如果只缺自动 P/T 依赖：

```bash
conda activate ecg_draw
pip install neurokit2
```

## 运行

```bash
python main.py
```

启动后点击“打开 .npy”，可以直接选择：

```text
sample_data/sample_maternal_fetal_ecg.npy
```

## MVP 功能

- 打开并校验 `.npy` 文件。
- 支持任意采样点长度的 `(batch, 10, samples)` 数据。
- 支持样本索引和第1/2/3组上下文切换。
- 在一个大图中纵向错位同步显示 10 个固定导联：`MECG_clean`、3 组 `AECG/MECG/FECG`。
- 初始横轴窗口为 10 秒，长数据可拖动查看。
- 每条波形使用 ECG paper 风格背景网格：
  - 横向小格 0.04 s，大格 0.20 s。
  - 纵向小格 0.1 mV，大格 0.5 mV。
- 使用 NeuroKit2 自动检测 `MECG_clean`、3 组 `MECG`、3 组 `FECG` 的 R 峰并叠加散点。
- 计算基础 HRV：R 峰数量、平均 RR、平均心率、SDNN、RMSSD、最小心率、最大心率。
- 支持人工修改 R 峰：
  - “添加R峰”：在当前导联点击新 R 峰位置。
  - “删除R峰”：点击错误 R 峰附近。
  - “移动R峰”：先点击错误 R 峰，再点击新位置。
- 鼠标两点标注：
  - P波宽度、QRS宽度、T波宽度、PR间期、QT间期等宽度/间期类标注会计算 `width_ms`。
  - P波高度、T波高度、幅度等高度/幅度类标注会计算 `amplitude_mv`。
- 时间标注和幅度标注拆成两个按钮。
- 自动 P/T/QRS：
  - R峰、QRS、P/T 自动标记均使用 NeuroKit2。
  - QRS 宽度来自 NeuroKit2 delineation 的 R onset/offset，幅度由对应 R 峰和基线计算。
  - P/T 基于 NeuroKit2 `ecg_delineate()` 粗略标记范围和高度。
  - 自动生成后可在表格中直接修改时间和值，宽度和幅度会自动重算并刷新图形。
- 可以分别通过“显示母亲自动标记”和“显示胎儿自动标记”控制图上自动标记显示；表格记录仍会保留。
- P波、T波、QRS/其他标注使用不同颜色；母亲/胎儿导联用不同色系区分。
- 标注阴影按该波段内真实波形的最高/最低值绘制，不再使用人为分层高度。
- 标注结果表支持选中一条或多条记录后批量删除，也支持直接编辑标注字段。
- 支持选中标注后点击“修改范围”，再在图上重新点击起点和终点来修正波段范围。
- 标注结果显示在表格中，并可导出 CSV。

## 使用说明

1. 打开 `.npy` 文件。
2. 选择样本。组选择作为当前组上下文保留，用于标注导出字段。
3. 在“标注导联”中选择要标注的 10 个导联之一。
4. 选择波形类型。
5. 点击“时间标注”或“幅度标注”。
6. 在大图中对应导联附近点击两个点：
   - 时间标注：点击起点和终点。
   - 幅度标注：点击基线点和峰值点。
7. 自动 P/T/QRS：点击“自动P/T/QRS”，生成粗标记；用“显示母亲自动标记”和“显示胎儿自动标记”分别控制图上显示。
8. 修改 R 峰：使用“添加R峰”“删除R峰”“移动R峰”按钮。
9. 修改标注：直接编辑标注结果表中的时间和值，或选中记录后点击“修改范围”在图上重新点起止位置；程序会自动更新宽度、高度并重绘。
10. 视图跳转：点击“返回初始视图”回到最初 10 秒窗口。
11. 删除标注：在标注结果表选中一条或多条记录，点击“删除选中标注”。
12. 需要保存时点击“导出CSV”。

## 项目结构

```text
.
├── main.py
├── requirements.txt
├── README.md
├── AGENTS.md
├── sample_data/
│   └── sample_maternal_fetal_ecg.npy
└── src/
    ├── app.py
    ├── data_loader.py
    ├── signal_processing.py
    ├── hrv.py
    ├── annotation.py
    ├── ecg_grid.py
    ├── plotting.py
    └── utils.py
```
