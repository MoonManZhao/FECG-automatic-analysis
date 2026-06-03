# Project Notes

- This is a PySide6 desktop MVP for maternal-fetal ECG signal viewing and annotation.
- Keep GUI, plotting, data loading, signal processing, HRV, and annotation export logic separated under `src/`.
- Data shape is fixed to `(batch, 10, 2500)` with `fs = 250 Hz`.
- Avoid adding a database or web backend for the MVP.
