"""
ptorch_nn.py

PyTorch 版推理封装，完整移植 libtorch_nn.h。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from nn_input_map import (
    nn_input_1_rotate_lambert,
    nn_input_2_normalization_dv,
    nn_input_2_normalization_tmin,
)

# -------------------------- 读取 scaler CSV ---------------------------------
def read_scaler_csv(filename: str):
    means, scales = [], []
    with Path(filename).open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                try:
                    means.append(float(row[0]))
                    scales.append(float(row[1]))
                except ValueError:
                    pass
    return means, scales

# -------------------------- 辅助：向量→Tensor -------------------------------
def tensor_from_vector(data: List[List[float]]) -> Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


# -------------------------- 核心推理 ----------------------------------------
def _predict_tensor(
    X_raw: List[List[float]],
    X_mean: Tensor,
    X_scale: Tensor,
    Y_mean: Tensor,
    Y_scale: Tensor,
    module: torch.jit.ScriptModule,
) -> torch.Tensor:
    X = tensor_from_vector(X_raw)
    X_norm = (X - X_mean) / X_scale
    with torch.inference_mode():
        preds_norm = module(X_norm)
    return preds_norm * Y_scale + Y_mean


def fast_predict_vector(
    raw_data: List[List[float]],
    X_mean: Tensor,
    X_scale: Tensor,
    Y_mean: Tensor,
    Y_scale: Tensor,
    module: torch.jit.ScriptModule,
    mode: int = 1,
) -> List[float]:
    v0_norms = []
    std_data = []
    for row in raw_data:
        if mode == 1:
            std_row, v0 = nn_input_2_normalization_dv(row)
        else:
            std_row, v0 = nn_input_2_normalization_tmin(row)
        std_data.append(std_row)
        v0_norms.append(v0)

    preds = _predict_tensor(std_data, X_mean, X_scale, Y_mean, Y_scale, module).squeeze(1)
    return (preds * torch.tensor(v0_norms)).tolist()


def fast_predict_vector_raw_data(
    raw_data: List[List[float]],
    X_mean: Tensor,
    X_scale: Tensor,
    Y_mean: Tensor,
    Y_scale: Tensor,
    module: torch.jit.ScriptModule,
    mode: int = 1,
) -> List[float]:
    mapped = [nn_input_1_rotate_lambert(row) for row in raw_data]
    return fast_predict_vector(mapped, X_mean, X_scale, Y_mean, Y_scale, module, mode)


# -------------------------- 读模型 + scaler --------------------------------
def ptorch_model_read(model_dir: str):
    model_path = Path(model_dir)
    X_mean, X_scale = (torch.tensor(v, dtype=torch.float32).unsqueeze(0) for v in read_scaler_csv(model_path / "scaler_X.csv"))
    Y_mean_val, Y_scale_val = read_scaler_csv(model_path / "scaler_Y.csv")
    Y_mean = torch.tensor(Y_mean_val[0], dtype=torch.float32)
    Y_scale = torch.tensor(Y_scale_val[0], dtype=torch.float32)

    module = torch.jit.load(model_path / "model_cpu.pt", map_location="cpu")
    module.eval()

    return X_mean, X_scale, Y_mean, Y_scale, module
