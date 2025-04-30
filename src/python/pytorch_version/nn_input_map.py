"""
nn_input_map.py

更新：调用 lambert() 的新接口，直接接收返回值。
其余内容与上一批一致；若您已保存旧版，请整体替换。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from array_math import array_norm2, rv_rotate
from lambert_esa import lambert


# ---------- 两个归一化函数 ---------------------------------------------------
def nn_input_2_normalization_dv(x_raw: List[float]) -> Tuple[List[float], float]:
    mu = x_raw[17]
    L0 = x_raw[0]
    T0 = math.sqrt(L0**3 / mu)
    V0 = L0 / T0
    A0 = L0 / T0**2

    x_std = [
        x_raw[1],
        x_raw[2],
        x_raw[3],
        x_raw[4] / V0,
        x_raw[5] / V0,
        *x_raw[6:10],
        *x_raw[10:14],
        x_raw[14] / T0,
        x_raw[15] / A0,
        x_raw[16] / V0,
    ]
    return x_std, V0


def nn_input_2_normalization_tmin(x_raw: List[float]) -> Tuple[List[float], float]:
    mu = x_raw[17]
    L0 = x_raw[0]
    T0 = math.sqrt(L0**3 / mu)
    V0 = L0 / T0
    A0 = L0 / T0**2

    x_std = [
        x_raw[1],
        x_raw[2],
        x_raw[3],
        x_raw[4] / V0,
        x_raw[5] / V0,
        *x_raw[6:10],
        *x_raw[10:14],
        x_raw[14] / T0,
        x_raw[15] / A0,
        x_raw[16] / V0,
    ]
    return x_std, T0


# ---------- 真近点角 ---------------------------------------------------------
def f_true(rv: np.ndarray, mu: float) -> float:
    rx, ry, rz, vx, vy, vz = rv
    h = np.cross(rv[:3], rv[3:])
    r = math.sqrt(rx**2 + ry**2 + rz**2)
    e_vec = np.cross(rv[3:], h) / mu - rv[:3] / r
    e = np.linalg.norm(e_vec)
    cosf = np.clip(np.dot(e_vec, rv[:3]) / (e * r), -1.0, 1.0)
    f = math.acos(cosf)
    return 2 * math.pi - f if np.dot(rv[:3], rv[3:]) < 0 else f


# ---------- 输入映射：旋转 + Lambert ----------------------------------------
def nn_input_1_rotate_lambert(x_raw: List[float]) -> List[float]:
    rv0 = np.array(x_raw[:6], dtype=float)
    rvt = np.array(x_raw[6:12], dtype=float)
    dt, mass, Tmax, Isp, mu = x_raw[12:17]

    rv0NU, rvtNU = np.empty(6), np.empty(6)
    rv_rotate(rv0, rvt, rv0NU, rvtNU)

    # Lambert 解
    dv0, dvt, a, e, flag = lambert(
        rv0NU[:3],
        rvtNU[:3],
        dt,
        -(np.cross(rv0NU[:3], rv0NU[3:]) / array_norm2(np.cross(rv0NU[:3], rv0NU[3:]))),
        mu,
    )
    if flag < 1:
        raise RuntimeError("Lambert solver failed")

    # Δv
    dv0_vec = rv0NU[3:] - dv0
    dvt_vec = rvtNU[3:] - dvt
    # 真实近点角
    rv_dep = np.concatenate((rv0NU[:3], dv0))
    f = f_true(rv_dep, mu)

    # Δv → 球坐标
    def cartesian_to_spherical(v):
        r = np.linalg.norm(v)
        az = math.atan2(v[1], v[0])
        el = 0.0 if r == 0 else math.acos(v[2] / r)
        return r, az, el

    dv0_r, dv0_az, dv0_el = cartesian_to_spherical(dv0_vec)
    dvt_r, dvt_az, dvt_el = cartesian_to_spherical(dvt_vec)

    return [
        a,
        e,
        math.cos(f),
        math.sin(f),
        dv0_r,
        dvt_r,
        math.cos(dv0_az),
        math.sin(dv0_az),
        math.cos(dv0_el),
        math.sin(dv0_el),
        math.cos(dvt_az),
        math.sin(dvt_az),
        math.cos(dvt_el),
        math.sin(dvt_el),
        dt,
        Tmax / mass,
        Isp,
        mu,
    ]
