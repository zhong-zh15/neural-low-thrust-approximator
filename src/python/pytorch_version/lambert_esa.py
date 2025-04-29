"""
lambert_esa.py

纯 Python / NumPy 版 Lambert 求解器，逐行翻译自 Lambert_esa.h。
返回 (v1, v2, a, e, flag) 与 C++ 变量含义完全一致。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

PI2 = 2.0 * math.pi


def x2tof(x: float, s: float, c: float, lw: int, N: int) -> float:
    """子函数：给定 x 计算 TOF。对应 C++ x2tof。"""
    am = s / 2.0
    a = am / (1.0 - x * x)

    if x < 1.0:  # 椭圆
        temp = max((s - c) / (2.0 * a), 0.0)
        beta = 2.0 * math.asin(math.sqrt(temp))
        if lw:
            beta = -beta
        alfa = 2.0 * math.acos(x)
        tof = a * math.sqrt(a) * (
            (alfa - math.sin(alfa)) - (beta - math.sin(beta)) + N * PI2
        )
    else:  # 双曲
        alfa = 2.0 * math.log(x + math.sqrt(x * x - 1.0))
        temp = max((s - c) / (-2.0 * a), 0.0)
        temp = math.sqrt(temp)
        beta = 2.0 * math.log(temp + math.sqrt(temp * temp + 1.0))
        if lw:
            beta = -beta
        tof = -a * math.sqrt(-a) * (
            (math.sinh(alfa) - alfa) - (math.sinh(beta) - beta)
        )
    return tof


def lambert(
    R1: np.ndarray,
    R2: np.ndarray,
    tf: float,
    unith: np.ndarray,
    mu: float,
    way: int = 0,
    N: int = 0,
    branch: int = 0,
    Maxiter: int = 200,
    tol: float = 1.0e-11,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    完整 Lambert 求解器。

    Parameters
    ----------
    R1, R2 : shape=(3,) 位置向量
    tf     : 转移时间
    unith  : 参考平面法向
    mu     : 引力常数
    way    : 0=顺行, 1=逆行
    N      : 周回数
    branch : 多周时左右支
    Maxiter, tol : 迭代控制

    Returns
    -------
    v1, v2 : 发射 / 到达速度向量
    a, e   : 椭圆半长轴与偏心率
    flag   : <1 失败, 1 成功, 2 共线退化
    """
    # ---------------- 非法时间 -----------------
    if tf <= 0.0:
        return np.zeros(3), np.zeros(3), 0.0, 0.0, -1

    # ---------------- 无量纲化 ------------------
    R1 = np.array(R1, dtype=float)
    R2 = np.array(R2, dtype=float)
    R = np.linalg.norm(R1)
    V_unit = math.sqrt(mu / R)
    T_unit = R / V_unit

    r1 = R1 / R
    r2 = R2 / R
    tf_nd = tf / T_unit

    # ---------------- 几何量 --------------------
    r2mod = np.linalg.norm(r2)
    cos_theta = np.clip(np.dot(r1, r2) / r2mod, -1.0, 1.0)
    theta = math.acos(cos_theta)

    cross_prod = np.cross(r1, r2)
    if cross_prod[2] < 0.0:
        theta = PI2 - theta
    if way == 1:
        theta = PI2 - theta
    lw = int(theta > math.pi + 1e-14)

    c = math.sqrt(1.0 + r2mod**2 - 2.0 * r2mod * math.cos(theta))
    s = (1.0 + r2mod + c) / 2.0
    am = s / 2.0
    lam = math.sqrt(r2mod) * math.cos(theta / 2.0) / s

    # ---------------- 特殊共线 ------------------
    if c <= 1.0e-14:
        a = tf / (2.0 * (1 + N) * math.pi)
        a = (a * a * mu) ** (1.0 / 3.0)
        a *= R
        e = 0.0
        v1 = np.cross(unith, R1)
        v2 = v1.copy()
        return v1, v2, a, e, 2

    # ---------------- 变量替换 ------------------
    if N == 0:
        x1, x2 = -0.5233, 0.5233
        y1 = math.log(x2tof(x1, s, c, lw, N)) - math.log(tf_nd)
        y2 = math.log(x2tof(x2, s, c, lw, N)) - math.log(tf_nd)
        err, i = 1.0, 0
        while err > tol and abs(y2 - y1) > 1e-14 and i < Maxiter:
            i += 1
            xnew = (x1 * y2 - y1 * x2) / (y2 - y1)
            ynew = math.log(x2tof(math.exp(xnew) - 1.0, s, c, lw, N)) - math.log(tf_nd)
            x1, y1, x2, y2 = x2, y2, xnew, ynew
            err = abs(x1 - xnew)
        x = math.exp(x2) - 1.0
    else:  # 多周
        if branch == 0:
            inn1, inn2 = -0.5234, -0.2234
        else:
            inn1, inn2 = 0.7234, 0.5234
        x1 = math.tan(inn1 * math.pi / 2.0)
        x2 = math.tan(inn2 * math.pi / 2.0)
        y1 = x2tof(inn1, s, c, lw, N) - tf_nd
        y2 = x2tof(inn2, s, c, lw, N) - tf_nd
        err, i = 1.0, 0
        while err > tol and abs(y2 - y1) > 1e-14 and i < Maxiter:
            i += 1
            xnew = (x1 * y2 - y1 * x2) / (y2 - y1)
            ynew = x2tof(math.atan(xnew) * 2 / math.pi, s, c, lw, N) - tf_nd
            x1, y1, x2, y2 = x2, y2, xnew, ynew
            err = abs(x1 - xnew)
        x = math.atan(x2) * 2 / math.pi

    # 未收敛
    if i >= Maxiter:
        return np.zeros(3), np.zeros(3), 0.0, 0.0, 0

    # ---------------- 计算椭圆参数 --------------
    a = am / (1.0 - x * x)
    if x < 1.0:  # 椭圆
        temp = max((s - c) / (2.0 * a), 0.0)
        beta = 2.0 * math.asin(math.sqrt(temp))
        if lw:
            beta = -beta
        alfa = 2.0 * math.acos(x)
        psi = 0.5 * (alfa - beta)
        eta2 = 2.0 * a * math.sin(psi)**2 / s
        eta = math.sqrt(eta2)
    else:  # 双曲
        temp = max((c - s) / (2.0 * a), 0.0)
        temp = math.sqrt(temp)
        beta = 2.0 * math.log(temp + math.sqrt(temp * temp + 1.0))
        if lw:
            beta = -beta
        alfa = 2.0 * math.log(x + math.sqrt(x * x - 1.0))
        psi = 0.5 * (alfa - beta)
        eta2 = -2.0 * a * math.sinh(psi)**2 / s
        eta = math.sqrt(eta2)

    p = r2mod / (am * eta2) * math.sin(theta / 2.0) ** 2
    sigma1 = (2.0 * lam * am - (lam + x * eta)) / (eta * math.sqrt(am))
    e = math.sqrt(max(1.0 - p / a, 0.0))

    # 旋转平面法向
    if np.linalg.norm(cross_prod) < 1.0e-14:
        ih = np.array(unith, dtype=float)
        flag = 2
    else:
        ih = cross_prod / np.linalg.norm(cross_prod)
        flag = 1
    if lw:
        ih = -ih

    vr1 = sigma1
    vt1 = math.sqrt(p)
    v1tan = np.cross(ih, r1)
    v2tan = np.cross(ih, r2) / r2mod

    vt2 = vt1 / r2mod
    vr2 = -vr1 + (vt1 - vt2) / math.tan(theta / 2.0)

    v1 = V_unit * (vr1 * r1 + vt1 * v1tan)
    v2 = V_unit * (vr2 / r2mod * r2 + vt2 * v2tan)

    return v1, v2, a * R, e, flag
