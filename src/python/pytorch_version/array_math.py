"""
array_math.py

直接对应 C++ 头文件 array_math.h 的全部函数与别名，保持逐行等价。
所有向量一律用 numpy.ndarray[shape=(3,)] 或更长一维数组；
标量用 float 。
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

# ---------- 基础数组运算 --------------------------------------------------


def array_copy(B: np.ndarray, A: Sequence[float]) -> None:
    """将 N 维向量 A 复制到 B"""
    B[:] = A


def array_add(C: np.ndarray, A: Sequence[float], B: Sequence[float]) -> None:
    """C[i] = A[i] + B[i]"""
    C[:] = np.add(A, B)


def array_minus(C: np.ndarray, A: Sequence[float], B: Sequence[float]) -> None:
    """C[i] = A[i] - B[i]"""
    C[:] = np.subtract(A, B)


def array_multi(C: np.ndarray, B: Sequence[float], A: float) -> None:
    """C[i] = B[i] * A"""
    C[:] = np.multiply(B, A)


def array_dot(A: Sequence[float], B: Sequence[float]) -> float:
    """点积"""
    return float(np.dot(A, B))


def array_cross(C: np.ndarray, A: Sequence[float], B: Sequence[float]) -> None:
    """3 维向量叉乘 C = A × B"""
    C[:] = np.cross(A, B)


def array_norm2(B: Sequence[float]) -> float:
    """欧氏范数 (L2)"""
    return float(np.linalg.norm(B))


# ---------- 更高级的 3D 工具 -------------------------------------------------

Vector_RorV = np.ndarray  # == (3,)
Matrix_Rotate = np.ndarray  # == (3,3)


def norm(v: Vector_RorV) -> float:
    return float(np.linalg.norm(v))


def dot(a: Vector_RorV, b: Vector_RorV) -> float:
    return float(np.dot(a, b))


def cross(a: Vector_RorV, b: Vector_RorV) -> Vector_RorV:
    return np.cross(a, b)


def unit_vector(v: Vector_RorV) -> Vector_RorV:
    return v / norm(v)


def rotation_matrix(a: Vector_RorV, b: Vector_RorV) -> Matrix_Rotate:
    """
    轴-角公式（Rodrigues）:
    构造把向量 a 旋到向量 b 的 3×3 旋转矩阵。
    """
    u = unit_vector(a)
    v = unit_vector(b)
    theta = math.acos(np.clip(dot(u, v), -1.0, 1.0))
    if theta < 1.0e-6:  # 近似同向
        return np.eye(3)
    w = unit_vector(cross(u, v))
    ct, st = math.cos(theta), math.sin(theta)
    wx, wy, wz = w
    R = np.array(
        [
            [ct + wx * wx * (1 - ct), wx * wy * (1 - ct) - wz * st, wx * wz * (1 - ct) + wy * st],
            [wy * wx * (1 - ct) + wz * st, ct + wy * wy * (1 - ct), wy * wz * (1 - ct) - wx * st],
            [wz * wx * (1 - ct) - wy * st, wz * wy * (1 - ct) + wx * st, ct + wz * wz * (1 - ct)],
        ],
        dtype=float,
    )
    return R


def multiply(M: Matrix_Rotate, v: Vector_RorV) -> Vector_RorV:
    """矩阵-向量乘法"""
    return M @ v


def rotate_x(theta: float) -> Matrix_Rotate:
    """绕 X 轴旋转 θ（弧度）"""
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], dtype=float)


def rv_rotate(
    rv0: Sequence[float],
    rv1: Sequence[float],
    rv0_rotated: np.ndarray | None = None,
    rv1_rotated: np.ndarray | None = None,
) -> Matrix_Rotate:
    """
    复制并完全对应 C++ 的 rv_rotate。
    若未提供 *rotated* 输出数组，则返回旋转后的两个 6 维向量。
    函数总是返回最终旋转矩阵 R_total。
    """
    rv0 = np.asarray(rv0, dtype=float)
    rv1 = np.asarray(rv1, dtype=float)

    P, V = rv0[:3], rv0[3:]
    n_vec = np.cross(rv0[:3], rv1[:3])

    if np.linalg.norm(n_vec) > 0.0:
        n_unit = n_vec / np.linalg.norm(n_vec)
        vdotp = np.dot(rv0[3:], n_unit)
        v_n = n_unit * vdotp
        v_t = rv0[3:] - v_n
        V = v_t

    P1, V1 = rv1[:3], rv1[3:]
    target = np.array([1.0, 0.0, 0.0])

    R1 = rotation_matrix(P, target)
    P_rot, V_rot = R1 @ P, R1 @ V
    P1_rot, V1_rot = R1 @ P1, R1 @ V1

    phi = -math.atan2(V_rot[2], V_rot[1])
    Rx = rotate_x(phi)
    R_total = Rx @ R1

    # 恢复真正的 V，再做最终旋转
    V = rv0[3:]
    P_rot = R_total @ P
    V_rot = R_total @ V
    P1_rot = R_total @ P1
    V1_rot = R_total @ V1

    if rv0_rotated is not None and rv1_rotated is not None:
        rv0_rotated[:3], rv0_rotated[3:] = P_rot, V_rot
        rv1_rotated[:3], rv1_rotated[3:] = P1_rot, V1_rot
    else:
        rv0_rotated = np.concatenate((P_rot, V_rot))
        rv1_rotated = np.concatenate((P1_rot, V1_rot))

    return R_total
