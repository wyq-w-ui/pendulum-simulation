"""
sindy_discover.py
数据驱动动力学发现模块：
利用数值平滑求导构建正交特征候选库，结合 STLSQ 稀疏回归自适应提取控制微分方程
内置多层回退列名兼容引擎与共线性抑制优化，杜绝虚假大系数抵消
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 跨平台字体兼容设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def stlsq_regressor(Theta, y, threshold=0.5, max_iter=25):
    """
    序贯阈值最小二乘法 (STLSQ)
    通过迭代阈值截断压制弱相关特征项，保留主导动力学骨架
    """
    xi, _, _, _ = np.linalg.lstsq(Theta, y, rcond=None)
    
    for _ in range(max_iter):
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0.0
        active_indices = ~small_indices
        if not np.any(active_indices):
            break
        xi[active_indices], _, _, _ = np.linalg.lstsq(Theta[:, active_indices], y, rcond=None)
        
    return xi


def build_library(theta, omega):
    """
    构建非线性动力学特征候选库：
    包含线性摆角、角速度黏滞阻尼、正弦非线性回复力、二次方湍流空气阻尼
    去除了易与 sin(theta) 产生泰勒多重共线性的 theta^3 项
    """
    X_dict = {
        "\\theta": theta,
        "\\dot{\\theta}": omega,
        "\\sin(\\theta)": np.sin(theta),
        "|\\dot{\\theta}|\\dot{\\theta}": np.abs(omega) * omega
    }
    feature_names = list(X_dict.keys())
    Theta = np.column_stack(list(X_dict.values()))
    return Theta, feature_names


def discover_governing_equation(df, lambda_sparse=0.5):
    """
    从时序数据中无预设提取二阶常微分动力学方程
    """
    df_eval = df.copy()

    # ---------- 1. 多层稳健时间列识别 ----------
    time_col = None
    for col in df_eval.columns:
        c_low = str(col).lower().strip()
        if c_low in ["time", "t", "t_eval", "t_s"] or "time" in c_low:
            time_col = col
            break
    if time_col is None:
        time_col = df_eval.columns[0]
    df_eval["time"] = pd.to_numeric(df_eval[time_col], errors='coerce')

    # ---------- 2. 多层稳健摆角列识别并转为标准弧度 ----------
    theta_col = None
    # 优先找明确带 rad 的列
    for col in df_eval.columns:
        c_low = str(col).lower().strip()
        if "theta" in c_low and "rad" in c_low:
            theta_col = col
            df_eval["theta_rad"] = pd.to_numeric(df_eval[theta_col], errors='coerce')
            break

    # 其次找带 deg / angle 的列并转为弧度
    if theta_col is None:
        for col in df_eval.columns:
            c_low = str(col).lower().strip()
            if "deg" in c_low or "angle" in c_low:
                theta_col = col
                df_eval["theta_rad"] = np.radians(pd.to_numeric(df_eval[theta_col], errors='coerce'))
                break

    # 再次匹配通用 theta 列
    if theta_col is None:
        for col in df_eval.columns:
            c_low = str(col).lower().strip()
            if "theta" in c_low:
                theta_col = col
                df_eval["theta_rad"] = pd.to_numeric(df_eval[theta_col], errors='coerce')
                break

    # 最终保底：取第二列
    if "theta_rad" not in df_eval.columns:
        df_eval["theta_rad"] = pd.to_numeric(df_eval.iloc[:, 1], errors='coerce')

    # 清除 NaN
    df_eval = df_eval.dropna(subset=["time", "theta_rad"]).copy().reset_index(drop=True)

    t = df_eval["time"].values
    theta_raw = df_eval["theta_rad"].values

    # 单位自适应幅值保护校验
    if np.max(np.abs(theta_raw)) > 3.14:
        theta_raw = np.radians(theta_raw)

    # 自适应 Savitzky-Golay 滤波平滑求导
    n_pts = len(t)
    dt = float(np.mean(np.diff(t))) if n_pts > 1 else 0.00416
    window_len = min(31, n_pts if n_pts % 2 != 0 else n_pts - 1)
    if window_len < 7:
        window_len = 7 if n_pts >= 7 else (n_pts if n_pts % 2 != 0 else n_pts - 1)

    theta_smooth = savgol_filter(theta_raw, window_length=window_len, polyorder=3)
    omega_smooth = np.gradient(theta_smooth, dt)
    alpha_target = np.gradient(omega_smooth, dt)

    Theta, feature_names = build_library(theta_smooth, omega_smooth)
    xi = stlsq_regressor(Theta, alpha_target, threshold=lambda_sparse)

    return xi, feature_names


def run_identifiability_ablation():
    """
    可辨识性消融实验：生成并保存消融边界曲线图
    """
    os.makedirs("figures", exist_ok=True)
    amp_deg = np.linspace(5, 80, 50)
    amp_rad = np.radians(amp_deg)
    # sin(theta) 与 theta 的相对偏离百分比
    dev = np.abs(amp_rad - np.sin(amp_rad)) / amp_rad * 100.0

    plt.figure(figsize=(7, 4))
    plt.plot(amp_deg, dev, 'r-', lw=2, label=r"$\frac{|\theta - \sin\theta|}{\theta} \times 100\%$")
    plt.axhline(5.0, color='gray', linestyle='--', label="5% Significance Horizon (~31°)")
    plt.xlabel("Amplitude (°)")
    plt.ylabel("Nonlinear Deviation (%)")
    plt.title("Large-Angle Identifiability Boundary")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/identifiability_curve.png", dpi=300)
    plt.close()
