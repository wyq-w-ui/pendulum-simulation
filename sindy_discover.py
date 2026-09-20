"""
sindy_discover.py
数据驱动动力学发现模块：
利用数值平滑求导构建特征候选库，结合 STLSQ 稀疏回归自适应发现控制微分方程
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def stlsq_regressor(Theta, y, threshold=0.1, max_iter=20):
    """
    序贯阈值最小二乘法 (STLSQ)
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
    构建非线性动力学候选库：[theta, omega, sin(theta), |omega|*omega, theta^3]
    """
    X_dict = {
        "\\theta": theta,
        "\\dot{\\theta}": omega,
        "\\sin(\\theta)": np.sin(theta),
        "|\\dot{\\theta}|\\dot{\\theta}": np.abs(omega) * omega,
        "\\theta^3": theta**3
    }
    feature_names = list(X_dict.keys())
    Theta = np.column_stack(list(X_dict.values()))
    return Theta, feature_names

def discover_governing_equation(df, lambda_sparse=0.15):
    """
    从轨迹时序数据中无预设提取二阶常微分方程
    """
    df_eval = df.copy()
    if "time" not in df_eval.columns and "t" in df_eval.columns:
        df_eval["time"] = df_eval["t"]
    if "theta_rad" not in df_eval.columns and "theta" in df_eval.columns:
        df_eval["theta_rad"] = df_eval["theta"]

    t = df_eval["time"].values
    theta_raw = df_eval["theta_rad"].values

    # 自适应选取 Savitzky-Golay 滤波窗口
    n_pts = len(t)
    dt = float(np.mean(np.diff(t)))
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
    可辨识性消融实验：生成并保存消融曲线图
    """
    os.makedirs("figures", exist_ok=True)
    amp_deg = np.linspace(5, 80, 50)
    amp_rad = np.radians(amp_deg)
    # sin(theta) 与 theta 的非线性偏离度
    dev = np.abs(amp_rad - np.sin(amp_rad)) / amp_rad * 100.0

    plt.figure(figsize=(7, 4))
    plt.plot(amp_deg, dev, 'r-', lw=2, label=r"$\frac{|\theta - \sin\theta|}{\theta} \times 100\%$")
    plt.axhline(5.0, color='gray', linestyle='--', label="5% 显著性差异阈值 (~31°)")
    plt.xlabel("摆动振幅 (°)")
    plt.ylabel("非线性偏离百分比 (%)")
    plt.title("大摆角模型可辨识性消融边界")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/identifiability_curve.png", dpi=300)
    plt.close()
