"""
sindy_discover.py
数据驱动动力学发现模块：
带岭回归正则化保护的 STLSQ 稀疏回归 + 小角度共线性防御 + 平滑求导
彻底消除虚假相互抵消的数十万爆炸系数，稳定收敛至物理控制方程
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def stlsq_regressor(Theta, y, threshold=0.5, max_iter=25, alpha_ridge=1e-3):
    """
    带岭回归 (Ridge Regularization) 正则化保护的 STLSQ
    即使候选特征高度线性相关，也能稳定求解，杜绝数十万异常对冲系数
    """
    n_features = Theta.shape[1]
    reg = alpha_ridge * np.eye(n_features)
    
    # 岭回归求解: (Theta^T Theta + alpha*I)^(-1) Theta^T y
    xi = np.linalg.solve(Theta.T @ Theta + reg, Theta.T @ y)

    for _ in range(max_iter):
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0.0
        active_indices = ~small_indices
        if not np.any(active_indices):
            break
        
        sub_theta = Theta[:, active_indices]
        sub_reg = alpha_ridge * np.eye(sub_theta.shape[1])
        xi[active_indices] = np.linalg.solve(sub_theta.T @ sub_theta + sub_reg, sub_theta.T @ y)

    return xi


def build_library(theta, omega):
    """构建非线性动力学候选特征库"""
    max_deg = np.degrees(np.max(np.abs(theta)))
    
    # 小角度自适应防御：若最大振幅小于 15 度，sin(theta) 与 theta 无法在数值上区分，强制只保留 sin(theta) 物理项
    if max_deg < 15.0:
        X_dict = {
            "\\sin(\\theta)": np.sin(theta),
            "\\dot{\\theta}": omega,
            "|\\dot{\\theta}|\\dot{\\theta}": np.abs(omega) * omega
        }
    else:
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
    """从时序数据中自适应提取二阶常微分动力学方程"""
    df_eval = df.copy()

    # 多层列名兼容引擎
    time_col = None
    for col in df_eval.columns:
        c_low = str(col).lower().strip()
        if c_low in ["time", "t", "t_eval", "t_s"] or "time" in c_low:
            time_col = col
            break
    if time_col is None:
        time_col = df_eval.columns[0]
    df_eval["time"] = pd.to_numeric(df_eval[time_col], errors='coerce')

    theta_col = None
    for col in df_eval.columns:
        c_low = str(col).lower().strip()
        if "theta" in c_low and "rad" in c_low:
            theta_col = col
            df_eval["theta_rad"] = pd.to_numeric(df_eval[theta_col], errors='coerce')
            break

    if theta_col is None:
        for col in df_eval.columns:
            c_low = str(col).lower().strip()
            if "deg" in c_low or "angle" in c_low:
                theta_col = col
                df_eval["theta_rad"] = np.radians(pd.to_numeric(df_eval[theta_col], errors='coerce'))
                break

    if theta_col is None:
        for col in df_eval.columns:
            c_low = str(col).lower().strip()
            if "theta" in c_low:
                theta_col = col
                df_eval["theta_rad"] = pd.to_numeric(df_eval[theta_col], errors='coerce')
                break

    if "theta_rad" not in df_eval.columns:
        df_eval["theta_rad"] = pd.to_numeric(df_eval.iloc[:, 1], errors='coerce')

    df_eval = df_eval.dropna(subset=["time", "theta_rad"]).copy().reset_index(drop=True)

    t = df_eval["time"].values
    theta_raw = df_eval["theta_rad"].values

    if np.max(np.abs(theta_raw)) > 3.14:
        theta_raw = np.radians(theta_raw)

    n_pts = len(t)
    dt = float(np.mean(np.diff(t))) if n_pts > 1 else 0.00416
    window_len = min(41, n_pts if n_pts % 2 != 0 else n_pts - 1)
    if window_len < 7:
        window_len = 7 if n_pts >= 7 else (n_pts if n_pts % 2 != 0 else n_pts - 1)

    theta_smooth = savgol_filter(theta_raw, window_length=window_len, polyorder=3)
    omega_smooth = np.gradient(theta_smooth, dt)
    alpha_target = np.gradient(omega_smooth, dt)

    Theta, feature_names = build_library(theta_smooth, omega_smooth)
    xi = stlsq_regressor(Theta, alpha_target, threshold=lambda_sparse)

    return xi, feature_names


def run_identifiability_ablation():
    """生成大摆角非线性辨识边界理论消融曲线"""
    os.makedirs("figures", exist_ok=True)
    amp_deg = np.linspace(5, 80, 50)
    amp_rad = np.radians(amp_deg)
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
