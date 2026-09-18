"""
src/sindy_discover.py (优化版：增强多重共线性抑制与自适应 STLSQ)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# -------------------------------------------------------------
# 1. 构造非线性候选特征库 Theta
# -------------------------------------------------------------
def build_feature_library(theta, omega):
    theta = theta.flatten()
    omega = omega.flatten()

    X = np.column_stack([
        np.ones_like(theta),                  # 1
        theta,                                # θ
        theta**2,                             # θ²
        theta**3,                             # θ³
        np.sin(theta),                        # sin(θ)
        np.cos(theta),                        # cos(θ)
        omega,                                # ω
        np.abs(omega) * omega,                # |ω|ω
        omega**2                              # ω²
    ])

    feature_names = [
        "1", "θ", "θ²", "θ³", "sin(θ)", "cos(θ)", "ω", "|ω|ω", "ω²"
    ]
    return X, feature_names


# -------------------------------------------------------------
# 2. 改进型 STLSQ (加入 Tikhonov 岭正则化稳定初解)
# -------------------------------------------------------------
def run_stlsq_robust(X, y, threshold_ratio=0.15, alpha_ridge=1e-3, max_iter=20):
    """
    带岭回归先验阻尼的序贯阈值最小二乘法 (解决 theta 与 sin(theta) 共线性)
    """
    n, p = X.shape
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms == 0] = 1.0
    X_norm = X / col_norms

    # 1. 使用岭回归计算初解，防止强共线性导致系数爆炸
    # (X^T X + alpha * I)^-1 X^T y
    XtX = X_norm.T @ X_norm
    ridge_matrix = XtX + alpha_ridge * np.eye(p)
    xi = np.linalg.solve(ridge_matrix, X_norm.T @ y)

    # 2. 确定自适应截断阈值 (基于最大特征系数的相对比例)
    threshold = np.max(np.abs(xi)) * threshold_ratio

    # 3. 序贯迭代剔除
    for _ in range(max_iter):
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0.0

        big_indices = ~small_indices
        if not np.any(big_indices):
            break

        X_sub = X_norm[:, big_indices]
        # 在保留的稀疏子空间重新做最小二乘
        xi_sub, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
        xi[big_indices] = xi_sub

    # 4. 反归一化
    xi_final = xi / col_norms
    return xi_final


# -------------------------------------------------------------
# 3. 单次时序 SINDy 方程识别
# -------------------------------------------------------------
def discover_governing_equation(df):
    t = df["time"].values
    theta_raw = df["theta_raw_rad"].values
    dt = t[1] - t[0]

    # SG 滤波平滑与数值导数
    w_len = 37 if len(t) >= 37 else (len(t) // 2 * 2 + 1)
    theta_smooth = savgol_filter(theta_raw, window_length=w_len, polyorder=3)
    omega_smooth = savgol_filter(theta_raw, window_length=w_len, polyorder=3, deriv=1, delta=dt)
    alpha_smooth = savgol_filter(theta_raw, window_length=w_len, polyorder=3, deriv=2, delta=dt)

    X_lib, names = build_feature_library(theta_smooth, omega_smooth)
    xi = run_stlsq_robust(X_lib, alpha_smooth, threshold_ratio=0.15)

    print("=" * 72)
    print("SINDy 动力学方程自动辨识结果 (抗共线性优化):")
    print("=" * 72)
    print(f"{'候选项':^10} | {'辨识系数 ξ':^14} | {'筛选状态':^12} | {'物理意义'}")
    print("-" * 72)

    equation_terms = []
    for name, coef in zip(names, xi):
        status = "保留 [✔]" if abs(coef) > 1e-4 else "置零 [✘]"
        desc = ""
        if name == "sin(θ)":
            desc = "非线性恢复力 (理论值 -g/L ≈ -19.61)"
        elif name == "θ":
            desc = "小角近似恢复力"
        elif name == "ω":
            desc = "线性阻尼 (轴承微小摩擦)"
        elif name == "|ω|ω":
            desc = "二次阻尼 (空气湍流阻力)"
        elif name in ["θ²", "cos(θ)", "ω²"]:
            desc = "偶对称陷阱项 (理论必须为 0)"

        print(f"{name:^10} | {coef:>13.4f}  | {status:^12} | {desc}")
        if abs(coef) > 1e-4:
            equation_terms.append(f"({coef:+.4f})·{name}")

    discovered_eq = "d²θ/dt² = " + " ".join(equation_terms)
    print("=" * 72)
    print(f"识别出的微分方程:\n  {discovered_eq}")
    print("=" * 72)

    return xi, names


# -------------------------------------------------------------
# 4. 可辨识性阈值消融实验
# -------------------------------------------------------------
def run_identifiability_ablation(df, num_subsamples=25):
    theta_raw_deg = np.abs(df["theta_raw_deg"].values)
    t = df["time"].values
    dt = t[1] - t[0]
    theta_raw_rad = df["theta_raw_rad"].values

    w_len = 37
    theta_s = savgol_filter(theta_raw_rad, window_length=w_len, polyorder=3)
    omega_s = savgol_filter(theta_raw_rad, window_length=w_len, polyorder=3, deriv=1, delta=dt)
    alpha_s = savgol_filter(theta_raw_rad, window_length=w_len, polyorder=3, deriv=2, delta=dt)

    X_full, names = build_feature_library(theta_s, omega_s)
    sin_idx = names.index("sin(θ)")
    theta_idx = names.index("θ")

    # 扫描角度上限
    theta_max_levels = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    p_selected = []

    print("\n正在运行可辨识性消融实验 (扫描振幅上限并执行重采样统计)...")

    for th_max in theta_max_levels:
        valid_mask = theta_raw_deg <= (th_max + 1.0)
        indices = np.where(valid_mask)[0]

        if len(indices) < 150:
            p_selected.append(0.0)
            continue

        success_count = 0
        for _ in range(num_subsamples):
            # 随机子集抽样
            sample_idx = np.random.choice(indices, size=int(len(indices) * 0.8), replace=True)
            X_sub = X_full[sample_idx]
            y_sub = alpha_s[sample_idx]

            xi_sub = run_stlsq_robust(X_sub, y_sub, threshold_ratio=0.15)

            # 成功判据：sin(θ) 项被激活，且线性 θ 项被置零（或明显弱于 sin(θ)）
            if abs(xi_sub[sin_idx]) > 8.0 and abs(xi_sub[theta_idx]) < 1e-4:
                success_count += 1

        prob = (success_count / num_subsamples) * 100.0
        p_selected.append(prob)
        print(f"  振幅截断上限 θ_max = {th_max:>4.1f}° | sin(θ) 正确辨识概率 P = {prob:>5.1f}%")

    # 绘制曲线
    plt.figure(figsize=(7.5, 4.5), dpi=150)
    plt.plot(theta_max_levels, p_selected, 'o-', color="crimson", lw=2.0, markersize=6, label="Selection Probability $P(\\theta_{max})$")
    plt.axhline(90, color="gray", linestyle="--", lw=1.2, label="Confidence (90%)")
    plt.title("Identifiability Horizon of $\\sin\\theta$ vs Linear $\\theta$", fontsize=12)
    plt.xlabel("Max Amplitude Limit $\\theta_{max}$ (deg)", fontsize=11)
    plt.ylabel("sin(θ) Selection Frequency (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.xlim(0, 65)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/identifiability_curve.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\n[OK] 可辨识性消融曲线已更新保存至: {fig_path}")
    plt.show()


if __name__ == "__main__":
    real_csv = "data/theta_t.csv"
    mock_csv = "data/mock_trajectory.csv"

    target_csv = real_csv if os.path.exists(real_csv) else mock_csv
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"未找到数据文件，请先运行 simulate.py 或 track.py 生成数据！")

    print(f"[*] SINDy 当前分析数据源: {target_csv}")
    df_data = pd.read_csv(target_csv)

    discover_governing_equation(df_data)
    run_identifiability_ablation(df_data, num_subsamples=30)