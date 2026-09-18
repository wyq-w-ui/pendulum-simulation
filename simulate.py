"""
src/simulate.py
大摆角单摆非线性动力学仿真与含噪数据生成模块
功能：
1. 求解真实工况下的单摆常微分方程（重力项 + 线性轴摩擦 + 二次空气阻力）：
   d²θ/dt² = - (g/L)*sin(θ) - b*(dθ/dt) - c*|dθ/dt|*(dθ/dt)
2. 注入模拟视觉测量的亚像素高斯噪声（RMS ~ 0.05°）
3. 导出标准 CSV 数据供后续拟合与方程发现模块直接吞入
4. 绘制并保存仿真轨迹与相图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 从第一部分导入理论计算基准（同一包内或通过 sys 导入）
try:
    from src.pendulum_theory import exact_period, G_STANDARD
except ModuleNotFoundError:
    from pendulum_theory import exact_period, G_STANDARD


def pendulum_dynamics(t, y, L, g, b, c):
    """
    单摆全阻尼动力学方程 (一阶方程组形式):
    y[0] = theta (角位移, rad)
    y[1] = omega (角速度, rad/s)
    """
    theta, omega = y
    dtheta = omega
    # 物理恢复力项与耗散阻尼项
    domega = -(g / L) * np.sin(theta) - b * omega - c * np.abs(omega) * omega
    return [dtheta, domega]


def generate_simulation_data(
        theta0_deg=60.0,
        L=0.50,
        g=G_STANDARD,
        b=0.020,  # 线性阻尼系数 (s^-1, 轴承微小摩擦)
        c=0.005,  # 二次阻尼系数 (rad^-1, 介质空气湍流阻力)
        t_span=(0.0, 12.0),
        fps=240,  # 模拟手机 240 fps 高速慢动作采样
        noise_deg_rms=0.05,  # 视觉亚像素测量噪声 RMS (度)
        random_seed=42
):
    """
    生成单摆动力学仿真时序数据
    """
    np.random.seed(random_seed)

    theta0_rad = np.radians(theta0_deg)
    y0 = [theta0_rad, 0.0]  # 静止无初速释放

    # 构建高密度等间隔时间采样点
    num_points = int((t_span[1] - t_span[0]) * fps) + 1
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    # 高精度 Runge-Kutta 求解 ODE
    sol = solve_ivp(
        fun=pendulum_dynamics,
        t_span=t_span,
        y0=y0,
        args=(L, g, b, c),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-11
    )

    t = sol.t
    theta_clean_rad = sol.y[0]
    omega_clean = sol.y[1]

    # 模拟视觉亚像素测量高斯白噪声
    noise_rad = np.radians(np.random.normal(0.0, noise_deg_rms, size=len(t)))
    theta_noisy_rad = theta_clean_rad + noise_rad

    # 真实物理角加速度（供理论比对）
    alpha_clean = -(g / L) * np.sin(theta_clean_rad) - b * omega_clean - c * np.abs(omega_clean) * omega_clean

    # 封装为标准 DataFrame
    df = pd.DataFrame({
        "time": t,
        "theta_raw_rad": theta_noisy_rad,
        "theta_raw_deg": np.degrees(theta_noisy_rad),
        "theta_clean_rad": theta_clean_rad,
        "omega_clean_rad_s": omega_clean,
        "alpha_clean_rad_s2": alpha_clean,
        "quality": 1  # 模拟数据质量标记为 1 (优秀)
    })

    return df, (L, g, b, c, theta0_deg)


def save_and_plot_simulation(df, meta_params, output_csv="data/mock_trajectory.csv",
                             figure_path="figures/simulated_trajectory.png"):
    """
    保存生成的数据并绘制动力学衰减曲线与相空间轨迹图
    """
    L, g, b, c, theta0_deg = meta_params

    # 创建保存目录
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    # 导出 CSV
    df.to_csv(output_csv, index=False)
    print(
        f"[OK] 仿真数据已成功导出至: {output_csv} (共 {len(df)} 帧，采样率 {len(df) / (df['time'].iloc[-1] - df['time'].iloc[0]):.1f} fps)")

    # 绘图展示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    # 子图 1: 时域阻尼衰减曲线
    ax1.plot(df["time"], df["theta_raw_deg"], color="silver", lw=1.0, alpha=0.8, label="Simulated Camera (Noisy)")
    ax1.plot(df["time"], np.degrees(df["theta_clean_rad"]), color="navy", lw=1.5, label="Ground Truth (Clean)")
    ax1.set_title(f"Damped Oscillation Trajectory ($\ architectural \\theta_0 = {theta0_deg}^\\circ$)", fontsize=11)
    ax1.set_xlabel("Time $t$ (s)", fontsize=10)
    ax1.set_ylabel("Angle $\\theta$ (deg)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # 子图 2: (theta, omega) 相图轨迹 (Phase Portrait)
    ax2.plot(df["theta_clean_rad"], df["omega_clean_rad_s"], color="darkgreen", lw=1.2)
    ax2.set_title("Phase Portrait $(\\theta, \\dot{\\theta})$ [Damped Inward Spiral]", fontsize=11)
    ax2.set_xlabel("$\\theta$ (rad)", fontsize=10)
    ax2.set_ylabel("$\\dot{\\theta}$ (rad/s)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    print(f"[OK] 仿真物理图表已保存至: {figure_path}")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("开始生成大摆角单摆动力学仿真数据...")
    print("设定真值: L=0.500 m, g=9.80665 m/s², 轴摩擦 b=0.020 s^-1, 空气阻力 c=0.005 rad^-1")
    print("=" * 60)

    # 模拟以 60° 释放、在 240 fps 下录制 12 秒
    sim_df, params = generate_simulation_data(
        theta0_deg=60.0,
        L=0.50,
        g=G_STANDARD,
        b=0.020,
        c=0.005,
        t_span=(0.0, 12.0),
        fps=240,
        noise_deg_rms=0.05
    )

    save_and_plot_simulation(sim_df, params)