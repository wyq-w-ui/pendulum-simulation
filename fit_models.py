"""
src/fit_models.py
大摆角单摆多模型常微分方程 (ODE) 全局拟合与 AIC/BIC 评估模块
功能：
1. 读取时序数据 (data/mock_trajectory.csv 或实际视觉提取数据)
2. 基于 scipy.integrate.solve_ivp 和 scipy.optimize.least_squares 对 M1~M4 进行参数反演
3. 统计各模型残差、RMSE，并计算 AIC 与 BIC 信息准则
4. 绘制各模型时域拟合曲线与残差子图 (Residual Plots)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# -------------------------------------------------------------
# 1. 四大候选动力学 ODE 定义
# -------------------------------------------------------------
def ode_m1(t, y, a):
    """M1: 小角线性模型 d²θ/dt² = -a*θ"""
    theta, omega = y
    return [omega, -a * theta]


def ode_m2(t, y, a):
    """M2: 理想非线性模型 d²θ/dt² = -a*sin(θ)"""
    theta, omega = y
    return [omega, -a * np.sin(theta)]


def ode_m3(t, y, a, b):
    """M3: 线性阻尼模型 d²θ/dt² = -a*sin(θ) - b*ω"""
    theta, omega = y
    return [omega, -a * np.sin(theta) - b * omega]


def ode_m4(t, y, a, b, c):
    """M4: 包含二次速度阻尼 d²θ/dt² = -a*sin(θ) - b*ω - c*|ω|*ω"""
    theta, omega = y
    return [omega, -a * np.sin(theta) - b * omega - c * np.abs(omega) * omega]


# -------------------------------------------------------------
# 2. 通用 ODE 轨迹积分与残差函数
# -------------------------------------------------------------
def simulate_trajectory(ode_func, t_eval, y0, params):
    """前向数值积分产生预测轨迹"""
    sol = solve_ivp(
        fun=ode_func,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        args=tuple(params),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    if not sol.success or len(sol.y[0]) != len(t_eval):
        # 求解发散或异常时返回极大惩罚
        return np.full_like(t_eval, 1e5)
    return sol.y[0]


def fit_single_model(ode_func, t_eval, theta_meas, y0_guess, p0, bounds, param_names):
    """
    通用拟合器：同时优化初始角度 y0[0] 与物理方程参数
    """
    # 待优化变量向量：[theta_0, *params]
    x0 = [y0_guess[0]] + list(p0)
    lower_b = [y0_guess[0] - np.radians(2.0)] + list(bounds[0])
    upper_b = [y0_guess[0] + np.radians(2.0)] + list(bounds[1])

    def residuals(x):
        theta_init = x[0]
        params = x[1:]
        theta_pred = simulate_trajectory(ode_func, t_eval, [theta_init, y0_guess[1]], params)
        return theta_pred - theta_meas

    res = least_squares(residuals, x0, bounds=(lower_b, upper_b), method="trf")
    best_theta0 = res.x[0]
    best_params = res.x[1:]

    # 重新生成最佳拟合轨迹
    best_traj = simulate_trajectory(ode_func, t_eval, [best_theta0, y0_guess[1]], best_params)
    res_vec = best_traj - theta_meas
    rss = np.sum(res_vec ** 2)
    n = len(theta_meas)
    k = len(best_params) + 1  # 自由参数个数 (含 theta0)

    # 信息准则计算
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    rmse_deg = np.degrees(np.sqrt(rss / n))

    fit_info = {
        "params": dict(zip(param_names, best_params)),
        "theta0_deg": np.degrees(best_theta0),
        "trajectory": best_traj,
        "residuals": res_vec,
        "rss": rss,
        "rmse_deg": rmse_deg,
        "k": k,
        "aic": aic,
        "bic": bic
    }
    return fit_info


# -------------------------------------------------------------
# 3. 多模型批量拟合与信息准则评价
# -------------------------------------------------------------
def run_model_comparison(data_path="data/mock_trajectory.csv", t_max=10.0, L_val=0.50):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据文件 {data_path}，请先运行 simulate.py 生成数据！")

    df = pd.read_csv(data_path)
    # 兼容列名：无论叫 't' 还是 'time' 都统一为 'time'
    if "time" not in df.columns and "t" in df.columns:
        df["time"] = df["t"]
    
    # 兼容摆角列名：无论叫 'theta' 还是 'theta_rad' 均可识别
    if "theta_rad" not in df.columns and "theta" in df.columns:
        df["theta_rad"] = df["theta"]

    df_fit = df[df["time"] <= t_max].copy()
    t_data = df_fit["time"].values
    theta_data = df_fit["theta_rad"].values

    y0_init = [theta_data[0], 0.0]

    # 4 个模型的配置字典
    model_configs = {
        "M1 (Linear)": {
            "func": ode_m1,
            "p0": [18.0],
            "bounds": ([5.0], [30.0]),
            "names": ["a"]
        },
        "M2 (Ideal Nonlinear)": {
            "func": ode_m2,
            "p0": [18.0],
            "bounds": ([5.0], [30.0]),
            "names": ["a"]
        },
        "M3 (Linear Damping)": {
            "func": ode_m3,
            "p0": [18.0, 0.01],
            "bounds": ([5.0, 0.0], [30.0, 0.5]),
            "names": ["a", "b"]
        },
        "M4 (Quadratic Damping)": {
            "func": ode_m4,
            "p0": [18.0, 0.01, 0.005],
            "bounds": ([5.0, 0.0, 0.0], [30.0, 0.5, 0.1]),
            "names": ["a", "b", "c"]
        }
    }

    results = {}
    print("=" * 80)
    print(f"正在对时序数据执行 M1~M4 动力学全局拟合 (数据点数: {len(t_data)}, 拟合时长: {t_max}s)...")
    print("=" * 80)

    for name, cfg in model_configs.items():
        print(f"正在优化 {name} ...")
        fit_info = fit_single_model(
            cfg["func"], t_data, theta_data, y0_init, cfg["p0"], cfg["bounds"], cfg["names"]
        )
        results[name] = fit_info

    # 找到最优 AIC 作为基准
    min_aic = min(r["aic"] for r in results.values())
    min_bic = min(r["bic"] for r in results.values())

    # 输出排位榜
    print("\n" + "=" * 90)
    print(f"{'模型名称':^22} | {'参数反演估计值':^26} | {'RMSE (°)':^10} | {'ΔAIC':^10} | {'ΔBIC':^10}")
    print("-" * 90)
    for name, r in results.items():
        param_str = ", ".join([f"{k}={v:.4f}" for k, v in r["params"].items()])
        delta_aic = r["aic"] - min_aic
        delta_bic = r["bic"] - min_bic
        print(f"{name:<22} | {param_str:<26} | {r['rmse_deg']:>9.4f} | {delta_aic:>9.2f} | {delta_bic:>9.2f}")
    print("=" * 90)
    print("注：ΔAIC / ΔBIC = 0.00 代表统计学上的最优模型；> 10 代表极强淘汰证据。")

    # ---------------------------------------------------------
    # 4. 绘图输出：拟合轨迹与残差分析图
    # ---------------------------------------------------------
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(11, 6.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # 绘制实测散点
    ax_main.scatter(t_eval, np.degrees(theta_meas), s=3, color="lightgray", label="Measured θ(t)", alpha=0.7)

    colors = ["#999999", "#ff7f0e", "#2ca02c", "#d62728"]
    linestyles = [":", "--", "-.", "-"]

    for (name, r), color, ls in zip(results.items(), colors, linestyles):
        ax_main.plot(t_eval, np.degrees(r["trajectory"]), label=f"{name}", color=color, linestyle=ls, lw=1.5)
        # 残差图
        ax_res.plot(t_eval, np.degrees(r["residuals"]), label=f"{name}", color=color, linestyle=ls, lw=1.2)

    ax_main.set_ylabel("Angle $\\theta$ (deg)", fontsize=11)
    ax_main.set_title("Global Trajectory Fit Comparison (M1 - M4)", fontsize=12)
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, linestyle="--", alpha=0.5)

    ax_res.set_ylabel("Residual (deg)", fontsize=10)
    ax_res.set_xlabel("Time $t$ (s)", fontsize=11)
    ax_res.axhline(0, color="black", linestyle="-", lw=0.8)
    ax_res.grid(True, linestyle="--", alpha=0.5)

    os.makedirs("figures", exist_ok=True)
    figure_path = "figures/model_comparison_fit.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    print(f"\n[OK] 模型对比与残差图已保存至: {figure_path}")
    plt.show()

    return results


if __name__ == "__main__":
    # 优先读取实测轨迹数据，若无则自动回退至仿真数据
    real_csv = "data/theta_t.csv"
    mock_csv = "data/mock_trajectory.csv"

    target_csv = real_csv if os.path.exists(real_csv) else mock_csv
    print(f"[*] 当前数据输入源: {target_csv}")

    run_model_comparison(data_path=target_csv, t_max=10.0)
