"""
fit_models.py
基于整体常微分方程数值积分的全局轨迹非线性拟合与模型比较 (AIC/BIC)
对比 M1(线性)、M2(简谐大角非线性)、M3(线性阻尼非线性)、M4(二次阻尼非线性)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# 配置跨平台字体兼容策略，防止负号与字符异常
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# 1. 动力学模型微分方程定义
def ode_m1(t, y, omega0_sq):
    """M1: 线性无阻尼"""
    theta, omega = y
    return [omega, -omega0_sq * theta]


def ode_m2(t, y, omega0_sq):
    """M2: 非线性无阻尼 (理想大摆角)"""
    theta, omega = y
    return [omega, -omega0_sq * np.sin(theta)]


def ode_m3(t, y, omega0_sq, gamma):
    """M3: 非线性 + 线性黏滞阻尼"""
    theta, omega = y
    return [omega, -omega0_sq * np.sin(theta) - 2 * gamma * omega]


def ode_m4(t, y, omega0_sq, gamma, beta):
    """M4: 非线性 + 线性黏滞 + 二次空气阻力"""
    theta, omega = y
    return [omega, -omega0_sq * np.sin(theta) - 2 * gamma * omega - beta * np.abs(omega) * omega]


# 2. 通用数值积分求解轨迹封装
def integrate_trajectory(ode_func, t_eval, y0, params):
    sol = solve_ivp(
        fun=lambda t, y: ode_func(t, y, *params),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    if not sol.success:
        return np.full_like(t_eval, np.nan)
    return sol.y[0]


# 3. 单模型非线性最小二乘拟合
def fit_single_model(ode_func, t_data, theta_data, y0_init, p0, bounds, param_names):
    n_pts = len(t_data)
    k_params = len(p0)

    def fit_wrapper(t, *params):
        return integrate_trajectory(ode_func, t, y0_init, params)

    popt, _ = curve_fit(fit_wrapper, t_data, theta_data, p0=p0, bounds=bounds, maxfev=2000)

    # 模拟最优轨迹与残差统计
    theta_pred = integrate_trajectory(ode_func, t_data, y0_init, popt)
    residuals = theta_data - theta_pred
    rss = np.sum(residuals**2)
    rmse_deg = np.degrees(np.sqrt(rss / n_pts))

    # 赤池信息量 (AIC) 与贝叶斯信息量 (BIC)
    rss_safe = max(rss, 1e-12)
    aic = n_pts * np.log(rss_safe / n_pts) + 2 * k_params
    bic = n_pts * np.log(rss_safe / n_pts) + k_params * np.log(n_pts)

    param_dict = {name: val for name, val in zip(param_names, popt)}
    return {
        "params": param_dict,
        "rmse_deg": rmse_deg,
        "aic": aic,
        "bic": bic,
        "theta_pred": theta_pred,
        "residuals": residuals
    }


# 4. 主控对决流水线
def run_model_comparison(data_path="data/theta_t.csv", t_max=10.0, L_val=0.50):
    df = pd.read_csv(data_path)

    # 兼容列名映射
    if "time" not in df.columns and "t" in df.columns:
        df["time"] = df["t"]
    if "theta_rad" not in df.columns and "theta" in df.columns:
        df["theta_rad"] = df["theta"]

    df_fit = df[df["time"] <= t_max].copy()
    t_data = df_fit["time"].values
    theta_data = df_fit["theta_rad"].values

    y0_init = [theta_data[0], 0.0]
    omega0_sq_init = 9.80665 / L_val

    # 规范标准模型名称（全英文，杜绝云端乱码）
    models = {
        "M1 (Linear, Undamped)": {
            "func": ode_m1,
            "p0": [omega0_sq_init],
            "bounds": ([5.0], [40.0]),
            "names": ["omega0_sq"]
        },
        "M2 (Nonlinear, Undamped)": {
            "func": ode_m2,
            "p0": [omega0_sq_init],
            "bounds": ([5.0], [40.0]),
            "names": ["omega0_sq"]
        },
        "M3 (Viscous Damping)": {
            "func": ode_m3,
            "p0": [omega0_sq_init, 0.01],
            "bounds": ([5.0, 0.0], [40.0, 1.0]),
            "names": ["omega0_sq", "gamma"]
        },
        "M4 (Quadratic Drag)": {
            "func": ode_m4,
            "p0": [omega0_sq_init, 0.01, 0.005],
            "bounds": ([5.0, 0.0, 0.0], [40.0, 1.0, 0.5]),
            "names": ["omega0_sq", "gamma", "beta"]
        }
    }

    results = {}
    for name, cfg in models.items():
        results[name] = fit_single_model(
            cfg["func"], t_data, theta_data, y0_init, cfg["p0"], cfg["bounds"], cfg["names"]
        )

    # 绘制模型对比图与时域残差图
    os.makedirs("figures", exist_ok=True)
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]}
    )

    ax_main.scatter(t_data, np.degrees(theta_data), s=4, color="lightgray", label=r"Measured $\theta(t)$", alpha=0.7)

    colors = {
        "M1 (Linear, Undamped)": "tab:orange",
        "M2 (Nonlinear, Undamped)": "tab:green",
        "M3 (Viscous Damping)": "tab:blue",
        "M4 (Quadratic Drag)": "tab:red"
    }

    for name, res in results.items():
        ax_main.plot(
            t_data,
            np.degrees(res["theta_pred"]),
            label=f"{name} (RMSE={res['rmse_deg']:.2f}°)",
            color=colors[name],
            linewidth=1.5
        )
        ax_res.plot(t_data, np.degrees(res["residuals"]), label=name, color=colors[name], linewidth=1.0)

    # 标准学术坐标轴与图名定义
    ax_main.set_ylabel(r"Angle $\theta$ ($^\circ$)")
    ax_main.set_title("Dynamic Models Comparison (M1 - M4 Global ODE Trajectory Fitting)")
    ax_main.grid(True, linestyle="--", alpha=0.5)
    ax_main.legend(loc="upper right", fontsize=9)

    ax_res.set_xlabel("Time $t$ (s)")
    ax_res.set_ylabel(r"Residuals ($^\circ$)")
    ax_res.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_res.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/model_comparison_fit.png", dpi=300)
    plt.close()

    return results
