"""
fit_models.py
基于整体常微分方程数值积分的全局轨迹非线性拟合与模型比较 (AIC/BIC)
特性：
1. 稳健的多层列名模糊匹配与自动保底，杜绝仿真与实测 CSV 的 KeyError
2. 角度/弧度自适应单位校验与换算
3. 动态释放点（波峰）检测与时间归零，清除平衡点死锁
4. 规范参数搜索边界，避免阻尼系数贴死边界
5. 纯英文与 LaTeX 规范学术绘图，消除 Linux 云端方框乱码
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# 配置跨平台基础字体，防止负号与字符异常
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# 1. 动力学模型常微分方程定义
def ode_m1(t, y, omega0_sq):
    """M1: 线性简谐（无阻尼）"""
    theta, omega = y
    return [omega, -omega0_sq * theta]


def ode_m2(t, y, omega0_sq):
    """M2: 非线性简谐（理想大摆角无阻尼）"""
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


# 2. 数值积分求解轨迹封装
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

    try:
        popt, _ = curve_fit(fit_wrapper, t_data, theta_data, p0=p0, bounds=bounds, maxfev=3000)
    except Exception:
        popt = np.array(p0)

    # 模拟最优轨迹并计算残差统计
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

    # ---------- 多级自适应列名兼容引擎 ----------
    # 1. 提取时间列
    time_col = None
    for col in df.columns:
        c_low = str(col).lower().strip()
        if c_low in ["time", "t", "t_eval", "t_s"] or "time" in c_low:
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]
    df["time"] = pd.to_numeric(df[time_col], errors='coerce')

    # 2. 提取摆角列并转换为弧度
    theta_col = None
    # 优先匹配明确为弧度的列
    for col in df.columns:
        c_low = str(col).lower().strip()
        if "theta" in c_low and "rad" in c_low:
            theta_col = col
            df["theta_rad"] = pd.to_numeric(df[theta_col], errors='coerce')
            break

    # 其次匹配角度单位列 (deg/angle)
    if theta_col is None:
        for col in df.columns:
            c_low = str(col).lower().strip()
            if "deg" in c_low or "angle" in c_low:
                theta_col = col
                df["theta_rad"] = np.radians(pd.to_numeric(df[theta_col], errors='coerce'))
                break

    # 再次匹配通用 theta 列
    if theta_col is None:
        for col in df.columns:
            c_low = str(col).lower().strip()
            if "theta" in c_low:
                theta_col = col
                df["theta_rad"] = pd.to_numeric(df[theta_col], errors='coerce')
                break

    # 最终绝对索引兜底：取第二列
    if "theta_rad" not in df.columns:
        df["theta_rad"] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

    # 清除 NaN
    df = df.dropna(subset=["time", "theta_rad"]).copy().reset_index(drop=True)

    t_all = df["time"].values
    theta_raw_arr = df["theta_rad"].values

    # 3. 自动单位幅值校验
    max_val = np.max(np.abs(theta_raw_arr))
    if max_val > 3.14:  # 若大于 pi 说明误将度当成了弧度，补做转换
        theta_all_rad = np.radians(theta_raw_arr)
    else:
        theta_all_rad = theta_raw_arr

    # 4. 动静检测切除：跳过开头静止/手持阶段
    diff_theta = np.abs(np.diff(theta_all_rad))
    motion_indices = np.where(diff_theta > 0.003)[0]
    first_move_idx = motion_indices[0] if len(motion_indices) > 0 else 0

    # 5. 搜索第一个释放波峰极值点，时间对齐归零
    search_end = min(len(theta_all_rad), first_move_idx + 300)
    local_segment = np.abs(theta_all_rad[first_move_idx:search_end])
    start_idx = first_move_idx + int(np.argmax(local_segment)) if len(local_segment) > 0 else 0

    df_valid = df.iloc[start_idx:].copy().reset_index(drop=True)
    df_valid["time"] = df_valid["time"] - df_valid["time"].iloc[0]

    df_fit = df_valid[df_valid["time"] <= t_max].copy()
    t_data = df_fit["time"].values

    if max_val > 3.14:
        theta_data = np.radians(df_fit["theta_rad"].values)
    else:
        theta_data = df_fit["theta_rad"].values

    # 初始状态设置：初始速度为 0，初始角位移为释放幅值
    y0_init = [theta_data[0], 0.0]
    omega0_sq_init = 9.80665 / max(L_val, 0.05)

    models = {
        "M1 (Linear, Undamped)": {
            "func": ode_m1,
            "p0": [omega0_sq_init],
            "bounds": ([1.0], [80.0]),
            "names": ["omega0_sq"]
        },
        "M2 (Nonlinear, Undamped)": {
            "func": ode_m2,
            "p0": [omega0_sq_init],
            "bounds": ([1.0], [80.0]),
            "names": ["omega0_sq"]
        },
        "M3 (Viscous Damping)": {
            "func": ode_m3,
            "p0": [omega0_sq_init, 0.005],
            "bounds": ([1.0, 0.0], [80.0, 0.5]),
            "names": ["omega0_sq", "gamma"]
        },
        "M4 (Quadratic Drag)": {
            "func": ode_m4,
            "p0": [omega0_sq_init, 0.005, 0.001],
            "bounds": ([1.0, 0.0, 0.0], [80.0, 0.5, 0.5]),
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

    theta_data_deg = np.degrees(theta_data)
    ax_main.scatter(t_data, theta_data_deg, s=6, color="darkgray", label=r"Measured $\theta(t)$", alpha=0.8)

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
            linewidth=1.6
        )
        ax_res.plot(t_data, np.degrees(res["residuals"]), label=name, color=colors[name], linewidth=1.1)

    ax_main.set_ylabel(r"Angle $\theta$ ($^\circ$)")
    ax_main.set_title("Dynamic Models Comparison (M1 - M4 Global ODE Trajectory Fitting)")
    ax_main.grid(True, linestyle="--", alpha=0.5)
    ax_main.legend(loc="upper right", fontsize=8.5)

    ax_res.set_xlabel("Time $t$ (s)")
    ax_res.set_ylabel(r"Residuals ($^\circ$)")
    ax_res.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_res.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/model_comparison_fit.png", dpi=300)
    plt.close()

    return results
