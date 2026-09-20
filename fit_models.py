"""
fit_models.py
基于整体常微分方程数值积分的全局轨迹非线性拟合与模型比较 (AIC/BIC)
已整合：
1. 动静检测自动切除视频开头静止/手持阶段
2. 自动搜索首个物理释放波峰，消灭 theta ≡ 0 平衡点死锁
3. 弧度/角度单位自适应纠偏，杜绝 0.19° 等微小异常量级
4. 规范阻尼参数拟合边界，避免 gamma 贴死边界出现骤死衰减
5. 纯英文与 LaTeX 规范学术绘图，杜绝 Linux 云端方块乱码
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# 配置跨平台字体回退，防止负号与字符异常
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

   # 1. 自动兼容时间列名 (time 或 t)
    for col in ["time", "t", "t_eval"]:
        if col in df.columns:
            df["time"] = df[col]
            break

    # 2. 自动兼容角度列名 (theta_rad, theta, theta_deg, theta_sim_deg 等)
    for col in ["theta_rad", "theta", "theta_deg", "theta_sim_deg", "theta_clean_deg"]:
        if col in df.columns:
            if "deg" in col:
                df["theta_rad"] = np.radians(df[col])
            else:
                df["theta_rad"] = df[col]
            break

    t_all = df["time"].values
    theta_raw_arr = df["theta_rad"].values

    # 1. 自动单位纠偏：判别输入是弧度制还是角度制
    max_val = np.max(np.abs(theta_raw_arr))
    if max_val > 3.14:
        theta_all_rad = np.radians(theta_raw_arr)
    else:
        theta_all_rad = theta_raw_arr

    # 2. 动静检测切除：跳过开头静止/手持阶段
    diff_theta = np.abs(np.diff(theta_all_rad))
    motion_indices = np.where(diff_theta > 0.003)[0]
    first_move_idx = motion_indices[0] if len(motion_indices) > 0 else 0

    # 3. 搜索第一个物理释放极值点（波峰对齐），确定真实释放起点
    search_end = min(len(theta_all_rad), first_move_idx + 300)
    local_segment = np.abs(theta_all_rad[first_move_idx:search_end])
    start_idx = first_move_idx + int(np.argmax(local_segment))

    # 截取从有效释放起点开始的时序数据并重置时间原点
    df_valid = df.iloc[start_idx:].copy().reset_index(drop=True)
    df_valid["time"] = df_valid["time"] - df_valid["time"].iloc[0]

    df_fit = df_valid[df_valid["time"] <= t_max].copy()
    t_data = df_fit["time"].values
    
    # 拟合内核全程采用标准弧度
    if max_val > 3.14:
        theta_data = np.radians(df_fit["theta_rad"].values)
    else:
        theta_data = df_fit["theta_rad"].values

    # 极值释放点初速度为 0，初始角度为释放角幅值，打破死锁
    y0_init = [theta_data[0], 0.0]
    omega0_sq_init = 9.80665 / max(L_val, 0.05)

    # 动力学模型空间与自适应参数搜索范围
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

    # 绘图显示转换回角度制 (°)
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

    # 标准学术排版
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
