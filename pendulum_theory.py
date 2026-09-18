"""
src/pendulum_theory.py
大摆角单摆动力学理论基准计算模块
功能：
1. 自研 AGM (算术-几何平均) 迭代算法计算第一类完全椭圆积分 K(k)
2. 计算无阻尼理想单摆在大摆角下的理论精确周期 T(theta_0)
3. 生成理论对比基准表与理论周期上扬图
"""

import numpy as np
import matplotlib.pyplot as plt

# 常用标准物理常量
G_STANDARD = 9.80665  # 标准重力加速度 (m/s^2)


def elliptic_k_agm(k, tol=1e-15, max_iter=100):
    """
    使用算术-几何平均 (AGM) 算法迭代求解第一类完全椭圆积分 K(k)。
    理论定义: K(k) = \int_0^{\pi/2} 1 / \sqrt{1 - k^2 \sin^2\phi} d\phi

    参数:
        k: 椭圆模数 (scalar 或 numpy array), 取值范围需在 [0, 1) 内
        tol: 迭代收敛容差
        max_iter: 最大迭代步数

    返回:
        K(k) 的高精度数值解
    """
    k = np.asarray(k, dtype=np.float64)
    if np.any(np.abs(k) >= 1.0):
        raise ValueError("椭圆模数 k 必须满足 |k| < 1")

    # 初始算术平均数与几何平均数
    # a_0 = 1, b_0 = \sqrt{1 - k^2}
    a = np.ones_like(k)
    b = np.sqrt(1.0 - k ** 2)

    for _ in range(max_iter):
        a_next = 0.5 * (a + b)
        b_next = np.sqrt(a * b)

        # 检查收敛条件
        if np.all(np.abs(a - b) < tol):
            break
        a, b = a_next, b_next

    # 极限值: K(k) = \pi / (2 * a_\infty)
    return np.pi / (2.0 * a)


def exact_period(theta0_rad, L=0.50, g=G_STANDARD):
    """
    计算无阻尼大摆角单摆的精确物理周期。
    精确公式: T(theta_0) = 4 * \sqrt{L/g} * K( \sin(theta_0 / 2) )

    参数:
        theta0_rad: 初始释放摆角 (弧度制, scalar 或 numpy array)
        L: 等效摆长 (m)
        g: 当地重力加速度 (m/s^2)

    返回:
        精确周期 T (s)
    """
    theta0_rad = np.asarray(theta0_rad, dtype=np.float64)
    k = np.sin(theta0_rad / 2.0)
    K_val = elliptic_k_agm(k)
    omega0 = np.sqrt(g / L)
    return (4.0 / omega0) * K_val


def small_angle_period(L=0.50, g=G_STANDARD):
    """
    计算小角近似简谐振动周期: T_0 = 2\pi * \sqrt{L/g}
    """
    return 2.0 * np.pi * np.sqrt(L / g)


def generate_theory_benchmark_table(L=0.50, g=G_STANDARD):
    """
    复现开题报告第 3.2 节的理论基准对照表
    """
    angles_deg = np.array([5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 80.0])
    angles_rad = np.radians(angles_deg)

    T0 = small_angle_period(L, g)
    T_exact = exact_period(angles_rad, L, g)

    # 相对差异计算
    sin_diff_ratio = (1.0 - np.sin(angles_rad) / angles_rad) * 100.0  # (1 - sinθ/θ)
    t_increase_ratio = ((T_exact - T0) / T0) * 100.0  # 相对 T0 增幅

    print("=" * 68)
    print(f"理论基准对照表 (摆长 L = {L:.3f} m, g = {g:.5f} m/s², T0 = {T0:.4f} s)")
    print("=" * 68)
    print(f"{'初始摆角 θ0':^12} | {'1 - sinθ/θ':^12} | {'精确周期 T(s)':^14} | {'相对 T0 增幅':^14}")
    print("-" * 68)
    for i in range(len(angles_deg)):
        print(
            f"{angles_deg[i]:>10.1f}° | {sin_diff_ratio[i]:>11.2f}% | {T_exact[i]:>13.4f} s | {t_increase_ratio[i]:>13.2f}%")
    print("=" * 68)


def plot_theory_curves(L=0.50, g=G_STANDARD, save_path="figures/theory_period_curve.png"):
    """
    绘制大摆角理论周期随初始摆角的变化曲线（含精确解与小角恒定线对比）
    """
    angles_deg = np.linspace(0.1, 85.0, 300)
    angles_rad = np.radians(angles_deg)

    T0 = small_angle_period(L, g)
    T_exact = exact_period(angles_rad, L, g)

    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(angles_deg, T_exact, 'r-', lw=2.0, label=r"Exact: $T(\theta_0) = 4\sqrt{L/g} \cdot K(\sin(\theta_0/2))$")
    plt.axhline(T0, color='blue', linestyle='--', lw=1.5, label=r"Small Angle: $T_0 = 2\pi\sqrt{L/g}$")

    plt.title(f"Nonlinear Period vs Initial Amplitude (L = {L:.2f} m)", fontsize=13)
    plt.xlabel(r"Initial Amplitude $\theta_0$ (degrees)", fontsize=11)
    plt.ylabel("Period $T$ (seconds)", fontsize=11)
    plt.xlim(0, 85)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10.5)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300)
        print(f"理论对比图已保存至: {save_path}")
    except Exception:
        pass
    plt.show()


if __name__ == "__main__":
    # 直接运行该文件进行自检验证
    generate_theory_benchmark_table(L=0.50, g=G_STANDARD)
    plot_theory_curves(L=0.50, g=G_STANDARD)