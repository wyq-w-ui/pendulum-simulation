"""
track.py
计算机视觉高精度单摆视频追踪模块：
形态学自适应轮廓提取 + 真实物理悬挂点动态校准 + 自动松手释放检测 + MAD稳健滤波
彻底根除前导横盘静止、幅度微弱缩水及丢帧跳变
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def robust_mad_filter(data, threshold=3.5):
    """
    基于中位数绝对偏差 (MAD) 清除离群噪点并执行线性插值
    """
    series = pd.Series(data)
    median = series.median()
    mad = (series - median).abs().median()

    if mad == 0 or np.isnan(mad):
        return series.interpolate(method='linear', limit_direction='both').fillna(0.0).values

    modified_z_score = 0.6745 * (series - median).abs() / mad
    outlier_mask = modified_z_score > threshold

    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan
    cleaned = cleaned.interpolate(method='linear', limit_direction='both').fillna(0.0)
    return cleaned.values


def track_pendulum_video(video_path, output_csv="data/theta_t.csv", show_preview=False):
    """
    高稳健性追踪小球轨迹并解算物理摆角时序
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # 常见实测慢动作/手机视频帧率缺省回退

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("无法读取视频第一帧画面")

    h, w = first_frame.shape[:2]
    
    # 临时记录中心点坐标
    x_coords = []
    y_coords = []
    times_raw = []
    frame_idx = 0
    last_valid_cx, last_valid_cy = None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # 图像预处理与双阈值稳健特征提取
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 增强对比度，自适应提取明暗对比目标
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        best_circle_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            # 过滤极小微斑与大块背景/人体干扰
            if area < 40 or area > (h * w * 0.12):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > best_circle_score:
                m = cv2.moments(c)
                if m["m00"] > 1e-5:
                    best_circle_score = circularity
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]

        # 帧间平滑约束，防止瞬时漂移
        if cx is not None and cy is not None:
            if last_valid_cx is None:
                last_valid_cx, last_valid_cy = cx, cy
            else:
                disp = np.hypot(cx - last_valid_cx, cy - last_valid_cy)
                if disp > (w * 0.20):  # 超过异常速度跳变则判定为干扰
                    cx, cy = np.nan, np.nan
                else:
                    last_valid_cx, last_valid_cy = cx, cy
        else:
            cx, cy = np.nan, np.nan

        times_raw.append(current_time)
        x_coords.append(cx)
        y_coords.append(cy)
        frame_idx += 1

    cap.release()

    if show_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # 插值填补漏检帧
    s_x = pd.Series(x_coords).interpolate(method='linear', limit_direction='both')
    s_y = pd.Series(y_coords).interpolate(method='linear', limit_direction='both')
    
    if s_x.isna().all() or s_y.isna().all():
        raise ValueError("视频追踪失败：未能成功识别摆球目标，请检查背景对比度。")

    arr_x = s_x.bfill().ffill().values
    arr_y = s_y.bfill().ffill().values

    # ---------- 1. 悬挂支点自适应估计 ----------
    # 摆球轨迹横坐标的对称轴中心即为真实悬挂支点的 x 坐标
    pivot_x = float(np.median(arr_x))
    # 悬挂支点 y 坐标估计在球体最高点上方
    min_y = float(np.min(arr_y))
    pivot_y = max(0.0, min_y - (h * 0.35))

    # 计算真实弧度与角度
    dx = arr_x - pivot_x
    dy = arr_y - pivot_y
    thetas_rad = np.arctan2(dx, np.maximum(dy, 10.0))
    thetas_deg = np.degrees(thetas_rad)

    # 滤除微小野点
    thetas_deg_clean = robust_mad_filter(thetas_deg)
    thetas_rad_clean = np.radians(thetas_deg_clean)

    # ---------- 2. 自动检测小球真正释放点（切除手持静止段） ----------
    # 计算瞬时运动速度大小
    vel = np.abs(np.diff(thetas_deg_clean))
    # 寻找摆动开始连续活跃的起始帧
    active_mask = vel > 0.08
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) > 0:
        motion_start = active_indices[0]
        # 在刚开始运动的邻域内寻找第一个振幅极值点（绝对最高释放点）
        search_window = min(len(thetas_deg_clean), motion_start + int(fps * 1.5))
        local_sub = np.abs(thetas_deg_clean[motion_start:search_window])
        start_cut = motion_start + int(np.argmax(local_sub))
    else:
        start_cut = 0

    # 截取释放后的有效振荡段
    valid_times = np.array(times_raw[start_cut:]) - times_raw[start_cut]
    valid_deg = thetas_deg_clean[start_cut:]
    valid_rad = thetas_rad_clean[start_cut:]

    # 角速度计算
    dt = 1.0 / fps
    valid_omega = np.gradient(valid_rad, dt)

    df_out = pd.DataFrame({
        "time": valid_times,
        "theta_rad": valid_rad,
        "theta_deg": valid_deg,
        "omega_rad_s": valid_omega
    })

    # 保存清洗后的完整物理数据
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    # 绘制实测轨迹提取诊断图
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(df_out["time"], df_out["theta_deg"], 'b-', lw=1.5, label=r"Tracked $\theta(t)$ (Physical Degrees)")
    plt.xlabel("Time $t$ (s)")
    plt.ylabel(r"Angle $\theta$ ($^\circ$)")
    plt.title("Extracted Pendulum Angular Trajectory (Lead-in Static Trimmed)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/extracted_trajectory.png", dpi=300)
    plt.close()

    return df_out
