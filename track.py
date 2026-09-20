"""
track.py
计算机视觉高精度单摆视频追踪模块：
形态学双阈值轮廓提取 + 轨迹几何圆弧反推真实支点 + 动态释放点切除 + MAD稳健滤波
彻底根除摆角微弱缩水、前导静止横盘与背景误检
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def robust_mad_filter(data, threshold=3.5):
    """基于中位数绝对偏差 (MAD) 清除离群噪点并执行线性插值"""
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


def fit_circle_arc(x, y):
    """
    利用 Kasa 算法进行代数圆弧拟合，反演真实悬挂支点 (x_pivot, y_pivot)
    圆方程: (x - a)^2 + (y - b)^2 = R^2
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    cx = c[0] / 2.0
    cy = c[1] / 2.0
    R = np.sqrt(c[2] + cx**2 + cy**2)
    return cx, cy, R


def track_pendulum_video(video_path, output_csv="data/theta_t.csv", show_preview=False):
    """高稳健性追踪小球轨迹并解算物理真实摆角时序"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("无法读取视频第一帧画面")

    h, w = first_frame.shape[:2]
    
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 30 or area > (h * w * 0.15):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            # 综合圆度与面积得分
            score = circularity * np.sqrt(area)
            if score > best_score:
                m = cv2.moments(c)
                if m["m00"] > 1e-5:
                    best_score = score
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]

        # 帧间物理移动连续性检验
        if cx is not None and cy is not None:
            if last_valid_cx is None:
                last_valid_cx, last_valid_cy = cx, cy
            else:
                disp = np.hypot(cx - last_valid_cx, cy - last_valid_cy)
                if disp > (w * 0.25):
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

    # 插值填补
    s_x = pd.Series(x_coords).interpolate(method='linear', limit_direction='both').bfill().ffill()
    s_y = pd.Series(y_coords).interpolate(method='linear', limit_direction='both').bfill().ffill()
    
    arr_x = s_x.values
    arr_y = s_y.values

    # 1. 真实几何悬挂支点估算
    try:
        # 使用动态摆动段进行圆弧拟合求解真实摆心
        dx_span = np.max(arr_x) - np.min(arr_x)
        if dx_span > 20:  # 存在有效摆动跨度
            pivot_x, pivot_y, fit_R = fit_circle_arc(arr_x, arr_y)
            # 若反推支点在画面下方则修正为上方
            if pivot_y > np.min(arr_y):
                pivot_x = float(np.median(arr_x))
                pivot_y = max(0.0, float(np.min(arr_y)) - (h * 0.5))
        else:
            pivot_x = float(np.median(arr_x))
            pivot_y = 0.0
    except Exception:
        pivot_x = float(np.median(arr_x))
        pivot_y = max(0.0, float(np.min(arr_y)) - (h * 0.5))

    # 2. 计算真实物理摆角
    dx = arr_x - pivot_x
    dy = arr_y - pivot_y
    thetas_rad = np.arctan2(dx, np.maximum(dy, 10.0))
    thetas_deg = np.degrees(thetas_rad)

    thetas_deg_clean = robust_mad_filter(thetas_deg)
    thetas_rad_clean = np.radians(thetas_deg_clean)

    # 3. 动态检测松手释放点（切除静止段）
    # 以像素绝对速度为基准检测
    pixel_vel = np.hypot(np.diff(arr_x), np.diff(arr_y))
    vel_threshold = max(2.0, np.percentile(pixel_vel, 25))
    moving_mask = pixel_vel > vel_threshold
    moving_indices = np.where(moving_mask)[0]

    if len(moving_indices) > 0:
        first_motion = moving_indices[0]
        # 在运动开始后 1.5 秒内搜索初始释放最大幅值点
        search_window = min(len(thetas_deg_clean), first_motion + int(fps * 1.5))
        local_segment = np.abs(thetas_deg_clean[first_motion:search_window])
        start_cut = first_motion + int(np.argmax(local_segment)) if len(local_segment) > 0 else first_motion
    else:
        start_cut = 0

    valid_times = np.array(times_raw[start_cut:]) - times_raw[start_cut]
    valid_deg = thetas_deg_clean[start_cut:]
    valid_rad = thetas_rad_clean[start_cut:]

    # 如果提取出的最大幅值仍然小于 2 度，发出警告
    if np.max(np.abs(valid_deg)) < 2.0:
        print("[WARNING] 警告：提取摆角幅值过小，请确保拍摄视频有清晰正向大摆角！")

    dt = 1.0 / fps
    valid_omega = np.gradient(valid_rad, dt)

    df_out = pd.DataFrame({
        "time": valid_times,
        "theta_rad": valid_rad,
        "theta_deg": valid_deg,
        "omega_rad_s": valid_omega
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

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
