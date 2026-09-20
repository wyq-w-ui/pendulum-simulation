"""
track.py
计算机视觉高精度单摆视频追踪模块：
形态学阈值轮廓提取 + 轨迹连续性防丢补偿 + 极坐标动态解算 + MAD 稳健滤波
彻底杜绝背景减除造成的丢帧与频繁归零跳变
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

    if mad == 0:
        return series.interpolate(method='linear', limit_direction='both').values

    modified_z_score = 0.6745 * (series - median).abs() / mad
    outlier_mask = modified_z_score > threshold

    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan
    cleaned = cleaned.interpolate(method='linear', limit_direction='both')
    return cleaned.values


def track_pendulum_video(video_path, output_csv="data/theta_t.csv", show_preview=False):
    """
    稳健跟踪单摆小球轨迹并解算角位移时间序列
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 240.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("无法读取视频第一帧画面")

    h, w = first_frame.shape[:2]
    pivot_x, pivot_y = w / 2.0, 0.0  # 默认悬挂支点在画面上方居中

    times = []
    thetas_raw = []
    frame_idx = 0
    last_valid_cx, last_valid_cy = None, None

    # 重设读取指针到起点
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # 转为灰度图像并高斯模糊滤波
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 结合 Otsu 与形态学闭操作提取高对比度暗色或亮色球体
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        best_circle_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            # 过滤过小噪点与过大的人手/支架轮廓
            if area < 30 or area > (h * w * 0.15):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue

            # 圆度计算: 4*pi*Area / (Perimeter^2)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 优先选择圆度最高的目标
            if circularity > best_circle_score:
                m = cv2.moments(c)
                if m["m00"] > 1e-5:
                    best_circle_score = circularity
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]

        # ----------------- 连续性约束防跳水保护 -----------------
        if cx is not None and cy is not None:
            # 首次记录有效位置
            if last_valid_cx is None:
                last_valid_cx, last_valid_cy = cx, cy
            else:
                # 检查帧间位移是否在合理物理极限内
                disp = np.hypot(cx - last_valid_cx, cy - last_valid_cy)
                if disp > (w * 0.25):  # 发生剧烈跳变时判定为噪点误检
                    cx, cy = np.nan, np.nan
                else:
                    last_valid_cx, last_valid_cy = cx, cy
        else:
            cx, cy = np.nan, np.nan

        # 解算摆角，缺失帧存入 np.nan 待统一插值，决不存 0
        if not np.isnan(cx):
            dx = cx - pivot_x
            dy = cy - pivot_y
            theta_val = np.arctan2(dx, max(dy, 1.0))
        else:
            theta_val = np.nan

        times.append(current_time)
        thetas_raw.append(theta_val)
        frame_idx += 1

    cap.release()

    if show_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # 插值填补漏检帧
    s_theta = pd.Series(thetas_raw).interpolate(method='linear', limit_direction='both')
    if s_theta.isna().all():
        raise ValueError("视频追踪失败：未能识别到有效小球目标，请增强对比度或调整背景。")

    s_theta = s_theta.fillna(0.0)
    thetas_deg = np.degrees(s_theta.values)
    # MAD 清洗突变野点
    thetas_deg_clean = robust_mad_filter(thetas_deg)
    thetas_rad_clean = np.radians(thetas_deg_clean)

    # 计算角速度
    dt = 1.0 / fps
    omega_clean = np.gradient(thetas_rad_clean, dt)

    df_out = pd.DataFrame({
        "time": times,
        "theta_rad": thetas_rad_clean,
        "omega_rad_s": omega_clean,
        "theta_deg": thetas_deg_clean
    })

    # 保存 CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    # 绘制提取轨迹图
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(df_out["time"], df_out["theta_deg"], 'b-', label=r"Tracked $\theta(t)$ (Cleaned)")
    plt.xlabel("Time $t$ (s)")
    plt.ylabel(r"Angle $\theta$ ($^\circ$)")
    plt.title("Extracted Pendulum Angular Trajectory")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/extracted_trajectory.png", dpi=300)
    plt.close()

    return df_out
