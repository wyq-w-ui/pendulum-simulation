"""
track.py
计算机视觉高精度测角模块：
亚像素质心追踪 + 极坐标动态解算 + MAD 稳健异常点清洗与插值
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def robust_mad_filter(data, threshold=3.5):
    """
    基于中位数绝对偏差 (MAD) 的稳健异常值检测与三次样条/线性插值清洗
    """
    series = pd.Series(data)
    median = series.median()
    mad = (series - median).abs().median()
    
    if mad == 0:
        return series.values
        
    modified_z_score = 0.6745 * (series - median).abs() / mad
    outlier_mask = modified_z_score > threshold
    
    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan
    cleaned = cleaned.interpolate(method='linear', limit_direction='both')
    return cleaned.values

def track_pendulum_video(video_path, output_csv="data/theta_t.csv", show_preview=False):
    """
    单摆视频全自动跟踪与角度提取
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 240.0  # 默认慢动作帧率

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("无法读取视频第一帧")

    h, w = first_frame.shape[:2]
    pivot_x, pivot_y = w / 2.0, 0.0  # 默认悬挂点在顶部中央

    times = []
    thetas_raw = []
    frame_idx = 0

    back_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time <= 0 and frame_idx > 0:
            current_time = frame_idx / fps

        fg_mask = back_sub.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            m = cv2.moments(c)
            if m["m00"] > 1e-4:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                dx = cx - pivot_x
                dy = cy - pivot_y
                theta_val = np.arctan2(dx, dy)
                times.append(current_time)
                thetas_raw.append(theta_val)

        frame_idx += 1

    cap.release()

    # 安全处理窗口销毁，防止云端 headless 崩溃
    if show_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if len(thetas_raw) < 10:
        raise ValueError("视频提取到的有效运动帧过少，请检查视频清晰度或光照。")

    thetas_deg = np.degrees(thetas_raw)
    thetas_deg_clean = robust_mad_filter(thetas_deg)
    thetas_rad_clean = np.radians(thetas_deg_clean)

    # 简易中心差分计算角速度
    dt = 1.0 / fps
    omega_clean = np.gradient(thetas_rad_clean, dt)

    df_out = pd.DataFrame({
        "t": times[:len(thetas_rad_clean)],
        "theta_rad": thetas_rad_clean,
        "omega_rad_s": omega_clean,
        "theta_deg": thetas_deg_clean
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    # 导出诊断图表
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(df_out["t"], df_out["theta_deg"], 'b-', label="提取角度 (清洗后)")
    plt.xlabel("时间 t (s)")
    plt.ylabel("摆角 θ (°)")
    plt.title("视频逐帧提取时序曲线")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/extracted_trajectory.png", dpi=300)
    if show_preview:
        plt.show()
    plt.close()

    return df_out
