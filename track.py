"""
src/track.py
大摆角单摆高速视频视觉测角与轨迹提取模块
功能：
1. 读取 MP4 视频逐帧图像及时间戳 (PTS)
2. 利用前段静止画面提取固定悬点 (x0, y0) 与背景模型
3. 结合亮度/背景差分与灰度加权质心提取摆球亚像素坐标 (x(t), y(t))
4. 几何计算带符号摆角: theta(t) = atan2(x - x0, y - y0)
5. 基于 MAD 稳健统计检测异常帧并局部平滑插值，导出 data/theta_t.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# -------------------------------------------------------------
# 1. 稳健统计 MAD (中位数绝对偏差) 异常检测
# -------------------------------------------------------------
def detect_outliers_mad(data, threshold=3.5):
    """
    使用中位数绝对偏差 (MAD) 识别一维数组中的离群点。
    MAD = 1.4826 * median(|r_i - median(r)|)
    """
    median = np.median(data)
    mad = 1.4826 * np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros_like(data, dtype=bool)
    diff = np.abs(data - median)
    return diff > (threshold * mad)


# -------------------------------------------------------------
# 2. 亚像素灰度加权质心定位
# -------------------------------------------------------------
def get_weighted_centroid(crop_gray, crop_diff, threshold_val):
    """
    在摆球 ROI 局部窗口内，计算以背景差异亮度为权重的灰度加权质心
    """
    # 差异越大说明越偏向摆球本体
    weights = np.maximum(0.0, crop_diff.astype(np.float32) - threshold_val)
    total_w = np.sum(weights)
    if total_w <= 1e-5:
        return None, None
    y_indices, x_indices = np.indices(crop_gray.shape)
    cx = np.sum(x_indices * weights) / total_w
    cy = np.sum(y_indices * weights) / total_w
    return cx, cy


# -------------------------------------------------------------
# 3. 核心视觉跟踪处理函数
# -------------------------------------------------------------
def track_pendulum_video(
        video_path,
        output_csv="data/theta_t.csv",
        pivot_manual=None,  # 若已知转轴坐标传入 (x0, y0)，否则从视频交互/前段检测
        static_seconds=2.0,  # 用前几秒静止画面构建背景与寻找参考标记
        show_preview=True
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps_prop = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_prop if fps_prop > 0 else 240.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"正在载入视频: {video_path}")
    print(f"分辨率: {width}x{height} | 标称帧率: {fps:.1f} fps | 总帧数: {total_frames}")

    # ---------------- Step A: 提取背景模型 ----------------
    static_frames_count = min(int(fps * static_seconds), max(total_frames // 10, 15))
    sample_frames = []

    print("正在分析前段静止背景与悬点基准...")
    for _ in range(static_frames_count):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample_frames.append(gray)

    if len(sample_frames) == 0:
        raise ValueError("无法读取视频前序帧，请检查文件格式是否有效。")

    # 像素级时间中位数背景
    background_model = np.median(np.stack(sample_frames, axis=0), axis=0).astype(np.uint8)

    # 确定悬点坐标 (x0, y0)
    if pivot_manual is not None:
        pivot_x, pivot_y = pivot_manual
    else:
        # 默认假设转轴位于画面上方正中附近，取水平中心
        pivot_x = width / 2.0
        pivot_y = height * 0.15
        print(f"[提示] 未指定精确悬点，使用默认估算坐标: ({pivot_x:.1f}, {pivot_y:.1f})")

    # 重置视频游标至第 0 帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t_list = []
    x_list = []
    y_list = []
    theta_list = []
    quality_list = []

    frame_idx = 0
    print("开始逐帧亚像素识别与角度解算...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取逐帧 PTS 时间戳 (优先读取毫秒，转秒)
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec > 0:
            current_t = pos_msec / 1000.0
        else:
            current_t = frame_idx / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background_model)

        # 动态二值化分割
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        # 形态学滤波去除孤立杂点
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx_sub, cy_sub = None, None
        quality_flag = 1  # 1 为优质帧，0 为可疑/丢失帧

        if contours:
            # 提取面积最大的连通域 (摆球)
            max_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_c)

            if area > 40:  # 面积阈值，过滤杂斑
                bx, by, bw, bh = cv2.boundingRect(max_c)
                # 扩展 ROI 边距
                pad = 4
                x1 = max(0, bx - pad)
                y1 = max(0, by - pad)
                x2 = min(width, bx + bw + pad)
                y2 = min(height, by + bh + pad)

                crop_g = gray[y1:y2, x1:x2]
                crop_d = diff[y1:y2, x1:x2]

                local_cx, local_cy = get_weighted_centroid(crop_g, crop_d, threshold_val=20)
                if local_cx is not None:
                    cx_sub = x1 + local_cx
                    cy_sub = y1 + local_cy

        if cx_sub is None or cy_sub is None:
            # 丢失摆球帧
            cx_sub = x_list[-1] if len(x_list) > 0 else pivot_x
            cy_sub = y_list[-1] if len(y_list) > 0 else (pivot_y + 100.0)
            quality_flag = 0

        # 解算相对竖直方向的有符号摆角 (弧度)
        dx = cx_sub - pivot_x
        dy = cy_sub - pivot_y
        theta_rad = np.arctan2(dx, dy)

        t_list.append(current_t)
        x_list.append(cx_sub)
        y_list.append(cy_sub)
        theta_list.append(theta_rad)
        quality_list.append(quality_flag)

        # 预览画面
        if show_preview and (frame_idx % 2 == 0):
            cv2.circle(frame, (int(pivot_x), int(pivot_y)), 5, (0, 0, 255), -1)
            cv2.line(frame, (int(pivot_x), int(pivot_y)), (int(cx_sub), int(cy_sub)), (255, 120, 0), 2)
            cv2.circle(frame, (int(cx_sub), int(cy_sub)), 6, (0, 255, 0), -1)
            deg_display = np.degrees(theta_rad)
            cv2.putText(
                frame, f"Frame: {frame_idx} | Theta: {deg_display:+.2f} deg",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cv2.imshow("Pendulum Tracking (Press Q to exit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户主动中断视频处理。")
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ---------------- Step B: MAD 稳健统计异常帧清洗 ----------------
    t_arr = np.array(t_list)
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    theta_arr = np.array(theta_list)
    q_arr = np.array(quality_list)

    # 计算转轴到球心的瞬时像素半径 r(t)
    r_arr = np.sqrt((x_arr - pivot_x) ** 2 + (y_arr - pivot_y) ** 2)
    radius_outliers = detect_outliers_mad(r_arr, threshold=3.5)

    # 标记异常帧 (面积丢失或半径剧烈跳变)
    total_anomalies = (q_arr == 0) | radius_outliers
    q_arr[total_anomalies] = 0

    valid_idx = np.where(~total_anomalies)[0]
    outlier_idx = np.where(total_anomalies)[0]

    anomaly_ratio = len(outlier_idx) / len(t_arr) * 100.0
    print("-" * 65)
    print(f"数据清洗报告: 总帧数 {len(t_arr)} | 异常帧数 {len(outlier_idx)} ({anomaly_ratio:.2f}%)")
    print(f"有效摆动半径均值: {np.mean(r_arr[valid_idx]):.2f} px | 标准差: {np.std(r_arr[valid_idx]):.2f} px")
    print("-" * 65)

    # 局部三次样条插值修补异常帧
    if len(outlier_idx) > 0 and len(valid_idx) > 10:
        cs = CubicSpline(t_arr[valid_idx], theta_arr[valid_idx])
        theta_arr[outlier_idx] = cs(t_arr[outlier_idx])
        print("已使用三次样条完成异常帧时序平滑修补。")

    # ---------------- Step C: 导出标准 CSV ----------------
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out = pd.DataFrame({
        "time": t_arr,
        "theta_raw_rad": theta_arr,
        "theta_raw_deg": np.degrees(theta_arr),
        "x_px": x_arr,
        "y_px": y_arr,
        "radius_px": r_arr,
        "quality": q_arr
    })
    df_out.to_csv(output_csv, index=False)
    print(f"[OK] 轨迹数据提取完毕，已成功保存至: {output_csv}")

    # 绘制提取出的角度曲线
    plt.figure(figsize=(9, 4), dpi=150)
    plt.plot(t_arr, np.degrees(theta_arr), color="teal", lw=1.2, label="Extracted $\\theta(t)$")
    if len(outlier_idx) > 0:
        plt.scatter(
            t_arr[outlier_idx], np.degrees(theta_arr[outlier_idx]),
            color="red", s=15, zorder=3, label="Interpolated Outliers"
        )
    plt.title("Extracted Pendulum Oscillation Trajectory", fontsize=11)
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Angle $\\theta$ (deg)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/extracted_trajectory.png", dpi=300)
    plt.show()

    return df_out


if __name__ == "__main__":
    # 测试运行时，将视频路径替换为你手机传输过来的真实 MP4 文件
    test_video = "data/raw_pendulum_video.mp4"

    if not os.path.exists(test_video):
        print(f"未检测到测试视频 {test_video}。")
        print("建议将拍摄的手机 120/240 fps 慢动作视频命名为 'raw_pendulum_video.mp4' 并放进 data/ 文件夹中。")
    else:
        # 若视频中悬点标记的像素坐标确定，可直接传参，如 pivot_manual=(640, 150)
        track_pendulum_video(test_video, output_csv="data/theta_t.csv", show_preview=True)