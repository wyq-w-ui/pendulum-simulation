"""
main_pipeline.py
大摆角单摆动力学全自动处理流水线 (One-Click Pipeline)
功能：
1. 自动检查并调用 track.py 提取视频轨迹 (若无视频则自动提示/回退)
2. 自动调用 fit_models.py 进行 M1~M4 轨迹拟合与 AIC/BIC 优选
3. 自动调用 sindy_discover.py 运行 STLSQ 方程发现与可辨识性消融
4. 汇总全流程关键结论与图表路径，形成可交付闭环
"""

import os
import sys
import pandas as pd

# 引入核心算法模块
# 将原先的 from src.track import ... 改为直接同级导入：
from track import track_pendulum_video
from fit_models import run_model_comparison
from sindy_discover import discover_governing_equation, run_identifiability_ablation


def run_full_pipeline(
    video_path="data/raw_pendulum_video.mp4",
    output_csv="data/theta_t.csv",
    mock_csv="data/mock_trajectory.csv",
    t_fit_max=10.0
):
    print("=" * 80)
    print("      大摆角单摆动力学 AI 辅助实验 —— 全自动分析流水线启动")
    print("=" * 80)

    # ---------------- 步骤 1：视频追踪与角度提取 ----------------
    print("\n>>> [阶段 1/3] 视觉特征追踪与角度序列提取...")
    if os.path.exists(video_path):
        print(f"[*] 检测到实验视频: {video_path}，开始自动逐帧亚像素识别...")
        track_pendulum_video(video_path, output_csv=output_csv, show_preview=False)
        active_csv = output_csv
    elif os.path.exists(output_csv):
        print(f"[*] 检测到已提取的实测数据表: {output_csv}，跳过视觉识别直接接入...")
        active_csv = output_csv
    elif os.path.exists(mock_csv):
        print(f"[!] 未检测到实拍视频或 {output_csv}，流水线自动启用仿真数据: {mock_csv}")
        active_csv = mock_csv
    else:
        print("[错误] 未找到任何可用数据源 (视频 / 实测CSV / 仿真CSV)！请先放入视频或运行 simulate.py。")
        sys.exit(1)

    print(f"[✔] 阶段 1 完成，当前动力学分析数据源: {active_csv}")

    # ---------------- 步骤 2：ODE 轨迹整体拟合与 AIC/BIC ----------------
    print("\n>>> [阶段 2/3] 常微分方程 (ODE) 轨迹全局拟合与模型比较...")
    fit_results = run_model_comparison(data_path=active_csv, t_max=t_fit_max)
    print("[✔] 阶段 2 完成，已生成 M1~M4 拟合残差图与模型打分表。")

    # ---------------- 步骤 3：SINDy 方程辨识与消融实验 ----------------
    print("\n>>> [阶段 3/3] SINDy 稀疏回归动力学方程发现与可辨识性消融...")
    df_data = pd.read_csv(active_csv)
    discover_governing_equation(df_data)
    run_identifiability_ablation(df_data, num_subsamples=30)
    print("[✔] 阶段 3 完成，已生成运动微分方程与可辨识性阈值曲线。")

    # ---------------- 最终完成提示 ----------------
    print("\n" + "=" * 80)
    print("                  ★ 全流程自动化分析全部完成！★")
    print("=" * 80)
    print("核心成果图表已保存至 figures/ 目录:")
    print("  1. figures/model_comparison_fit.png  (M1~M4 拟合与残差时序对比)")
    print("  2. figures/identifiability_curve.png (sinθ 可辨识性置信度阈值曲线)")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()