"""
app.py
大摆角单摆动力学 AI 辅助实验 —— 一键交互分析网页 (双模式：仿真演练 / 实测视频)
彻底解决 session_state 串台、旧数据残留与图文脱节问题
"""

import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

from track import track_pendulum_video
from fit_models import run_model_comparison
from sindy_discover import discover_governing_equation, run_identifiability_ablation
from pendulum_theory import small_angle_period, exact_period
from simulate import generate_simulation_data, save_and_plot_simulation

st.set_page_config(
    page_title="大摆角单摆非线性动力学AI辨识系统",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 大摆角单摆动力学：从手机视频到微分方程")
st.caption("AI+物理实验创新赛道 | 计算机视觉测角 · 全局ODE拟合 · SINDy稀疏方程发现")

st.sidebar.header("⚙️ 实验物理参数")
L_input = st.sidebar.number_input("等效摆长 L (m)", value=0.50, min_value=0.10, max_value=2.00, step=0.01)
g_input = st.sidebar.number_input("当地重力加速度 g (m/s²)", value=9.80665, step=0.0001, format="%.5f")
t_fit_max = st.sidebar.slider("轨迹拟合分析时长 (秒)", min_value=3.0, max_value=20.0, value=10.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🕹️ 数据源模式切换")

def clear_mode_cache():
    """切换模式时清理历史分析结果，防止旧数据污染"""
    for key in ["active_df", "csv_path", "run_done", "active_mode"]:
        if key in st.session_state:
            del st.session_state[key]

data_mode = st.sidebar.radio(
    "选择分析模式:",
    ("数字仿真演练模式 (无需视频，一键生成)", "实拍视频分析模式 (待上传实测视频)"),
    on_change=clear_mode_cache
)

# 1. 数字仿真演练模式
if data_mode == "数字仿真演练模式 (无需视频，一键生成)":
    st.subheader("1. 数字靶场：动力学仿真生成")
    st.info("💡 当前为无视频演练模式：利用 Runge-Kutta 数值求解器生成大摆角阻尼轨迹，用于标准基准验证。")

    col_sim_1, col_sim_2 = st.columns([3, 1])
    with col_sim_1:
        sim_angle = st.slider("初始释放角 θ₀ (度)", min_value=15.0, max_value=80.0, value=60.0, step=5.0)
    with col_sim_2:
        btn_run_sim = st.button("🚀 启动全流程分析", type="primary", use_container_width=True)

    if btn_run_sim:
        with st.spinner("正在生成仿真轨迹并运行后端算法链路..."):
            sim_df, meta_params = generate_simulation_data(
                theta0_deg=sim_angle,
                L=L_input,
                g=g_input,
                t_span=(0.0, 12.0),
                fps=240,
                noise_deg_rms=0.05
            )
            csv_path = "data/mock_trajectory.csv"
            save_and_plot_simulation(sim_df, meta_params, output_csv=csv_path)
            
            st.session_state["active_df"] = sim_df
            st.session_state["csv_path"] = csv_path
            st.session_state["run_done"] = True
            st.session_state["active_mode"] = "sim"

# 2. 实拍视频分析模式
else:
    st.subheader("1. 实验输入：拖拽上传慢动作视频")
    uploaded_file = st.file_uploader(
        "拖入手机拍摄的大摆角单摆慢动作视频 (.mp4)",
        type=["mp4", "mov", "avi"],
        help="建议使用 120 或 240 fps 录制，初始摆角 ≤ 80°"
    )
    if uploaded_file is not None:
        video_temp_path = "temp_input_video.mp4"
        with open(video_temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col_v, col_btn = st.columns([2, 1])
        with col_v:
            st.video(uploaded_file)
        with col_btn:
            st.write("### 准备就绪")
            st.write(f"文件大小: {uploaded_file.size / (1024 * 1024):.2f} MB")
            btn_run_video = st.button("🚀 启动视频全自动提取与分析", type="primary", use_container_width=True)

        if btn_run_video:
            with st.spinner("正在逐帧提取摆球亚像素质心与角度..."):
                csv_path = "data/theta_t.csv"
                if os.path.exists(csv_path):
                    try:
                        os.remove(csv_path)
                    except Exception:
                        pass
                df_video = track_pendulum_video(video_temp_path, output_csv=csv_path, show_preview=False)
                
                st.session_state["active_df"] = df_video
                st.session_state["csv_path"] = csv_path
                st.session_state["run_done"] = True
                st.session_state["active_mode"] = "video"
    else:
        st.info("💡 提示：实拍视频模式下请拖入 .mp4 文件；若目前暂无实测视频，可在左侧切换为【数字仿真演练模式】先睹为快。")

# 3. 结果呈现主控模块
if st.session_state.get("run_done", False) and "active_df" in st.session_state:
    df_data = st.session_state["active_df"]
    target_csv = st.session_state["csv_path"]
    curr_mode = st.session_state.get("active_mode", "sim")

    with st.spinner("正在执行四大模型常微分拟合与 SINDy 微分方程辨识..."):
        fit_results = run_model_comparison(data_path=target_csv, t_max=t_fit_max, L_val=L_input)
        xi_res, names_res = discover_governing_equation(df_data, lambda_sparse=0.5)

    st.success("🎉 全链路动力学分析执行完毕！")
    st.write("---")
    st.subheader("2. 核心科学成果与数据交付")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 轨迹拟合与模型比较 (AIC/BIC)",
        "🧠 SINDy 动力学方程发现",
        "📈 角度序列与相图",
        "📐 理论椭圆积分对照"
    ])

    with tab1:
        st.markdown("#### 四大动力学模型横向对决与残差演化")
        if os.path.exists("figures/model_comparison_fit.png"):
            st.image("figures/model_comparison_fit.png", use_container_width=True)

        summary_data = []
        min_aic = min(r["aic"] for r in fit_results.values())
        min_bic = min(r["bic"] for r in fit_results.values())
        for name, r in fit_results.items():
            p_str = ", ".join([f"{k}={v:.4f}" for k, v in r["params"].items()])
            summary_data.append({
                "模型名称": name,
                "反演参数估计": p_str,
                "RMSE (°)": f"{r['rmse_deg']:.4f}",
                "ΔAIC": f"{r['aic'] - min_aic:.2f}",
                "ΔBIC": f"{r['bic'] - min_bic:.2f}"
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        st.caption("注：ΔAIC / ΔBIC = 0.00 为统计学最优模型；> 10 代表极强淘汰证据。")

    with tab2:
        st.markdown("#### 数据驱动无预设自动发现的微分方程")
        eq_terms = []
        for name, coef in zip(names_res, xi_res):
            if abs(coef) > 1e-4:
                eq_terms.append(f"({coef:+.4f}) \\cdot {name}")
        latex_expr = " \\ddot{\\theta} = " + (" ".join(eq_terms) if eq_terms else "0")
        st.latex(latex_expr)

        st.markdown("#### 可辨识性阈值消融曲线 (区分 sinθ 与线性 θ 的振幅界限)")
        run_identifiability_ablation()
        if os.path.exists("figures/identifiability_curve.png"):
            st.image("figures/identifiability_curve.png", width=720)

    with tab3:
        st.markdown("#### 时域时序波形与向心耗散相图")
        if curr_mode == "sim" and os.path.exists("figures/simulated_trajectory.png"):
            st.image("figures/simulated_trajectory.png", use_container_width=True)
        elif curr_mode == "video" and os.path.exists("figures/extracted_trajectory.png"):
            st.image("figures/extracted_trajectory.png", use_container_width=True)
            
        st.markdown("##### 提取的时序数据表 (前 100 行)")
        st.dataframe(df_data.head(100), use_container_width=True)

    with tab4:
        st.markdown("#### 理论大摆角精确周期对比 (AGM 迭代第一类完全椭圆积分)")
        T0 = small_angle_period(L_input, g_input)
        st.info(f"摆长 L = {L_input:.3f} m 对应的理想小角简谐周期 T₀ = {T0:.4f} s")
        if os.path.exists("figures/theory_period_curve.png"):
            st.image("figures/theory_period_curve.png", width=720)
