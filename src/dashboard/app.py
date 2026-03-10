import streamlit as st
st.set_page_config(
    page_title="FabSight - Smart Fab AI Platform",
    page_icon="🔬",
    layout="wide"
)

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

if platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from src.process_map import get_sensor_label, get_process_info, PROCESS_ORDER, PROCESS_THRESHOLDS
from src.prediction.risk_scorer import PreFailureRiskScorer
from src.agents.pipeline import FabAgentPipeline

@st.cache_data
def load_data():
    X = pd.read_csv("data/raw/X_processed.csv")
    y = pd.read_csv("data/raw/y.csv").squeeze()
    y = -y
    return X, y

# 사이드바
st.sidebar.title("FabSight")
st.sidebar.markdown("**Smart Fab AI Platform**")
st.sidebar.markdown("---")
contamination = st.sidebar.slider("Contamination", 0.01, 0.15, 0.07, 0.01)
run_analysis = st.sidebar.button("Run Analysis", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis Pipeline**")
st.sidebar.markdown("1. Detection Agent\n\n2. Diagnosis Agent\n\n3. Action Agent\n\n4. Report Agent")

# 메인 헤더
st.title("FabSight")
st.markdown("**Smart Semiconductor Fab Monitoring & Anomaly Diagnosis System**")
st.markdown("---")

X, y = load_data()
anomaly_count = int((y == -1).sum())
total_count = len(y)
normal_count = total_count - anomaly_count
anomaly_rate = anomaly_count / total_count * 100

# ── ALERT SYSTEM ──────────────────────────────────────────────────────────────
if os.path.exists("data/raw/top5_sensors.csv"):
    top5_df_alert = pd.read_csv("data/raw/top5_sensors.csv")
    alerts = []
    for _, row in top5_df_alert.iterrows():
        sid = int(row['sensor'])
        info = get_process_info(sid)
        proc = info["process"]
        score = float(row['shap_score'])
        thresh = PROCESS_THRESHOLDS.get(proc, {"warning": 0.6, "critical": 0.8})
        normalized = min(score / 0.03, 1.0)
        if normalized >= thresh["critical"]:
            alerts.append({
                "level": "CRITICAL",
                "process": proc,
                "param": info["param"],
                "stage": info["stage"],
                "score": normalized
            })
        elif normalized >= thresh["warning"]:
            alerts.append({
                "level": "WARNING",
                "process": proc,
                "param": info["param"],
                "stage": info["stage"],
                "score": normalized
            })

    if alerts:
        critical_alerts = [a for a in alerts if a["level"] == "CRITICAL"]
        warning_alerts  = [a for a in alerts if a["level"] == "WARNING"]

        if critical_alerts:
            alert_msg = " | ".join(
                [f"🔴 [{a['process']}] {a['param']} — Risk {a['score']:.1%}" for a in critical_alerts]
            )
            st.error(f"**CRITICAL ALERT** {alert_msg}")
        if warning_alerts:
            warn_msg = " | ".join(
                [f"🟡 [{a['process']}] {a['param']} — Risk {a['score']:.1%}" for a in warning_alerts]
            )
            st.warning(f"**WARNING** {warn_msg}")
    else:
        st.success("✅ All processes operating within normal parameters.")
# ──────────────────────────────────────────────────────────────────────────────

# KPI 상단 배치
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Samples", f"{total_count:,}")
kpi2.metric("Normal Samples", f"{normal_count:,}", delta=f"{100-anomaly_rate:.1f}%")
kpi3.metric("Anomaly Samples", f"{anomaly_count}", delta=f"-{anomaly_rate:.1f}%", delta_color="inverse")
kpi4.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
kpi5.metric("Sensor Features", f"{X.shape[1]}")
st.markdown("---")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
    
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "FAB Monitoring",
    "SPC Control Chart",
    "Anomaly Detection & Risk Scoring",
    "Feature Analysis",
    "Agent Diagnosis Report",
    "Operation Log",
    "Stream Simulator"
])

# TAB 1: FAB 모니터링
with tab1:
    st.subheader("FAB Process Status Monitoring")
    st.markdown("Process-level equipment status overview (Digital Twin inspired)")
    st.markdown("---")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")

        process_status = {}
        for _, row in top5_df.iterrows():
            sid = int(row['sensor'])
            info = get_process_info(sid)
            proc = info["process"]
            score = float(row['shap_score'])
            thresh = PROCESS_THRESHOLDS.get(proc, {"warning": 0.6, "critical": 0.8})
            normalized = min(score / 0.03, 1.0)
            if normalized >= thresh["critical"]:
                status = "🔴 ANOMALY"
                color = "#ff4b4b"
            elif normalized >= thresh["warning"]:
                status = "🟡 WARNING"
                color = "#ffa500"
            else:
                status = "🟢 NORMAL"
                color = "#00cc44"
            if proc not in process_status or normalized > process_status[proc]["score"]:
                process_status[proc] = {
                    "status": status, "color": color,
                    "score": normalized, "param": info["param"],
                    "stage": info["stage"]
                }

        st.markdown("### Process Equipment Status")
        cols = st.columns(4)
        for i, proc in enumerate(PROCESS_ORDER):
            with cols[i]:
                if proc in process_status:
                    ps = process_status[proc]
                    st.markdown(f"""
                    <div style='background:{ps["color"]}22; border:2px solid {ps["color"]};
                    border-radius:8px; padding:16px; text-align:center;'>
                    <h3 style='color:{ps["color"]}; margin:0'>{proc}</h3>
                    <p style='font-size:11px; color:gray; margin:4px 0'>{ps["stage"]}</p>
                    <h4 style='margin:8px 0'>{ps["status"]}</h4>
                    <p style='font-size:12px; margin:0'>Risk: {ps["score"]:.1%}</p>
                    <p style='font-size:11px; color:gray; margin:0'>{ps["param"]}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:#f0f0f022; border:2px solid #cccccc;
                    border-radius:8px; padding:16px; text-align:center;'>
                    <h3 style='color:#cccccc; margin:0'>{proc}</h3>
                    <h4 style='margin:8px 0'>UNKNOWN</h4>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Risk Score by Process")
        fig_fab, ax_fab = plt.subplots(figsize=(10, 3))
        procs = list(process_status.keys())
        scores = [process_status[p]["score"] for p in procs]
        bar_colors = ["#ff4b4b" if s >= 0.8 else "#ffa500" if s >= 0.6 else "#00cc44" for s in scores]
        ax_fab.barh(procs, scores, color=bar_colors)
        ax_fab.axvline(0.6, color='orange', linestyle='--', linewidth=1, label='Warning Threshold')
        ax_fab.axvline(0.8, color='red', linestyle='--', linewidth=1, label='Critical Threshold')
        ax_fab.set_xlim(0, 1)
        ax_fab.set_xlabel('Risk Score')
        ax_fab.set_title('Anomaly Risk Score by Process')
        ax_fab.legend()
        st.pyplot(fig_fab)
        plt.close(fig_fab)
    else:
        st.info("Run feature importance analysis first: python src/analysis/feature_importance.py")

# TAB 2: SPC
with tab2:
    st.subheader("SPC Control Chart (Statistical Process Control)")
    st.markdown("3-sigma rule based anomaly detection using normal data baseline")

    sensor_idx = st.slider("Select Sensor", 0, X.shape[1]-1, 0)
    col = X.columns[sensor_idx]
    sensor = X[col]
    sensor_values = sensor.values
    y_values = y.values
    normal_sensor = sensor[y == 1]

    mean = normal_sensor.mean()
    std = normal_sensor.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['red' if label == -1 else 'steelblue' for label in y_values]
    ax.scatter(range(len(sensor_values)), sensor_values, c=colors, s=8, alpha=0.6)
    ax.axhline(mean, color='green', linewidth=1.5, linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axhline(ucl, color='red', linewidth=1.5, linestyle='--', label=f'UCL: {ucl:.2f}')
    ax.axhline(lcl, color='red', linewidth=1.5, linestyle='--', label=f'LCL: {lcl:.2f}')
    ax.set_title(f'SPC Control Chart - {col}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sensor Value (Normalized)')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    out_of_control = ((sensor > ucl) | (sensor < lcl)).sum()
    st.info(f"Sensor {col}: Out-of-control samples = **{out_of_control}** ({out_of_control/len(sensor)*100:.1f}%)")

# TAB 3: 이상탐지 & Risk Scoring
with tab3:
    st.subheader("Anomaly Detection & Pre-failure Risk Scoring")
    st.markdown("Isolation Forest anomaly detection + GBM-based failure risk prediction")

    if run_analysis:
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            if_model = IsolationForest(n_estimators=200, contamination=contamination,
                                       random_state=42, n_jobs=1)
            if_model.fit(X_train)
            if_scores = if_model.decision_function(X_test)
            if_preds = if_model.predict(X_test)

            risk_scorer = PreFailureRiskScorer()
            risk_metrics = risk_scorer.train(X, y)
            risk_scores_all = risk_scorer.predict_risk(X)
            risk_scores_test = risk_scorer.predict_risk(X_test)

        st.markdown("### Model Performance Comparison")
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_test_bin = (y_test == -1).astype(int)
        if_pred_bin = (if_preds == -1).astype(int)

        comparison_df = pd.DataFrame({
            "Model": ["Isolation Forest", "Risk Scorer (GBM)"],
            "Precision": [
                round(precision_score(y_test_bin, if_pred_bin, zero_division=0), 3),
                risk_metrics["precision"]
            ],
            "Recall": [
                round(recall_score(y_test_bin, if_pred_bin, zero_division=0), 3),
                risk_metrics["recall"]
            ],
            "F1": [
                round(f1_score(y_test_bin, if_pred_bin, zero_division=0), 3),
                risk_metrics["f1"]
            ],
            "ROC-AUC": ["-", risk_metrics["roc_auc"]]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("### Pre-failure Risk Score Distribution")
        col_a, col_b, col_c = st.columns(3)
        high_risk = int((risk_scores_test >= 0.7).sum())
        mid_risk  = int(((risk_scores_test >= 0.4) & (risk_scores_test < 0.7)).sum())
        low_risk  = int((risk_scores_test < 0.4).sum())
        col_a.metric("High Risk", f"{high_risk}")
        col_b.metric("Medium Risk", f"{mid_risk}")
        col_c.metric("Low Risk", f"{low_risk}")

        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
        c2_colors = ['red' if label == -1 else 'steelblue' for label in y_test.values]
        axes[0].scatter(range(len(if_scores)), if_scores, c=c2_colors, s=8, alpha=0.6)
        axes[0].axhline(0, color='orange', linewidth=1.5, linestyle='--', label='Decision Boundary')
        axes[0].set_title('Isolation Forest Anomaly Score')
        axes[0].set_xlabel('Sample Index')
        axes[0].legend()
        axes[1].hist(risk_scores_test[y_test.values == 1], bins=30, alpha=0.6,
                     color='steelblue', label='Normal')
        axes[1].hist(risk_scores_test[y_test.values == -1], bins=30, alpha=0.6,
                     color='red', label='Anomaly')
        axes[1].axvline(0.7, color='red', linestyle='--', label='High Risk')
        axes[1].axvline(0.4, color='orange', linestyle='--', label='Medium Risk')
        axes[1].set_title('Pre-failure Risk Score Distribution')
        axes[1].set_xlabel('Risk Score')
        axes[1].legend()
        st.pyplot(fig3)
        plt.close(fig3)

        st.session_state['if_scores'] = if_scores
        st.session_state['risk_scores'] = risk_scores_all
        st.session_state['analysis_done'] = True
    else:
        st.info("Click 'Run Analysis' in the sidebar to start.")

# TAB 4: SHAP + Root Cause Graph
with tab4:
    st.subheader("Feature Analysis (SHAP Feature Importance)")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")
        top5_df['Process'] = top5_df['sensor'].apply(
            lambda s: get_process_info(int(s))["process"]
        )
        top5_df['Parameter'] = top5_df['sensor'].apply(
            lambda s: get_sensor_label(int(s))
        )
        top5_df['Stage'] = top5_df['sensor'].apply(
            lambda s: get_process_info(int(s))["stage"]
        )

        st.markdown("#### Top 5 Sensors by SHAP Importance (Process Mapping)")
        st.dataframe(top5_df[['sensor','Parameter','Process','Stage','shap_score']],
                     use_container_width=True, hide_index=True)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        labels = [f"{row['Parameter']}" for _, row in top5_df.iterrows()]
        ax4.barh(labels[::-1], top5_df['shap_score'].values[::-1], color='steelblue')
        ax4.set_title('Top 5 Sensors - SHAP Importance (with Process Mapping)')
        ax4.set_xlabel('Mean |SHAP Value|')
        st.pyplot(fig4)
        plt.close(fig4)

        # ── ROOT CAUSE GRAPH ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Root Cause Analysis — Process-Level Impact")
        st.markdown("Aggregated SHAP impact score grouped by FAB process stage")

        process_impact = top5_df.groupby('Process')['shap_score'].sum().reset_index()
        process_impact.columns = ['Process', 'Total Impact']
        process_impact = process_impact.sort_values('Total Impact', ascending=False)

        # 공정별 임팩트 bar chart
        fig_rc, ax_rc = plt.subplots(figsize=(8, 4))
        impact_colors = []
        for _, row in process_impact.iterrows():
            thresh = PROCESS_THRESHOLDS.get(row['Process'], {"warning": 0.6, "critical": 0.8})
            normalized = min(row['Total Impact'] / 0.03, 1.0)
            if normalized >= thresh["critical"]:
                impact_colors.append("#ff4b4b")
            elif normalized >= thresh["warning"]:
                impact_colors.append("#ffa500")
            else:
                impact_colors.append("#00cc44")

        bars = ax_rc.bar(process_impact['Process'], process_impact['Total Impact'],
                         color=impact_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, process_impact['Total Impact']):
            ax_rc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax_rc.set_title('Root Cause — Anomaly Impact by Process', fontsize=13)
        ax_rc.set_xlabel('Process')
        ax_rc.set_ylabel('Cumulative SHAP Impact Score')
        ax_rc.set_ylim(0, process_impact['Total Impact'].max() * 1.2)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff4b4b', label='Critical'),
            Patch(facecolor='#ffa500', label='Warning'),
            Patch(facecolor='#00cc44', label='Normal'),
        ]
        ax_rc.legend(handles=legend_elements, loc='upper right')
        st.pyplot(fig_rc)
        plt.close(fig_rc)

        # 공정별 주요 센서 요약 카드
        st.markdown("#### Sensor Contribution by Process")
        proc_cols = st.columns(len(process_impact))
        for i, (_, prow) in enumerate(process_impact.iterrows()):
            proc = prow['Process']
            sensors_in_proc = top5_df[top5_df['Process'] == proc]
            thresh = PROCESS_THRESHOLDS.get(proc, {"warning": 0.6, "critical": 0.8})
            normalized = min(prow['Total Impact'] / 0.03, 1.0)
            if normalized >= thresh["critical"]:
                card_color = "#ff4b4b"
            elif normalized >= thresh["warning"]:
                card_color = "#ffa500"
            else:
                card_color = "#00cc44"

            sensor_list = "<br>".join(
                [f"• {r['Parameter']} ({r['shap_score']:.4f})" for _, r in sensors_in_proc.iterrows()]
            )
            with proc_cols[i]:
                st.markdown(f"""
                <div style='background:{card_color}22; border:2px solid {card_color};
                border-radius:8px; padding:12px; text-align:center;'>
                <h4 style='color:{card_color}; margin:0'>{proc}</h4>
                <p style='font-size:11px; margin:4px 0'>Impact: {prow['Total Impact']:.4f}</p>
                <p style='font-size:11px; color:#555; margin:0; text-align:left'>{sensor_list}</p>
                </div>""", unsafe_allow_html=True)
        # ──────────────────────────────────────────────────────────────────────

        if os.path.exists("data/raw/shap_summary.png"):
            st.markdown("---")
            st.image("data/raw/shap_summary.png", caption="SHAP Summary Plot")
    else:
        st.info("Run feature importance analysis first: python src/analysis/feature_importance.py")

# TAB 5: Agent 리포트
with tab5:
    st.subheader("Agent-based Anomaly Diagnosis Report")
    st.markdown("4-stage Agent pipeline: Detection → Diagnosis → Action → Report")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")

        if st.button("Execute Agent Pipeline", use_container_width=True):
            if_scores_input = st.session_state.get('if_scores', np.random.randn(100))
            risk_scores_input = st.session_state.get('risk_scores', np.random.rand(total_count))

            # 단계별 진행 표시
            progress_bar = st.progress(0, text="Initializing pipeline...")
            status_area = st.empty()

            # Stage 1
            status_area.markdown("⚙️ **Stage 1. Detection Agent** running...")
            progress_bar.progress(20, text="Stage 1/4 — Detection Agent")
            import time; time.sleep(0.5)

            # Stage 2
            status_area.markdown("⚙️ **Stage 2. Diagnosis Agent** running...")
            progress_bar.progress(45, text="Stage 2/4 — Diagnosis Agent")
            time.sleep(0.5)

            # Stage 3
            status_area.markdown("⚙️ **Stage 3. Action Agent** running...")
            progress_bar.progress(65, text="Stage 3/4 — Action Agent")
            time.sleep(0.5)

            # Stage 4
            status_area.markdown("⚙️ **Stage 4. Report Agent (GPT-4o-mini)** running...")
            progress_bar.progress(85, text="Stage 4/4 — Report Agent (LLM)")

            with st.spinner("Generating LLM report..."):
                pipeline = FabAgentPipeline()
                result = pipeline.run(if_scores_input, risk_scores_input, top5_df)

            progress_bar.progress(100, text="Pipeline complete.")
            status_area.success("✅ All 4 stages completed successfully.")

            det = result["detection"]
            dia = result["diagnosis"]
            act = result["action"]

            st.markdown("---")

            # Stage 1 결과 카드
            st.markdown("""
            <div style='background:#1a1a2e22; border-left:4px solid #4a90d9;
            border-radius:6px; padding:12px; margin-bottom:12px;'>
            <strong>Stage 1. Detection Agent</strong>
            </div>""", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("Anomaly Detected", f"{det['anomaly_count']}")
            d2.metric("High Risk Samples", f"{det['high_risk_count']}")
            d3.metric("Avg Risk Score", f"{det['avg_risk_score']:.3f}")

            # Stage 2 결과 카드
            st.markdown("""
            <div style='background:#1a2e1a22; border-left:4px solid #4ad94a;
            border-radius:6px; padding:12px; margin:12px 0;'>
            <strong>Stage 2. Diagnosis Agent</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"**Primary Process**: `{dia['primary_process']}`")
            st.markdown(f"**Affected Stages**: {', '.join(dia['affected_stages'])}")
            causes_df = pd.DataFrame(dia["root_causes"])
            st.dataframe(causes_df, use_container_width=True, hide_index=True)

            # Stage 3 결과 카드
            st.markdown("""
            <div style='background:#2e1a1a22; border-left:4px solid #d94a4a;
            border-radius:6px; padding:12px; margin:12px 0;'>
            <strong>Stage 3. Action Agent</strong>
            </div>""", unsafe_allow_html=True)
            priority_color = "#ff4b4b" if act['priority'] == "즉시 조치" else "#ffa500"
            st.markdown(f"**Priority**: <span style='color:{priority_color}; font-weight:bold'>{act['priority']}</span>",
                        unsafe_allow_html=True)
            for i, a in enumerate(act["recommended_actions"]):
                st.markdown(f"{i+1}. {a}")

            # Stage 4 결과 카드
            st.markdown("""
            <div style='background:#2e2a1a22; border-left:4px solid #d9a84a;
            border-radius:6px; padding:12px; margin:12px 0;'>
            <strong>Stage 4. Report Agent (GPT-4o-mini)</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown(result["report"])

# ReAct Agent 추론 로그
            st.markdown("---")
            st.markdown("### 🤖 ReAct Agent Reasoning Trace")
            st.caption("LLM이 스스로 판단한 Tool 호출 순서 및 결과")

            TOOL_META = {
                "analyze_anomaly":    ("", "#4a90d9", "Anomaly Analysis"),
                "diagnose_root_cause":("", "#4ad94a", "Root Cause Diagnosis"),
                "get_action_plan":    ("", "#d94a4a", "Action Planning"),
                "generate_report":    ("", "#d9a84a", "Report Generation"),
            }

            for log in result.get("react_log", []):
                tool = log["tool"]
                icon, color, label = TOOL_META.get(tool, ("", "#888", tool))
                with st.expander(f"Step {log['step']}: {label}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Input (LLM → Tool)**")
                        st.json(log["args"])
                    with col2:
                        st.markdown("**Output (Tool → LLM)**")
                        st.json(log["result"])

            report_path = "data/raw/agent_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"=== FabSight Agent Report ===\n")
                f.write(f"Generated: {result['log']['timestamp']}\n\n")
                f.write(result["report"])
            st.success("Report saved: data/raw/agent_report.txt")
    else:
        st.info("Run feature importance analysis first.")

# TAB 6: 운영 로그 + Anomaly History Chart
with tab6:
    st.subheader("FAB Operation Log")
    st.markdown("Agent pipeline execution history")

    log_path = "data/raw/operation_log.csv"
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = log_df.sort_values("timestamp", ascending=False).reset_index(drop=True)

        st.markdown(f"**Total records: {len(log_df)}**")

        if len(log_df) > 1:
            # ── ANOMALY HISTORY CHART ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Anomaly History Analysis")

            col_h1, col_h2 = st.columns(2)

            with col_h1:
                # 시간대별 anomaly count
                fig_h1, ax_h1 = plt.subplots(figsize=(6, 3))
                ax_h1.plot(range(len(log_df)), log_df['anomaly_count'].values[::-1],
                           marker='o', color='steelblue', linewidth=2, markersize=5)
                ax_h1.fill_between(range(len(log_df)),
                                   log_df['anomaly_count'].values[::-1],
                                   alpha=0.2, color='steelblue')
                ax_h1.set_title('Anomaly Count per Execution')
                ax_h1.set_xlabel('Execution Index')
                ax_h1.set_ylabel('Anomaly Count')
                st.pyplot(fig_h1)
                plt.close(fig_h1)

            with col_h2:
                # 공정별 이상 발생 빈도
                if 'primary_process' in log_df.columns:
                    proc_freq = log_df['primary_process'].value_counts()
                    fig_h2, ax_h2 = plt.subplots(figsize=(6, 3))
                    proc_colors = ["#ff4b4b" if p in ["CVD", "ETCH"] else "#ffa500"
                                   for p in proc_freq.index]
                    ax_h2.bar(proc_freq.index, proc_freq.values, color=proc_colors)
                    ax_h2.set_title('Anomaly Frequency by Process')
                    ax_h2.set_xlabel('Process')
                    ax_h2.set_ylabel('Count')
                    st.pyplot(fig_h2)
                    plt.close(fig_h2)

            # high risk trend
            st.markdown("#### High Risk Sample Trend")
            fig_h3, ax_h3 = plt.subplots(figsize=(12, 3))
            x_idx = range(len(log_df))
            ax_h3.bar(x_idx, log_df['high_risk_count'].values[::-1],
                      color='#ff4b4b', alpha=0.7, label='High Risk Count')
            ax_h3.plot(x_idx, log_df['anomaly_count'].values[::-1],
                       color='steelblue', marker='o', linewidth=1.5,
                       markersize=4, label='Total Anomaly Count')
            ax_h3.set_title('High Risk vs Anomaly Count Trend')
            ax_h3.set_xlabel('Execution Index')
            ax_h3.legend()
            st.pyplot(fig_h3)
            plt.close(fig_h3)

            st.markdown("---")
            # ─────────────────────────────────────────────────────────────────

            st.markdown("### Log Statistics")
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Executions", f"{len(log_df)}")
            s2.metric("Avg Anomaly Count", f"{log_df['anomaly_count'].mean():.0f}")
            s3.metric("Immediate Action Rate",
                      f"{(log_df['priority']=='즉시 조치').sum()/len(log_df)*100:.0f}%")

        st.markdown("---")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No operation logs yet. Run the Agent pipeline to start logging.")

# TAB 7: Stream Simulator
with tab7:
    st.subheader("Real-time Sensor Stream Simulator")
    st.markdown("Simulates live sensor data ingestion and anomaly detection in real-time")
    st.markdown("---")

    from src.simulator.stream_simulator import SensorStreamSimulator
    from sklearn.ensemble import IsolationForest
    import time

    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        stream_n = st.slider("Samples per tick", 5, 50, 20)
        stream_speed = st.slider("Speed (seconds)", 0.5, 3.0, 1.0, 0.5)
        run_stream = st.button("Start Stream", use_container_width=True)

    with col_s2:
        stream_placeholder = st.empty()

    if run_stream:
        simulator = SensorStreamSimulator(X, y, window_size=stream_n)
        if_model = IsolationForest(n_estimators=100, contamination=0.07,
                                   random_state=42, n_jobs=1)
        if_model.fit(X)

        risk_scorer = PreFailureRiskScorer()
        risk_scorer.train(X, y)

        history = []
        stop_placeholder = st.empty()

        for tick in range(15):
            X_window, y_window = simulator.get_random_sample(stream_n)
            scores = if_model.decision_function(X_window)
            preds  = if_model.predict(X_window)
            risks  = risk_scorer.predict_risk(X_window)

            anomaly_count = int((preds == -1).sum())
            high_risk     = int((risks >= 0.7).sum())
            avg_risk      = float(risks.mean())

            history.append({
                "tick": tick + 1,
                "anomaly_count": anomaly_count,
                "high_risk": high_risk,
                "avg_risk": round(avg_risk, 3),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            hist_df = pd.DataFrame(history)

            with stream_placeholder.container():
                st.markdown(f"**Tick {tick+1}/15** — {datetime.now().strftime('%H:%M:%S')}")

                m1, m2, m3 = st.columns(3)
                status_color = "🔴" if anomaly_count > 3 else "🟡" if anomaly_count > 0 else "🟢"
                m1.metric("Anomalies", f"{status_color} {anomaly_count}")
                m2.metric("High Risk", f"{high_risk}")
                m3.metric("Avg Risk Score", f"{avg_risk:.3f}")

                if len(hist_df) > 1:
                    fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 2.5))
                    axes_s[0].plot(hist_df["tick"], hist_df["anomaly_count"],
                                   marker='o', color='#ff4b4b', linewidth=2)
                    axes_s[0].fill_between(hist_df["tick"], hist_df["anomaly_count"],
                                           alpha=0.2, color='#ff4b4b')
                    axes_s[0].set_title("Anomaly Count per Tick")
                    axes_s[0].set_xlabel("Tick")

                    axes_s[1].plot(hist_df["tick"], hist_df["avg_risk"],
                                   marker='s', color='steelblue', linewidth=2)
                    axes_s[1].axhline(0.7, color='red', linestyle='--',
                                      linewidth=1, label='High Risk')
                    axes_s[1].axhline(0.4, color='orange', linestyle='--',
                                      linewidth=1, label='Medium Risk')
                    axes_s[1].set_title("Avg Risk Score per Tick")
                    axes_s[1].set_xlabel("Tick")
                    axes_s[1].legend(fontsize=8)
                    axes_s[1].set_ylim(0, 1)
                    st.pyplot(fig_s)
                    plt.close(fig_s)

                st.dataframe(hist_df.tail(5)[::-1].reset_index(drop=True),
                             use_container_width=True, hide_index=True)

            time.sleep(stream_speed)

        st.success("Stream simulation complete. 15 ticks processed.")