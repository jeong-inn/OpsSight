import streamlit as st
st.set_page_config(
    page_title="FabSight - Smart Fab AI Platform",
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples", f"{total_count:,}")
c2.metric("Anomaly Samples", f"{anomaly_count}")
c3.metric("Anomaly Rate", f"{anomaly_count/total_count*100:.1f}%")
c4.metric("Sensor Features", f"{X.shape[1]}")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "FAB Monitoring",
    "SPC Control Chart",
    "Anomaly Detection & Risk Scoring",
    "Feature Analysis",
    "Agent Diagnosis Report",
    "Operation Log"
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
                status = "🟡  WARNING"
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

# TAB 4: SHAP
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

        if os.path.exists("data/raw/shap_summary.png"):
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
            with st.spinner("Running agent pipeline..."):
                pipeline = FabAgentPipeline()
                result = pipeline.run(if_scores_input, risk_scores_input, top5_df)

            det = result["detection"]
            dia = result["diagnosis"]
            act = result["action"]

            st.markdown("---")
            st.markdown("### Stage 1. Detection Agent")
            d1, d2, d3 = st.columns(3)
            d1.metric("Anomaly Detected", f"{det['anomaly_count']}")
            d2.metric("High Risk Samples", f"{det['high_risk_count']}")
            d3.metric("Avg Risk Score", f"{det['avg_risk_score']:.3f}")

            st.markdown("### Stage 2. Diagnosis Agent")
            st.markdown(f"**Primary Process**: `{dia['primary_process']}`")
            st.markdown(f"**Affected Stages**: {', '.join(dia['affected_stages'])}")
            causes_df = pd.DataFrame(dia["root_causes"])
            st.dataframe(causes_df, use_container_width=True, hide_index=True)

            st.markdown("### Stage 3. Action Agent")
            st.markdown(f"**Priority**: `{act['priority']}`")
            for i, a in enumerate(act["recommended_actions"]):
                st.markdown(f"{i+1}. {a}")

            st.markdown("### Stage 4. Report Agent (GPT-4o-mini)")
            st.markdown(result["report"])

            report_path = "data/raw/agent_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"=== FabSight Agent Report ===\n")
                f.write(f"Generated: {result['log']['timestamp']}\n\n")
                f.write(result["report"])
            st.success("Report saved: data/raw/agent_report.txt")
    else:
        st.info("Run feature importance analysis first.")

# TAB 6: 운영 로그
with tab6:
    st.subheader("FAB Operation Log")
    st.markdown("Agent pipeline execution history")

    log_path = "data/raw/operation_log.csv"
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.markdown(f"**Total records: {len(log_df)}**")
        log_df = log_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        if len(log_df) > 1:
            st.markdown("---")
            st.markdown("### Log Statistics")
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Executions", f"{len(log_df)}")
            s2.metric("Avg Anomaly Count", f"{log_df['anomaly_count'].mean():.0f}")
            s3.metric("Immediate Action Rate",
                      f"{(log_df['priority']=='즉시 조치').sum()/len(log_df)*100:.0f}%")
    else:
        st.info("No operation logs yet. Run the Agent pipeline to start logging.")