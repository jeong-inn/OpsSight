import streamlit as st
st.set_page_config(
    page_title="OpsSight - Operational Monitoring & Validation Platform",
    page_icon="⚙️",
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
from src.core.runtime_controller import RuntimeController
from src.test_runner.scenario_loader import ScenarioLoader
from src.test_runner.scenario_executor import ScenarioExecutor
from src.test_runner.validator import ScenarioValidator

@st.cache_data
def load_data():
    X = pd.read_csv("data/raw/X_processed.csv")
    y = pd.read_csv("data/raw/y.csv").squeeze()
    y = -y
    return X, y

def build_runtime_snapshot_from_top_signals(controller, top5_path="data/raw/top5_sensors.csv"):
    if not os.path.exists(top5_path):
        return controller.evaluate_and_update(
            anomaly_score=0.12,
            risk_score=0.18,
            anomaly_count=0,
            persistent_fault=False,
            communication_delay=False,
            sensor_stuck=False,
            compound_fault=False,
        )

    top_df = pd.read_csv(top5_path)

    normalized_scores = []
    warning_count = 0
    critical_count = 0

    for _, row in top_df.iterrows():
        sid = int(row["sensor"])
        info = get_process_info(sid)
        proc = info["process"]
        shap_score = float(row["shap_score"])

        thresh = PROCESS_THRESHOLDS.get(proc, {"warning": 0.6, "critical": 0.8})
        normalized = min(shap_score / 0.03, 1.0)
        normalized_scores.append(normalized)

        if normalized >= thresh["critical"]:
            critical_count += 1
        elif normalized >= thresh["warning"]:
            warning_count += 1

    avg_score = float(np.mean(normalized_scores)) if normalized_scores else 0.0
    max_score = float(np.max(normalized_scores)) if normalized_scores else 0.0

    anomaly_count = warning_count + critical_count
    persistent_fault = critical_count >= 2
    communication_delay = warning_count >= 2
    sensor_stuck = critical_count >= 1 and avg_score > 0.75
    compound_fault = anomaly_count >= 3 and critical_count >= 1

    return controller.evaluate_and_update(
        anomaly_score=max_score,
        risk_score=avg_score,
        anomaly_count=anomaly_count,
        persistent_fault=persistent_fault,
        communication_delay=communication_delay,
        sensor_stuck=sensor_stuck,
        compound_fault=compound_fault,
    )

def render_runtime_snapshot(snapshot, controller):
    state_col1, state_col2, state_col3 = st.columns(3)
    state_col1.metric("Current State", snapshot.current_state)
    state_col2.metric("Alert Level", snapshot.alert_level)
    state_col3.metric("Last Event", snapshot.last_event or "-")

    if snapshot.alert_level == "CRITICAL":
        st.error(f"**Recommended Action**: {snapshot.recommended_action}")
    elif snapshot.alert_level == "WARNING":
        st.warning(f"**Recommended Action**: {snapshot.recommended_action}")
    elif snapshot.alert_level == "CAUTION":
        st.info(f"**Recommended Action**: {snapshot.recommended_action}")
    else:
        st.success(f"**Recommended Action**: {snapshot.recommended_action}")

    with st.expander("Runtime Decision Details", expanded=False):
        st.write("**Reasons**")
        for reason in snapshot.reasons:
            st.write(f"- {reason}")
        
        st.write("**Fault Persistence Tracking**")
        st.write(f"- warning_streak: {snapshot.warning_streak}")
        st.write(f"- critical_streak: {snapshot.critical_streak}")
        st.write(f"- normal_streak: {snapshot.normal_streak}")
        st.write(f"- persistent_critical: {snapshot.persistent_critical}")

        history_df = pd.DataFrame(controller.get_history())
        if not history_df.empty:
            st.write("**State Transition History**")
            st.dataframe(history_df, use_container_width=True, hide_index=True)

# 사이드바
st.sidebar.title("OpsSight")
st.sidebar.markdown("**State-Aware Operational Monitoring Platform**")
st.sidebar.markdown("---")
contamination = st.sidebar.slider("Detection Sensitivity", 0.01, 0.15, 0.07, 0.01)
run_analysis = st.sidebar.button("Run Monitoring Analysis", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Operational Support Pipeline**")
st.sidebar.markdown("1. Detection Agent\n\n2. Diagnosis Agent\n\n3. Action Agent\n\n4. Report Agent")

# 메인 헤더
st.title("OpsSight")
st.markdown("**State-Aware Operational Monitoring, Alerting & Validation System**")
st.markdown("---")

X, y = load_data()
anomaly_count = int((y == -1).sum())
total_count = len(y)
normal_count = total_count - anomaly_count
anomaly_rate = anomaly_count / total_count * 100

runtime_controller = RuntimeController()
runtime_controller.boot()
runtime_controller.arm()
runtime_controller.activate()

runtime_snapshot = build_runtime_snapshot_from_top_signals(runtime_controller)
render_runtime_snapshot(runtime_snapshot, runtime_controller)
st.markdown("---")

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
                [f"🔴 [{a['process']}] {a['param']} — Alert Score {a['score']:.1%}" for a in critical_alerts]
            )
            st.error(f"**CRITICAL ALERT** {alert_msg}")
        if warning_alerts:
            warn_msg = " | ".join(
                [f"🟡 [{a['process']}] {a['param']} — Alert Score {a['score']:.1%}" for a in warning_alerts]
            )
            st.warning(f"**WARNING** {warn_msg}")
    else:
        st.success("✅ All subsystems operating within normal parameters.")
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "System Status Monitoring",
    "Scenario Validation",
    "Alert Overview",
    "Operator Guidance Report",
    "Operation Log",
    "Scenario Stream Simulator",
    "Signal Impact Analysis"
])

# TAB 1: 시스템 상태 모니터링
with tab1:
    st.subheader("System Status Monitoring")
    st.markdown("Subsystem-level operational status overview.")
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
                status = "🔴 ABNORMAL"
                color = "#ff4b4b"
            elif normalized >= thresh["warning"]:
                status = "🟡 DEGRADED"
                color = "#ffa500"
            else:
                status = "🟢 NORMAL"
                color = "#00cc44"

            if proc not in process_status or normalized > process_status[proc]["score"]:
                process_status[proc] = {
                    "status": status,
                    "color": color,
                    "score": normalized,
                    "param": info["param"],
                    "stage": info["stage"]
                }

        st.markdown("### Subsystem Operational Status")
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
                    <p style='font-size:12px; margin:0'>Alert Score: {ps["score"]:.1%}</p>
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
        abnormal_count = sum(1 for v in process_status.values() if "ABNORMAL" in v["status"])
        degraded_count = sum(1 for v in process_status.values() if "DEGRADED" in v["status"])
        normal_count = sum(1 for v in process_status.values() if "NORMAL" in v["status"])

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Abnormal Subsystems", abnormal_count)
        summary_col2.metric("Degraded Subsystems", degraded_count)
        summary_col3.metric("Normal Subsystems", normal_count)

    else:
        st.info("Run signal impact analysis first: python src/analysis/feature_importance.py")
# TAB 2: Scenario Validation
with tab2:
    st.subheader("Scenario Validation")
    st.markdown("Execute predefined operational scenarios and compare expected vs actual results.")
    st.markdown("---")

    loader = ScenarioLoader()
    executor = ScenarioExecutor()
    validator = ScenarioValidator()

    scenario_files = loader.list_scenarios()

    if not scenario_files:
        st.warning("No scenario files found in the scenarios directory.")
    else:
        selected_scenario = st.selectbox(
            "Select Scenario",
            scenario_files,
            index=0
        )

        scenario = loader.load(selected_scenario)

        st.markdown("### Scenario Overview")
        overview_col1, overview_col2 = st.columns(2)
        with overview_col1:
            st.write(f"**Name**: {scenario.name}")
            st.write(f"**Description**: {scenario.description}")
            st.write(f"**Initial Actions**: {', '.join(scenario.initial_actions)}")
        with overview_col2:
            st.write(f"**Expected Final State**: {scenario.expected_final_state}")
            st.write(f"**Expected Alert Level**: {scenario.expected_alert_level}")
            st.write(f"**Event Count**: {len(scenario.events)}")

        if st.button("Run Scenario Validation", use_container_width=True):
            execution = executor.run(scenario)
            validation = validator.validate(
                expected_final_state=scenario.expected_final_state,
                actual_final_state=execution.final_state,
                expected_alert_level=scenario.expected_alert_level,
                actual_alert_level=execution.final_alert_level,
            )

            st.markdown("---")
            st.markdown("### Validation Result")

            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Expected State", scenario.expected_final_state)
            result_col2.metric("Actual State", execution.final_state)
            result_col3.metric("Validation", "PASS" if validation.passed else "FAIL")

            result_col4, result_col5 = st.columns(2)
            result_col4.metric("Expected Alert", scenario.expected_alert_level)
            result_col5.metric("Actual Alert", execution.final_alert_level)

            if validation.passed:
                st.success("Scenario validation passed. Expected and actual results match.")
            else:
                st.error("Scenario validation failed. Expected and actual results do not match.")

            st.markdown("### Recommended Action")
            if execution.final_alert_level == "CRITICAL":
                st.error(execution.recommended_action)
            elif execution.final_alert_level == "WARNING":
                st.warning(execution.recommended_action)
            elif execution.final_alert_level == "CAUTION":
                st.info(execution.recommended_action)
            else:
                st.success(execution.recommended_action)

            st.markdown("### Match Details")
            st.json(validation.details)

            st.markdown("### Step Results")
            step_df = pd.DataFrame(execution.step_results)
            st.dataframe(step_df, use_container_width=True, hide_index=True)
# TAB 3: 이상탐지 & Alert Scoring
with tab3:
    st.subheader("Alert Overview")
    st.markdown("Summarized alert priority view for operational decision support.")

    if run_analysis:
        with st.spinner("Running alert overview..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            risk_scorer = PreFailureRiskScorer()
            risk_scorer.train(X, y)
            risk_scores_test = risk_scorer.predict_risk(X_test)

        st.markdown("### Alert Priority Summary")
        col_a, col_b, col_c = st.columns(3)

        critical_count = int((risk_scores_test >= 0.7).sum())
        warning_count = int(((risk_scores_test >= 0.4) & (risk_scores_test < 0.7)).sum())
        low_count = int((risk_scores_test < 0.4).sum())

        col_a.metric("Critical Candidates", f"{critical_count}")
        col_b.metric("Warning Candidates", f"{warning_count}")
        col_c.metric("Low Priority Signals", f"{low_count}")

        st.markdown("### Alert Priority Score Distribution")
        fig_alert, ax_alert = plt.subplots(figsize=(8, 4))
        ax_alert.hist(
            risk_scores_test[y_test.values == 1],
            bins=30,
            alpha=0.6,
            label="Normal"
        )
        ax_alert.hist(
            risk_scores_test[y_test.values == -1],
            bins=30,
            alpha=0.6,
            label="Anomaly"
        )
        ax_alert.axvline(0.7, linestyle='--', linewidth=1, label='Critical Threshold')
        ax_alert.axvline(0.4, linestyle='--', linewidth=1, label='Warning Threshold')
        ax_alert.set_title("Alert Priority Score Distribution")
        ax_alert.set_xlabel("Alert Score")
        ax_alert.legend()
        st.pyplot(fig_alert)
        plt.close(fig_alert)

        st.info(
            "Alert priority is used as an operational support signal and is connected "
            "to the state-aware runtime decision flow."
        )
    else:
        st.info("Click 'Run Monitoring Analysis' in the sidebar to generate alert overview.")

# TAB 4: Agent 리포트
with tab4:
    st.subheader("Agent-based Operator Guidance Report")
    st.markdown("4-stage operational support pipeline: Detection → Diagnosis → Action → Report")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")

        if st.button("Execute Operational Support Pipeline", use_container_width=True):
            if_scores_input = st.session_state.get('if_scores', np.random.randn(100))
            risk_scores_input = st.session_state.get('risk_scores', np.random.rand(total_count))

            progress_bar = st.progress(0, text="Initializing pipeline...")
            status_area = st.empty()

            status_area.markdown("⚙️ **Stage 1. Detection Agent** running...")
            progress_bar.progress(20, text="Stage 1/4 — Detection Agent")
            import time; time.sleep(0.5)

            status_area.markdown("⚙️ **Stage 2. Diagnosis Agent** running...")
            progress_bar.progress(45, text="Stage 2/4 — Diagnosis Agent")
            time.sleep(0.5)

            status_area.markdown("⚙️ **Stage 3. Action Agent** running...")
            progress_bar.progress(65, text="Stage 3/4 — Action Agent")
            time.sleep(0.5)

            status_area.markdown("⚙️ **Stage 4. Report Agent (GPT-4o-mini)** running...")
            progress_bar.progress(85, text="Stage 4/4 — Report Agent (LLM)")

            with st.spinner("Generating operator guidance report..."):
                pipeline = FabAgentPipeline()
                result = pipeline.run(if_scores_input, risk_scores_input, top5_df)

            progress_bar.progress(100, text="Pipeline complete.")
            status_area.success("✅ All 4 stages completed successfully.")

            det = result["detection"]
            dia = result["diagnosis"]
            act = result["action"]

            st.markdown("---")

            st.markdown("""
            <div style='background:#1a1a2e22; border-left:4px solid #4a90d9;
            border-radius:6px; padding:12px; margin-bottom:12px;'>
            <strong>Stage 1. Detection Agent</strong>
            </div>""", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("Anomaly Detected", f"{det['anomaly_count']}")
            d2.metric("High Priority Samples", f"{det['high_risk_count']}")
            d3.metric("Avg Alert Score", f"{det['avg_risk_score']:.3f}")

            st.markdown("""
            <div style='background:#1a2e1a22; border-left:4px solid #4ad94a;
            border-radius:6px; padding:12px; margin:12px 0;'>
            <strong>Stage 2. Diagnosis Agent</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"**Primary Subsystem**: `{dia['primary_process']}`")
            st.markdown(f"**Affected Stages**: {', '.join(dia['affected_stages'])}")
            causes_df = pd.DataFrame(dia["root_causes"])
            st.dataframe(causes_df, use_container_width=True, hide_index=True)

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

            st.markdown("""
            <div style='background:#2e2a1a22; border-left:4px solid #d9a84a;
            border-radius:6px; padding:12px; margin:12px 0;'>
            <strong>Stage 4. Report Agent (GPT-4o-mini)</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown(result["report"])

            st.markdown("---")
            st.markdown("### 🤖 ReAct Agent Reasoning Trace")
            st.caption("LLM이 판단한 Tool 호출 순서 및 결과")

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
                f.write(f"=== OpsSight Operator Report ===\n")
                f.write(f"Generated: {result['log']['timestamp']}\n\n")
                f.write(result["report"])
            st.success("Report saved: data/raw/agent_report.txt")
    else:
        st.info("Run signal impact analysis first.")

# TAB 5: 운영 로그
with tab5:
    st.subheader("Operation Log")
    st.markdown("Operational support pipeline execution history")

    log_path = "data/raw/operation_log.csv"
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = log_df.sort_values("timestamp", ascending=False).reset_index(drop=True)

        st.markdown(f"**Total records: {len(log_df)}**")

        if len(log_df) > 1:
            st.markdown("---")
            st.markdown("### Abnormal Event History Analysis")

            col_h1, col_h2 = st.columns(2)

            with col_h1:
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
                if 'primary_process' in log_df.columns:
                    proc_freq = log_df['primary_process'].value_counts()
                    fig_h2, ax_h2 = plt.subplots(figsize=(6, 3))
                    proc_colors = ["#ff4b4b" if p in ["CVD", "ETCH"] else "#ffa500"
                                   for p in proc_freq.index]
                    ax_h2.bar(proc_freq.index, proc_freq.values, color=proc_colors)
                    ax_h2.set_title('Abnormal Event Frequency by Subsystem')
                    ax_h2.set_xlabel('Subsystem')
                    ax_h2.set_ylabel('Count')
                    st.pyplot(fig_h2)
                    plt.close(fig_h2)

            st.markdown("#### High Priority Trend")
            fig_h3, ax_h3 = plt.subplots(figsize=(12, 3))
            x_idx = range(len(log_df))
            ax_h3.bar(x_idx, log_df['high_risk_count'].values[::-1],
                      color='#ff4b4b', alpha=0.7, label='High Priority Count')
            ax_h3.plot(x_idx, log_df['anomaly_count'].values[::-1],
                       color='steelblue', marker='o', linewidth=1.5,
                       markersize=4, label='Total Anomaly Count')
            ax_h3.set_title('High Priority vs Anomaly Count Trend')
            ax_h3.set_xlabel('Execution Index')
            ax_h3.legend()
            st.pyplot(fig_h3)
            plt.close(fig_h3)

            st.markdown("---")

            st.markdown("### Log Statistics")
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Executions", f"{len(log_df)}")
            s2.metric("Avg Anomaly Count", f"{log_df['anomaly_count'].mean():.0f}")
            s3.metric("Immediate Action Rate",
                      f"{(log_df['priority']=='즉시 조치').sum()/len(log_df)*100:.0f}%")

        st.markdown("---")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No operation logs yet. Run the operational support pipeline to start logging.")

# TAB 6: Stream Simulator
with tab6:
    st.subheader("Scenario-based Signal Stream Simulator")
    st.markdown("Simulates live multi-signal data ingestion and abnormal condition monitoring in real time")
    st.markdown("---")

    from src.simulator.stream_simulator import SensorStreamSimulator
    from sklearn.ensemble import IsolationForest
    import time

    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        stream_n = st.slider("Signals per tick", 5, 50, 20)
        stream_speed = st.slider("Tick interval (seconds)", 0.5, 3.0, 1.0, 0.5)
        run_stream = st.button("Start Scenario Stream", use_container_width=True)

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
                m2.metric("High Priority", f"{high_risk}")
                m3.metric("Avg Alert Score", f"{avg_risk:.3f}")

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
                                      linewidth=1, label='High Priority')
                    axes_s[1].axhline(0.4, color='orange', linestyle='--',
                                      linewidth=1, label='Medium Priority')
                    axes_s[1].set_title("Avg Alert Score per Tick")
                    axes_s[1].set_xlabel("Tick")
                    axes_s[1].legend(fontsize=8)
                    axes_s[1].set_ylim(0, 1)
                    st.pyplot(fig_s)
                    plt.close(fig_s)

                st.dataframe(hist_df.tail(5)[::-1].reset_index(drop=True),
                             use_container_width=True, hide_index=True)

            time.sleep(stream_speed)

        st.success("Scenario stream simulation complete. 15 ticks processed.")

# TAB 7: SHAP + Root Cause Graph
with tab7:
    st.subheader("Signal Impact Analysis (SHAP Feature Importance)")

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

        st.markdown("#### Top 5 Signals by SHAP Importance (Subsystem Mapping)")
        st.dataframe(top5_df[['sensor','Parameter','Process','Stage','shap_score']],
                     use_container_width=True, hide_index=True)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        labels = [f"{row['Parameter']}" for _, row in top5_df.iterrows()]
        ax4.barh(labels[::-1], top5_df['shap_score'].values[::-1], color='steelblue')
        ax4.set_title('Top 5 Signals - SHAP Importance (with Subsystem Mapping)')
        ax4.set_xlabel('Mean |SHAP Value|')
        st.pyplot(fig4)
        plt.close(fig4)

        st.markdown("---")
        st.markdown("### Root Cause Analysis — Subsystem-Level Impact")
        st.markdown("Aggregated SHAP impact score grouped by subsystem")

        process_impact = top5_df.groupby('Process')['shap_score'].sum().reset_index()
        process_impact.columns = ['Process', 'Total Impact']
        process_impact = process_impact.sort_values('Total Impact', ascending=False)

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
        ax_rc.set_title('Root Cause — Abnormal Impact by Subsystem', fontsize=13)
        ax_rc.set_xlabel('Subsystem')
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

        st.markdown("#### Signal Contribution by Subsystem")
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

        if os.path.exists("data/raw/process_contribution.png"):
            st.markdown("---")
            st.markdown("### Subsystem Contribution Analysis")
            st.caption("서브시스템별 이상 기여도 분석 (SHAP 기반)")
            col_pct = st.columns(4)
            contributions = {"CVD": 40.3, "ETCH": 22.0, "CMP": 19.8, "LITHO": 17.9}
            for i, (proc, pct) in enumerate(contributions.items()):
                col_pct[i].metric(f"{proc}", f"{pct}%", "contribution")
            st.image("data/raw/process_contribution.png", caption="Subsystem Abnormal Contribution")

        if os.path.exists("data/raw/shap_summary.png"):
            st.markdown("---")
            st.image("data/raw/shap_summary.png", caption="SHAP Summary Plot")
    else:
        st.info("Run signal impact analysis first: python src/analysis/feature_importance.py")
