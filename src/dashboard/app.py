import streamlit as st
st.set_page_config(
    page_title="FabSight - Smart Fab AI Platform",
    page_icon="🔬",
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import platform
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# OS별 폰트
if platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from src.process_map import get_sensor_label, get_process_info, PROCESS_ORDER, PROCESS_THRESHOLDS
from src.prediction.risk_scorer import PreFailureRiskScorer
from src.agents.pipeline import FabAgentPipeline

# ─── 데이터 로드 ───
@st.cache_data
def load_data():
    X = pd.read_csv("data/raw/X_processed.csv")
    y = pd.read_csv("data/raw/y.csv").squeeze()
    y = -y
    return X, y

# ─── 사이드바 ───
st.sidebar.title("⚙️ FabSight")
st.sidebar.markdown("**Smart Fab AI Platform**")
st.sidebar.markdown("---")
contamination = st.sidebar.slider("Contamination", 0.01, 0.15, 0.07, 0.01)
run_analysis = st.sidebar.button("🚀 전체 분석 실행", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**분석 파이프라인**")
st.sidebar.markdown("1️⃣ Detection Agent\n\n2️⃣ Diagnosis Agent\n\n3️⃣ Action Agent\n\n4️⃣ Report Agent")

# ─── 메인 헤더 ───
st.title("🔬 FabSight")
st.markdown("**Smart Semiconductor Fab Monitoring & Anomaly Diagnosis System**")
st.markdown("---")

X, y = load_data()
anomaly_count = int((y == -1).sum())
total_count = len(y)

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 샘플", f"{total_count:,}개")
c2.metric("이상 샘플", f"{anomaly_count}개")
c3.metric("이상 비율", f"{anomaly_count/total_count*100:.1f}%")
c4.metric("센서 피처 수", f"{X.shape[1]}개")
st.markdown("---")

# ─── 탭 구성 ───
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 FAB 모니터링",
    "📈 SPC 관리도",
    "🤖 이상 탐지 & 위험도",
    "🔍 핵심 센서 분석",
    "🧠 Agent 진단 리포트",
    "📋 운영 로그"
])

# ═══════════════════════════════════════════
# TAB 1: FAB 상태 모니터링
# ═══════════════════════════════════════════
with tab1:
    st.subheader("🏭 FAB 공정 상태 모니터링")
    st.markdown("반도체 제조 공정별 실시간 설비 상태 현황 (Digital Twin inspired)")
    st.markdown("---")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")
        
        # 공정별 상태 계산
        process_status = {}
        for _, row in top5_df.iterrows():
            sid = int(row['sensor'])
            info = get_process_info(sid)
            proc = info["process"]
            score = float(row['shap_score'])
            thresh = PROCESS_THRESHOLDS.get(proc, {"warning": 0.6, "critical": 0.8})
            # shap score 정규화 (0~1)
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

        st.markdown("### 공정별 설비 상태")
        cols = st.columns(4)
        for i, proc in enumerate(PROCESS_ORDER):
            with cols[i]:
                if proc in process_status:
                    ps = process_status[proc]
                    st.markdown(f"""
                    <div style='background:{ps["color"]}22; border:2px solid {ps["color"]};
                    border-radius:10px; padding:16px; text-align:center;'>
                    <h3 style='color:{ps["color"]}; margin:0'>{proc}</h3>
                    <p style='font-size:11px; color:gray; margin:4px 0'>{ps["stage"]}</p>
                    <h4 style='margin:8px 0'>{ps["status"]}</h4>
                    <p style='font-size:12px; margin:0'>위험도: {ps["score"]:.1%}</p>
                    <p style='font-size:11px; color:gray; margin:0'>{ps["param"]}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:#f0f0f022; border:2px solid #cccccc;
                    border-radius:10px; padding:16px; text-align:center;'>
                    <h3 style='color:#cccccc; margin:0'>{proc}</h3>
                    <h4 style='margin:8px 0'>⚪ UNKNOWN</h4>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 공정별 위험도 비교")
        fig_fab, ax_fab = plt.subplots(figsize=(10, 3))
        procs = list(process_status.keys())
        scores = [process_status[p]["score"] for p in procs]
        bar_colors = ["#ff4b4b" if s >= 0.8 else "#ffa500" if s >= 0.6 else "#00cc44" for s in scores]
        ax_fab.barh(procs, scores, color=bar_colors)
        ax_fab.axvline(0.6, color='orange', linestyle='--', linewidth=1, label='Warning')
        ax_fab.axvline(0.8, color='red', linestyle='--', linewidth=1, label='Critical')
        ax_fab.set_xlim(0, 1)
        ax_fab.set_xlabel('위험도 Score')
        ax_fab.set_title('공정별 이상 위험도')
        ax_fab.legend()
        st.pyplot(fig_fab)
        plt.close(fig_fab)
    else:
        st.info("먼저 핵심 센서 분석을 실행해주세요. (터미널: python src/analysis/feature_importance.py)")

# ═══════════════════════════════════════════
# TAB 2: SPC 관리도
# ═══════════════════════════════════════════
with tab2:
    st.subheader("SPC 관리도 (Statistical Process Control)")
    st.markdown("정상 데이터 기준 3-sigma 룰 기반 이상탐지")

    sensor_idx = st.slider("센서 선택", 0, X.shape[1]-1, 0)
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
    ax.axhline(mean, color='green', linewidth=1.5, linestyle='--', label=f'평균: {mean:.2f}')
    ax.axhline(ucl, color='red', linewidth=1.5, linestyle='--', label=f'UCL: {ucl:.2f}')
    ax.axhline(lcl, color='red', linewidth=1.5, linestyle='--', label=f'LCL: {lcl:.2f}')
    ax.set_title(f'SPC 관리도 - {col}')
    ax.set_xlabel('샘플 인덱스')
    ax.set_ylabel('센서값 (정규화)')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    out_of_control = ((sensor > ucl) | (sensor < lcl)).sum()
    st.info(f"센서 {col}: 관리 한계 이탈 = **{out_of_control}개** ({out_of_control/len(sensor)*100:.1f}%)")

# ═══════════════════════════════════════════
# TAB 3: 이상 탐지 & 위험도
# ═══════════════════════════════════════════
with tab3:
    st.subheader("🤖 이상 탐지 & Pre-failure Risk Scoring")
    st.markdown("Isolation Forest 이상탐지 + GBM 기반 고장 위험도 예측")

    if run_analysis:
        with st.spinner("Isolation Forest + Risk Scorer 학습 중..."):
            # Isolation Forest
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            if_model = IsolationForest(n_estimators=200, contamination=contamination,
                                       random_state=42, n_jobs=1)
            if_model.fit(X_train)
            if_scores = if_model.decision_function(X_test)
            if_preds = if_model.predict(X_test)

            # Risk Scorer
            risk_scorer = PreFailureRiskScorer()
            risk_metrics = risk_scorer.train(X, y)
            risk_scores_all = risk_scorer.predict_risk(X)
            risk_scores_test = risk_scorer.predict_risk(X_test)

        # 성능 비교표
        st.markdown("### 📊 모델 성능 비교")
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_test_bin = (y_test == -1).astype(int)
        if_pred_bin = (if_preds == -1).astype(int)

        comparison_df = pd.DataFrame({
            "모델": ["Isolation Forest", "Risk Scorer (GBM)"],
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

        # 위험도 분포
        st.markdown("### 🎯 고장 위험도 분포 (Pre-failure Risk Score)")
        col_a, col_b, col_c = st.columns(3)
        high_risk = int((risk_scores_test >= 0.7).sum())
        mid_risk  = int(((risk_scores_test >= 0.4) & (risk_scores_test < 0.7)).sum())
        low_risk  = int((risk_scores_test < 0.4).sum())
        col_a.metric("🔴 HIGH RISK", f"{high_risk}개")
        col_b.metric("🟡 MEDIUM RISK", f"{mid_risk}개")
        col_c.metric("🟢 LOW RISK", f"{low_risk}개")

        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
        # IF Score
        c2_colors = ['red' if label == -1 else 'steelblue' for label in y_test.values]
        axes[0].scatter(range(len(if_scores)), if_scores, c=c2_colors, s=8, alpha=0.6)
        axes[0].axhline(0, color='orange', linewidth=1.5, linestyle='--', label='Decision Boundary')
        axes[0].set_title('Isolation Forest Anomaly Score')
        axes[0].set_xlabel('Sample Index')
        axes[0].legend()
        # Risk Score
        axes[1].hist(risk_scores_test[y_test.values == 1], bins=30, alpha=0.6,
                     color='steelblue', label='정상')
        axes[1].hist(risk_scores_test[y_test.values == -1], bins=30, alpha=0.6,
                     color='red', label='이상')
        axes[1].axvline(0.7, color='red', linestyle='--', label='High Risk')
        axes[1].axvline(0.4, color='orange', linestyle='--', label='Medium Risk')
        axes[1].set_title('Pre-failure Risk Score 분포')
        axes[1].set_xlabel('Risk Score')
        axes[1].legend()
        st.pyplot(fig3)
        plt.close(fig3)

        # 세션에 저장 (Agent 탭에서 사용)
        st.session_state['if_scores'] = if_scores
        st.session_state['risk_scores'] = risk_scores_all
        st.session_state['analysis_done'] = True
    else:
        st.info("👈 사이드바에서 '🚀 전체 분석 실행'을 눌러주세요.")

# ═══════════════════════════════════════════
# TAB 4: SHAP 핵심 센서
# ═══════════════════════════════════════════
with tab4:
    st.subheader("🔍 핵심 센서 분석 (SHAP Feature Importance)")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")

        # 공정명 매핑 추가
        top5_df['공정'] = top5_df['sensor'].apply(
            lambda s: get_process_info(int(s))["process"]
        )
        top5_df['파라미터'] = top5_df['sensor'].apply(
            lambda s: get_sensor_label(int(s))
        )
        top5_df['공정단계'] = top5_df['sensor'].apply(
            lambda s: get_process_info(int(s))["stage"]
        )

        st.markdown("#### 이상에 영향을 미친 Top 5 센서 (공정 매핑)")
        st.dataframe(top5_df[['sensor','파라미터','공정','공정단계','shap_score']],
                     use_container_width=True, hide_index=True)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        labels = [f"{row['파라미터']}" for _, row in top5_df.iterrows()]
        ax4.barh(labels[::-1], top5_df['shap_score'].values[::-1], color='steelblue')
        ax4.set_title('Top 5 Sensors - SHAP Importance (공정 매핑)')
        ax4.set_xlabel('Mean |SHAP Value|')
        st.pyplot(fig4)
        plt.close(fig4)

        if os.path.exists("data/raw/shap_summary.png"):
            st.image("data/raw/shap_summary.png", caption="SHAP Summary Plot")
    else:
        st.info("터미널에서 먼저 실행: python src/analysis/feature_importance.py")

# ═══════════════════════════════════════════
# TAB 5: Agent 진단 리포트
# ═══════════════════════════════════════════
with tab5:
    st.subheader("🧠 Agent 기반 이상 진단 리포트")
    st.markdown("Detection → Diagnosis → Action → Report 4단계 Agent 파이프라인")

    if os.path.exists("data/raw/top5_sensors.csv"):
        top5_df = pd.read_csv("data/raw/top5_sensors.csv")

        if st.button("🤖 Agent 파이프라인 실행", use_container_width=True):
            if_scores_input = st.session_state.get(
                'if_scores', np.random.randn(100)
            )
            risk_scores_input = st.session_state.get(
                'risk_scores', np.random.rand(total_count)
            )
            with st.spinner("Agent 파이프라인 실행 중..."):
                pipeline = FabAgentPipeline()
                result = pipeline.run(if_scores_input, risk_scores_input, top5_df)

            # 결과 표시
            det = result["detection"]
            dia = result["diagnosis"]
            act = result["action"]

            st.markdown("---")
            st.markdown("### 1️⃣ Detection Agent 결과")
            d1, d2, d3 = st.columns(3)
            d1.metric("이상 탐지", f"{det['anomaly_count']}개")
            d2.metric("고위험 샘플", f"{det['high_risk_count']}개")
            d3.metric("평균 위험도", f"{det['avg_risk_score']:.3f}")

            st.markdown("### 2️⃣ Diagnosis Agent 결과")
            st.markdown(f"**주요 영향 공정**: `{dia['primary_process']}`")
            st.markdown(f"**영향 공정 단계**: {', '.join(dia['affected_stages'])}")
            causes_df = pd.DataFrame(dia["root_causes"])
            st.dataframe(causes_df, use_container_width=True, hide_index=True)

            st.markdown("### 3️⃣ Action Agent 결과")
            st.markdown(f"**우선순위**: `{act['priority']}`")
            for i, a in enumerate(act["recommended_actions"]):
                st.markdown(f"{i+1}. {a}")

            st.markdown("### 4️⃣ Report Agent 결과 (GPT-4o-mini)")
            st.markdown(result["report"])

            # 저장
            report_path = "data/raw/agent_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"=== FabSight Agent Report ===\n")
                f.write(f"생성 시각: {result['log']['timestamp']}\n\n")
                f.write(result["report"])
            st.success("✅ 리포트 저장 완료: data/raw/agent_report.txt")
    else:
        st.info("먼저 핵심 센서 분석을 실행해주세요.")

# ═══════════════════════════════════════════
# TAB 6: 운영 로그
# ═══════════════════════════════════════════
with tab6:
    st.subheader("📋 FAB 운영 로그")
    st.markdown("Agent 파이프라인 실행 이력")

    log_path = "data/raw/operation_log.csv"
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.markdown(f"**총 {len(log_df)}건의 분석 기록**")

        # 최신순 정렬
        log_df = log_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        # 통계
        if len(log_df) > 1:
            st.markdown("---")
            st.markdown("### 로그 통계")
            s1, s2, s3 = st.columns(3)
            s1.metric("총 분석 횟수", f"{len(log_df)}회")
            s2.metric("평균 이상 탐지", f"{log_df['anomaly_count'].mean():.0f}개")
            s3.metric("즉시조치 비율",
                      f"{(log_df['priority']=='즉시 조치').sum()/len(log_df)*100:.0f}%")
    else:
        st.info("아직 운영 로그가 없습니다. Agent 파이프라인을 실행하면 자동으로 기록됩니다.")
