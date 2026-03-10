# src/agents/pipeline.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DetectionAgent:
    """이상 탐지 결과 취합"""
    def run(self, anomaly_scores: np.ndarray, risk_scores: np.ndarray,
            threshold: float = 0.0) -> dict:
        anomaly_flags = (anomaly_scores < threshold).astype(int)
        return {
            "anomaly_count": int(anomaly_flags.sum()),
            "total_count": len(anomaly_flags),
            "anomaly_rate": round(float(anomaly_flags.mean()), 4),
            "avg_risk_score": round(float(risk_scores.mean()), 4),
            "high_risk_count": int((risk_scores >= 0.7).sum()),
        }

class DiagnosisAgent:
    """핵심 센서 기반 근본 원인 분석"""
    def run(self, top5_df: pd.DataFrame) -> dict:
        from src.process_map import get_sensor_label, get_process_info
        causes = []
        for _, row in top5_df.iterrows():
            sid = int(row['sensor'])
            info = get_process_info(sid)
            causes.append({
                "sensor_id": sid,
                "label": get_sensor_label(sid),
                "process": info["process"],
                "stage": info["stage"],
                "shap_score": round(float(row['shap_score']), 4),
            })
        # 가장 영향 큰 공정
        top_process = causes[0]["process"] if causes else "UNKNOWN"
        return {
            "root_causes": causes,
            "primary_process": top_process,
            "affected_stages": list(set(c["stage"] for c in causes)),
        }

class ActionAgent:
    """공정별 조치 우선순위 추천"""
    ACTION_DB = {
        "CVD":   ["챔버 압력 센서 점검", "가스 유량 컨트롤러 확인", "챔버 클리닝 스케줄 검토"],
        "ETCH":  ["플라즈마 파워 안정성 확인", "RF 매칭 네트워크 점검", "가스 흐름 균일성 체크"],
        "CMP":   ["웨이퍼 온도 분포 측정", "슬러리 공급량 점검", "패드 컨디셔너 상태 확인"],
        "LITHO": ["얼라인먼트 오프셋 캘리브레이션", "렌즈 클리닝 상태 점검", "스테이지 진동 측정"],
    }

    def run(self, diagnosis: dict) -> dict:
        primary = diagnosis["primary_process"]
        actions = self.ACTION_DB.get(primary, ["설비 전반 점검 필요"])
        return {
            "primary_process": primary,
            "recommended_actions": actions,
            "priority": "즉시 조치" if diagnosis.get("root_causes", [{}])[0].get("shap_score", 0) > 0.02 else "모니터링",
        }

class ReportAgent:
    """최종 운영자 리포트 생성 (LLM)"""
    def run(self, detection: dict, diagnosis: dict, action: dict) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        causes_text = "\n".join([
            f"  - {c['label']} (공정: {c['process']}, SHAP: {c['shap_score']})"
            for c in diagnosis["root_causes"]
        ])
        prompt = f"""
당신은 반도체 FAB 운영 AI 어시스턴트입니다.

[탐지 결과]
- 전체 샘플: {detection['total_count']}개
- 이상 탐지: {detection['anomaly_count']}개 ({detection['anomaly_rate']*100:.1f}%)
- 고위험 샘플: {detection['high_risk_count']}개
- 평균 위험도: {detection['avg_risk_score']:.3f}

[근본 원인 분석]
주요 영향 공정: {diagnosis['primary_process']}
핵심 센서:
{causes_text}

[권장 조치]
우선순위: {action['priority']}
조치 항목:
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(action['recommended_actions']))}

위 내용을 바탕으로 FAB 운영자를 위한 간결한 이상 분석 리포트를 작성해주세요.
다음 구조로 작성하세요:
1. 📊 이상 상황 요약
2. 🔍 근본 원인 분석
3. 🛠️ 권장 조치 순서
4. ⚠️ 추가 모니터링 권고

한국어로 작성하세요.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        return response.choices[0].message.content

class FabAgentPipeline:
    """Detection → Diagnosis → Action → Report 4단계 파이프라인"""
    def __init__(self):
        self.detection_agent = DetectionAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.action_agent = ActionAgent()
        self.report_agent = ReportAgent()

    def run(self, anomaly_scores: np.ndarray, risk_scores: np.ndarray,
            top5_df: pd.DataFrame) -> dict:
        detection = self.detection_agent.run(anomaly_scores, risk_scores)
        diagnosis = self.diagnosis_agent.run(top5_df)
        action = self.action_agent.run(diagnosis)
        report = self.report_agent.run(detection, diagnosis, action)

        # 운영 로그 저장
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "anomaly_count": detection["anomaly_count"],
            "high_risk_count": detection["high_risk_count"],
            "primary_process": diagnosis["primary_process"],
            "priority": action["priority"],
        }
        self._save_log(log_entry)

        return {
            "detection": detection,
            "diagnosis": diagnosis,
            "action": action,
            "report": report,
            "log": log_entry,
        }

    def _save_log(self, entry: dict):
        log_path = "data/raw/operation_log.csv"
        df_new = pd.DataFrame([entry])
        if os.path.exists(log_path):
            df_old = pd.read_csv(log_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(log_path, index=False)
