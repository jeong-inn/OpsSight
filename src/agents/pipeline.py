import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Tool 정의 ──────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_anomaly",
            "description": "이상 탐지 결과를 분석하고 심각도를 판단합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "anomaly_rate": {"type": "number", "description": "이상 탐지 비율 (0~1)"},
                    "high_risk_count": {"type": "integer", "description": "고위험 샘플 수"},
                    "avg_risk_score": {"type": "number", "description": "평균 위험도 점수"}
                },
                "required": ["anomaly_rate", "high_risk_count", "avg_risk_score"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_root_cause",
            "description": "SHAP 기반 핵심 센서를 분석하여 근본 원인 공정을 진단합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_process": {"type": "string", "description": "주요 영향 공정 (CVD/ETCH/CMP/LITHO)"},
                    "top_sensor_id": {"type": "integer", "description": "가장 영향력 높은 센서 ID"},
                    "shap_score": {"type": "number", "description": "최고 SHAP 점수"}
                },
                "required": ["primary_process", "top_sensor_id", "shap_score"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_action_plan",
            "description": "공정별 조치 계획을 조회하고 우선순위를 결정합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "process": {"type": "string", "description": "조치가 필요한 공정"},
                    "severity": {"type": "string", "enum": ["critical", "warning", "normal"], "description": "심각도"}
                },
                "required": ["process", "severity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "최종 운영자 리포트를 생성합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "이상 상황 요약"},
                    "root_cause": {"type": "string", "description": "근본 원인"},
                    "actions": {"type": "string", "description": "권장 조치"},
                    "monitoring": {"type": "string", "description": "추가 모니터링 권고"}
                },
                "required": ["summary", "root_cause", "actions", "monitoring"]
            }
        }
    }
]

# ── Tool 실행 함수 ──────────────────────────────────────
ACTION_DB = {
    "CVD":   ["챔버 압력 센서 점검", "가스 유량 컨트롤러 확인", "챔버 클리닝 스케줄 검토"],
    "ETCH":  ["플라즈마 파워 안정성 확인", "RF 매칭 네트워크 점검", "가스 흐름 균일성 체크"],
    "CMP":   ["웨이퍼 온도 분포 측정", "슬러리 공급량 점검", "패드 컨디셔너 상태 확인"],
    "LITHO": ["얼라인먼트 오프셋 캘리브레이션", "렌즈 클리닝 상태 점검", "스테이지 진동 측정"],
}

def execute_tool(name: str, args: dict, context: dict) -> str:
    if name == "analyze_anomaly":
        rate = args["anomaly_rate"]
        severity = "critical" if rate > 0.1 else "warning" if rate > 0.05 else "normal"
        context["severity"] = severity
        return json.dumps({
            "severity": severity,
            "interpretation": f"이상률 {rate*100:.1f}% — {'즉시 조치 필요' if severity == 'critical' else '모니터링 강화' if severity == 'warning' else '정상 범위'}",
            "high_risk_count": args["high_risk_count"]
        }, ensure_ascii=False)

    elif name == "diagnose_root_cause":
        process = args["primary_process"]
        context["primary_process"] = process
        return json.dumps({
            "diagnosis": f"{process} 공정에서 이상 징후 감지",
            "sensor_id": args["top_sensor_id"],
            "impact_level": "높음" if args["shap_score"] > 0.02 else "중간",
            "detail": f"SHAP 점수 {args['shap_score']:.4f} — 해당 센서가 불량 예측에 가장 큰 영향"
        }, ensure_ascii=False)

    elif name == "get_action_plan":
        process = args["process"]
        severity = args["severity"]
        actions = ACTION_DB.get(process, ["설비 전반 점검 필요"])
        context["actions"] = actions
        return json.dumps({
            "process": process,
            "priority": "즉시 조치" if severity == "critical" else "모니터링",
            "actions": actions
        }, ensure_ascii=False)

    elif name == "generate_report":
        report = f"""## FAB 이상 진단 리포트
**생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 1. 이상 상황 요약
{args['summary']}

### 2. 근본 원인 분석
{args['root_cause']}

### 3. 권장 조치 순서
{args['actions']}

### 4. 추가 모니터링 권고
{args['monitoring']}
"""
        context["report"] = report
        return json.dumps({"status": "success", "report_length": len(report)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── ReAct Agent ────────────────────────────────────────
class FabReActAgent:
    """LLM이 Tool 호출 여부를 스스로 판단하는 ReAct Agent"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.context = {}

    def run(self, detection_data: dict, diagnosis_data: dict) -> dict:
        causes = diagnosis_data.get("root_causes", [{}])
        top = causes[0] if causes else {}

        system_prompt = """당신은 반도체 FAB 운영 AI Agent입니다.
주어진 데이터를 분석하여 반드시 다음 순서로 Tool을 호출하세요:
1. analyze_anomaly — 이상 심각도 판단
2. diagnose_root_cause — 근본 원인 진단
3. get_action_plan — 조치 계획 수립
4. generate_report — 최종 리포트 생성
각 Tool의 결과를 확인한 후 다음 Tool을 호출하세요."""

        user_message = f"""FAB 센서 데이터 분석을 시작하세요.

탐지 데이터:
- 전체 샘플: {detection_data['total_count']}개
- 이상 탐지: {detection_data['anomaly_count']}개
- 이상률: {detection_data['anomaly_rate']}
- 고위험 샘플: {detection_data['high_risk_count']}개
- 평균 위험도: {detection_data['avg_risk_score']}

진단 데이터:
- 주요 공정: {diagnosis_data.get('primary_process', 'UNKNOWN')}
- 최고 영향 센서 ID: {top.get('sensor_id', 0)}
- SHAP 점수: {top.get('shap_score', 0)}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        tool_calls_log = []
        max_iterations = 6

        for i in range(max_iterations):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.2
            )

            msg = response.choices[0].message

            # Tool 호출 없으면 종료
            if not msg.tool_calls:
                break

            messages.append(msg)

            # Tool 실행
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args, self.context)
                tool_calls_log.append({
                    "step": i + 1,
                    "tool": tc.function.name,
                    "args": args,
                    "result": json.loads(result)
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        return {
            "tool_calls_log": tool_calls_log,
            "report": self.context.get("report", "리포트 생성 실패"),
            "severity": self.context.get("severity", "unknown"),
            "primary_process": self.context.get("primary_process", "UNKNOWN"),
            "actions": self.context.get("actions", []),
        }


# ── 기존 호환성 유지 (FabAgentPipeline) ───────────────
class DetectionAgent:
    def run(self, anomaly_scores, risk_scores, threshold=0.0):
        anomaly_flags = (anomaly_scores < threshold).astype(int)
        return {
            "anomaly_count": int(anomaly_flags.sum()),
            "total_count": len(anomaly_flags),
            "anomaly_rate": round(float(anomaly_flags.mean()), 4),
            "avg_risk_score": round(float(risk_scores.mean()), 4),
            "high_risk_count": int((risk_scores >= 0.7).sum()),
        }

class DiagnosisAgent:
    def run(self, top5_df):
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
        top_process = causes[0]["process"] if causes else "UNKNOWN"
        return {
            "root_causes": causes,
            "primary_process": top_process,
            "affected_stages": list(set(c["stage"] for c in causes)),
        }

class ActionAgent:
    ACTION_DB = ACTION_DB

    def run(self, diagnosis):
        primary = diagnosis["primary_process"]
        actions = self.ACTION_DB.get(primary, ["설비 전반 점검 필요"])
        return {
            "primary_process": primary,
            "recommended_actions": actions,
            "priority": "즉시 조치" if diagnosis.get("root_causes", [{}])[0].get("shap_score", 0) > 0.02 else "모니터링",
        }

class ReportAgent:
    def run(self, detection, diagnosis, action):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        causes_text = "\n".join([
            f"  - {c['label']} (공정: {c['process']}, SHAP: {c['shap_score']})"
            for c in diagnosis["root_causes"]
        ])
        prompt = f"""당신은 반도체 FAB 운영 AI 어시스턴트입니다.
[탐지 결과]
- 전체 샘플: {detection['total_count']}개
- 이상 탐지: {detection['anomaly_count']}개 ({detection['anomaly_rate']*100:.1f}%)
- 고위험 샘플: {detection['high_risk_count']}개
[근본 원인 분석]
주요 영향 공정: {diagnosis['primary_process']}
핵심 센서:\n{causes_text}
[권장 조치]
{chr(10).join(f'{i+1}. {a}' for i, a in enumerate(action['recommended_actions']))}
FAB 운영자를 위한 간결한 이상 분석 리포트를 한국어로 작성하세요."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.3
        )
        return response.choices[0].message.content

class FabAgentPipeline:
    def __init__(self):
        self.detection_agent = DetectionAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.action_agent = ActionAgent()
        self.report_agent = ReportAgent()
        self.react_agent = FabReActAgent()

    def run(self, anomaly_scores, risk_scores, top5_df):
        detection = self.detection_agent.run(anomaly_scores, risk_scores)
        diagnosis = self.diagnosis_agent.run(top5_df)
        action = self.action_agent.run(diagnosis)

        # ReAct Agent 실행
        react_result = self.react_agent.run(detection, diagnosis)

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
            "report": react_result["report"],
            "react_log": react_result["tool_calls_log"],
            "log": log_entry,
        }

    def _save_log(self, entry):
        log_path = "data/raw/operation_log.csv"
        df_new = pd.DataFrame([entry])
        if os.path.exists(log_path):
            df_old = pd.read_csv(log_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(log_path, index=False)
