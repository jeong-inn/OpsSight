# 🔬 FabSight
**Smart Semiconductor Fab Monitoring & Anomaly Diagnosis System**

> 반도체 설비 센서 데이터 기반 이상 감지 · 고장 위험 예측 · Agent 진단 · 운영 관제 플랫폼

---


##  프로젝트 개요

반도체 제조 공정에서 설비 이상은 수율 저하와 직결됩니다.  
FabSight는 SECOM 공정 데이터(590개 센서, 1,567 샘플)를 기반으로  
**통계 기반 이상탐지 → ML 이상탐지 → 고장 위험 예측 → Agent 자동 진단**까지  
Autonomous Fab 관점의 AI 운영 시스템을 구현한 프로젝트입니다.

---


##  시스템 아키텍처
```
📡 SECOM Sensor Data (590 sensors, 1,567 samples)
           │
           ▼
   ┌─────────────────┐
   │   Preprocessing  │  결측치 처리, 분산 0 제거, StandardScaler
   └────────┬────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐   ┌──────────────────┐
│   SPC   │   │ Isolation Forest  │  비지도 ML 이상탐지
│ 관리도  │   └────────┬─────────┘
└─────────┘            │
                       ▼
              ┌─────────────────────┐
              │ Pre-failure Risk     │  GBM 기반 고장 위험도 예측
              │ Scoring (GBM)        │  (0~1 확률 출력)
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  SHAP Feature        │  공정별 핵심 센서 추출
              │  Importance          │  CVD / ETCH / CMP / LITHO
              └────────┬────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │      Agent Pipeline          │
        │  Detection → Diagnosis       │
        │  → Action → Report           │
        └────────┬─────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Streamlit        │  FAB 모니터링 / 운영 로그
        │ Dashboard        │  / 모델 비교 / Agent 리포트
        └─────────────────┘
```
![FabSight Architecture](docs/architecture.png)

---


## 주요 기능

###  FAB 공정 상태 모니터링
- LITHO / CVD / ETCH / CMP 공정별 실시간 설비 상태 카드
- 정상 / 경고 / 이상 3단계 색상 표시
- 공정별 위험도 비교 차트

###  SPC 관리도 (Statistical Process Control)
- 정상 데이터 기준 3-sigma UCL/LCL 계산
- 센서별 관리 한계 이탈 샘플 시각화

###  이상 탐지 & Pre-failure Risk Scoring
- **Isolation Forest**: 비지도 학습 기반 이상탐지
- **GBM Risk Scorer**: 고장 위험도 0~1 확률 예측
- HIGH / MEDIUM / LOW 위험 등급 분류
- 모델 성능 비교표 (Precision / Recall / F1 / ROC-AUC)

###  핵심 센서 분석 (SHAP + 공정별 기여도)
- SHAP Feature Importance 기반 Top 5 센서 추출
- 센서 번호 → 공정명 매핑 (CVD / ETCH / CMP / LITHO)
- 센서 중요도 → 공정 기여도로 집계 (Anomaly Contribution Analysis)

**공정별 이상 기여도 분석 결과:**

| 공정 | 기여도 | 핵심 센서 |
|---|---|---|
| CVD | 40.3% | Sensor_31, Sensor_419 |
| ETCH | 22.0% | Sensor_487 |
| CMP | 19.8% | Sensor_545 |
| LITHO | 17.9% | Sensor_59 |

> CVD 공정이 전체 이상의 40.3%를 차지 → 조치 우선순위 1순위
> 단순 센서 중요도에서 공정 단위 분석으로 확장하여 운영자 의사결정 지원

###  Agent 기반 이상 진단 (4단계 파이프라인)
| Agent | 역할 | 입력 | 출력 |
|---|---|---|---|
| Detection Agent | 이상 탐지 결과 취합 | 센서 데이터 | anomaly count, risk score |
| Diagnosis Agent | 근본 원인 분석 | SHAP + 공정 매핑 | root cause sensors |
| Action Agent | 조치 우선순위 추천 | 진단 결과 | 공정별 점검 항목 |
| Report Agent | 운영자 리포트 생성 | 전체 분석 결과 | LLM 자동 리포트 |

###  운영 로그
- Agent 실행 이력 자동 저장 (timestamp, 이상 수, 주요 공정, 우선순위)

---


##  기술 스택

| 분류 | 기술 |
|---|---|
| Language | Python 3.9 |
| ML/AI | Scikit-learn, SHAP, GradientBoosting |
| LLM | OpenAI GPT-4o-mini |
| Dashboard | Streamlit |
| Data | SECOM Dataset (UCI ML Repository) |

---


##  모델 성능

### 최종 모델 성능

| 모델 | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| SPC (3-sigma) | - | - | - | - |
| Isolation Forest + PCA | 0.120 | 0.120 | 0.120 | - |
| Risk Scorer (GBM + SMOTE + PCA + Threshold) | 0.079 | 0.476 | 0.135 | 0.584 |

---

### 실험 설계 (GBM Ablation Study)

클래스 불균형(14:1) 문제를 단계적으로 해결한 실험 결과:

| Setting | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (GBM) | 0.250 | 0.095 | 0.138 | 0.729 |
| + SMOTE | 0.333 | 0.048 | 0.083 | 0.652 |
| + SMOTE + PCA | 0.182 | 0.095 | 0.125 | 0.535 |
| + SMOTE + PCA + Threshold Tuning | 0.079 | **0.476** | 0.135 | 0.584 |

**설계 근거:**
- SMOTE 단독: 고차원(446차원)에서 합성 샘플 품질 저하 → Recall 오히려 감소
- PCA 추가: 50차원으로 축소 후 SMOTE 효과 회복
- Threshold Tuning: 운영 임계값 0.2 + class weight 8 적용으로 Recall 0.095 → 0.476 (5배 개선)
- **ROC-AUC 하락은 의도된 트레이드오프**: 반도체 공정에서 미탐지(False Negative)는 수율 손실로 직결되므로 Recall 최대화 전략 선택. 오탐지 비용 < 미탐지 비용.

---


##  실행 방법
```bash
# 1. 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env에 OPENAI_API_KEY 입력

# 3. 전처리 실행
python src/preprocessing/preprocess.py

# 4. SHAP 분석
python src/analysis/feature_importance.py

# 5. 대시보드 실행
PYTHONPATH=$(pwd) streamlit run src/dashboard/app.py
```

---


## 📁 프로젝트 구조
```
fabsight/
├── data/raw/              # 처리된 데이터 및 분석 결과
├── src/
│   ├── preprocessing/     # 데이터 전처리
│   ├── detection/         # SPC, Isolation Forest
│   ├── analysis/          # SHAP Feature Importance
│   ├── prediction/        # Pre-failure Risk Scorer (GBM)
│   ├── agents/            # Agent Pipeline (4단계)
│   ├── simulator/         # 센서 스트림 시뮬레이터
│   ├── dashboard/         # Streamlit 앱
│   └── process_map.py     # 공정 매핑 테이블
├── .env.example
├── requirements.txt
└── README.md
```

---
##  운영 시나리오

### Scenario 1 — CVD 공정 이상 감지 및 자동 진단
```
1. CVD 챔버 압력 센서(Sensor_31) 드리프트 발생
2. Isolation Forest → 이상 샘플 탐지
3. GBM Risk Scorer → 위험도 0.85 (HIGH)
4. SHAP 분석 → CVD 공정 기여도 40.3% 확인
5. ReAct Agent → analyze_anomaly → diagnose_root_cause → get_action_plan 순서로 자동 판단
6. 운영자에게 "챔버 압력 센서 점검 / 가스 유량 컨트롤러 확인" 조치 리포트 전달
```

### Scenario 2 — ETCH 공정 위험도 상승 모니터링
```
1. ETCH 플라즈마 관련 센서(Sensor_487) 이상 징후
2. SPC 관리도 → UCL 이탈 감지
3. Risk Score 점진적 상승 → WARNING 알림 발생
4. Digital Twin Simulator → ETCH 공정 drift 누적 시뮬레이션
5. 사전 예방 조치 → "RF 매칭 네트워크 점검 / 가스 흐름 균일성 체크"
```

### Scenario 3 — 다중 공정 동시 이상
```
1. CVD + CMP 동시 이상 징후 → 고위험 샘플 급증
2. CRITICAL 배너 자동 표시
3. Agent → 복합 근본 원인 분석 → 우선순위 기반 조치 계획 생성
4. 운영 로그 자동 저장 → 이력 추적
```

---

##  설계 의도 및 기술적 고려사항

- **SPC → IF 순서**: 정규분포 가정의 SPC 한계를 보완하기 위해 비지도 ML 기반 Isolation Forest 병행
- **SHAP 도입 이유**: 446차원 고차원 센서 데이터에서 IF 성능 저하(차원의 저주) → 핵심 센서 추출로 해석 가능성 확보
- **Agent 구조 선택**: OpenAI Function Calling 기반 ReAct 패턴 구현. LLM이 analyze_anomaly / diagnose_root_cause / get_action_plan / generate_report Tool을 스스로 판단하여 순서대로 호출하고 각 결과를 컨텍스트로 누적해 최종 리포트 생성.
- **센서 익명화 대응**: 실무 환경에서는 EDD 연동 + LLM RAG 구조로 센서명 자동 변환 아키텍처 확장 가능
- **Risk Scoring 방식**: SECOM은 정적 샘플 데이터로 시계열 예측이 부적합 → 현재 센서 상태 기반 고장 위험도 분류로 설계

##  Limitations & Future Work

### 현재 한계
- **데이터**: SECOM은 공개 데이터셋으로 실제 Fab 환경과 차이 존재. 센서명 익명화로 도메인 해석 제한
- **Agent 구조**: Tool-guided orchestration에 가까움. 완전 자유 탐색형 ReAct는 아님
- **Digital Twin**: 실제 설비 physics 기반이 아닌 경량 파라미터 시뮬레이션
- **모델 성능**: 고차원 데이터 특성상 F1 0.135 수준. Recall 최대화 전략으로 보완

### Future Work
- 실제 Fab 센서 데이터 연동 시 EDD + LLM RAG로 센서명 자동 해석 아키텍처 확장 가능
- LangGraph 기반 멀티 에이전트 협업 구조로 확장
- 시계열 센서 데이터 확보 시 LSTM 기반 예측 모델 추가
- 실제 설비 물리 모델 기반 Digital Twin 고도화

## Demo

![FabSight Demo](docs/fabsight-demo.gif)

> Alert System, FAB 공정 모니터링, SHAP Root Cause Analysis, Agent 진단 파이프라인 시연

---
