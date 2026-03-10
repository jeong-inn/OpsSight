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

###  핵심 센서 분석 (SHAP)
- SHAP Feature Importance 기반 Top 5 센서 추출
- 센서 번호 → 공정명 매핑 (CVD_Chamber_Pressure 등)

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

| 모델 | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| SPC (3-sigma) | - | - | - | - |
| Isolation Forest | 0.095 | 0.065 | 0.077 | - |
| Risk Scorer (GBM) | 0.077 | 0.032 | 0.045 | 0.73 |

> SECOM 데이터셋 특성상 446개 고차원 센서로 인한 차원의 저주로 탐지 성능이 낮게 나타남.  
> 이를 해결하기 위해 SHAP 기반 핵심 센서 추출 → Agent 진단 구조로 보완.

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

##  설계 의도 및 기술적 고려사항

- **SPC → IF 순서**: 정규분포 가정의 SPC 한계를 보완하기 위해 비지도 ML 기반 Isolation Forest 병행
- **SHAP 도입 이유**: 446차원 고차원 센서 데이터에서 IF 성능 저하(차원의 저주) → 핵심 센서 추출로 해석 가능성 확보
- **Agent 구조 선택**: 단순 LLM 프롬프트 호출이 아닌 역할 분리된 4단계 파이프라인으로 확장성 및 유지보수성 확보
- **센서 익명화 대응**: 실무 환경에서는 EDD 연동 + LLM RAG 구조로 센서명 자동 변환 아키텍처 확장 가능
- **Risk Scoring 방식**: SECOM은 정적 샘플 데이터로 시계열 예측이 부적합 → 현재 센서 상태 기반 고장 위험도 분류로 설계
