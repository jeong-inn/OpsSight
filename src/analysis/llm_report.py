import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generate_report(top5_sensors, anomaly_count, total_count):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sensor_list = "\n".join([
        f"- 센서 {row['sensor']}: SHAP score {row['shap_score']:.4f}"
        for row in top5_sensors
    ])

    prompt = f"""
당신은 반도체 제조 설비 운영 및 데이터 분석 전문가입니다.
아래는 공정 이상 탐지 분석 결과입니다.

[분석 결과]
- 전체 샘플 수: {total_count}개
- 이상 탐지 샘플 수: {anomaly_count}개 ({anomaly_count/total_count*100:.1f}%)

[이상에 영향을 미친 Top 5 센서 (SHAP 기준)]
{sensor_list}

위 결과를 바탕으로 현장 운영자가 즉시 활용할 수 있는 조치 리포트를 작성해주세요.
데이터의 센서명은 보안상 익명화(숫자)되어 있으므로, 특정 물리량(온도, 압력 등)으로
단정 짓지 말고 통계적 중요도와 조치 우선순위 관점에서 서술해야 합니다.

리포트는 아래 형식으로 작성해주세요:

1. 이상 상황 요약 (탐지된 불량 비율 및 현재 상태 요약)
2. 주요 원인 분석 (SHAP 스코어가 높은 핵심 센서들의 기여도 설명)
3. 권장 점검 순서 (운영자가 실제로 취해야 할 3단계 액션 플랜)
4. 추가 모니터링 권고사항 (재발 방지를 위한 조언)

한국어로, 현장 엔지니어가 바로 읽고 조치할 수 있도록 명확하고 전문적인 톤으로 작성해주세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    top5_df = pd.read_csv("data/raw/top5_sensors.csv")
    top5_sensors = top5_df.to_dict('records')

    y = pd.read_csv("data/raw/y.csv").squeeze()
    y = -y
    anomaly_count = int((y == -1).sum())
    total_count = len(y)

    print("LLM 리포트 생성 중...")
    report = generate_report(top5_sensors, anomaly_count, total_count)

    print("\n" + "="*50)
    print("운영자 이상 분석 리포트")
    print("="*50)
    print(report)

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n리포트 저장 완료: data/raw/report.txt")