import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'AppleGothic'  # Mac 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

def get_shap_values(model, X_test):
    print("SHAP 값 계산 중... (1~2분 소요)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap 버전에 따라 리스트로 반환될 수 있음 → 방어 코드
    if isinstance(shap_values, list):
        shap_values_2d = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_2d = shap_values

    return shap_values_2d

def get_top_sensors(shap_values_2d, feature_names, top_n=10):
    mean_abs_shap = np.abs(shap_values_2d).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    return [(feature_names[i], mean_abs_shap[i]) for i in top_indices]

def plot_shap_summary(shap_values_2d, X_test, save_path="data/raw/shap_summary.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_2d, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP Summary Plot 저장: {save_path}")

def plot_top_sensors_bar(top_sensors, save_path="data/raw/top_sensors.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = [s[0] for s in top_sensors]
    values = [s[1] for s in top_sensors]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], values[::-1], color='steelblue')
    ax.set_title('Top Sensors by SHAP Importance', fontsize=14)
    ax.set_xlabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Top Sensors 차트 저장: {save_path}")

if __name__ == "__main__":
    X = pd.read_csv("data/raw/X_processed.csv")
    y = pd.read_csv("data/raw/y.csv").squeeze()
    y = -y  # SECOM 라벨 변환

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    contamination = (y_train == -1).sum() / len(y_train)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    print("모델 학습 완료")

    shap_values_2d = get_shap_values(model, X_test)

    top_sensors = get_top_sensors(shap_values_2d, list(X_test.columns), top_n=10)
    print("\n=== 이상에 영향을 미치는 Top 10 센서 ===")
    for i, (sensor, score) in enumerate(top_sensors, 1):
        print(f"{i:2d}. 센서 {sensor:>6} | SHAP score: {score:.4f}")

    plot_shap_summary(shap_values_2d, X_test)
    plot_top_sensors_bar(top_sensors)

    top5_df = pd.DataFrame(top_sensors[:5], columns=['sensor', 'shap_score'])
    top5_df.to_csv("data/raw/top5_sensors.csv", index=False)
    print("\nTop 5 센서 저장 완료: data/raw/top5_sensors.csv")
    print("\nFeature Importance 완료!")