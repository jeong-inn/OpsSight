import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def compute_spc(X, y, feature_idx=0):
    sensor = X.iloc[:, feature_idx]
    normal_sensor = sensor[y == 1]
    mean = normal_sensor.mean()
    std = normal_sensor.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    return sensor, mean, ucl, lcl

def detect_anomalies_spc(X, y, threshold=3):
    """
    벡터화 연산 기반 SPC 이상 탐지 (정상 데이터 기준 UCL/LCL 계산)
    threshold: 몇 개 이상 센서가 범위 이탈 시 이상 판정
    """
    normal_X = X[y == 1]
    means = normal_X.mean()
    stds = normal_X.std()

    ucls = means + 3 * stds
    lcls = means - 3 * stds

    out_of_control = (X > ucls) | (X < lcls)

    # std==0 센서 제외
    valid_cols = stds[stds > 0].index
    out_of_control = out_of_control[valid_cols]

    anomaly_counts = out_of_control.sum(axis=1)
    anomaly_flags = anomaly_counts >= threshold

    return anomaly_flags, anomaly_counts

def plot_spc(X, y, feature_idx=0, save_path="data/raw/spc_chart.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sensor, mean, ucl, lcl = compute_spc(X, y, feature_idx)
    col_name = X.columns[feature_idx]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ['red' if label == -1 else 'steelblue' for label in y]
    ax.scatter(range(len(sensor)), sensor, c=colors, s=10, alpha=0.6)

    ax.axhline(mean, color='green', linewidth=1.5, linestyle='--')
    ax.axhline(ucl, color='red', linewidth=1.5, linestyle='--')
    ax.axhline(lcl, color='red', linewidth=1.5, linestyle='--')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', label='정상'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='이상'),
        Line2D([0], [0], color='green', linestyle='--', label=f'평균: {mean:.2f}'),
        Line2D([0], [0], color='red', linestyle='--', label=f'UCL/LCL: ±{ucl-mean:.2f}'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f'SPC 관리도 - 센서 {col_name}', fontsize=14)
    ax.set_xlabel('샘플 인덱스')
    ax.set_ylabel('센서값 (정규화)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"SPC 차트 저장: {save_path}")

if __name__ == "__main__":
    X = pd.read_csv("data/raw/X_processed.csv")
    y = pd.read_csv("data/raw/y.csv").squeeze()

    anomaly_flags, anomaly_counts = detect_anomalies_spc(X, y, threshold=3)

    print(f"SPC 기반 이상 탐지 샘플 수: {anomaly_flags.sum()}")
    print(f"전체 샘플 수: {len(anomaly_flags)}")
    print(f"실제 이상 샘플 수 (정답): {(y == -1).sum()}")

    plot_spc(X, y, feature_idx=0)
    print("SPC 완료!")
    