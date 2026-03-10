import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
import os

def train_isolation_forest(X_train, contamination=0.07):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)

    print("\n=== Isolation Forest 성능 평가 (Test Data) ===")
    print(classification_report(y_test, predictions,
                                  target_names=['이상(-1)', '정상(1)']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # ROC-AUC (이상=-1 → 1, 정상=1 → 0으로 변환)
    y_binary = np.where(y_test == -1, 1, 0)
    roc_auc = roc_auc_score(y_binary, -scores)
    avg_precision = average_precision_score(y_binary, -scores)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")

    return predictions, scores, roc_auc

def plot_roc_pr(scores, y_test, save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    y_binary = np.where(y_test == -1, 1, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_binary, -scores)
    auc = roc_auc_score(y_binary, -scores)
    axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_binary, -scores)
    ap = average_precision_score(y_binary, -scores)
    axes[1].plot(recall, precision, color='darkorange', lw=2, label=f'AP = {ap:.4f}')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "if_roc_pr.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC/PR 차트 저장: {save_path}")

def plot_anomaly_scores(scores, y_test, save_path="data/raw/if_scores.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['red' if label == -1 else 'steelblue' for label in y_test]
    ax.scatter(range(len(scores)), scores, c=colors, s=10, alpha=0.6)
    ax.axhline(0, color='orange', linewidth=1.5, linestyle='--',
               label='Decision Boundary')
    ax.set_title('Isolation Forest Anomaly Score')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"IF 스코어 차트 저장: {save_path}")

if __name__ == "__main__":
    X = pd.read_csv("data/raw/X_processed.csv")
    y = pd.read_csv("data/raw/y.csv").squeeze()

    y = -y  # SECOM 라벨 → Scikit-Learn 표준으로 변환 (-1:정상, 1:이상 → 1:정상, -1:이상)


    # Train/Test Split (stratify로 이상 비율 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Test 이상 샘플 수: {(y_test == -1).sum()}")

    print("\nIsolation Forest 학습 중...")
    contamination = (y_train == -1).sum() / len(y_train)
    print(f"contamination 설정값: {contamination:.4f}")
    model = train_isolation_forest(X_train, contamination=contamination)
    predictions, scores, roc_auc = evaluate_model(model, X_test, y_test)

    plot_anomaly_scores(scores, y_test)
    plot_roc_pr(scores, y_test)

    print("\nIsolation Forest 완료!")