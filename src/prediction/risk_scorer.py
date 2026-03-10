# src/prediction/risk_scorer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

class PreFailureRiskScorer:
    """
    설비 상태 기반 고장 위험도 조기 예측 모델
    SECOM 센서 데이터로 이상 발생 가능성(risk score)을 0~1 사이 확률로 출력
    """
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.is_trained = False
        self.metrics = {}

    def train(self, X: pd.DataFrame, y: pd.Series):
        # y: 1=정상, -1=이상 → 0=정상, 1=이상으로 변환
        y_bin = (y == -1).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bin, test_size=0.3, random_state=42, stratify=y_bin
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # 성능 평가
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        self.metrics = {
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 3),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 3),
        }
        return self.metrics

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """각 샘플의 고장 위험도 (0~1) 반환"""
        if not self.is_trained:
            raise ValueError("모델을 먼저 학습시켜주세요.")
        return self.model.predict_proba(X)[:, 1]

    def get_risk_level(self, score: float) -> str:
        if score >= 0.7:
            return "🔴 HIGH"
        elif score >= 0.4:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"
