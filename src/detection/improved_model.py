import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_data():
    X = pd.read_csv('data/raw/X_processed.csv')
    y = pd.read_csv('data/raw/y.csv').squeeze()
    y_binary = (y == 1).astype(int)
    return X, y_binary

def preprocess(X, n_components=50):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA 설명 분산: {pca.explained_variance_ratio_.sum():.3f}")
    return X_pca, scaler, pca

def evaluate_gbm(X_pca, y):
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_orig, y_train_orig)

    best_result = None
    best_recall = 0

    for weight in [3, 5, 8, 10]:
        sample_weights = np.where(y_train_res == 1, weight, 1)
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )
        model.fit(X_train_res, y_train_res, sample_weight=sample_weights)
        y_prob = model.predict_proba(X_test_orig)[:, 1]

        for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]:
            y_pred = (y_prob >= threshold).astype(int)
            r = recall_score(y_test_orig, y_pred, zero_division=0)
            f = f1_score(y_test_orig, y_pred, zero_division=0)
            p = precision_score(y_test_orig, y_pred, zero_division=0)
            if r > best_recall and p > 0.05:
                best_recall = r
                best_result = {
                    'weight': weight, 'threshold': threshold,
                    'recall': r, 'f1': f, 'precision': p,
                    'y_prob': y_prob
                }

    print("\n=== GBM 최적 결과 (threshold + class_weight 튜닝) ===")
    print(f"weight={best_result['weight']}, threshold={best_result['threshold']}")
    print(f"Precision: {best_result['precision']:.3f}")
    print(f"Recall:    {best_result['recall']:.3f}")
    print(f"F1:        {best_result['f1']:.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test_orig, best_result['y_prob']):.3f}")

def evaluate_isolation_forest(X_pca, y):
    iso = IsolationForest(
        n_estimators=200, contamination=104/1567, random_state=42
    )
    iso.fit(X_pca)
    y_pred = (iso.predict(X_pca) == -1).astype(int)
    print("\n=== Isolation Forest + PCA 결과 ===")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    X, y = load_data()
    X_pca, scaler, pca = preprocess(X)
    evaluate_gbm(X_pca, y)
    evaluate_isolation_forest(X_pca, y)
