import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_raw_data():
    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    url_labels = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

    print("데이터 다운로드 중...")
    X = pd.read_csv(url_data, delim_whitespace=True, header=None)
    y = pd.read_csv(url_labels, delim_whitespace=True, header=None)
    y = y[0]  # 라벨 추출 (1: 정상, -1: 이상)
    return X, y

def preprocess(X, y):
    print(f"원본 shape: {X.shape}")

    # 1. 결측치 50% 이상 컬럼 제거
    missing_ratio = X.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index
    X = X.drop(columns=cols_to_drop)
    print(f"결측치 50% 이상 컬럼 제거 후: {X.shape}")

    # 2. 남은 결측치 중앙값 대체
    X = X.fillna(X.median())
    print(f"남은 결측치 수: {X.isnull().sum().sum()}")

    # 3. 분산 0 컬럼 제거
    zero_var_cols = X.columns[X.var() == 0]
    X = X.drop(columns=zero_var_cols)
    print(f"분산 0 컬럼 제거 후: {X.shape}")

    # 4. 정규화
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("정규화 완료")

    return X_scaled, y, scaler

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    X, y = load_raw_data()
    X_processed, y, scaler = preprocess(X, y)

    print("\n최종 데이터 shape:", X_processed.shape)
    print("정상 샘플 수 (1):", (y == 1).sum())
    print("이상 샘플 수 (-1):", (y == -1).sum())

    X_processed.to_csv("data/raw/X_processed.csv", index=False)
    y.to_csv("data/raw/y.csv", index=False)
    print("\n전처리 완료. 파일 저장됨.")