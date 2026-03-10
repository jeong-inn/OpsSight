import pandas as pd
import requests
import io

# SECOM 데이터 직접 다운로드
url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
url_labels = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

print("데이터 다운로드 중...")
X = pd.read_csv(url_data, sep=' ', header=None)
y = pd.read_csv(url_labels, sep=' ', header=None)

print("데이터 shape:", X.shape)
print("라벨 분포:")
print(y[0].value_counts())

missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
print("결측치 비율:", round(missing_ratio, 4))