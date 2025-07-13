import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

# 1. 로그 CSV 파일 불러오기 (user_id, url_path, access_time 포함)
df = pd.read_csv("user_logs.csv")

# 2. 시간 순으로 정렬
df = df.sort_values(by=['user_id', 'access_time'])

# 3. URL을 숫자로 변환 (예: dashboard → 0, account → 1, ...)
url_map = {url: idx for idx, url in enumerate(df['url_path'].unique())}
df['url_code'] = df['url_path'].map(url_map)

# 4. 사용자별 URL 시퀀스 생성
MAX_SEQ_LEN = 10
def encode_sequence(group):
    codes = group['url_code'].tolist()
    return codes[:MAX_SEQ_LEN] + [-1]*(MAX_SEQ_LEN - len(codes))

sequences = []
user_ids = []

for user_id, group in df.groupby('user_id'):
    if len(group) < 2:
        continue
    encoded = encode_sequence(group)
    sequences.append(encoded)
    user_ids.append(user_id)

X = np.array(sequences)

# 5. 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Isolation Forest 모델 학습
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

# 7. 모델 및 스케일러 저장
os.makedirs("model", exist_ok=True)
with open("model/isolation_forest.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model/url_map.pkl", "wb") as f:
    pickle.dump(url_map, f)

print("✅ 모델 학습 완료 및 저장 완료.")
