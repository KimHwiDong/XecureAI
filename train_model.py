import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os
import numpy as np

# 1. 데이터 로드 (user_id, url_path, access_time 포함)
df = pd.read_csv("user_logs.csv")

# 2. 시퀀스 기반 특징: 사용자별 이동 순서를 시간순으로 정렬
sequence_df = df.sort_values(by=['user_id', 'access_time'])

# 3. 시퀀스를 고유한 패턴으로 치환 (URL을 숫자로 매핑)
url_map = {url: idx for idx, url in enumerate(df['url_path'].unique())}
sequence_df['url_code'] = sequence_df['url_path'].map(url_map)

# 4. 시퀀스 인코딩 함수 정의
def encode_sequence(group):
    return [url_map[url] for url in group['url_path']]

# 5. 사용자별 시퀀스 코드 리스트 생성
user_encoded = []
for user_id, group in sequence_df.groupby('user_id'):
    encoded = encode_sequence(group)
    user_encoded.append({'user_id': user_id, 'encoded_sequence': encoded})
user_encoded = pd.DataFrame(user_encoded)

# 6. 시퀀스를 고정 길이 벡터로 변환 (앞 10개, 부족하면 -1로 패딩)
MAX_SEQ_LEN = 10
def pad_sequence(seq):
    padded = seq[:MAX_SEQ_LEN] + [-1] * (MAX_SEQ_LEN - len(seq))
    return padded

user_encoded['padded_sequence'] = user_encoded['encoded_sequence'].apply(pad_sequence)

X = np.vstack(user_encoded['padded_sequence'].values)

# 7. 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Isolation Forest 모델 학습 (이상치 10% 가정)
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
user_encoded["is_anomaly"] = model.fit_predict(X_scaled)
user_encoded["is_anomaly"] = user_encoded["is_anomaly"].apply(lambda x: 0 if x == 1 else 1)  # 1=이상치, 0=정상

# 9. 모델 및 스케일러 저장
os.makedirs("model", exist_ok=True)
with open("model/seq_isolation_forest.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/seq_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 10. 결과 저장 및 출력
user_encoded[['user_id', 'encoded_sequence', 'is_anomaly']].to_csv("model/seq_isolation_results.csv", index=False)
print("\n✅ 시퀀스 기반 Isolation Forest 모델 학습 완료 및 저장됨.")
print("📄 결과 저장: model/seq_isolation_results.csv")
