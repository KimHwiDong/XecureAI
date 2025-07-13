import pickle
import pandas as pd

# 예측 대상 URL 정보 (유저가 접근한 페이지 등)
user_id = "user1"
url_path = "/settings/profile"
url_len = len(url_path)
slash_count = url_path.count("/")
visit_count = 5  # 예: DB에서 조회하거나 평균값 사용

# 전처리기 및 모델 로드
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/isolation_forest.pkl", "rb") as f:
    model = pickle.load(f)

# 특징 벡터 생성 및 스케일링
X_test = pd.DataFrame([[url_len, slash_count, visit_count]], columns=["url_len", "slash_count", "visit_count"])
X_scaled = scaler.transform(X_test)

# 예측 (1: 정상, -1: 이상치)
raw_pred = model.predict(X_scaled)[0]
result = 0 if raw_pred == 1 else 1

# 결과 출력
print("\n✅ 예측 결과:")
print("정상 접근" if result == 0 else "비정상 접근")
