import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

users = [f"user{i}" for i in range(1, 6)]
urls_normal = [
    "/dashboard",
    "/account",
    "/settings/profile",
    "/settings/security",
    "/finance/statement",
    "/transfer",
]
urls_abnormal = [
    "/unauthorized/access",
    "/internal/debug",
    "/error/logs",
    "/api/private/override",
    "/hidden/page",
]

data = []

start_time = datetime.now() - timedelta(days=30)

for user in users:
    # 정상 URL 방문 (반복 방문 많음)
    for url in urls_normal:
        visits = random.randint(3, 10)  # 3~10회 방문 (정상)
        for _ in range(visits):
            timestamp = start_time + timedelta(minutes=random.randint(0, 60*24*30))
            data.append([user, url, timestamp.strftime("%Y-%m-%d %H:%M:%S")])

    # 비정상 URL 방문 (적은 방문 횟수)
    for url in urls_abnormal:
        visits = random.randint(0, 2)  # 0~2회 방문 (비정상)
        for _ in range(visits):
            timestamp = start_time + timedelta(minutes=random.randint(0, 60*24*30))
            data.append([user, url, timestamp.strftime("%Y-%m-%d %H:%M:%S")])

# DataFrame 생성
df = pd.DataFrame(data, columns=["user_id", "url_path", "access_time"])

# CSV 저장
df.to_csv("user_logs.csv", index=False)

print("✅ user_logs.csv 파일 생성 완료")
print(df.head(10))
