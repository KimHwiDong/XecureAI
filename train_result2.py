import pandas as pd

def main():
    # 데이터 불러오기
    df = pd.read_csv("user_logs.csv")

    # 유저별로 그룹핑 후 방문 흐름 출력
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values(by='access_time')  # 시간순 정렬
        url_sequence = group['url_path'].tolist()

        if len(url_sequence) < 2:
            continue  # 너무 짧으면 출력 안 함

        # 방문 경로를 " → " 로 연결
        path_str = " → ".join(url_sequence)

        print(f"{user_id} 방문 흐름:")
        print(path_str)
        print("-" * 50)

if __name__ == "__main__":
    main()
