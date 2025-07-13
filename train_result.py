import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:
    matplotlib.rc('font', family='NanumGothic')

matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("user_logs.csv")

# 저장 폴더 생성
os.makedirs("user_graphs", exist_ok=True)

# 유저별 반복
for user_id, group in df.groupby('user_id'):
    group = group.sort_values(by='access_time')
    url_sequence = group['url_path'].tolist()

    if len(url_sequence) < 2:
        continue  # 너무 짧은 시퀀스는 생략

    # 그래프 생성
    G = nx.DiGraph()
    for i in range(len(url_sequence) - 1):
        src = url_sequence[i]
        dst = url_sequence[i + 1]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)

    # 그래프 시각화
    plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=1800,
            font_size=10, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"{user_id}의 URL 방문 흐름")
    plt.tight_layout()

    # 이미지 저장
    filename = f"user_graphs/{user_id}_url_graph.png"
    plt.savefig(filename)
    plt.close()
