from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model/isolation_forest.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/url_map.pkl", "rb") as f:
    url_map = pickle.load(f)

MAX_SEQ_LEN = 10

@app.route('/api/decision', methods=['POST'])
def decision():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        url_sequence = data.get('urlSequence')

        if not user_id:
            print("요청에 userId가 없습니다.")
            return "F"

        if not url_sequence:
            print("요청에 urlSequence가 없습니다.")
            return "F"

        # URL 문자열을 숫자로 변환
        encoded = [url_map.get(url, -1) for url in url_sequence]
        print(f"Encoded URL sequence: {encoded}")

        # 시퀀스를 길이에 맞게 자르거나 패딩
        if len(encoded) > MAX_SEQ_LEN:
            encoded = encoded[-MAX_SEQ_LEN:]
            print(f"시퀀스가 너무 깁니다. 최근 {MAX_SEQ_LEN}개로 자릅니다.")
        else:
            encoded = [-1] * (MAX_SEQ_LEN - len(encoded)) + encoded
            print(f"시퀀스가 짧습니다. 앞쪽에 -1로 패딩합니다.")

        print(f"최종 입력 시퀀스: {encoded}")

        scaled = scaler.transform([encoded])
        prediction = model.predict(scaled)

        print(f"예측 결과: {prediction[0]}")
        return "T" if prediction[0] == 1 else "F"

    except Exception as e:
        print(f"서버 오류 발생: {str(e)}")
        return "F"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
