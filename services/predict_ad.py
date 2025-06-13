import sys
from datetime import datetime

import joblib
import pandas as pd

if len(sys.argv) != 3:
    print("❌ 사용법: python services/predict_ad.py '<timestamp>' '<text>'")
    print("예시: python services/predict_ad.py '2025-06-10 12:34:56' '선팅할인행사중입니다'")
    sys.exit(1)

timestamp = sys.argv[1]
text = sys.argv[2]

try:
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    time_only = dt.strftime("%H:%M:%S")
except ValueError:
    print("❌ timestamp 형식이 잘못되었습니다. 예: 2025-06-10 12:34:56")
    sys.exit(1)

model = joblib.load("models/ad_classifier.pkl")

X_input = pd.DataFrame([{
    "text": text,
    "hour": dt.hour,
    "minute": dt.minute
}])

prediction = model.predict(X_input)[0]
probability = model.predict_proba(X_input)[0][prediction]

label = "광고" if prediction == 1 else "일반 멘트"

print("🕒 시간:", time_only)
print("🗣️ 멘트:", text)
print("🔍 예측 결과:", label)
print(f"📊 신뢰도: {probability:.2%}")