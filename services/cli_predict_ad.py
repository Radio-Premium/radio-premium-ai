import sys
from datetime import datetime

from services.predict_ad import predict_ad

if len(sys.argv) != 3:
    print("사용법: python services/cli_predict_ad.py '<timestamp>' '<text>'")
    print("예시: python services/cli_predict_ad.py '2025-06-10 12:34:56' '선팅할인행사중입니다'")
    sys.exit(1)

timestamp = sys.argv[1]
text = sys.argv[2]

try:
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    time_only = dt.strftime("%H:%M:%S")
except ValueError:
    print("timestamp 형식이 잘못되었습니다. 예: 2025-06-10 12:34:56")
    sys.exit(1)

result = predict_ad(timestamp, text)

print("시간:", time_only)
print("멘트:", text)
print("예측 결과:", result["label"])
print(f"신뢰도: {result['confidence']:.2%}")