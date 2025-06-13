from datetime import datetime
import joblib
import pandas as pd

from services.constants import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict_ad(timestamp: str, text: str) -> dict:
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    X_input = pd.DataFrame([{
        "text": text,
        "hour": dt.hour,
        "minute": dt.minute
    }])

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][prediction]

    return {
        "isAd": bool(prediction),
        "confidence": float(probability),
        "label": "광고" if prediction == 1 else "일반 멘트"
    }