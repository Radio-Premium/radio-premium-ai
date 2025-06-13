import glob
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

csv_files = glob.glob("data/whisper_labeled_*.csv")
df_list = [pd.read_csv(file, header=None, names=["timestamp", "id", "text", "label"], quotechar='"') for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

def is_irregular_ad(row):
    minute = row["minute"]
    return ((0 <= minute < 27) or (30 <= minute < 57)) and row["label"] == 1

irregular_ads = df[df.apply(is_irregular_ad, axis=1)]

boosted_ads = pd.concat([irregular_ads] * 2, ignore_index=True)

df = pd.concat([df, boosted_ads], ignore_index=True)

X_text = df["text"]
X_time = df[["hour", "minute"]]
y = df["label"]

classes = np.array(sorted(df["label"].unique()))
weights = compute_class_weight("balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))

classes = np.array(sorted(df["label"].unique()))
weights = compute_class_weight("balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(ngram_range=(1, 2)), "text"),
        ("time", OneHotEncoder(handle_unknown='ignore'), ["hour", "minute"])
    ],
    transformer_weights={
        "text": 2.0,
        "time": 0.1
    }
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=class_weights[0] / class_weights[1],
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    ))
])

X_combined = pd.concat([X_text, X_time], axis=1)
model.fit(X_combined, y)

y_proba = model.predict_proba(X_combined)[:, 1]
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)
print(f"Classification Report (threshold={threshold}):")
print(classification_report(y, y_pred, target_names=["Not Ad", "Ad"]))

wrong = df[y != y_pred]
print("\n잘못 분류된 샘플:")
print(wrong[["timestamp", "text", "label"]].head(10))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ad_classifier.pkl")
print("XGBoost 광고 탐지 모델이 저장되었습니다! → models/ad_classifier.pkl")
