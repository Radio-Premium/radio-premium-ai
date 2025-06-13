import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

whisper_labeled_csv_files = glob.glob("data/whisper_labeled_*.csv")

df_list = []
for file in whisper_labeled_csv_files:
    df_temp = pd.read_csv(file, header=None, names=["timestamp", "id", "text", "label"], quotechar='"')
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)

df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
X_text = df["text"]
X_time = df[["hour", "minute"]]
y = df["label"]

classes = np.array(sorted(df["label"].unique()))
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(ngram_range=(1, 2)), "text"),
        ("time", OneHotEncoder(handle_unknown='ignore'), ["hour", "minute"])
    ],
    transformer_weights={
        "text": 2.0,
        "time": 1.0
    }
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight=class_weights))
])

X_combined = pd.concat([X_text, X_time], axis=1)
model.fit(X_combined, y)

y_pred = model.predict(X_combined)
print("📊 Classification Report:")
print(classification_report(y, y_pred, target_names=["Not Ad", "Ad"]))

wrong = df[y != y_pred]
print("\n❌ 잘못 분류된 샘플:")
print(wrong[["timestamp", "text", "label"]].head(10))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/ad_classifier.pkl")
print("🎉 Logistic Regression 광고 탐지 모델이 저장되었습니다! → models/ad_classifier.pkl")