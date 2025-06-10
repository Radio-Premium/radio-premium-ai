import csv
from datetime import datetime
import os

LOG_FILE_PATH = "data/whisper_logs.csv"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

def save_transcription_log(text, userId):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE_PATH, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, userId, text])