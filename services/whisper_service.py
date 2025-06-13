import os
import csv
import re
import queue
import threading
import subprocess
import warnings
from datetime import datetime

import socketio
import numpy
import whisper
from dotenv import load_dotenv

from .logger import save_transcription_log
from services.process_registry import active_processes
from services.ad_predictor import predict_ad

load_dotenv()
BACKEND_API_URL = os.getenv("BACKEND_API_URL")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

LOG_FILE_PATH = "data/whisper_logs.csv"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

def save_transcription_log(text, userId):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE_PATH, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, userId, text])

model = whisper.load_model("base", device="cpu")
text_queue = queue.Queue()

socketio = socketio.Client()

@socketio.event(namespace="/whisper")
def connect():
    print("Whisper namespace connected")

@socketio.event(namespace="/whisper")
def disconnect():
    print("Whisper namespace disconnected")

socketio.connect(BACKEND_API_URL, namespaces=["/whisper"])

def transcribe_radio_stream(url, userId, channelId):
  def worker():
    try:
      process = subprocess.Popen(
        [
          "ffmpeg", "-i", url,
          "-af", "highpass=f=200, lowpass=f=3000, afftdn, dynaudnorm",
          "-f", "wav", "-acodec", "pcm_s16le",
          "-ar", "16000", "-ac", "1", "-loglevel", "quiet", "-"
        ],
        stdout=subprocess.PIPE
      )
      active_processes[userId] = process

      sample_rate = 16000
      chunk_duration = 5
      chunk_size = sample_rate * 2 * chunk_duration

      while True:
        data = process.stdout.read(chunk_size)
        if not data:
          break

        audio = numpy.frombuffer(data, numpy.int16).astype(numpy.float32) / 32768.0
        result = model.transcribe(audio, language="ko")
        text = result.get("text", "").strip()

        if text:
          clear_spaces_text = re.sub(r"\s+", "", text)
          save_transcription_log(clear_spaces_text, userId)

          # 광고 예측
          timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          prediction_result = predict_ad(timestamp, clear_spaces_text)

          # 소켓 전송
          socketio.emit("transcribedRadioText", {
            "text": clear_spaces_text,
            "userId": userId,
            "channelId": channelId,
            "isAd": prediction_result["isAd"],
            "confidence": prediction_result["confidence"]
          }, namespace="/whisper")

          print(f"[추출 결과] {text} | 광고 여부: {prediction_result['label']} ({prediction_result['confidence']:.2%})")

    except Exception as e:
      print(f"[whisper_service] 오류: {e}")

  thread = threading.Thread(target=worker, daemon=True)
  thread.start()
  